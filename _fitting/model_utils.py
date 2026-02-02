import pymc as pm
import numpy as np
from patsy import dmatrix
import re
import os
import matplotlib.pyplot as plt
import arviz as az
import xarray as xr
import time
import pandas as pd
import warnings
from _fitting.fitting_utils import hist_plot, CI_plot, CI_plot_alt, CI_plot_both, plot_posteriors_side_by_side, plot_spline

###
def abbrev_surveillance(name):
    if name is None:
        return "nosurv"
    base = "surv"
    if "urban" in name:
        base = "urb_surv"
    weight = "p" if "pop_weighted" in name else "u"
    return f"{base}_{weight}"

def abbrev_urbanisation(name):
    if name is None:
        return "nourb"
    base = "urb"
    weight = "p" if "pop_weighted" in name else "u"
    std = "_std" if "std" in name else ""
    return f"{base}_{weight}{std}"

def abbrev_stat(stat):
    # remove spaces
    s = stat.replace(" ", "")
    
    # lag extraction: "(k)"
    lag = re.search(r"\((\d+)\)", s)
    lag_str = f"({lag.group(1)})" if lag else ""
    
    # weighting
    if "pop_weighted" in s:
        w = "p"
    elif "unweighted" in s:
        w = "u"
    else:
        w = ""
    
    # remove weighting + lag from base
    base = re.sub(r"_?(pop_weighted|unweighted).*", "", s)
    
    return f"{base}_{w}{lag_str}"

def model_settings_to_name(settings):
    surv = abbrev_surveillance(settings.get("surveillance_name"))
    urb = abbrev_urbanisation(settings.get("urbanisation_name"))
    
    stats = settings.get("stat_names", [])
    if len(stats) == 0:
        stat_str = "nostat"
    else:
        stat_str = "+".join(abbrev_stat(s) for s in stats)
    
    deg = settings.get("degree")
    k = settings.get("num_knots")
    
    knot_map = {"quantile": "q", "uniform": "u"}
    kt = knot_map.get(settings.get("knot_type"), settings.get("knot_type"))
    
    orth = "o" if settings.get("orthogonal") else "no"
    if len(stats) == 0:
        return f"[{surv} {urb}] [{stat_str}] []"
    else:
        return f"[{surv} {urb}] [{stat_str}] [{deg},{k},{kt},{orth}]"

def settings_to_var_names(model_settings):
    v = ['intercept', 'alpha']
    if model_settings['urbanisation_name'] is not None:
        v = v + ['beta_u']
    v = v + [f'sigma_w({stat_name})' for stat_name in model_settings['stat_names']]
    v = v + [f'w({stat_name})' for stat_name in model_settings['stat_names']]
    return v

def elpd_to_xr(elpd):
    return xr.Dataset(
        {k: xr.DataArray(v) for k, v in elpd.items()}
    )

def elpd_to_row(eval_waic, eval_loo, model_name, data_name):
    return {
        "model_name": model_name,
        "data_name": data_name,

        # WAIC
        "waic": float(eval_waic.elpd_waic),
        #"p_waic": float(eval_waic.p_waic),
        "waic_se": float(eval_waic.se),
        "waic_warning": int(eval_waic.warning),

        # LOO
        "loo": float(eval_loo.elpd_loo),
        #"p_loo": float(eval_loo.p_loo),
        "loo_se": float(eval_loo.se),

        # diagnostics
        "n_pareto_k_bad": int(np.sum(eval_loo.pareto_k>0.7)),
        "n_pareto_k_very_bad": int(np.sum(eval_loo.pareto_k>1)),
        "pareto_k_mean": float(eval_loo.pareto_k.mean()),
    }
###
def model_fit(data, data_name, model_settings, outpath, n_chains=4, n_draws=500, n_tune=500, sampler="nutpie"):
    model_name = model_settings_to_name(model_settings)
    folder_name = f'{data_name}/{model_name}'
    data_path = os.path.join(outpath, f'{data_name}/')
    os.makedirs(data_path, exist_ok=True)
    report_path = os.path.join(outpath, f'{data_name}/reports/')
    os.makedirs(report_path, exist_ok=True)
    output_path = os.path.join(outpath, folder_name)
    os.makedirs(output_path, exist_ok=True)

    model, model_B, model_knot_list = build_model(data.copy(), **model_settings)
    with model:
        s0 = time.time()
        idata = pm.sample(tune=n_tune,
                          draws=n_draws,
                          chains=n_chains,
                          random_seed=42,
                          discard_tuned_samples=True,
                          nuts_sampler=sampler,
                          store_divergences=True,
                          progressbar=False)
        s1 = time.time()
        pm.compute_log_likelihood(idata, progressbar=False)
        s2 = time.time()
        print(f'\nPosterior Sampling {s1 - s0:.2f} seconds')
        print(f'Log Likelihood Compute {s2 - s1:.2f} seconds \n')

    #### Time Metrics
    metrics_df = pd.DataFrame([{
        "model_name": model_name,
        "data_name": data_name,
        "sampling_time_sec": s1 - s0,
        "log_likelihood_time_sec": s2 - s1,
        "n_chains": n_chains,
        "n_draws": n_draws,
        "n_tune": n_tune,
        "sampler": sampler,
    }])
    # inner
    inner_metrics_file = os.path.join(output_path, "model_timings.csv")
    if os.path.exists(inner_metrics_file):
        metrics_df.to_csv(inner_metrics_file, mode="a", header=False, index=False)
    else:
        metrics_df.to_csv(inner_metrics_file, index=False)
    # outer
    outer_metrics_file = os.path.join(data_path, "model_timings.csv")
    if os.path.exists(outer_metrics_file):
        metrics_df.to_csv(outer_metrics_file, mode="a", header=False, index=False)
    else:
        metrics_df.to_csv(outer_metrics_file, index=False)
    ####

    #### WAIC and PSIS LOO
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eval_waic = az.waic(idata)
        eval_psis_loo_elpd = az.loo(idata)
    # dataframes (inner and outer)
    wl_df = pd.DataFrame([elpd_to_row(eval_waic, eval_psis_loo_elpd, model_name, data_name)])
    inner_wl_file = os.path.join(output_path, "model_elpd_metrics.csv")
    outer_wl_file = os.path.join(data_path, "model_elpd_metrics.csv")
    wl_df.to_csv(inner_wl_file, index=False)
    if os.path.exists(outer_wl_file):
        wl_df.to_csv(outer_wl_file, mode="a", header=False, index=False)
    else:
        wl_df.to_csv(outer_wl_file, index=False)

    khat_fig = az.plot_khat(eval_psis_loo_elpd).get_figure()
    fig_file = os.path.join(output_path, f"khat.png")
    khat_fig.savefig(fig_file, bbox_inches="tight")
    plt.close(khat_fig)
    ####

    # Save inference data
    idata_file = os.path.join(output_path, "idata.nc")
    idata.to_netcdf(idata_file)

    # Save summary table
    var_names = settings_to_var_names(model_settings)
    summary_df = az.summary(idata, var_names=var_names)
    summary_file = os.path.join(output_path, "summary.csv")
    summary_df.to_csv(summary_file)

    # Plot and save traces
    trace_axes = az.plot_trace(idata, var_names=var_names)
    # trace_axes is an ndarray of matplotlib Axes
    figs = {ax.get_figure() for ax in trace_axes.ravel()}  # unique figures
    for i, fig in enumerate(figs):
        fig_file = os.path.join(output_path, f"trace.png")
        fig.savefig(fig_file, bbox_inches="tight")
        plt.close(fig)

    # Plot and save splines
    for stat_name in model_settings['stat_names']:
        fig = plot_spline(
            idata,
            stat_name,
            f'w({stat_name})',
            f'sigma_w({stat_name})',
            model_B[stat_name],
            data[stat_name],
            knots=model_knot_list[stat_name],
            show_basis=True,
            basis_scale=8,
            invert_log=False
        );
        if isinstance(fig, plt.Figure):
            fig_file = os.path.join(output_path, f"spline_{stat_name}.png")
            fig.savefig(fig_file, bbox_inches="tight")
            plt.close(fig)

    create_html_report([output_path, report_path], model_name=model_name, n_draws=n_draws)
    return

def ess_style(x, n_draws):
    if isinstance(x, (int, float)):
        if x < n_draws / 5:
            return "background-color: red;"
        elif x < n_draws / 4:
            return "background-color: yellow;"
        else:
            return "background-color: lightgreen;"
    return ""

def create_html_report(folder_path, model_name, n_draws, title=None):
    """
    Generate a simple HTML report by combining CSV tables and images
    in a folder. Assumes files are named:
    - model_timings.csv
    - summary.csv
    - model_elpd_metrics.csv
    - trace.png
    - khat.png
    - spline_*.png
    """
    out_file = []
    for path in folder_path:
        out_file.append(os.path.join(path, f"_report_[{model_name}].html"))
    if title is None:
        title = f"Model Report: {model_name}"

    html_parts = [
        f"<html><head><title>{title}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; font-size: 12px; line-height: 1.2; margin: 8px; text-align:center; }",
        "h1, h2, h3, h4 { margin: 4px 0 8px 0; font-weight: normal; }",
        "table { border-collapse: collapse; font-size: 15px; margin: 0 auto 12px auto; width: 80%; }",
        "table th, table td { border: 1px solid #aaa; padding: 4px 6px; line-height: 1.2; text-align: center; }",
        "img { max-width: 80%; margin: 8px auto; display: block; }",
        "p { margin: 4px 0; }",
        "</style></head><body>"
    ]
    html_parts.append(f"<h1>{title}</h1>")

    # --- Tables ---
    table_files = ["model_timings.csv", "summary.csv", "model_elpd_metrics.csv"]
    for tfile in table_files:
        tpath = os.path.join(folder_path[0], tfile)
        if os.path.exists(tpath):
            df = pd.read_csv(tpath)
            df = df.round(2)

            # --- Conditional coloring for summary.csv ---
            if tfile == "summary.csv":
                # determine numeric columns
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                # exclude 'ess_b' and 'a' from rounding
                round_cols = [c for c in numeric_cols if c not in ["ess_bulk", "ess_tail"]]

                fmt_dict = {c: "{:.2f}" for c in round_cols}  # 2 decimal places
                for c in ["ess_bulk", "ess_tail"]:
                    if c in df.columns:
                        df[c] = df[c].astype(int)
                        fmt_dict[c] = "{:d}"  # integer format
                # Apply red background to r_hat >= 1.01
                df_html = (df.style.format(fmt_dict)
                            .map(lambda x: "background-color: red;" 
                                 if isinstance(x, (int, float)) and x >= 1.01 else "background-color: lightgreen;",
                                 subset=["r_hat"])
                            .map(lambda x: ess_style(x, n_draws),
                                 subset=["ess_bulk", "ess_tail"])
                                 ).to_html()
                #df_html = df.style.format(fmt_dict).map(
                    #lambda x: "background-color: red;" if isinstance(x, (int, float)) and x >= 1.01 else "",
                    #subset=["r_hat"]
                #).to_html()
            elif tfile =="model_elpd_metrics.csv":
                df_html = (df.style.format(fmt_dict)
                            .map(lambda x: "background-color: red;" 
                                 if isinstance(x, (int, float)) and x >= 1 else "background-color: lightgreen;",
                                 subset=["waic_warning"])
                            .map(lambda x: "background-color: red;" 
                                 if isinstance(x, (int, float)) and x > 0 else "background-color: lightgreen;",
                                 subset=["n_pareto_k_bad", "n_pareto_k_very_bad"])
                            .map(lambda x: "background-color: yellow;", subset=["waic", "loo"])).to_html()
            else:
                df_html = df.to_html(index=False, escape=False, border=0)

            html_parts.append(f"<h2>{tfile}</h2>")
            html_parts.append(df_html)

    # --- Images ---
    # 1) trace.png
    trace_file = os.path.join(folder_path[0], "trace.png")
    if os.path.exists(trace_file):
        html_parts.append("<h2>Trace Plot</h2>")
        html_parts.append(f'<img src="{os.path.basename(trace_file)}" style="max-width:100%;">')

    # 2) spline plots (all spline_*.png)
    spline_files = sorted([f for f in os.listdir(folder_path[0]) if f.startswith("spline_") and f.endswith(".png")])
    for sf in spline_files:
        html_parts.append(f"<h2>{sf}</h2>")
        html_parts.append(f'<img src="{sf}" style="max-width:100%;">')

    # 3) khat.png
    khat_file = os.path.join(folder_path[0], "khat.png")
    if os.path.exists(khat_file):
        html_parts.append("<h2>Pareto k Diagnostics</h2>")
        html_parts.append(f'<img src="{os.path.basename(khat_file)}" style="max-width:100%;">')

    

    html_parts.append("</body></html>")

    # --- write file ---
    for o in out_file:
        with open(o, "w") as f:
            f.write("\n".join(html_parts))

    print(f"HTML report written to: {out_file}")

###

def build_model(data, stat_names, degree=3, num_knots = 3, knot_type='quantile', orthogonal=True,
                surveillance_name='urban_surveillance_pop_weighted', urbanisation_name='urbanisation_pop_weighted_std'):
    model = pm.Model()
    with model:
        # Priors
        alpha = pm.Exponential("alpha", 0.5)
        intercept = pm.Normal("intercept", mu=0, sigma=2.5)
        if urbanisation_name is not None:
            beta_u = pm.Normal("beta_u", mu=0, sigma=1)

        # splines
        knot_list = {}
        B = {}
        sigma_w = {}
        w = {}
        f = {}
        for stat_name in stat_names:
            d = data[stat_name]
            if knot_type=='equispaced':
                knot_list[stat_name] = np.linspace(np.min(d), np.max(d), num_knots+2)[1:-1]
            elif knot_type=='quantile':
                knot_list[stat_name] = np.percentile(d, np.linspace(0, 100, num_knots + 2))[1:-1]
            else:
                print('knot_list must be quantile or equispaced')

            B[stat_name] = dmatrix(f"bs(s, knots=knots, degree=degree, include_intercept=False)-1",
                        {"s": data[stat_name], "knots": knot_list[stat_name], "degree":degree})
            if orthogonal:
                B[stat_name] = np.asarray(B[stat_name])
                B[stat_name] = (B[stat_name] - B[stat_name].mean(axis=0)) / B[stat_name].std(axis=0)
                B[stat_name], _ = np.linalg.qr(B[stat_name])
        
            # Spline coefficients
            sigma_w[stat_name] = pm.HalfNormal(f"sigma_w({stat_name})", sigma=0.5)
            w[stat_name] = pm.Normal(f"w({stat_name})", mu=0, sigma=1, size=B[stat_name].shape[1], dims="splines")
        
            # Spline contribution (with scaled mean to zero soft constraint)
            f_raw = pm.math.dot(B[stat_name], sigma_w[stat_name]* w[stat_name])
            f_mean = pm.math.mean(f_raw)
            f_var  = pm.math.mean((f_raw - f_mean) ** 2)
            f_std = pm.math.sqrt(f_var + 1e-6)
            pm.Potential(
            f"f_centred_prior({stat_name})",
            pm.logp(pm.Normal.dist(mu=0.0, sigma=0.01), f_mean/(f_std+1e-6)))
            f[stat_name] = f_raw

        # Link
        log_mu = intercept + pm.math.log(data['population'])
        if surveillance_name is not None:
            log_mu += pm.math.log(data[surveillance_name]+1e-3)
        if urbanisation_name is not None:
            log_mu += beta_u*data[urbanisation_name]
        for stat_name in stat_names:
            log_mu += f[stat_name]

        # Likelihood
        y_obs = pm.NegativeBinomial('y_obs', mu=pm.math.exp(log_mu), alpha=alpha, observed=data['cases'])

    return model, B, knot_list

def data_settings_to_name(s):
    admin = s["admin"]
    start = f"{s['start_year']}{s['start_month']:02d}"
    end = f"{s['end_year']}{s['end_month']:02d}"
    return f"a{admin}_{start}_{end}"