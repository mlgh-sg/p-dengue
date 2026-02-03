import pymc as pm
import numpy as np
from patsy import dmatrix
import re
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import arviz as az
import xarray as xr
import time
import pandas as pd
import warnings
from _fitting.fitting_utils import hist_plot, CI_plot, CI_plot_alt, CI_plot_both, plot_posteriors_side_by_side, plot_spline
from glob import glob

###
import base64
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
    
    # check if _log is present
    has_log = "_log" in s
    
    # weighting
    if "pop_weighted" in s:
        w = "p"
    elif "unweighted" in s:
        w = "u"
    else:
        w = ""
    
    # remove weighting and lag, keep everything else
    base = re.sub(r"_?(pop_weighted|unweighted).*", "", s)
    
    # reattach _log if it was in original
    if has_log and not base.endswith("_log"):
        base += "_log"
    
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
        return f"[{surv}__{urb}][{stat_str}][]"
    else:
        return f"[{surv}__{urb}][{stat_str}][{deg},{k},{kt},{orth}]"

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
def model_fit(data, data_name, model_settings, outpath, n_chains=4, n_draws=500, n_tune=500, sampler="nutpie", invert_log=False, task=None, replace=False):
    
    if task is None:
        data_path = os.path.join(outpath, f'{data_name}/')
    else:
        data_path = os.path.join(outpath, f'{data_name}[{task}]/')
    os.makedirs(data_path, exist_ok=True)
    
    model_name = model_settings_to_name(model_settings)

    idata_path = os.path.join(data_path, 'idata')
    os.makedirs(idata_path, exist_ok=True)

    report_path = os.path.join(data_path, f'reports/')
    os.makedirs(report_path, exist_ok=True)

    metrics_path = os.path.join(data_path, f'metrics')
    os.makedirs(metrics_path, exist_ok=True)

    output_path = os.path.join(data_path, f'outputs/{model_name}')
    os.makedirs(output_path, exist_ok=True)

    # if idata already exists, skip
    idata_file = os.path.join(idata_path, f"idata_[{model_name}].nc")
    if not replace:
        if os.path.exists(idata_file):
            print(f"Skipping {model_name}, data already exists.")
            create_html_report(output_path, model_name=model_name, n_draws=n_draws, reports_folder=report_path, replace=False)
            return

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
    inner_metrics_file = os.path.join(output_path, "_model_timings.csv")
    if os.path.exists(inner_metrics_file):
        metrics_df.to_csv(inner_metrics_file, mode="a", header=False, index=False)
    else:
        metrics_df.to_csv(inner_metrics_file, index=False)
    # outer
    outer_metrics_file = os.path.join(data_path, "_model_timings.csv")
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

    # Save pointwise values for later comparison
    pointwise_file = os.path.join(metrics_path, f"_metrics[{model_name}].npz")
    np.savez(
        pointwise_file,
        waic_pointwise=eval_waic.waic_i.values,
        loo_pointwise=eval_psis_loo_elpd.loo_i.values,
        pareto_k=eval_psis_loo_elpd.pareto_k.values
    )

    # dataframes (inner and outer)
    wl_df = pd.DataFrame([elpd_to_row(eval_waic, eval_psis_loo_elpd, model_name, data_name)])
    inner_wl_file = os.path.join(output_path, "_model_elpd_metrics.csv")
    outer_wl_file = os.path.join(data_path, "_model_elpd_metrics.csv")
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
            invert_log=invert_log
        );
        if isinstance(fig, plt.Figure):
            fig_file = os.path.join(output_path, f"spline_{stat_name}.png")
            fig.savefig(fig_file, bbox_inches="tight")
            plt.close(fig)

    create_html_report(output_path, model_name=model_name, n_draws=n_draws, reports_folder=report_path)
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

def create_html_report(model_folder, model_name, n_draws, reports_folder=None, title=None, replace=False):
    """
    Generate HTML report for a single model.

    Args:
        model_folder: path to the model_name folder containing csv/images
        model_name: name of the model
        n_draws: number of draws for ESS coloring
        reports_folder: if provided, also generate a report in this folder
        title: optional HTML title
    """

    # Paths for output HTML files
    out_files = [os.path.join(model_folder, f"report_[{model_name}].html")]
    if reports_folder:
        os.makedirs(reports_folder, exist_ok=True)
        out_files.append(os.path.join(reports_folder, f"report_[{model_name}].html"))
    # If not replacing, check if files exist
    if not replace:
        if all(os.path.exists(f) for f in out_files):
            print(f"Skipping HTML report for {model_name}, report already exists.")
            return

    if title is None:
        title = f"Model Report: {model_name}"

    # --- Read CSVs ---
    table_files = ["_model_timings.csv", "summary.csv", "_model_elpd_metrics.csv"]
    csv_html_parts = []
    for tfile in table_files:
        tpath = os.path.join(model_folder, tfile)
        if os.path.exists(tpath):
            df = pd.read_csv(tpath).round(2)
            # apply formatting only if relevant columns exist
            int_cols = ["ess_bulk", "ess_tail", "waic_warning", "n_pareto_k_bad", "n_pareto_k_very_bad"]
            fmt_dict = {c: "{:.2f}" for c in df.select_dtypes(include="number").columns if c not in int_cols}
            for c in int_cols:
                if c in df.columns:
                    df[c] = df[c].astype(int)
                    fmt_dict[c] = "{:d}"
            # Apply styling for summary.csv
            if tfile == "summary.csv":
                df_html = (df.style.format(fmt_dict)
                    .map(lambda x: "background-color: red;" if isinstance(x, (int, float)) and x >= 1.01 else "background-color: lightgreen;",
                         subset=["r_hat"] if "r_hat" in df.columns else [])
                    .map(lambda x: ess_style(x, n_draws),
                         subset=["ess_bulk", "ess_tail"] if "ess_bulk" in df.columns else [])
                    ).to_html()
            elif tfile == "_model_elpd_metrics.csv":
                df_html = (df.style.format(fmt_dict)
                    .map(lambda x: "background-color: red;" if isinstance(x, (int, float)) and x >= 1 else "background-color: lightgreen;",
                         subset=["waic_warning"] if "waic_warning" in df.columns else [])
                    .map(lambda x: "background-color: red;" if isinstance(x, (int, float)) and x > 0 else "background-color: lightgreen;",
                         subset=["n_pareto_k_bad", "n_pareto_k_very_bad"] if "n_pareto_k_bad" in df.columns else [])
                    .map(lambda x: "background-color: yellow;", subset=["waic", "loo"] if "waic" in df.columns else [])
                    ).to_html()
            else:
                df_html = df.to_html(index=False, escape=False, border=0)

            csv_html_parts.append(f"<h2>{tfile}</h2>\n{df_html}")

    # --- Images ---
    img_files = []
    # trace.png
    trace_path = os.path.join(model_folder, "trace.png")
    if os.path.exists(trace_path):
        img_files.append(("Trace Plot", trace_path))
    # khat.png
    khat_path = os.path.join(model_folder, "khat.png")
    if os.path.exists(khat_path):
        img_files.append(("Pareto k Diagnostics", khat_path))
    # spline_*.png
    for sf in sorted([f for f in os.listdir(model_folder) if f.startswith("spline_") and f.endswith(".png")]):
        sf_path = os.path.join(model_folder, sf)
        img_files.append((sf, sf_path))

    # --- Assemble HTML ---
    html_base = [
        f"<html><head><title>{title}</title>",
        "<style>",
        "body { font-family: Arial; font-size: 12px; line-height: 1.2; margin: 8px; text-align:center; }",
        "h1, h2 { margin: 4px 0 8px 0; font-weight: normal; }",
        "table { border-collapse: collapse; font-size: 15px; margin: 0 auto 12px auto; width: 80%; }",
        "table th, table td { border: 1px solid #aaa; padding: 4px 6px; text-align: center; }",
        "img { max-width: 80%; margin: 8px auto; display: block; }",
        "</style></head><body>",
        f"<h1>{title}</h1>"
    ]
    html_base.extend(csv_html_parts)
    html_parts = html_base.copy()
    # Add images as base64
    for caption, path in img_files:
        html_parts.append(f"<h2>{caption}</h2>")
        with open(path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
            html_parts.append(f'<img src="data:image/png;base64,{img_data}" style="max-width:100%;">')
    
    html_parts.append("</body></html>")

    # --- Write HTML files ---
    for html_file in out_files:
        with open(html_file, "w") as f:
            f.write("\n".join(html_parts))

    print(f"HTML reports written to: {', '.join(out_files)}")


#############

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

###########################

def compare_models(outpath, data_name, task, metric="loo"):
    """
    Compare multiple models using pointwise ELPD values.
    Mimics ArviZ's compare() output format.
    
    Args:
        outpath: base path to model fits
        data_name: name of the data folder (e.g., 'a2_201601_201912')
        task: subfolder name (e.g., 'variable_selection')
        metric: 'loo' or 'waic'
    
    Returns:
        DataFrame with model comparison results ranked by ELPD
    """
    
    # Find all npz files
    metrics_path = os.path.join(outpath, f'{data_name}[{task}]/metrics')
    npz_files = [os.path.join(metrics_path, f) for f in os.listdir(metrics_path)]
    
    if len(npz_files) == 0:
        raise ValueError(f"No files found in {metrics_path}")
    
    print(f"Comparing models using {metric.upper()}")
    print("="*100)
    
    # Load all models
    models = {}
    for npz_file in npz_files:
        model_name = os.path.basename(npz_file)[9:-5]  # Extract from _metrics[...].npz
        data = np.load(npz_file)
        models[model_name] = data[f'{metric}_pointwise']
    
    # Build comparison dataframe
    results = []
    for name, pointwise in models.items():
        results.append({
            'model': name,
            f'{metric}': pointwise.sum(),
            f'p_{metric}': len(pointwise),  # effective number of parameters
            f'{metric}_se': np.std(pointwise) * np.sqrt(len(pointwise)),
        })
    
    df = pd.DataFrame(results)
    
    # Sort by ELPD (higher is better for elpd_loo, lower is better for looic)
    # ArviZ reports as negative (looic), but we keep as positive (elpd)
    df = df.sort_values(f'{metric}', ascending=False).reset_index(drop=True)
    
    # Add rank
    df.insert(0, 'rank', range(len(df)))
    
    # Calculate differences from best model (rank 0)
    best_name = df.iloc[0]['model']
    best_pointwise = models[best_name]
    
    d_metric = []
    d_se = []
    weight = []
    
    for idx, row in df.iterrows():
        current_pointwise = models[row['model']]
        
        # Difference from best (best - current, so negative means worse)
        diff_pointwise = best_pointwise - current_pointwise
        diff = diff_pointwise.sum()
        diff_se = np.std(diff_pointwise) * np.sqrt(len(diff_pointwise))
        
        d_metric.append(diff)
        d_se.append(diff_se)
        
        # Akaike weight
        weight.append(np.exp(-0.5 * diff))
    
    # Normalize weights
    weight = np.array(weight)
    weight = weight / weight.sum()
    
    df[f'd{metric}'] = d_metric
    df['dse'] = d_se
    df['weight'] = weight
    
    # Set model as index (like ArviZ does)
    df = df.set_index('model')
    
    # Reorder columns to match ArviZ output
    column_order = ['rank', f'{metric}', f'p_{metric}', f'd{metric}', 'weight', f'{metric}_se', 'dse']
    df = df[column_order]
    df = df.round(2)
    
    print(df.to_string())
    print("="*100)
    
    return df