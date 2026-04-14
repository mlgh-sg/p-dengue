# if using conda env for jax GPU
import subprocess, os; os.environ["CUDA_VISIBLE_DEVICES"] = str(max([(int(l.split(',')[0]), int(l.split(',')[1])) for l in subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'], capture_output=True, text=True).stdout.strip().split('\n')], key=lambda x: x[1])[0])
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pymc as pm
import pymc.math as pmm
import arviz as az
from patsy import dmatrix
import nutpie
import time
from IPython.display import display
from pymc.variational.callbacks import CheckParametersConvergence
import io
import base64
import re

# if using uv env for CPU
# import pytensor
 #pytensor.config.mode='NUMBA'

import pytensor.tensor as pt
from pytensor.gradient import disconnected_grad
from scipy.sparse.linalg import eigsh
az.style.use("arviz-darkgrid")
import warnings
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", module="arviz")
import seaborn as sns
from scipy.special import erf
from multiprocessing import Pool
from _fitting.spline_utils import *



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
    p = settings.get("penalty_parameters", {}).get("p")
    
    knot_map = {"quantile": "q", "uniform": "u"}
    kt = knot_map.get(settings.get("knot_type"), settings.get("knot_type"))
    
    if len(stats) == 0:
        return f"[{surv}__{urb}][{stat_str}][]"
    else:
        return f"[{surv}__{urb}][{stat_str}][{k}, {p}]"
    

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64

def plot_link(x, idata, var_names=['zi_b0', 'zi_b1'], link='logit'):
    idata_posterior = idata.posterior
    x_mean = np.mean(x)
    x_std_dev = np.std(x)
    
    if np.all([v in idata_posterior for v in var_names]):
        b0_samples = idata_posterior[var_names[0]].values.flatten()
        b1_samples = idata_posterior[var_names[1]].values.flatten()

        x0_samples = -b0_samples / (b1_samples) * x_std_dev + x_mean
        link_samples = np.array([b0 + b1 * ((x - x_mean) / x_std_dev) for b0, b1 in zip(b0_samples, b1_samples)])
    elif all(v in idata_posterior for v in ['zi_c', 'zi_b01', 'zi_b11']):
        c_samples = idata_posterior['zi_c'].values.flatten()
        b01_samples = idata_posterior['zi_b01'].values.flatten()
        b11_samples = idata_posterior['zi_b11'].values.flatten()
        x0_samples = c_samples.copy() * x_std_dev + x_mean
        link_samples = np.array([np.maximum(0, ((x - x_mean) / x_std_dev) - c) * (b01 - b11) + b11 * (((x - x_mean) / x_std_dev) - c) for c, b01, b11 in zip(c_samples, b01_samples, b11_samples)])
    elif all(v in idata_posterior for v in ['zi_c', 'zi_b1', 'zi_b1d']):
        c_samples = idata_posterior['zi_c'].values.flatten()
        b1_samples = idata_posterior['zi_b1'].values.flatten()
        b1d_samples = idata_posterior['zi_b1d'].values.flatten()
        x0_samples = c_samples.copy() * x_std_dev + x_mean
        link_samples = np.array([np.maximum(0, ((x - x_mean) / x_std_dev) - c) * b1d + b1 * (((x - x_mean) / x_std_dev) - c) for c, b1, b1d in zip(c_samples, b1_samples, b1d_samples)])
        
    x0_mean = np.mean(x0_samples)
    x0_lower = np.percentile(x0_samples, 2.5)
    x0_upper = np.percentile(x0_samples, 97.5)

    def logit_to_prob(logit):
        return 1 / (1 + np.exp(-logit))
    def probit_to_prob(probit):
        return 0.5 * (1 + erf(probit / np.sqrt(2)))
    if link == 'logit':
        link_samples = logit_to_prob(link_samples)
    elif link == 'probit':
        link_samples = probit_to_prob(link_samples)
    
    link_mean = link_samples.mean(axis=0)
    link_lower5 = np.percentile(link_samples, 25, axis=0)
    link_upper5 = np.percentile(link_samples, 75, axis=0)
    link_lower = np.percentile(link_samples, 2.5, axis=0)
    link_upper = np.percentile(link_samples, 97.5, axis=0)

    # reorder
    id = np.argsort(x)
    x = x[id]
    link_mean = link_mean[id]
    link_lower5 = link_lower5[id]
    link_upper5 = link_upper5[id]
    link_lower = link_lower[id]
    link_upper = link_upper[id]
    link_hdi = np.array([link_lower, link_upper]).T
    plt.figure(figsize=(8, 5))

    plt.axvline(x0_mean, color='red', linestyle='--', label='Mean x0')
    plt.axvline(x0_lower, color='red', linestyle=':', label='95% CI x0')
    plt.axvline(x0_upper, color='red', linestyle=':')

    plt.plot(x, link_mean, label='Mean Link', color='blue')
    plt.fill_between(x, link_hdi[:, 0], link_hdi[:, 1], color='blue', alpha=0.3, label='95% HDI')
    plt.xlabel('x')
    plt.ylabel('Link(psi)')
    plt.title(f'Posterior of Inverse {link.capitalize()} vs x')
    plt.legend()
    plt.grid()
    return plt.gcf()

def plot_link_spline(x, idata, stat_name, B, knots, var_names=['zi_b0', 'zi_b1'], link='logit'):
    id = np.argsort(x)
    x = x[id]
    x_mean = np.mean(x)
    x_std_dev = np.std(x)
    
    idata_posterior = idata.posterior
    if np.all([v in idata_posterior for v in var_names]):
        b0_samples = idata_posterior[var_names[0]].values.flatten()
        b1_samples = idata_posterior[var_names[1]].values.flatten()

        x0_samples = -b0_samples / (b1_samples) * x_std_dev + x_mean
        link_samples = np.array([b0 + b1 * ((x - x_mean) / x_std_dev) for b0, b1 in zip(b0_samples, b1_samples)])
    elif all(v in idata_posterior for v in ['zi_c', 'zi_b01', 'zi_b11']):
        c_samples = idata_posterior['zi_c'].values.flatten()
        b01_samples = idata_posterior['zi_b01'].values.flatten()
        b11_samples = idata_posterior['zi_b11'].values.flatten()
        x0_samples = c_samples.copy() * x_std_dev + x_mean
        link_samples = np.array([np.maximum(0, ((x - x_mean) / x_std_dev) - c) * (b01 - b11) + b11 * (((x - x_mean) / x_std_dev) - c) for c, b01, b11 in zip(c_samples, b01_samples, b11_samples)])
    elif all(v in idata_posterior for v in ['zi_c', 'zi_b1', 'zi_b1d']):
        c_samples = idata_posterior['zi_c'].values.flatten()
        b1_samples = idata_posterior['zi_b1'].values.flatten()
        b1d_samples = idata_posterior['zi_b1d'].values.flatten()
        x0_samples = c_samples.copy() * x_std_dev + x_mean
        link_samples = np.array([np.maximum(0, ((x - x_mean) / x_std_dev) - c) * b1d + b1 * (((x - x_mean) / x_std_dev) - c) for c, b1, b1d in zip(c_samples, b1_samples, b1d_samples)])
        
    x0_mean = np.mean(x0_samples)
    x0_lower = np.percentile(x0_samples, 2.5)
    x0_upper = np.percentile(x0_samples, 97.5)

    def logit_to_prob(logit):
        return 1 / (1 + np.exp(-logit))
    def probit_to_prob(probit):
        return 0.5 * (1 + erf(probit / np.sqrt(2)))
    if link == 'logit':
        link_samples = logit_to_prob(link_samples)
    elif link == 'probit':
        link_samples = probit_to_prob(link_samples)

    ###
    knots_local = np.array(knots, copy=True)
    B_local = np.array(B, copy=True, order="F")
    data_local = np.array(x, copy=True)
    B_plot = B_local[id]
    
    # Extract posterior samples
    w_samples = idata_posterior[f'w({stat_name})'].stack(draws=("chain", "draw")).values  # (n_basis, n_draws)
    # sigma_w_samples = idata_posterior[f'sigma_w({stat_name})'].stack(draws=("chain", "draw")).values  # (n_draws,)

    f_samples = (B_plot @ w_samples)
    s_samples = np.exp(f_samples)
    ###
    link_samples = link_samples * s_samples.T

    link_mean = link_samples.mean(axis=0)
    # link_lower5 = np.percentile(link_samples, 25, axis=0)
    # link_upper5 = np.percentile(link_samples, 75, axis=0)
    link_lower = np.percentile(link_samples, 2.5, axis=0)
    link_upper = np.percentile(link_samples, 97.5, axis=0)

    link_hdi = np.array([link_lower, link_upper]).T
    plt.figure(figsize=(8, 5))
    
    for i_, k_ in enumerate(knots_local):
        plt.axvline(k_, color='green', linestyle='--',
                label='Knots' if i_ == 0 else None)
    #plt.axvline(knots_local, color='green', linestyle='--', label='Knots')

    plt.axvline(x0_mean, color='red', linestyle='--', label='Mean x0')
    plt.axvline(x0_lower, color='red', linestyle=':', label='95% CI x0')
    plt.axvline(x0_upper, color='red', linestyle=':')

    plt.plot(x, link_mean, label='Mean Link', color='blue')
    plt.fill_between(x, link_hdi[:, 0], link_hdi[:, 1], color='blue', alpha=0.3, label='95% HDI')
    plt.xlabel('x')
    plt.ylabel('Link(psi)')
    plt.title(f'Posterior of Multiplicative Effect (Spline, {link.capitalize()} Link)')
    plt.legend()
    plt.grid()
    return plt.gcf()

def plot_exp_spline_(x, idata, stat_name, B, knots):
    id = np.argsort(x)
    x = x[id]
    x_mean = np.mean(x)
    x_std_dev = np.std(x)

    ###
    knots_local = np.array(knots, copy=True)
    B_local = np.array(B, copy=True, order="F")
    # data_local = np.array(x, copy=True)
    B_plot = B_local[id]
    
    # Extract posterior samples
    w_samples = idata.posterior[f'w({stat_name})'].stack(draws=("chain", "draw")).values  # (n_basis, n_draws)
    #sigma_w_samples = idata.posterior[f'sigma_w({stat_name})'].stack(draws=("chain", "draw")).values  # (n_draws,)

    f_samples = (B_plot @ w_samples)
    s_samples = np.exp(f_samples)
    ###
    link_samples = s_samples.T

    link_mean = link_samples.mean(axis=0)
    # link_lower5 = np.percentile(link_samples, 25, axis=0)
    # link_upper5 = np.percentile(link_samples, 75, axis=0)
    link_lower = np.percentile(link_samples, 2.5, axis=0)
    link_upper = np.percentile(link_samples, 97.5, axis=0)

    link_hdi = np.array([link_lower, link_upper]).T
    plt.figure(figsize=(8, 7))
    
    for i_, k_ in enumerate(knots_local):
        plt.axvline(k_, color='green', linestyle='--',
                label='Knots' if i_ == 0 else None)

    plt.ylim(-0.01, 1.8)
    plt.plot(x, link_mean, label='Mean Link', color='blue')
    plt.fill_between(x, link_hdi[:, 0], link_hdi[:, 1], color='blue', alpha=0.3, label='95% HDI')
    plt.xlabel('x')
    plt.ylabel('Multiplicative Effect (Spline)')
    plt.title(f'Posterior of Multiplicative Effect (Spline)')
    plt.legend()
    plt.grid()
    return plt.gcf()

def plot_exp_spline(x, idata, stat_name, B, knots, data=None, freq_transform=lambda x: x, invert_log=True):
    id = np.argsort(x)
    x = x[id]

    data_local = x

    knots_local = np.array(knots, copy=True)
    B_local = np.array(B, copy=True, order="F")
    B_plot = B_local[id]
    
    w_samples = idata.posterior[f'w({stat_name})'].stack(draws=("chain", "draw")).values
    f_samples = B_plot @ w_samples
    s_samples = np.exp(f_samples)
    link_samples = s_samples.T

    link_mean = link_samples.mean(axis=0)
    link_lower = np.percentile(link_samples, 2.5, axis=0)
    link_upper = np.percentile(link_samples, 97.5, axis=0)
    link_lower5 = np.percentile(link_samples, 25, axis=0)
    link_upper5 = np.percentile(link_samples, 75, axis=0)
    link_hdi = np.array([link_lower, link_upper]).T

    y_max = np.minimum(np.maximum(np.ceil(link_mean.max() / 0.5) * 0.5, 1.0), 10.0)
    #print(y_max)
    y_min = -0.01
    y_range = y_max - y_min
    label_overhead_inches = 1.5
    fig_height = (y_range / 0.5) * 1.5 + label_overhead_inches
    fig_width = 8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # inverse log trasformation
    if (invert_log) & (stat_name[0:2] == 'tp'):
        plot_knots = (np.exp(knots_local) - 1e-6) * 1000
        if data is not None:
            plot_data = (np.exp(data_local) - 1e-6) * 1000
    else:
        plot_knots = knots_local
        if data is not None:
            plot_data = data_local

    # histogram with knot-defined bins
    if data is not None:
        raw_vals = plot_data
        bins = np.concatenate([[raw_vals.min()], plot_knots, [raw_vals.max()]])
        counts, edges = np.histogram(raw_vals, bins=bins)
        transformed = freq_transform(counts)
        # scale to y_range for visibility
        transformed_scaled = transformed / transformed.max() * 1.0
        ax.bar(edges[:-1], transformed_scaled, width=np.diff(edges), align='edge',
               color='green', alpha=0.3, label=f'density', zorder=0)

    for i_, k_ in enumerate(plot_knots):
        ax.axvline(k_, color='green', linestyle='--', label='Knots' if i_ == 0 else None)

    ax.set_ylim(y_min, y_max)
    ax.plot(plot_data, link_mean, label='Mean Link', color='blue')
    ax.fill_between(plot_data, link_hdi[:, 0], link_hdi[:, 1], color='blue', alpha=0.3, label='95% HDI')
    ax.fill_between(plot_data, link_lower5, link_upper5, color='blue', alpha=0.3, label='50% HDI')
    ax.set_xlabel('x')
    ax.set_yticks(np.arange(0.0, y_max + 0.5, 0.25))
    ax.set_ylabel('Multiplicative Effect (Spline)')
    ax.set_title(f'Posterior of Multiplicative Effect (Spline)')
    ax.legend()
    ax.grid()

    return fig

units = {'t2':'C˚', 'rh':'%RH', 'tp':'mm'}
units_log = {'t2':'C˚', 'rh':'%RH', 'tp':'log(m)'}

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

def plot_spline_Bknots(idata, stat_name,
                       var, sigma_var, B, data, knots,
                       figsize=(10,5), show_basis=False, basis_scale=1, invert_log=False, centred_w=True):
    # work on local copies to avoid mutating caller data
    knots_local = np.array(knots, copy=True)
    B_local = np.array(B, copy=True, order="F")
    data_local = np.array(data, copy=True)

    index = np.argsort(data_local)
    data_plot = data_local[index]
    B_plot = B_local[index, :]

    # Extract posterior samples
    w_samples = idata.posterior[var].stack(draws=("chain", "draw")).values  # (n_basis, n_draws)
    # sigma_w_samples = idata.posterior[sigma_var].stack(draws=("chain", "draw")).values  # (n_draws,)

    # Compute spline contributions for each draw
    if centred_w:
        f_samples = (B_plot @ w_samples)  # (n_plot, n_draws)
    #else:
        #f_samples = (B_plot @ w_samples) * sigma_w_samples  # (n_plot, n_draws)

    # Compute mean and credible intervals
    f_mean = f_samples.mean(axis=1)
    f_25 = np.percentile(f_samples, 25, axis=1)
    f_75 = np.percentile(f_samples, 75, axis=1)
    f_025 = np.percentile(f_samples, 2.5, axis=1)
    f_975 = np.percentile(f_samples, 97.5, axis=1)

    # Create figure/axes explicitly
    fig, ax = plt.subplots(figsize=figsize)

    # inverse log trasformation
    if (invert_log) & (stat_name[0:2] == 'tp'):
        plot_knots = (np.exp(knots_local) - 1e-6) * 1000
        data_plot = (np.exp(data_plot) - 1e-6) * 1000
    else:
        plot_knots = knots_local

    ax.vlines(plot_knots, ymin=np.min(f_025), ymax=np.max(f_975), label='knots', lw=0.8, alpha=0.7)
    if show_basis:
        for i in range(B_plot.shape[1]):
            ax.plot(data_plot,
                    np.max(f_975) +
                    (np.max(f_975) - np.min(f_025)) * basis_scale * (
                        (B_plot[:, i] - np.min(B_plot[:, i]))/(np.max(B_plot[:, i]) - np.min(B_plot[:, i])) + 0.05),
                        alpha=0.99, linestyle=':')

    # Main lines and ribbons
    ax.plot(data_plot, f_mean, color='red', label='Mean spline effect')
    ax.fill_between(data_plot, f_025, f_25, color='red', alpha=0.3, label='95% CI')
    ax.fill_between(data_plot, f_25, f_75, color='blue', alpha=0.3, label='50% CI')
    ax.fill_between(data_plot, f_75, f_975, color='red', alpha=0.3)

    abbrev_stat_name = abbrev_stat(stat_name)
    if (invert_log) & (stat_name[0:2] == 'tp'):
        abbrev_stat_name = abbrev_stat_name.replace("_log", "")
        xlab = f'{abbrev_stat_name} ({units[stat_name[0:2]]})'
    else:
        xlab = f'{abbrev_stat_name} ({units_log[stat_name[0:2]]})'

    ax.set_xlabel(xlab)
    ax.set_ylabel('Spline contribution')
    # ax.set_ylim(-0.5, 3.5)
    ax.legend()

    return fig

def elpd_to_row(eval_waic, eval_loo, model_name, data_name):
    return {"model_name": model_name,
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
            "pareto_k_mean": float(eval_loo.pareto_k.mean())}

def go(data, m, model_dict, idata_dict, time_dict, B_dict,
       knot_list_dict, link_dict, stat_names_dict, var_names_dict, n_divergences_dict,
       tune=1000, draws=4000, target_accept=0.8, max_treedepth=10, max_energy_error=1000, compute_idata=True,
       show = {'summary': True, 'trace': True, 'pair': True, 'metrics': True,
               'spline': True, 'exp_spline': True, 'link': True, 'link_spline': True,
               'divergences': True}):
    
    var_names = var_names_dict[m]
    stat_names = stat_names_dict[m]
    with model_dict[m]:
        if compute_idata:
            s0 = time.time()
            idata_dict[m] = pm.sample(
                tune=tune,
                draws=draws,
                chains=4,
                cores=4,
                discard_tuned_samples=True,
                store_divergences=True,
                nuts_sampler="nutpie",
                target_accept = target_accept,
                max_treedepth = max_treedepth,
                nuts_sampler_kwargs={"max_energy_error": max_energy_error}
            )
            s1 = time.time()
            pm.compute_log_likelihood(idata_dict[m], progressbar=False)
            s2 = time.time()
            time_dict[m] = (s1 - s0, s2 - s1)
            n_divergences = int(idata_dict[m].sample_stats["diverging"].sum())
            n_divergences_dict[m] = n_divergences
    n_divergences = n_divergences_dict[m]
    # ---------- Summary table ----------
    if show['summary']:
        summary_df = az.summary(idata_dict[m], var_names=var_names)
        summary_html = summary_df.to_html()

    # ---------- Trace plot ----------
    if show['trace']:
        tp_var_names = [v for v in var_names if 'true_scale' not in v]
        fig_trace = az.plot_trace(idata_dict[m], var_names=tp_var_names)
        fig_trace = fig_trace.ravel()[0].figure
        trace_img = fig_to_base64(fig_trace)
    # ---------- Pair plot ----------
    if show['pair']:
        az.rcParams["plot.max_subplots"] = 200
        pp_var_names = [v for v in var_names if 'true_scale' not in v]
        ax = az.plot_pair(
            idata_dict[m],
            var_names=pp_var_names,
            textsize=14,
            divergences=True)

        for i in range(ax.shape[0]):
            ax[i, 0].yaxis.label.set_rotation(0)
            ax[i, 0].yaxis.label.set_ha('right')
        for j in range(ax.shape[1]):
            ax[0, j].xaxis.label.set_rotation(45)
            ax[0, j].xaxis.label.set_ha('right')
        fig_pair = ax.ravel()[0].figure
        pair_img = fig_to_base64(fig_pair)

    #### WAIC and PSIS LOO
    if show['metrics']:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eval_waic = az.waic(idata_dict[m])
            eval_psis_loo_elpd = az.loo(idata_dict[m])
        wl_df = pd.DataFrame([elpd_to_row(eval_waic, eval_psis_loo_elpd, m, 'd')])
        wl_html = wl_df.to_html()

    # ---------- Spline plot ----------
    if show['spline']:
        spline_imgs = []
        if stat_names is not None:
            for stat_name in stat_names:
                fig_spline = plot_spline_Bknots(
                    idata_dict[m],
                    stat_name,
                    f'w({stat_name})',
                    f'sigma_w({stat_name})',
                    B_dict[m][stat_name],
                    data[stat_name].values,
                    knots=knot_list_dict[m][stat_name],
                    show_basis=True,
                    invert_log=True,
                    centred_w=True
                )
                spline_imgs.append(fig_to_base64(fig_spline))
    if show['exp_spline']:
        exp_spline_imgs = []
        if stat_names is not None:
            for stat_name in stat_names:
                fig_exp_spline = plot_exp_spline(
                    data[stat_name].values,
                    idata_dict[m],
                    stat_name,
                    B_dict[m][stat_name],
                    knot_list_dict[m][stat_name],
                    data=data,
                )
                exp_spline_imgs.append(fig_to_base64(fig_exp_spline))
    
    if show['link']:
        if link_dict[m] is not None:
            link = link_dict[m]['link']
            link_stat_name = link_dict[m]['link_stat_name']
            zi_img = fig_to_base64(plot_link(data[link_stat_name].values, idata_dict[m],
                                            var_names=['zi_b0', 'zi_b1'], link=link))
        else:
            zi_img = None
    else:
            zi_img = None

    if show['link_spline']:
        if (link_dict[m] is not None)&(stat_names is not None):
            if link_dict[m]['link_stat_name'] in stat_names:
                zi_s_img = fig_to_base64(plot_link_spline(data[link_dict[m]['link_stat_name']].values, idata_dict[m],
                                                    link_dict[m]['link_stat_name'], B_dict[m][link_dict[m]['link_stat_name']],
                                                    knot_list_dict[m][link_dict[m]['link_stat_name']], link=link_dict[m]['link']))
        else:
            zi_s_img = None
    else:
        zi_s_img = None

    #---Divergences plot---
    if show['divergences']&(n_divergences > 0):
        posterior = idata_dict[m].posterior.to_dataframe().reset_index()
        stats = idata_dict[m].sample_stats.to_dataframe().reset_index()
        df = posterior.merge(stats, on=["chain","draw"])

        #posterior = az.extract(idata_dict[m], combined=True).to_pandas()
        #stats = idata_dict[m].sample_stats.to_dataframe()
        #df = posterior.join(stats)
        sns.pairplot( df, vars=pp_var_names, hue="diverging",
                    corner=True, diag_kind='kde', plot_kws={"alpha":0.5, 's':1}, diag_kws={"common_norm": False})
        div_img = fig_to_base64(plt.gcf())
    else:
        div_img = None

    # ---------- Build HTML ----------
    html_parts = []

    # --- Precompute reusable values ---
    total_samples = (
        idata_dict[m].posterior.sizes['draw'] *
        idata_dict[m].posterior.sizes['chain']
    )
    div_pct = n_divergences / total_samples * 100

    # --- Header ---
    html_parts.append(f"""
    <html>
    <head>
        <title>Model Report: {m}</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            h1 {{ margin-bottom: 10px; }}
            img {{ margin-top: 20px; max-width: 100%; }}
            table {{ border-collapse: collapse; }}
            th, td {{ padding: 6px 8px; }}
        </style>
    </head>
    <body>
        <h1>Model Report: {m}</h1>

        <h2>Timing</h2>
        <p>Posterior Sampling: {time_dict[m][0]:.2f} seconds</p>
        <p>Log Likelihood Compute: {time_dict[m][1]:.2f} seconds</p>
    """)

    # --- Summary ---
    if show['summary']:
        html_parts.append(f"""
        <h2>Summary</h2>
        {summary_html}
        <p>Divergences: {n_divergences} out of {total_samples} samples ({div_pct:.2f}%)</p>
        """)

    # --- Metrics ---
    if show['metrics']:
        html_parts.append(f"""
        <h2>WAIC and PSIS LOO</h2>
        {wl_html}
        """)

    # --- Trace ---
    if show['trace']:
        html_parts.append(f"""
        <h2>Trace Plot</h2>
        <img src="data:image/png;base64,{trace_img}">
        """)

    # --- Pair ---
    if show['pair']:
        html_parts.append(f"""
        <h2>Pair Plot</h2>
        <img src="data:image/png;base64,{pair_img}">
        """)

    # --- Spline ---
    if show['spline']:
        imgs = "".join(f'<img src="data:image/png;base64,{img}">' for img in spline_imgs)
        html_parts.append(f"""
        <h2>Spline Plot</h2>
        {imgs}
        """)

    # --- Exponential Spline ---
    if show['exp_spline']:
        imgs = "".join(f'<img src="data:image/png;base64,{img}">' for img in exp_spline_imgs)
        html_parts.append(f"""
        <h2>Exponential Spline Plot</h2>
        {imgs}
        """)

    # --- Link ---
    if show['link']:
        img_html = f'<img src="data:image/png;base64,{zi_img}">' if zi_img is not None else ""
        html_parts.append(f"""
        <h2>ZI link Plot</h2>
        {img_html}
        """)

    # --- Link spline ---
    if show['link_spline']:
        img_html = f'<img src="data:image/png;base64,{zi_s_img}">' if zi_s_img is not None else ""
        html_parts.append(f"""
        <h2>ZI link with Spline Plot</h2>
        {img_html}
        """)

    # --- Divergences ---
    if show['divergences']:
        img_html = (
            f'<img src="data:image/png;base64,{div_img}">'
            if n_divergences > 0
            else "<p>No divergences detected.</p>"
        )
        html_parts.append(f"""
        <h2>Divergences Plot</h2>
        <p>Divergences: {n_divergences} out of {total_samples} samples ({div_pct:.2f}%)</p>
        {img_html}
        """)

    # --- Footer ---
    html_parts.append("""
    </body>
    </html>
    """)

    html_content = "".join(html_parts)

    with open(f"{m}.html", "w") as f:
        f.write(html_content)

    print(f"Saved report to {m}.html")
    
    return model_dict[m], B_dict[m], knot_list_dict[m], stat_names_dict[m], var_names_dict[m], link_dict[m], idata_dict[m], time_dict[m], n_divergences_dict[m]

def difference_matrix(n, order=1):
    """
    Construct a kth-order finite difference matrix of size (n-order, n).

    Parameters
    ----------
    n : int
        Length of the coefficient vector.
    order : int
        Order of the difference.

    Returns
    -------
    D : ndarray
        Difference matrix of shape (n-order, n)
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    if order >= n:
        raise ValueError("order must be < n")

    D = np.eye(n)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D

def time_models(model_dict, idata_dict, models_list, iter):
    # Compare models
    times = {m: [] for m in models_list}
    for m in models_list:
        for i in range(iter):
            with model_dict[m]:
                s0 = time.time()
                idata_dict[m] = pm.sample(tune=1000, draws=4000, chains=4, cores=4, progressbar=False,
                                            discard_tuned_samples=True, nuts_sampler="nutpie", store_divergences=True)
                s1 = time.time()
                pm.compute_log_likelihood(idata_dict[m], progressbar=False)
                s2 = time.time()
                times[m].append(s1 - s0)
        print(f"Model {m}: Mean time = {np.mean(times[m]):.2f} seconds")
    mean_times = {m: np.mean(times[m]) for m in models_list}
    return times, mean_times 

def _time_worker(task):
    model_dict, idata_dict, m, iter = task
    times, mean_times = time_models(model_dict, idata_dict, [m], iter)
    return m, times[m], mean_times[m]


def time_models_parallel(model_dict, idata_dict, models_list, iter, n_workers=4):

    tasks = [(model_dict, idata_dict, m, iter) for m in models_list]
    with Pool(n_workers) as p:
        results = p.map(_time_worker, tasks)
    # Reassemble results
    times = {}
    mean_times = {}

    for m, t, mt in results:
        times[m] = t
        mean_times[m] = mt
    return times, mean_times

def build_var_names(alpha_parameters, intercept_parameters, link, link_type,
                    b1_parameters, c_parameters, stat_names, spline_implementation,
                    penalty_order, penalty_parameters, beta_u_parameters):
    var_names = []

    # Always present
    if intercept_parameters is not None:
        var_names.append('intercept')
    if alpha_parameters is not None:
        var_names.append('alpha')

    # Urbanisation
    if beta_u_parameters is not None:
        var_names.append('beta_u')

    # Spline weights
    if stat_names is not None:
        if spline_implementation == 'svd':
            for stat_name in stat_names:
                # var_names.append(f'sigma_w({stat_name})')
                var_names.append(f'w({stat_name})')
            if penalty_order is not None:
                for stat_name in stat_names:
                    # var_names.append(f'p_({stat_name})')
                    # var_names.append(f'p({stat_name})')
                    var_names.append(f"pen({stat_name})")
                    # var_names.append(f'pot({stat_name})')
                    # var_names.append(f'pot_unit({stat_name})')
                    # var_names.append(f'smoothness({stat_name})')
                    # var_names.append(f'full_smoothness({stat_name})')
                    # var_names.append(f'alt_smoothness({stat_name})')
        elif spline_implementation == 'rw2':
            for stat_name in stat_names:
                var_names.append(f'w({stat_name})')
            if penalty_order is not None:
                for stat_name in stat_names:
                    var_names.append(f'p_({stat_name})')
                    var_names.append(f'p({stat_name})')
        elif spline_implementation == 'spectral':
            for stat_name in stat_names:
                var_names.append(f'w({stat_name})')
            if penalty_order is not None:
                for stat_name in stat_names:
                    var_names.append(f'sigma_pen({stat_name})')
    # Zero-inflation / link variables
    if link == 'logit':
        if b1_parameters is not None:
            var_names.append('zi_b1')
        if c_parameters is not None:
            var_names.append('zi_c')
        var_names.append('zi_b0')
        var_names.append('zi_c_true_scale')

    return var_names

def build_model_name(alpha_type, alpha_parameters,
                     intercept_type, intercept_parameters,
                     link, link_stat_name, link_type,
                     b1_type, b1_parameters,
                     c_type, c_parameters,
                     stat_names, num_knots, knot_type, degree,
                     spline_implementation, spline_type, spline_parameters,
                     penalty_order, penalty_type, penalty_parameters, penalty_std,
                     cutoff, beta_u_type, beta_u_parameters,
                     exclude=None,
                     surveillance_name=None,
                     urbanisation_name='urbanisation_pop_weighted_std'):

    surv = abbrev_surveillance(surveillance_name)
    urb = abbrev_urbanisation(urbanisation_name)
    stats = stat_names or []
    if len(stats) == 0:
        stat_str = "nostat"
    else:
        stat_str = "+".join(abbrev_stat(s) for s in stats)

    exclude = exclude or []

    def fmt_params(params):
        return "(" + ",".join(f"{k}={v}" for k, v in params.items()) + ")"

    parts = []
    
    # Spline stats 0
    if stat_names is not None and 'stats' not in exclude:
        # stats_str = "+".join(stat_names)
        # parts.append(f"stats[{stats_str}]")
        parts.append(f"knots({num_knots},{knot_type},deg{degree},{spline_implementation})") #,spline={spline_type}{',' + fmt_params(spline_parameters) if spline_parameters else ''})")

    # Penalty 1
    if penalty_order is not None and 'penalty' not in exclude:
        #pen_str = f"pen(ord={penalty_order}"
        pen_str = f"pen(o{penalty_order}"
        #if penalty_std is not None:
            #pen_str += f",std={penalty_std}"
        #pen_str += f",type={penalty_type}"
        if penalty_type == 'halfnormal' and penalty_parameters is not None:
            pen_str += "," + ",".join(f"{k}={v}" for k, v in penalty_parameters.items())
        pen_str += ")"
        parts.append(pen_str)

    # Link 2
    if link is not None and 'link' not in exclude:
        parts.append(f"link({link_stat_name},{link},{link_type},cut={cutoff})")

    # Alpha prior 3
    if alpha_parameters is not None and 'alpha' not in exclude:
        parts.append(f"alpha({alpha_type},{fmt_params(alpha_parameters)})")

    # Intercept prior 4
    if intercept_parameters is not None and 'intercept' not in exclude:
        parts.append(f"intercept({intercept_type},{fmt_params(intercept_parameters)})")

    # b1 prior 5
    if link is not None and b1_parameters is not None and 'b1' not in exclude:
        parts.append(f"b1({b1_type},{fmt_params(b1_parameters)})")

    # c prior 6
    if link is not None and c_parameters is not None and 'c' not in exclude:
        parts.append(f"c({c_type},{fmt_params(c_parameters)})")

    # Urbanisation prior 7
    if beta_u_parameters is not None and 'beta_u' not in exclude:
        parts.append(f"betau({beta_u_type},{fmt_params(beta_u_parameters)})")

    parts = [parts[i] for i in [2, 1, 0, 3, 4, 5, 6, 7, 8] if i < len(parts)]
    # parts = [f"[{surv}, {urb}][{stat_str}]"] + parts
    parts = [f"[{stat_str}]"] + parts
    return "__".join(parts)

def build_sig_spline_p_model_(data,
                             alpha_type, alpha_parameters,
                             intercept_type, intercept_parameters,
                             beta_u_type, beta_u_parameters,
                             link, link_stat_name, link_type,
                             b1_type, b1_parameters,
                             c_type, c_parameters,
                             stat_names, num_knots, knot_type, degree,
                             spline_implementation, spline_type, spline_parameters,
                             penalty_order, penalty_type, penalty_parameters, penalty_std,
                             cutoff, 
                             exclude=None,
                             surveillance_name=None,
                             urbanisation_name='urbanisation_pop_weighted_std'):

    m = build_model_name(alpha_type, alpha_parameters,
                        intercept_type, intercept_parameters,
                        link, link_stat_name, link_type,
                        b1_type, b1_parameters,
                        c_type, c_parameters,
                        stat_names, num_knots, knot_type, degree,
                        spline_implementation, spline_type, spline_parameters,
                        penalty_order, penalty_type, penalty_parameters, penalty_std,
                        cutoff, beta_u_type, beta_u_parameters,
                        exclude=exclude)

    model = pm.Model()
    with model:
        # Priors
        if alpha_type == 'exponential':
            alpha = pm.Exponential("alpha", lam=alpha_parameters['lam'])
        elif alpha_type == 'gamma':
            alpha = pm.Gamma("alpha", alpha=alpha_parameters['a'], beta=alpha_parameters['b'])
        if intercept_type == 'normal':
            intercept = pm.Normal("intercept", mu=intercept_parameters['mu'], sigma=intercept_parameters['sigma'])
        if urbanisation_name is not None:
            beta_u = pm.Normal("beta_u", mu=beta_u_parameters['mu'], sigma=beta_u_parameters['sigma'])
        
        # splines
        B = None
        knot_list = None
        if stat_names is not None:
            knot_list = {}
            B = {}
            sigma_w = {}
            w = {}
            f = {}
            for stat_name in stat_names:
                num_knots_ = num_knots
                d = data[stat_name].values
                # d = np.clip(d, np.percentile(d, 0.1), np.percentile(d, 99.9))
                if stat_name == link_stat_name:
                    num_knots_ = int(num_knots * (np.max(d)-cutoff) / (np.max(d)-np.min(d)))
                    d = np.clip(d, cutoff, None)  
                if knot_type=='equispaced':
                    knot_list[stat_name] = np.linspace(np.min(d), np.max(d), num_knots_+2)[1:-1]
                    # knot_list[stat_name] = np.linspace(np.min(d), np.max(d), num_knots+2)[:]
                elif knot_type=='quantile':
                    knot_list[stat_name] = np.percentile(np.unique(d), np.linspace(0, 100, num_knots_ + 2))[1:-1]
                else:
                    print('knot_list must be quantile or equispaced')

                B_full = dmatrix(f"bs(s, knots=knots, degree=degree, include_intercept=True)-1",
                        {"s": d, "knots": knot_list[stat_name], "degree":degree})

                if spline_implementation == 'spectral':
                    if (penalty_order is not None) and (penalty_order == 2):
                        k = B_full.shape[1]  # number of B-spline basis functions
                        D = difference_matrix(k, order=penalty_order)
                        P = D.T @ D
                        P_ = np.linalg.pinv(P)
                        BP_Bt = B_full @ P_ @ B_full.T

                        K = penalty_order  # null space dimension = penalty order
                        r = k - K          # rank of penalised part
                        V_pen, U_pen = eigsh(BP_Bt, k=r, which='LM')  # LM = Largest Magnitude

                        ###
                        X_pen = U_pen @ np.diag(np.sqrt(V_pen))
                        X_pen = np.ascontiguousarray(X_pen)

                        #
                        # Fix
                        X_pen = X_pen - X_pen.mean(axis=0)  # centre

                        v1 = np.arange(1, k+1, dtype=float)
                        f1 = B_full @ v1
                        f1 = f1 - f1.mean()
                        for j in range(X_pen.shape[1]):
                            col = X_pen[:, j]
                            f1 = f1 - (f1 @ col) / (col @ col) * col
                        f1 = f1 / np.linalg.norm(f1)
                        X1 = f1[:, None]
                        #z = data[stat_name].values  # the covariate values
                        #z_centred = z - z.mean()
                        #X1 = z_centred / (np.max(z_centred) - np.min(z_centred))  # scale to range [-0.5, 0.5]
                        #l = np.arange(1, B_full.shape[1]+1)
                        #X1 = B_full@l  # unpenalised component (the null space of the penalty)
                        X1 = (X1 - X1.mean()) / (np.max(X1) - np.min(X1))  # scale
                        #X1 = X1[:, None]  # (n x 1)
                        X1 = np.ascontiguousarray(X1)
                        ###

                        B[stat_name] = np.ascontiguousarray(np.hstack([X_pen, X1]))  # (n x (r+1))
                        #tau = pm.Gumbel(f"tau({stat_name})", alpha=2.0, beta=1.0)
                        sigma_pen = pm.HalfNormal(f"sigma_pen({stat_name})", sigma=10.0)
                        w_pen = pm.Normal(f"w_pen({stat_name})", mu=0, sigma=sigma_pen, size=r, dims="splines_pen")
                        w1 = pm.Normal(f"w1({stat_name})", mu=0, sigma=1.0)
                        w[stat_name] = pm.Deterministic(f"w({stat_name})", pt.concatenate([w_pen, [w1]]))
                        f[stat_name] = pm.math.dot(B[stat_name], w[stat_name])
                    else:
                        raise ValueError("spectral spline implementation requires a penalty")
                if spline_implementation == 'rw2':
                    if (penalty_order is not None) and (penalty_order == 2):
                        B[stat_name] = np.ascontiguousarray(B_full)  # ensure B is C-contiguous for PyMC
                        k = B_full.shape[1]  # number of B-spline basis functions
                        # Centre B so curve is mean zero
                        B_centred = B_full - B_full.mean(axis=0)
                        B[stat_name] = np.ascontiguousarray(B_centred)

                        # Penalty scale
                        if penalty_type == 'halfnormal':
                            p_ = pm.HalfNormal(f"p_({stat_name})", sigma=1.0)
                            h = (np.max(d) - np.min(d)) / (num_knots + 1)
                            p = pm.Deterministic(f'p({stat_name})', p_ * penalty_parameters['sigma'] / np.sqrt(h))

                            # RW2 via AR(2) with coefficients [2, -1]
                            # w0 fixed at 0, w1 free
                            w1 = pm.Normal(f"w1({stat_name})", mu=0, sigma=spline_parameters['sigma_w1'])
                            # w_i = 2*w_{i-1} - w_{i-2} + epsilon_i, epsilon_i ~ N(0, p^2)
                            w[stat_name] = pm.AR(
                                f"w({stat_name})",
                                rho=[2.0, -1.0],
                                sigma=p,
                                init_dist=pm.Normal.dist(mu=pt.stack([0.0, w1]), sigma=1e-6, shape=2),
                                steps=k-2
                            )
                            
                            f[stat_name] = pm.math.dot(B[stat_name], w[stat_name])
                    else:
                        raise ValueError("rw2 spline implementation requires a penalty_order to be = 2")

                elif spline_implementation == 'svd':
                    B_full_centred = B_full - B_full.mean(axis=0)  # centre the spline basis functions
                    U, S, Vt = np.linalg.svd(B_full_centred, full_matrices=False)
                    k = len(S)
                    r = np.sum(S > 1e-10)
                    U_r = U[:, :r]
                    S_r = S[:r]
                    Vt_r = Vt[:r, :]
                    #print(Vt[-1,:])
                    X_r = U_r @ np.diag(S_r)
                    X_r = np.ascontiguousarray(X_r)  # ensure X_r is C-contiguous for PyMC
                    B[stat_name] = X_r

                    # Spline coefficients
                    if spline_type == 'halfnormal':
                        sigma_w[stat_name] = pm.Deterministic(f"sigma_w({stat_name})", pt.as_tensor_variable(10.0))
                        #sigma_w[stat_name] = pm.HalfNormal(f"sigma_w({stat_name})", sigma=spline_parameters['sigma_w_sigma'])
                    elif spline_type == 'halfstudentt':
                        sigma_w[stat_name] = pm.Deterministic(f"sigma_w({stat_name})", pt.as_tensor_variable(10.0))
                        #sigma_w[stat_name] = pm.HalfStudentT(f"sigma_w({stat_name})", nu=spline_parameters['sigma_w_nu'], sigma=spline_parameters['sigma_w_sigma'])
                    w[stat_name] = pm.Normal(f"w({stat_name})", mu=0, sigma=sigma_w[stat_name], size=B[stat_name].shape[1], dims="splines")
                    # ws0 = B[stat_name].shape[1]
                    # w_ = pm.Normal(f"w_({stat_name})", mu=0, sigma=1.0, size=B[stat_name].shape[1], dims="splines")
                    # w[stat_name] = pm.Deterministic(f"w({stat_name})", w_ * sigma_w[stat_name])

                    f[stat_name] = pm.math.dot(B[stat_name], w[stat_name])

                    if penalty_order is not None:
                        if penalty_type == 'halfnormal':
                            #p_ = pm.HalfNormal(f"p_({stat_name})", sigma=1.0)
                            h = (np.max(d) - np.min(d)) / (num_knots + 1)
                            #p = pm.Deterministic(f'p({stat_name})', p_ * penalty_parameters['sigma'])
                            p = pm.Deterministic(f'p({stat_name})', pt.as_tensor_variable(penalty_parameters['p']))

                        D = difference_matrix(k, order=penalty_order)
                        DV = D @ Vt_r.T
                        DV = np.ascontiguousarray(DV)
                        DV = pt.as_tensor_variable(DV)

                        d_eval = np.linspace(np.min(d), np.max(d), 10000)
                        h_eval = (np.max(d) - np.min(d)) / 10000
                        B_eval = dmatrix(f"bs(s, knots=knots, degree=degree, include_intercept=True)-1",
                        {"s": d_eval, "knots": knot_list[stat_name], "degree":degree})
                        B_eval = B_eval - B_eval.mean(axis=0)  # centre
                        # _, _, Vt_eval = np.linalg.svd(B_full_centred, full_matrices=False)
                        # Vt_r_eval = Vt_eval[:r, :]
                        # print(B_eval.shape, Vt_r_eval.T.shape, B[stat_name].shape[1])
                        f_eval = B_eval @ Vt_r.T @ w[stat_name]  # evaluate the spline
                        f_dd = pt.diff(f_eval, n=2) / h_eval**2  # second derivative via finite differences
                        alt_smoothness = pm.Deterministic(f"alt_smoothness({stat_name})", pt.mean(f_dd**2))
                        
                        if penalty_std:
                            # w_std = pt.std(w[stat_name])
                            # w_std = pt.std(w_)
                            # Dw = pt.dot(DV, w[stat_name] / w_std)
                            # Dw = pt.dot(DV, w[stat_name] / )
                            
                            #Vw = Vt_r.T @ w[stat_name]
                            # w_scale = pt.dot(w[stat_name], w[stat_name]) / ws0 vvv should multiply by V so in original weight space
                            # w_scale = pt.dot(Vw, Vw) / ws0
                            #w_scale = pt.mean(pt.abs(Vw))**2 + 1e-6
                            DVw = pt.dot(DV, w[stat_name])
                            DVw1 = DVw[:-1]
                            DVw2 = DVw[1:]
                            scale = pt.mean(f[stat_name]**2) + 1e-6
                            int_d2f2 = 1/h**3*(2/3*pt.dot(DVw, DVw) + 1/6*pt.dot(DVw1, DVw2))
                            pm.Potential(f"spline_penalty({stat_name})", (-pt.log(p) -1/2*int_d2f2/scale/p**2) * r)
                            pot = pm.Deterministic(f"pot({stat_name})", (-pt.log(p) -1/2*int_d2f2/scale/p**2) * r)
                            pot_unit = pm.Deterministic(f"pot_unit({stat_name})", pot/r)
                            smoothness = pm.Deterministic(f"smoothness({stat_name})", 1/h**3*(2/3*pt.dot(DVw, DVw))/scale)
                            full_smoothness = pm.Deterministic(f"full_smoothness({stat_name})",
                                                               1/h**3*(2/3*pt.dot(DVw, DVw) + 1/6*pt.dot(DVw1, DVw2))/scale)
                        else:
                            DVw = pt.dot(DV, w[stat_name])
                            DVw1 = DVw[:-1]
                            DVw2 = DVw[1:]
                            int_d2f2 = 1/h**3*(2/3*pt.dot(DVw, DVw) + 1/6*pt.dot(DVw1, DVw2))
                            pm.Potential(f"spline_penalty({stat_name})", (-pt.log(p) -1/2*int_d2f2/p**2) * r)
                            pot = pm.Deterministic(f"pot({stat_name})", (-pt.log(p) -1/2*int_d2f2/p**2) * r)
                            pot_unit = pm.Deterministic(f"pot_unit({stat_name})", pot/r)
                            smoothness = pm.Deterministic(f"smoothness({stat_name})", 1/h**3*(2/3*pt.dot(DVw, DVw)))
                            full_smoothness = pm.Deterministic(f"full_smoothness({stat_name})",
                                                               1/h**3*(2/3*pt.dot(DVw, DVw) + 1/6*pt.dot(DVw1, DVw2)))

        # Link
        log_mu = intercept + pm.math.log(data['population'])
        surveillance_name = None
        if surveillance_name is not None:
            log_mu += pm.math.log(pm.math.max(data[surveillance_name], pm.math.log(1e-3)))
        if urbanisation_name is not None:
            log_mu += beta_u*data[urbanisation_name]
        if stat_names is not None:
            for stat_name in stat_names:
                log_mu += f[stat_name]

        # Zero-inflation component
        if link is None:
            y_obs = pm.NegativeBinomial('y_obs', mu=pm.math.exp(log_mu), alpha=alpha, observed=data['cases'])
        else:
            x = data[link_stat_name].values
            x_mean = np.mean(x)
            x_std_dev = np.std(x)
            x_std = (x - x_mean) / x_std_dev
            if link == 'logit':
                if b1_type == 'halfnormal':
                    zi_b1 = pm.HalfNormal("zi_b1", sigma=b1_parameters['sigma'])
                if c_type == 'normal':
                    zi_c = pm.Normal("zi_c", mu=c_parameters['mu'], sigma=c_parameters['sigma'])
                zi_b0 = pm.Deterministic("zi_b0", -zi_c * zi_b1)
                zi_x = zi_b0 + zi_b1 * x_std

                #zi_b0_true_scale = pm.Deterministic("zi_b0_true_scale", - zi_b1/x_std_dev*(zi_c*x_std_dev)+x_mean)
                #zi_b1_true_scale = pm.Deterministic("zi_b1_true_scale", zi_b1 / x_std_dev)
                zi_c_true_scale = pm.Deterministic("zi_c_true_scale", zi_c * x_std_dev + x_mean)
            
                # Likelihood
                if link_type == 'multiplicative':
                    y_obs = pm.NegativeBinomial('y_obs', mu=pm.math.invlogit(zi_x) * pm.math.exp(log_mu), alpha=alpha, observed=data['cases'])
                elif link_type == 'additive':
                    y_obs = pm.ZeroInflatedNegativeBinomial('y_obs', psi=pm.math.invlogit(zi_x), mu=pm.math.exp(log_mu), alpha=alpha, observed=data['cases'])

    # m += f"__non-centred-w"
    # m += "boundary_knots"
    m = m[:200]
    
    return model, m, B, knot_list

def build_sig_spline_p_model(data,
                             alpha_type, alpha_parameters,
                             intercept_type, intercept_parameters,
                             beta_u_type, beta_u_parameters,
                             link, link_stat_name, link_type,
                             b1_type, b1_parameters,
                             c_type, c_parameters,
                             stat_names, num_knots, knot_type, degree,
                             spline_implementation, spline_type, spline_parameters,
                             penalty_order, penalty_type, penalty_parameters, penalty_std,
                             cutoff, 
                             exclude=None,
                             surveillance_name=None,
                             urbanisation_name='urbanisation_pop_weighted_std'):

    m = build_model_name(alpha_type, alpha_parameters,
                        intercept_type, intercept_parameters,
                        link, link_stat_name, link_type,
                        b1_type, b1_parameters,
                        c_type, c_parameters,
                        stat_names, num_knots, knot_type, degree,
                        spline_implementation, spline_type, spline_parameters,
                        penalty_order, penalty_type, penalty_parameters, penalty_std,
                        cutoff, beta_u_type, beta_u_parameters,
                        exclude=exclude)

    model = pm.Model()
    with model:
        # Priors
        if alpha_type == 'exponential':
            alpha = pm.Exponential("alpha", lam=alpha_parameters['lam'])
        elif alpha_type == 'gamma':
            alpha = pm.Gamma("alpha", alpha=alpha_parameters['a'], beta=alpha_parameters['b'])
        if intercept_type == 'normal':
            intercept = pm.Normal("intercept", mu=intercept_parameters['mu'], sigma=intercept_parameters['sigma'])
        if urbanisation_name is not None:
            beta_u = pm.Normal("beta_u", mu=beta_u_parameters['mu'], sigma=beta_u_parameters['sigma'])
        
        # splines
        B = None
        knot_list = None
        if stat_names is not None:
            knot_list = {}
            B = {}
            sigma_w = {}
            w = {}
            f = {}
            for stat_name in stat_names:
                num_knots_ = num_knots
                d = data[stat_name].values
                # d = np.clip(d, np.percentile(d, 0.1), np.percentile(d, 99.9))
                if stat_name == link_stat_name:
                    num_knots_ = int(num_knots * (np.max(d)-cutoff) / (np.max(d)-np.min(d)))
                    d = np.clip(d, cutoff, None)  
                if knot_type=='equispaced':
                    knot_list[stat_name] = np.linspace(np.min(d), np.max(d), num_knots_+2)[1:-1]
                    # knot_list[stat_name] = np.linspace(np.min(d), np.max(d), num_knots+2)[:]
                elif knot_type=='quantile':
                    knot_list[stat_name] = np.percentile(np.unique(d), np.linspace(0, 100, num_knots_ + 2))[1:-1]
                else:
                    print('knot_list must be quantile or equispaced')

                B_full = dmatrix(f"bs(s, knots=knots, degree=degree, include_intercept=True)-1",
                        {"s": d, "knots": knot_list[stat_name], "degree":degree})

                if spline_implementation == 'svd':
                    B_full_centred = B_full - B_full.mean(axis=0)  # centre the spline basis functions
                    U, S, Vt = np.linalg.svd(B_full_centred, full_matrices=False)
                    k = len(S)
                    r = np.sum(S > 1e-10)
                    U_r = U[:, :r]
                    S_r = S[:r]
                    Vt_r = Vt[:r, :]
                    X_r = U_r @ np.diag(S_r)
                    X_r = np.ascontiguousarray(X_r)  # ensure X_r is C-contiguous for PyMC
                    B[stat_name] = X_r

                    # Spline coefficients
                    if spline_type == 'halfnormal':
                        sigma_w[stat_name] = pm.Deterministic(f"sigma_w({stat_name})", pt.as_tensor_variable(10.0))
                        #sigma_w[stat_name] = pm.HalfNormal(f"sigma_w({stat_name})", sigma=spline_parameters['sigma_w_sigma'])
                    elif spline_type == 'halfstudentt':
                        sigma_w[stat_name] = pm.Deterministic(f"sigma_w({stat_name})", pt.as_tensor_variable(10.0))
                        #sigma_w[stat_name] = pm.HalfStudentT(f"sigma_w({stat_name})", nu=spline_parameters['sigma_w_nu'], sigma=spline_parameters['sigma_w_sigma'])
                    w[stat_name] = pm.Normal(f"w({stat_name})", mu=0, sigma=sigma_w[stat_name], size=B[stat_name].shape[1], dims="splines")
                    # ws0 = B[stat_name].shape[1]
                    # w_ = pm.Normal(f"w_({stat_name})", mu=0, sigma=1.0, size=B[stat_name].shape[1], dims="splines")
                    # w[stat_name] = pm.Deterministic(f"w({stat_name})", w_ * sigma_w[stat_name])

                    f[stat_name] = pm.math.dot(B[stat_name], w[stat_name])

                    if penalty_order is not None:
                        if penalty_type == 'halfnormal':
                            #p_ = pm.HalfNormal(f"p_({stat_name})", sigma=1.0)
                            h = (np.max(d) - np.min(d)) / (num_knots + 1)
                            #p = pm.Deterministic(f'p({stat_name})', p_ * penalty_parameters['sigma'])
                            p = pm.Deterministic(f'p({stat_name})', pt.as_tensor_variable(penalty_parameters['p']))

                        D = difference_matrix(k, order=penalty_order)
                        DV = D @ Vt_r.T
                        DV = np.ascontiguousarray(DV)
                        DV = pt.as_tensor_variable(DV)
                        
                        if penalty_std:
                            raise ValueError("penalty_std=True is not allowed")
                        else:
                            DVw = pt.dot(DV, w[stat_name])
                            DVw1 = DVw[:-1]
                            DVw2 = DVw[1:]
                            int_d2f2 = 1/h**3*(2/3*pt.dot(DVw, DVw) + 1/6*pt.dot(DVw1, DVw2))
                            pm.Potential(f"spline_penalty({stat_name})", (-pt.log(p) -1/2*int_d2f2/p**2) * r)
                            pot = pm.Deterministic(f"pot({stat_name})", (-pt.log(p) -1/2*int_d2f2/p**2) * r)
                            pot_unit = pm.Deterministic(f"pot_unit({stat_name})", pot/r)
                            smoothness = pm.Deterministic(f"smoothness({stat_name})", 1/h**3*(2/3*pt.dot(DVw, DVw)))
                            full_smoothness = pm.Deterministic(f"full_smoothness({stat_name})",
                                                               1/h**3*(2/3*pt.dot(DVw, DVw) + 1/6*pt.dot(DVw1, DVw2)))

        # Link
        log_mu = intercept + pm.math.log(data['population'])
        surveillance_name = None
        if surveillance_name is not None:
            log_mu += pm.math.log(pm.math.max(data[surveillance_name], pm.math.log(1e-3)))
        if urbanisation_name is not None:
            log_mu += beta_u*data[urbanisation_name]
        if stat_names is not None:
            for stat_name in stat_names:
                log_mu += f[stat_name]

        # Zero-inflation component
        if link is None:
            y_obs = pm.NegativeBinomial('y_obs', mu=pm.math.exp(log_mu), alpha=alpha, observed=data['cases'])
        else:
            x = data[link_stat_name].values
            x_mean = np.mean(x)
            x_std_dev = np.std(x)
            x_std = (x - x_mean) / x_std_dev
            if link == 'logit':
                if b1_type == 'halfnormal':
                    zi_b1 = pm.HalfNormal("zi_b1", sigma=b1_parameters['sigma'])
                if c_type == 'normal':
                    zi_c = pm.Normal("zi_c", mu=c_parameters['mu'], sigma=c_parameters['sigma'])
                zi_b0 = pm.Deterministic("zi_b0", -zi_c * zi_b1)
                zi_x = zi_b0 + zi_b1 * x_std

                #zi_b0_true_scale = pm.Deterministic("zi_b0_true_scale", - zi_b1/x_std_dev*(zi_c*x_std_dev)+x_mean)
                #zi_b1_true_scale = pm.Deterministic("zi_b1_true_scale", zi_b1 / x_std_dev)
                zi_c_true_scale = pm.Deterministic("zi_c_true_scale", zi_c * x_std_dev + x_mean)
            
                # Likelihood
                if link_type == 'multiplicative':
                    y_obs = pm.NegativeBinomial('y_obs', mu=pm.math.invlogit(zi_x) * pm.math.exp(log_mu), alpha=alpha, observed=data['cases'])
                elif link_type == 'additive':
                    y_obs = pm.ZeroInflatedNegativeBinomial('y_obs', psi=pm.math.invlogit(zi_x), mu=pm.math.exp(log_mu), alpha=alpha, observed=data['cases'])

    # m += f"__non-centred-w"
    # m += "boundary_knots"
    m = m[:200]
    
    return model, m, B, knot_list

def build_model_equispaced_exact(data,
                             alpha_type, alpha_parameters,
                             intercept_type, intercept_parameters,
                             beta_u_type, beta_u_parameters,
                             link, link_stat_name, link_type,
                             b1_type, b1_parameters,
                             c_type, c_parameters,
                             stat_names, num_knots, knot_type, degree,
                             spline_implementation, spline_type, spline_parameters,
                             penalty_order, penalty_type, penalty_parameters, penalty_std,
                             cutoff, 
                             exclude=None,
                             surveillance_name=None,
                             urbanisation_name='urbanisation_pop_weighted_std'):

    m = build_model_name(alpha_type, alpha_parameters,
                        intercept_type, intercept_parameters,
                        link, link_stat_name, link_type,
                        b1_type, b1_parameters,
                        c_type, c_parameters,
                        stat_names, num_knots, knot_type, degree,
                        spline_implementation, spline_type, spline_parameters,
                        penalty_order, penalty_type, penalty_parameters, penalty_std,
                        cutoff, beta_u_type, beta_u_parameters,
                        exclude=exclude)

    model = pm.Model()
    with model:
        # Priors
        if alpha_type == 'exponential':
            alpha = pm.Exponential("alpha", lam=alpha_parameters['lam'])
        elif alpha_type == 'gamma':
            alpha = pm.Gamma("alpha", alpha=alpha_parameters['a'], beta=alpha_parameters['b'])
        if intercept_type == 'normal':
            intercept = pm.Normal("intercept", mu=intercept_parameters['mu'], sigma=intercept_parameters['sigma'])
        if urbanisation_name is not None:
            beta_u = pm.Normal("beta_u", mu=beta_u_parameters['mu'], sigma=beta_u_parameters['sigma'])
        
        # splines
        B = None
        knot_list = None
        if stat_names is not None:
            knot_list = {}
            B = {}
            sigma_w = {}
            w = {}
            f = {}
            for stat_name in stat_names:
                d = data[stat_name].values
                # d = np.clip(d, np.percentile(d, 0.1), np.percentile(d, 99.9))

                B_full, _, _, knot_list[stat_name] = eval_spline_basis_equispaced_numeric(degree, np.min(d), np.max(d), num_knots, d).values()

                if spline_implementation == 'svd':
                    B_full_centred = B_full - B_full.mean(axis=0)  # centre the spline basis functions
                    U, S, Vt = np.linalg.svd(B_full_centred, full_matrices=False)
                    k = len(S)
                    r = np.sum(S > 1e-10)
                    U_r = U[:, :r]
                    S_r = S[:r]
                    Vt_r = Vt[:r, :]
                    V_r = np.ascontiguousarray(Vt_r.T)
                    X_r = U_r @ np.diag(S_r)
                    X_r = np.ascontiguousarray(X_r)  # ensure X_r is C-contiguous for PyMC
                    B[stat_name] = X_r

                    # Spline coefficients
                    if spline_type == 'halfnormal':
                        sigma_w[stat_name] = pm.Deterministic(f"sigma_w({stat_name})", pt.as_tensor_variable(10.0))
                        #sigma_w[stat_name] = pm.HalfNormal(f"sigma_w({stat_name})", sigma=spline_parameters['sigma_w_sigma'])
                    elif spline_type == 'halfstudentt':
                        sigma_w[stat_name] = pm.Deterministic(f"sigma_w({stat_name})", pt.as_tensor_variable(10.0))
                        #sigma_w[stat_name] = pm.HalfStudentT(f"sigma_w({stat_name})", nu=spline_parameters['sigma_w_nu'], sigma=spline_parameters['sigma_w_sigma'])
                    w[stat_name] = pm.Normal(f"w({stat_name})", mu=0, sigma=sigma_w[stat_name], size=B[stat_name].shape[1], dims="splines")
                    # ws0 = B[stat_name].shape[1]
                    # w_ = pm.Normal(f"w_({stat_name})", mu=0, sigma=1.0, size=B[stat_name].shape[1], dims="splines")
                    # w[stat_name] = pm.Deterministic(f"w({stat_name})", w_ * sigma_w[stat_name])

                    f[stat_name] = pm.math.dot(B[stat_name], w[stat_name])

                    if penalty_order is not None:
                        if penalty_type == 'halfnormal':
                            h = (np.max(d) - np.min(d)) / (num_knots + 1)
                            #p_ = pm.HalfNormal(f"p_({stat_name})", sigma=1.0)
                            #p = pm.Deterministic(f'p({stat_name})', p_ * penalty_parameters['sigma'])
                            p = pm.Deterministic(f'p({stat_name})', pt.as_tensor_variable(penalty_parameters['p']))

                        #D = difference_matrix(k, order=penalty_order)
                        #DV = D @ Vt_r.T
                        #DV = np.ascontiguousarray(DV)
                        #DV = pt.as_tensor_variable(DV)
                        
                        if penalty_std:
                            raise ValueError("penalty_std=True is not allowed")
                        else:
                            D2 = difference_matrix(k, order=2)        # (k-2, k)
                            #print(D2)
                            D2V = np.ascontiguousarray(D2 @ V_r)        # (k-2, k)(k, r) = (k-2, r)
                            D2V_pt = pt.as_tensor_variable(D2V)
                            D2Vw = pt.dot(D2V_pt, w[stat_name])
                            int_f_dd_sq = (1 / (3 * h**3)) * (
                                            D2Vw[0]**2
                                            + 2 * pt.sum(D2Vw[1:-1]**2)
                                            + D2Vw[-1]**2
                                            + pt.sum(D2Vw[:-1] * D2Vw[1:])
                                        )
                            # Vw = Vt_r.T @ w[stat_name]
                            # d2Vw = Vw[2:] - 2*Vw[1:-1] + Vw[:-2]
                            # int_f_dd_sq = (1/(3*h**3))*(d2Vw[0]**2 + 2*pt.sum(d2Vw[1:-1]**2) + d2Vw[-1]**2 + pt.sum(d2Vw[0:-1]*d2Vw[1:]))
                            # int_d2f2 = 1/h**3*(2/3*pt.dot(DVw, DVw) + 1/6*pt.dot(DVw1, DVw2))
                            pm.Potential(f"spline_penalty({stat_name})", (-pt.log(p) -1/2*int_f_dd_sq/p**2) * r)
                            pen = pm.Deterministic(f"pen({stat_name})", (-pt.log(p) -1/2*int_f_dd_sq/p**2))
                            # pot = pm.Deterministic(f"pot({stat_name})", (-pt.log(p) -1/2*int_d2f2/p**2) * r)
                            # pot_unit = pm.Deterministic(f"pot_unit({stat_name})", pot/r)
                            # smoothness = pm.Deterministic(f"smoothness({stat_name})", 1/h**3*(2/3*pt.dot(DVw, DVw)))
                            #full_smoothness = pm.Deterministic(f"full_smoothness({stat_name})",
                                                               # 1/h**3*(2/3*pt.dot(DVw, DVw) + 1/6*pt.dot(DVw1, DVw2)))

        # Link
        log_mu = intercept + pm.math.log(data['population'])
        surveillance_name = None
        if surveillance_name is not None:
            log_mu += pm.math.log(pm.math.max(data[surveillance_name], pm.math.log(1e-3)))
        if urbanisation_name is not None:
            log_mu += beta_u*data[urbanisation_name]
        if stat_names is not None:
            for stat_name in stat_names:
                log_mu += f[stat_name]

        # Zero-inflation component
        if link is None:
            y_obs = pm.NegativeBinomial('y_obs', mu=pm.math.exp(log_mu), alpha=alpha, observed=data['cases'])
        else:
            raise ValueError("link is not allowed for this model")

    m = m[:200]
    
    return model, m, B, knot_list

def fit_sig_spline_p_model(data, data_name,
                           model_settings,
                           outpath, task,
                           n_chains=4, n_draws=500, n_tune=500, sampler="nutpie",
                           invert_log=False, centred_w=True,
                           check_report=True, check_idata=True, clear_idata=False,
                           basis_scale=1,
                           model_builder='build_sig_spline_p_model',
                           show = {'summary': True, 'trace': True, 'pair': True, 'metrics': True,
                            'spline': True, 'exp_spline': True, 'link': True, 'link_spline': True,
                            'divergences': True}):
    
    if model_builder == 'build_sig_spline_p_model':
        model, m, B, knot_list = build_sig_spline_p_model(data.copy(), **model_settings)
        model_name = m
    if model_builder == 'build_model_equispaced_exact':
        model, m, B, knot_list = build_model_equispaced_exact(data.copy(), **model_settings)
        model_name = m  
    var_names = build_var_names(model_settings['alpha_parameters'], model_settings['intercept_parameters'], model_settings['link'], model_settings['link_type'],
                                    model_settings['b1_parameters'], model_settings['c_parameters'], model_settings['stat_names'], model_settings['spline_implementation'],
                                    model_settings['penalty_order'], model_settings['penalty_parameters'], model_settings['beta_u_parameters'])

    data_path = os.path.join(outpath, f'{data_name}[{task}]/')
    os.makedirs(data_path, exist_ok=True)
    
    idata_path = os.path.join(data_path, 'idata')
    os.makedirs(idata_path, exist_ok=True)
    report_path = os.path.join(data_path, f'reports/')
    os.makedirs(report_path, exist_ok=True)
    metrics_path = os.path.join(data_path, f'metrics')
    os.makedirs(metrics_path, exist_ok=True)
    output_path = os.path.join(data_path, f'outputs/{model_name}')
    os.makedirs(output_path, exist_ok=True)

    # if report already exists, skip
    idata_file = os.path.join(idata_path, f"idata_[{model_name}].nc")
    report_file = os.path.join(report_path, f"report_[{model_name}].html")
    if check_report and os.path.exists(report_file):
        print(f"Skipping {model_name}, report already exists.")
        return
    
    with model:
        if check_idata and os.path.exists(idata_file):
            print(f"Skipping {model_name} idata compute, already exists.")
            idata = az.from_netcdf(idata_file)
            with open(os.path.join(output_path, "divergences.txt"), "r") as f:
                n_divergences = int(f.readline().strip())
            if "log_likelihood" not in idata.groups():
                print(f"Log likelihood missing, recomputing...")
                pm.compute_log_likelihood(idata, progressbar=False)
                # tmp_file = os.path.join(idata_path, f"temp_idata_[{model_name}].nc")
                # idata_thinned = idata.sel(draw=slice(None, None, 12))
                # idata_thinned.to_netcdf(tmp_file)
                # os.replace(tmp_file, idata_file)  # atomic replace
                idata.to_netcdf(idata_file)
                print('saved idata to', idata_file)
                times = (0.0, 0.0)
            else:
                with open(os.path.join(output_path, "times.txt"), "r") as f:
                    sampling_time = float(f.readline().strip())
                    log_likelihood_time = float(f.readline().strip())
                    times = (sampling_time, log_likelihood_time)
        else:
            target_accept = 0.8
            max_treedepth = 10
            max_energy_error = 1000

            s0 = time.time()
            idata = pm.sample(
                tune=n_tune,
                draws=n_draws,
                chains=n_chains,
                cores=n_chains,
                discard_tuned_samples=True,
                store_divergences=True,
                nuts_sampler="nutpie",
                target_accept = target_accept,
                max_treedepth = max_treedepth, # comment out below when using CPU
                nuts_sampler_kwargs={"max_energy_error": max_energy_error, 'backend': 'jax', 'gradient_backend': 'jax'},
                progressbar=True
            )
            s1 = time.time()
            pm.compute_log_likelihood(idata, progressbar=False)
            s2 = time.time()
            times = (s1 - s0, s2 - s1)
            n_divergences = int(idata.sample_stats["diverging"].sum())
            # save times and n_divergences to file, such that I can also read back later without loading the whole idata
            with open(os.path.join(output_path, "divergences.txt"), "w") as f:
                f.write(f"{n_divergences}\n")
            with open(os.path.join(output_path, "times.txt"), "w") as f:
                f.write(f"{times[0]}\n{times[1]}\n")
            # Save inference data
            idata_thinned = idata.sel(draw=slice(None, None, 8))
            idata_thinned.to_netcdf(idata_file)
            print('saved idata to', idata_file)
            print(f'\nPosterior Sampling {s1 - s0:.2f} seconds')
            print(f'Log Likelihood Compute {s2 - s1:.2f} seconds \n')


            #### Time Metrics
            metrics_df = pd.DataFrame([{"model_name": model_name,
                                        "data_name": data_name,
                                        "sampling_time_sec": s1 - s0,
                                        "log_likelihood_time_sec": s2 - s1,
                                        "n_chains": n_chains,
                                        "n_draws": n_draws,
                                        "n_tune": n_tune,
                                        "sampler": sampler}])
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
        eval_waic = az.waic(idata, pointwise=True)
        eval_psis_loo_elpd = az.loo(idata, pointwise=True)

    # Save pointwise values for later comparison
    pointwise_file = os.path.join(metrics_path, f"_metrics[{model_name}].npz")
    #np.savez(
        #pointwise_file,
        #waic_pointwise=eval_waic.waic_i.values,
        #loo_pointwise=eval_psis_loo_elpd.loo_i.values,
        #pareto_k=eval_psis_loo_elpd.pareto_k.values
    #)
    mtime = time.time()
    np.savez(
        pointwise_file,
        loo_elpd_loo = eval_psis_loo_elpd.elpd_loo,
        loo_se = eval_psis_loo_elpd.se,
        loo_p_loo = eval_psis_loo_elpd.p_loo,
        loo_n_samples = eval_psis_loo_elpd.n_samples,
        loo_n_data_points = eval_psis_loo_elpd.n_data_points,
        loo_warning = eval_psis_loo_elpd.warning,

        loo_pointwise=eval_psis_loo_elpd.loo_i.values,
        pareto_k=eval_psis_loo_elpd.pareto_k.values,

        waic_elpd_waic = eval_waic.elpd_waic,
        waic_se = eval_waic.se,
        waic_p_waic = eval_waic.p_waic,
        waic_warning = eval_waic.warning,

        waic_pointwise=eval_waic.waic_i.values,
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
    # khat_fig.savefig(fig_file, bbox_inches="tight")
    plt.close(khat_fig)
    ####
    idata_posterior = idata.posterior
    # ---------- Summary table ----------
    if show['summary']:
        filtered = []
        for v in var_names:
            if v in idata_posterior:
                vals = idata_posterior[v].values
                if np.std(vals) > 0:
                    filtered.append(v)
        summary_df = az.summary(idata, var_names=filtered)
        summary_file = os.path.join(output_path, "summary.csv")
        summary_df.to_csv(summary_file)
        summary_html = summary_df.to_html()
    else:
        summary_html = None

    # ---------- Trace plot ----------
    if show['trace']:
        fig_trace = az.plot_trace(idata, var_names=var_names)
        fig_trace = fig_trace.ravel()[0].figure
        fig_file = os.path.join(output_path, f"trace.png")
        # fig_trace.savefig(fig_file, bbox_inches="tight")
        plt.close(fig_trace)
        trace_img = fig_to_base64(fig_trace)
    else:
        trace_img = None

    # ---------- Pair plot ----------
    if show['pair']:
        az.rcParams["plot.max_subplots"] = 200
        ax = az.plot_pair(
            idata,
            var_names=var_names,
            textsize=14,
            divergences=True)

        for i in range(ax.shape[0]):
            ax[i, 0].yaxis.label.set_rotation(0)
            ax[i, 0].yaxis.label.set_ha('right')
        for j in range(ax.shape[1]):
            ax[0, j].xaxis.label.set_rotation(45)
            ax[0, j].xaxis.label.set_ha('right')
        fig_pair = ax.ravel()[0].figure
        fig_file = os.path.join(output_path, f"pair.png")
        # fig_pair.savefig(fig_file, bbox_inches="tight")
        plt.close(fig_pair)
        pair_img = fig_to_base64(fig_pair)
    else:
        pair_img = None

    # ---------- WAIC and PSIS LOO ----------
    if show['metrics']:
        # with warnings.catch_warnings():
            # warnings.simplefilter("ignore")
             #eval_waic = az.waic(idata)
            # eval_psis_loo_elpd = az.loo(idata)
        wl_df = pd.DataFrame([elpd_to_row(eval_waic, eval_psis_loo_elpd, m, 'd')])
        wl_html = wl_df.to_html()
    else:
        wl_html = None

    # ---------- Spline plot ----------
    stat_names = model_settings['stat_names']
    if show['spline']:
        spline_imgs = []
        if stat_names is not None:
            for stat_name in stat_names:
                fig_spline = plot_spline_Bknots(
                    idata,
                    stat_name,
                    f'w({stat_name})',
                    f'sigma_w({stat_name})',
                    B[stat_name],
                    data[stat_name].values,
                    knots=knot_list[stat_name],
                    show_basis=True,
                    invert_log=True,
                    centred_w=True
                )
                fig_spline_file = os.path.join(output_path, f"spline_{stat_name}.png")
                # fig_spline.savefig(fig_spline_file, bbox_inches="tight", dpi=200)
                plt.close(fig_spline)
                spline_imgs.append(fig_to_base64(fig_spline))
    else:
        spline_imgs = None

    if show['exp_spline']:
        exp_spline_imgs = []
        if stat_names is not None:
            for stat_name in stat_names:
                fig_exp_spline = plot_exp_spline(
                    data[stat_name].values,
                    idata,
                    stat_name,
                    B[stat_name],
                    knot_list[stat_name],
                    data=data
                )
                fig_exp_spline_file = os.path.join(output_path, f"spline_{stat_name}.png")
                # fig_exp_spline.savefig(fig_exp_spline_file, bbox_inches="tight", dpi=200)
                plt.close(fig_exp_spline)
                exp_spline_imgs.append(fig_to_base64(fig_exp_spline))
    else:
        exp_spline_imgs = None

    link = model_settings['link']
    link_stat_name = model_settings['link_stat_name']
    if show['link']:
        if link is not None:
            fig_link = plot_link(data[link_stat_name].values, idata,
                                            var_names=['zi_b0', 'zi_b1'], link=link)
            zi_img = fig_to_base64(fig_link)
        else:
            zi_img = None
    else:
        zi_img = None

    if (show['link_spline']&(link is not None)&(stat_names is not None)&(link_stat_name in stat_names)):
        fig_link_spline = plot_link_spline(data[link_stat_name].values, idata,
                                            link_stat_name, B[link_stat_name],
                                            knot_list[link_stat_name], link=link)
        zi_s_img = fig_to_base64(fig_link_spline)
    else:
        zi_s_img = None
        
    #--- Divergences plot ---
    if show['divergences']&(n_divergences > 0):
        posterior = idata_posterior.to_dataframe().reset_index()
        stats = idata.sample_stats.to_dataframe().reset_index()
        df = posterior.merge(stats, on=["chain","draw"])
        sns.pairplot( df, vars=var_names, hue="diverging",
                    corner=True, diag_kind='kde', plot_kws={"alpha":0.5, 's':1}, diag_kws={"common_norm": False})
        div_img = fig_to_base64(plt.gcf())
    else:
        div_img = None

    go_report(report_path, m, idata, n_divergences, times, show, summary_html, wl_html, trace_img, pair_img, spline_imgs, exp_spline_imgs, zi_img, zi_s_img, div_img)
    # create_html_report(output_path, model_name=model_name, n_draws=n_draws, reports_folder=report_path, replace=(not check_report), clear_images=True)
    if clear_idata:
        # delete nc file to save space
        os.remove(idata_file)
    return

def go_report(report_path, m, idata, n_divergences, times, show, summary_html, wl_html, trace_img, pair_img, spline_imgs, exp_spline_imgs, zi_img, zi_s_img, div_img):
    # ---------- Build HTML ----------
    html_parts = []

    # --- Precompute reusable values ---
    total_samples = (
        idata.posterior.sizes['draw'] *
        idata.posterior.sizes['chain']
    )
    div_pct = n_divergences / total_samples * 100

    # --- Header ---
    html_parts.append(f"""
    <html>
    <head>
        <title>Model Report: {m}</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            h1 {{ margin-bottom: 10px; }}
            img {{ margin-top: 20px; max-width: 100%; }}
            table {{ border-collapse: collapse; }}
            th, td {{ padding: 6px 8px; }}
        </style>
    </head>
    <body>
        <h1>Model Report: {m}</h1>

        <h2>Timing</h2>
        <p>Posterior Sampling: {times[0]:.2f} seconds</p>
        <p>Log Likelihood Compute: {times[1]:.2f} seconds</p>
    """)

    # --- Summary ---
    if show['summary']:
        html_parts.append(f"""
        <h2>Summary</h2>
        {summary_html}
        <p>Divergences: {n_divergences} out of {total_samples} samples ({div_pct:.2f}%)</p>
        """)

    # --- Metrics ---
    if show['metrics']:
        html_parts.append(f"""
        <h2>WAIC and PSIS LOO</h2>
        {wl_html}
        """)

    # --- Trace ---
    if show['trace']:
        html_parts.append(f"""
        <h2>Trace Plot</h2>
        <img src="data:image/png;base64,{trace_img}">
        """)

    # --- Pair ---
    if show['pair']:
        html_parts.append(f"""
        <h2>Pair Plot</h2>
        <img src="data:image/png;base64,{pair_img}">
        """)

    # --- Spline ---
    if show['spline']:
        imgs = "".join(f'<img src="data:image/png;base64,{img}">' for img in spline_imgs)
        html_parts.append(f"""
        <h2>Spline Plot</h2>
        {imgs}
        """)

    # --- Exponential Spline ---
    if show['exp_spline']:
        imgs = "".join(f'<img src="data:image/png;base64,{img}">' for img in exp_spline_imgs)
        html_parts.append(f"""
        <h2>Exponential Spline Plot</h2>
        {imgs}
        """)

    # --- Link ---
    if show['link']:
        img_html = f'<img src="data:image/png;base64,{zi_img}">' if zi_img is not None else ""
        html_parts.append(f"""
        <h2>ZI link Plot</h2>
        {img_html}
        """)

    # --- Link spline ---
    if show['link_spline']:
        img_html = f'<img src="data:image/png;base64,{zi_s_img}">' if zi_s_img is not None else ""
        html_parts.append(f"""
        <h2>ZI link with Spline Plot</h2>
        {img_html}
        """)

    # --- Divergences ---
    if show['divergences']:
        img_html = (
            f'<img src="data:image/png;base64,{div_img}">'
            if n_divergences > 0
            else "<p>No divergences detected.</p>"
        )
        html_parts.append(f"""
        <h2>Divergences Plot</h2>
        <p>Divergences: {n_divergences} out of {total_samples} samples ({div_pct:.2f}%)</p>
        {img_html}
        """)

    # --- Footer ---
    html_parts.append("""
    </body>
    </html>
    """)

    html_content = "".join(html_parts)

    report_file = os.path.join(report_path, f"report_[{m}].html")
    with open(report_file, "w") as f:
        f.write(html_content)

    print(f"Saved report to {report_file}")


def ess_style(x, n_draws):
    if isinstance(x, (int, float)):
        if x < n_draws / 5:
            return "background-color: red;"
        elif x < n_draws / 4:
            return "background-color: yellow;"
        else:
            return "background-color: lightgreen;"
    return ""

def create_html_report(model_folder, model_name, n_draws, reports_folder=None, title=None, replace=False, clear_images=False):
    """
    Generate HTML report for a single model.

    Args:
        model_folder: path to the model_name folder containing csv/images
        model_name: name of the model
        n_draws: number of draws for ESS coloring
        reports_folder: if provided, also generate a report in this folder
        title: optional HTML title
        clear_images: if True, remove image files after generating the report
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

    if clear_images:
        # remove images to save space
        for _, path in img_files:
            os.remove(path)
