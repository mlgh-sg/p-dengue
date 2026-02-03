import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from patsy import dmatrix
import re

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

def hist_plot(idata, figsize=(9,5), root=True):
    # comparing observed and predicted histograms
    fig, ax = plt.subplots(figsize=figsize)
    # az.plot_ppc(idata, ax=ax)
    # Adjust histogram bins
    ax.clear()
    obs = idata.observed_data['y_obs'].values
    ppc = idata.posterior_predictive['y_obs'].values.flatten()
    ax.hist(ppc, bins=np.concat([np.arange(20), np.arange(20,100, 5), np.arange(100, 300, 20)]), alpha=0.3, density=True, label='Predicted')
    ax.hist(obs, bins=np.concat([np.arange(20), np.arange(20,100, 5), np.arange(100, 300, 20)]), alpha=0.3, density=True, label='Observed')
    ax.set_xbound(lower=None, upper=200)
    if root:
        ax.set_yscale("function", functions=(np.sqrt, np.square))
    ax.legend()

def hist_plot_contrast(idata, figsize=(9, 5), root=True):

    fig, ax = plt.subplots(figsize=figsize)

    obs = idata.observed_data["y_obs"].values
    ppc = idata.posterior_predictive["y_obs"].values.flatten()

    bins = np.concatenate([
        np.arange(20),
        np.arange(20, 100, 5),
        np.arange(100, 300, 20)
    ])

    # Compute densities manually
    obs_hist, _ = np.histogram(obs, bins=bins, density=True)
    ppc_hist, _ = np.histogram(ppc, bins=bins, density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    overlap = np.minimum(obs_hist, ppc_hist)
    obs_only = obs_hist - overlap
    ppc_only = ppc_hist - overlap

    # Plot overlap
    ax.bar(
        bin_centers, overlap,
        width=np.diff(bins),
        color="purple", alpha=0.5, label="Overlap"
    )

    # Plot non-overlapping parts
    ax.bar(
        bin_centers, obs_only,
        bottom=overlap,
        width=np.diff(bins),
        color="black", alpha=1, label="Observed only"
    )

    ax.bar(
        bin_centers, ppc_only,
        bottom=overlap,
        width=np.diff(bins),
        color="red", alpha=1, label="Predicted only"
    )

    ax.set_xbound(lower=None, upper=200)

    if root:
        ax.set_yscale("function", functions=(np.sqrt, np.square))

    ax.legend()
    ax.set_xlabel("Cases")
    ax.set_ylabel("Density")

    plt.show()

def CI_plot(idata, figsize=(12,5)):
    # Get observed and posterior predictive samples
    obs = idata.observed_data['y_obs'].values
    ppc = idata.posterior_predictive['y_obs'].values  # (chain, draw, obs)

    # Flatten chains and draws
    ppc_flat = ppc.reshape(-1, ppc.shape[-1])  # (total_draws, obs)

    # Sort observations and get the sort indices
    sort_idx = np.argsort(obs)
    obs_sorted = obs[sort_idx]

    # Sort posterior predictive samples by the same indices
    ppc_sorted = ppc_flat[:, sort_idx]  # Each column corresponds to same obs

    # Compute credible intervals for each observation
    ppc_mean = ppc_sorted.mean(axis=0)
    ppc_lower = np.percentile(ppc_sorted, 2.5, axis=0)
    ppc_upper = np.percentile(ppc_sorted, 97.5, axis=0)

    print(np.mean((obs_sorted>=ppc_lower)&(obs_sorted<=ppc_upper)))
    # Plot (subsample for visibility with 24k points)
    step = max(1, len(obs) // 5000)  # Show ~2000 points
    x = np.arange(0, len(obs), step)

    plt.figure(figsize=figsize)
    plt.plot(x, obs_sorted[::step], 'ko', markersize=2, alpha=0.5, label='Observed')
    # plt.plot(x, ppc_mean[::step], 'r-', linewidth=1, alpha=0.7, label='Predicted mean')
    plt.fill_between(x, ppc_lower[::step], ppc_upper[::step], 
                    alpha=0.3, color='red', label='95% CI')
    plt.xlabel('Observation (sorted by observed value)')
    plt.ylabel('Cases')
    plt.legend()
    plt.title('Observed vs Predicted (sorted by observed value)')
    plt.grid(alpha=0.3)

def CI_plot_alt(idata, figsize=(12,5), root=True):
    # Get observed and posterior predictive samples
    obs = idata.observed_data['y_obs'].values
    ppc = idata.posterior_predictive['y_obs'].values  # (chain, draw, obs)

    # Flatten chains and draws
    ppc_flat = ppc.reshape(-1, ppc.shape[-1])  # (total_draws, obs)

    # Compute credible intervals for each observation
    ppc_mean = ppc_flat.mean(axis=0)
    ppc_lower = np.percentile(ppc_flat, 2.5, axis=0)
    ppc_upper = np.percentile(ppc_flat, 97.5, axis=0)

    # Sort lower and get the sort indices
    sort_idx = np.argsort(ppc_lower)
    obs_sorted = obs[sort_idx]
    ppc_mean = ppc_mean[sort_idx]
    ppc_lower = ppc_lower[sort_idx]
    ppc_upper = ppc_upper[sort_idx]
    # Sort posterior predictive samples by the same indices
    #ppc_sorted = ppc_flat[:, sort_idx]  # Each column corresponds to same obs
    print(np.mean((obs_sorted>=ppc_lower)&(obs_sorted<=ppc_upper)))
    

    # Plot (subsample for visibility with 24k points)
    step = max(1, len(obs) // 5000)  # Show ~2000 points
    x = np.arange(0, len(obs), step)

    plt.figure(figsize=figsize)
    plt.plot(x, obs_sorted[::step], 'ko', markersize=0.5, alpha=0.5, label='Observed')
    # plt.plot(x, ppc_mean[::step], 'r-', linewidth=1, alpha=0.7, label='Predicted mean')
    plt.fill_between(x, ppc_lower[::step], ppc_upper[::step], 
                    alpha=0.3, color='red', label='95% CI')
    plt.xlabel('Observation (sorted by posterior predictive lower CI bound)')
    plt.ylabel('Cases')
    plt.legend()
    plt.title('Observed vs Predicted (sorted by posterior predictive lower CI bound)')
    plt.grid(alpha=0.3)

def CI_plot_both(idata, max_points=5000, figsize=(12,9)):
    # Get observed and posterior predictive samples
    obs = idata.observed_data['y_obs'].values
    ppc = idata.posterior_predictive['y_obs'].values  # (chain, draw, obs)

    # Flatten chains and draws
    ppc_flat = ppc.reshape(-1, ppc.shape[-1])  # (total_draws, obs)

    step = max(1, len(obs) // max_points)

    fig, axes = plt.subplots(
        nrows=2, ncols=1,
        figsize=figsize,
        sharex=False
    )

    # =========================
    # Plot 1: sorted by observed
    # =========================
    sort_idx_obs = np.argsort(obs)
    obs_sorted = obs[sort_idx_obs]
    ppc_sorted = ppc_flat[:, sort_idx_obs]

    lower_obs = np.percentile(ppc_sorted, 2.5, axis=0)
    upper_obs = np.percentile(ppc_sorted, 97.5, axis=0)

    coverage_obs = np.mean(
        (obs_sorted >= lower_obs) & (obs_sorted <= upper_obs)
    )

    x1 = np.arange(0, len(obs_sorted), step)

    ax = axes[0]
    ax.plot(x1, obs_sorted[::step], 'ko', markersize=2, alpha=0.5, label='Observed')
    ax.fill_between(
        x1,
        lower_obs[::step],
        upper_obs[::step],
        alpha=0.3,
        color='red',
        label='95% CI'
    )
    ax.set_title(f'Sorted by observed value (coverage={coverage_obs:.3f})')
    ax.set_ylabel('Cases')
    ax.legend()
    ax.grid(alpha=0.3)

    # =========================================
    # Plot 2: sorted by posterior predictive CI
    # =========================================
    lower = np.percentile(ppc_flat, 2.5, axis=0)
    upper = np.percentile(ppc_flat, 97.5, axis=0)

    sort_idx_ppc = np.argsort(lower)
    obs_sorted2 = obs[sort_idx_ppc]
    lower_sorted = lower[sort_idx_ppc]
    upper_sorted = upper[sort_idx_ppc]

    coverage_ppc = np.mean(
        (obs_sorted2 >= lower_sorted) & (obs_sorted2 <= upper_sorted)
    )

    x2 = np.arange(0, len(obs_sorted2), step)

    ax = axes[1]
    ax.plot(x2, obs_sorted2[::step], 'ko', markersize=0.8, alpha=0.5, label='Observed')
    ax.fill_between(
        x2,
        lower_sorted[::step],
        upper_sorted[::step],
        alpha=0.3,
        color='red',
        label='95% CI'
    )
    ax.set_title(f'Sorted by posterior predictive lower CI')
    ax.set_xlabel('Observation index')
    ax.set_ylabel('Cases')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()

def CI_plot_both_coverage(idata, max_points=5000, figsize=(12, 9), n_bins=20):
    import numpy as np
    import matplotlib.pyplot as plt

    # Get observed and posterior predictive samples
    obs = idata.observed_data["y_obs"].values
    ppc = idata.posterior_predictive["y_obs"].values  # (chain, draw, obs)

    # Flatten chains and draws
    ppc_flat = ppc.reshape(-1, ppc.shape[-1])  # (total_draws, obs)

    step = max(1, len(obs) // max_points)

    fig, axes = plt.subplots(
        nrows=2, ncols=1,
        figsize=figsize,
        sharex=False
    )

    # =========================
    # Plot 1: sorted by observed
    # =========================
    sort_idx_obs = np.argsort(obs)
    obs_sorted = obs[sort_idx_obs]
    ppc_sorted = ppc_flat[:, sort_idx_obs]

    lower_obs = np.percentile(ppc_sorted, 2.5, axis=0)
    upper_obs = np.percentile(ppc_sorted, 97.5, axis=0)

    inside = (
        (obs_sorted >= lower_obs) &
        (obs_sorted <= upper_obs)
    )

    coverage_obs = inside.mean()

    x1 = np.arange(0, len(obs_sorted), step)

    ax = axes[0]
    ax.plot(x1, obs_sorted[::step], 'ko', markersize=2, alpha=0.5, label='Observed')
    ax.fill_between(
        x1,
        lower_obs[::step],
        upper_obs[::step],
        alpha=0.3,
        color='red',
        label='95% CI'
    )
    ax.set_title(f'Sorted by observed value (coverage={coverage_obs:.3f})')
    ax.set_ylabel('Cases')
    ax.grid(alpha=0.3)

    # ---- Discrete quantile-bin coverage (step function) ----
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(obs_sorted, quantiles)

    bin_left = []
    bin_right = []
    bin_coverages = []

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (obs_sorted >= bin_edges[i]) & (obs_sorted < bin_edges[i + 1])
        else:
            mask = (obs_sorted >= bin_edges[i]) & (obs_sorted <= bin_edges[i + 1])

        if mask.sum() == 0:
            continue

        idx = np.where(mask)[0]
        bin_left.append(idx.min())
        bin_right.append(idx.max())
        bin_coverages.append(inside[mask].mean())

    ax2 = ax.twinx()
    for l, r, c in zip(bin_left, bin_right, bin_coverages):
        ax2.fill_between(
            [l+0.2, r-0.2],
            [c, c],
            [0, 0],
            step="post",
            alpha=0.1,
            color="blue"
        )

    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('Coverage')
    ax2.axhline(0.95, color='blue', linestyle='--', alpha=0.3)

    # Legend handling
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color='red', lw=6, alpha=0.3, label='95% CI'),
        Line2D([0], [0], marker='o', color='k', lw=0, markersize=4, label='Observed'),
        Patch(facecolor='blue', alpha=0.5, label='Binned coverage')
    ]
    ax.legend(handles=legend_elements, loc='best')

    # =========================================
    # Plot 2: sorted by posterior predictive CI
    # =========================================
    lower = np.percentile(ppc_flat, 2.5, axis=0)
    upper = np.percentile(ppc_flat, 97.5, axis=0)

    sort_idx_ppc = np.argsort(lower)
    obs_sorted2 = obs[sort_idx_ppc]
    lower_sorted = lower[sort_idx_ppc]
    upper_sorted = upper[sort_idx_ppc]

    coverage_ppc = np.mean(
        (obs_sorted2 >= lower_sorted) & (obs_sorted2 <= upper_sorted)
    )

    x2 = np.arange(0, len(obs_sorted2), step)

    ax = axes[1]
    ax.plot(x2, obs_sorted2[::step], 'ko', markersize=0.8, alpha=0.5, label='Observed')
    ax.fill_between(
        x2,
        lower_sorted[::step],
        upper_sorted[::step],
        alpha=0.3,
        color='red',
        label='95% CI'
    )
    ax.set_title(f'Sorted by posterior predictive lower CI')
    ax.set_xlabel('Observation index')
    ax.set_ylabel('Cases')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.show()

def plot_posteriors_side_by_side(idata1, idata2, var_names=None, figsize=(12, 3), textsize=10):
    """
    Plot posterior distributions for the same variables from two InferenceData objects
    side by side (each variable has a row, two columns: idata1 and idata2).
    Vector variables starting with 'w' are plotted with all elements on the same subplot.
    """
    colors = plt.cm.tab10.colors  # cycle of 10 colors
    if var_names is None:
        var_names = list(idata1.posterior.data_vars)
    
    n_vars = len(var_names)
    fig, axes = plt.subplots(n_vars, 2, figsize=(figsize[0], figsize[1]*n_vars), squeeze=False)
    
    for i, var in enumerate(var_names):
        # Determine common x-limits
        if var[0] == 'w':
            dims = idata1.posterior[var].dims[2:]  # usually the spline dimension
            all_vals = np.concatenate([idata1.posterior[var].values.ravel(), idata2.posterior[var].values.ravel()])
        else:
            all_vals = np.concatenate([idata1.posterior[var].values.ravel(), idata2.posterior[var].values.ravel()])
        x_min, x_max = all_vals.min(), all_vals.max()
        
        # Handle vector variables like 'w1'
        if var[0] == 'w':
            for j in range(idata1.posterior[var].sizes[dims[0]]):
                color = colors[j % len(colors)]
                az.plot_posterior(idata1.posterior[var].sel({dims[0]: j}), ax=axes[i, 0], hdi_prob='hide', color=color, textsize=textsize)
                az.plot_posterior(idata2.posterior[var].sel({dims[0]: j}), ax=axes[i, 1], hdi_prob='hide', color=color, textsize=textsize)
        else:
            az.plot_posterior(idata1, var_names=[var], ax=axes[i, 0], color="blue", hdi_prob='hide', round_to=4, textsize=textsize)
            az.plot_posterior(idata2, var_names=[var], ax=axes[i, 1], color="red", hdi_prob='hide', round_to=4, textsize=textsize)
        
        axes[i, 0].set_title(f"{var} - idata1")
        axes[i, 1].set_title(f"{var} - idata2")
        axes[i, 0].set_xlim(x_min, x_max)
        axes[i, 1].set_xlim(x_min, x_max)

    plt.show()

units = {'t2':'C˚', 'rh':'%RH', 'tp':'mm'}
units_log = {'t2':'C˚', 'rh':'%RH', 'tp':'log(m)'}

def plot_spline(idata, stat_name, var, sigma_var, B, data, knots=None, figsize=(10,5), show_basis=False, basis_scale=4, orthogonal=True, invert_log=False):
    # Extract posterior samples
    w_samples = idata.posterior[var].stack(draws=("chain", "draw")).values  # (n_basis, n_draws)
    sigma_w_samples = idata.posterior[sigma_var].stack(draws=("chain", "draw")).values  # (n_draws,)

    # Compute spline contributions for each draw
    f_s1_samples = (np.asarray(B, order="F") @ w_samples) * sigma_w_samples  # broadcasting: (n_obs, n_draws)
    # print('function mean across samples: ', np.mean(f_s1_samples))
    #f_s1_samples = f_s1_samples - np.mean(f_s1_samples)

    # Compute mean and 95% credible intervals
    f_s1_mean = f_s1_samples.mean(axis=1)
    f_s1_lower5 = np.percentile(f_s1_samples, 25, axis=1)
    f_s1_upper5 = np.percentile(f_s1_samples, 75, axis=1)
    f_s1_lower = np.percentile(f_s1_samples, 2.5, axis=1)
    f_s1_upper = np.percentile(f_s1_samples, 97.5, axis=1)

    # Plot
    index = np.argsort(data)
    plt.figure(figsize=figsize)
    if knots is not None:
        if (invert_log)&(stat_name[0:2]=='tp'):
            plot_knots = (np.exp(knots)-1e-6)*1000
        else:
            plot_knots = knots
        plt.vlines(plot_knots, ymin=np.min(f_s1_lower), ymax=np.max(f_s1_upper), label='knots')
        if show_basis:
            x = np.linspace(np.min(data), np.max(data), 1000)
            if (invert_log)&(stat_name[0:2]=='tp'):
                xx = (np.exp(x)-1e-6)*1000
            else:
                 xx = x
            BB = dmatrix("bs(x, knots=knots, degree=3, include_intercept=False)-1", {"x": x, "knots": knots},)
            if orthogonal:
                BB = np.asarray(BB)
                BB = (BB - BB.mean(axis=0)) / BB.std(axis=0)
                BB, _ = np.linalg.qr(BB)
            for i in range(BB.shape[1]):
                plt.plot(xx,
                         (BB[:, i] - np.min(BB))*basis_scale + np.max(f_s1_upper) + (np.max(f_s1_upper)-np.min(f_s1_lower))*0.05,
                         alpha=0.99, linestyle=':')
            # plt.plot(x, (np.max(f_s1_upper)+(np.max(f_s1_upper)-np.min(f_s1_lower))*0.01)*np.ones(len(x)), c='white', alpha=0.8)

    x = np.array(data)[index]
    
    if (invert_log)&(stat_name[0:2]=='tp'):
            x = (np.exp(x)-1e-6) * 1000
    plt.plot(x, f_s1_mean[index], color='red', label='Mean spline effect')
    plt.fill_between(x, f_s1_lower[index], f_s1_lower5[index], color='red', alpha=0.3, label='95% CI')
    plt.fill_between(x, f_s1_lower5[index], f_s1_upper5[index], color='blue', alpha=0.3, label='50% CI')
    plt.fill_between(x, f_s1_upper5[index], f_s1_upper[index], color='red', alpha=0.3)

    abbrev_stat_name = abbrev_stat(stat_name)  
    if (invert_log)&(stat_name[0:2]=='tp'):
        abbrev_stat_name = abbrev_stat_name.replace("_log", "")
        xlab = f'{abbrev_stat_name} ({units[stat_name[0:2]]})'
    else:
        xlab = f'{abbrev_stat_name} ({units_log[stat_name[0:2]]})'

    plt.xlabel(xlab)
    plt.ylabel('Spline contribution')
    plt.legend()

    fig = plt.gcf()
    return fig
