import os
import pandas as pd
import base64
import io
import matplotlib.pyplot as plt

###
from _fitting._utils_style import color_ess
###


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64

def go_report(report_path, m, idata, n_divergences, times, show, summary_html, f_summary_html, wl_html, trace_img, pair_img, spline_imgs, exp_spline_imgs, zi_img, zi_s_img, div_img):
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
        <h2>f Summary</h2>
        {f_summary_html if f_summary_html is not None else ""}
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
                    .map(lambda x: color_ess(x, n_draws),
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