import os
import re
from io import StringIO
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.expanduser("~/atmos/OutputStorage")
OUT_DIR = os.path.expanduser("~/atmos/BA/Plots/aerosol_analysis_pub")
SAVE_DPI = 300
SHOW_PLOTS = False

STARS = {
    "18": "Epsilon_Eri",
    "24": "HD40307",
    "25": "HD85512",
    "26": "HD97658",
}
STAR_ORDER = ["Epsilon_Eri", "HD40307", "HD85512", "HD97658"]

RUN_LABELS = {
    "Run_1": "2.7 Gy star",
    "Run_2": "0.5 Gy star",
    "Run_3": "9.9 Gy star",
}
RUN_ORDER = ["Run_1", "Run_2", "Run_3"]

SCENARIO_TO_FSCALE = {
    "1": "0.75",
    "2": "1.00",
    "3": "1.50",
}
FSCALE_ORDER = ["0.75", "1.00", "1.50"]

FIGSIZE_CASE = (6.2, 7.2)
FIGSIZE_GRID = (11.0, 8.5)
HEATMAP_FIGSIZE = (7.5, 5.2)

# -----------------------------
# HELPERS
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize_for_filename(value):
    value = str(value)
    for old, new in [(" ", "_"), ("/", "_"), ("\\", "_"), ("=", "_"), (":", "_")]:
        value = value.replace(old, new)
    return value


def numeric_sort_key(value):
    try:
        return float(value)
    except Exception:
        return str(value)


def star_name_from_id(star_id):
    return STARS.get(str(star_id), str(star_id))


def discover_run_dirs(base_dir):
    if not os.path.isdir(base_dir):
        print(f"Base directory not found: {base_dir}")
        return []
    run_dirs = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and re.fullmatch(r"Run_\d+", name):
            run_dirs.append(name)
    return sorted(run_dirs, key=lambda x: int(x.split("_")[1]))


def save_figure(fig, out_path):
    ensure_dir(os.path.dirname(out_path))
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def pick_altitude_column(df):
    for candidate in ["ALT", "Z", "Alt", "alt", "z"]:
        if candidate in df.columns:
            return candidate
    return None


def to_numeric_array(series):
    return pd.to_numeric(series, errors="coerce").values


def finite_mask(x, y):
    return np.isfinite(x) & np.isfinite(y)


def finite_positive_mask(x, y):
    return np.isfinite(x) & np.isfinite(y) & (x > 0)


def integrate_curve(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def safe_log10_ratio(num, den):
    if np.isfinite(num) and np.isfinite(den) and num > 0 and den > 0:
        return np.log10(num / den)
    return np.nan


def format_case_label(run, scenario):
    return f"{RUN_LABELS.get(run, run)} | FSCALE={SCENARIO_TO_FSCALE.get(str(scenario), scenario)}"


# -----------------------------
# FILE DISCOVERY
# -----------------------------
def build_file_index(scale_dir):
    photochem_dir = os.path.join(scale_dir, "PHOTOCHEM_OUTPUT")
    scenario_files = defaultdict(lambda: {"hcaer_file": None})
    pattern = re.compile(r"^(Run_\d+)_([0-9]+(?:\.[0-9]+)?)_(\d+)_(.+)$")

    if os.path.isdir(photochem_dir):
        for fname in os.listdir(photochem_dir):
            match = pattern.match(fname)
            if not match:
                continue
            _, _, scenario, suffix = match.groups()
            full_path = os.path.join(photochem_dir, fname)
            if suffix == "hcaer.out":
                scenario_files[scenario]["hcaer_file"] = full_path

    return dict(scenario_files)


# -----------------------------
# PARSER
# -----------------------------
def parse_hcaer(file_path):
    if file_path is None or not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Failed to open {file_path}: {e}")
        return None

    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Z") or stripped.startswith("ALT"):
            header_idx = i
            break

    if header_idx is None:
        print(f"No aerosol header found in {file_path}")
        return None

    table_lines = [line.rstrip("\n") for line in lines[header_idx:] if line.strip()]

    try:
        df = pd.read_csv(
            StringIO("\n".join(table_lines)),
            sep=r"\s+",
            engine="python",
        )
    except Exception as e:
        print(f"Failed to parse aerosol table in {file_path}: {e}")
        return None

    alt_col = pick_altitude_column(df)
    if alt_col is None:
        print(f"No altitude column found in {file_path}")
        return None

    data = {"ALT": to_numeric_array(df[alt_col])}
    for col in df.columns:
        if col == alt_col:
            continue
        data[col] = to_numeric_array(df[col])
    return data


# -----------------------------
# DATASET BUILD
# -----------------------------
def build_dataset(base_dir):
    data = {}
    runs = discover_run_dirs(base_dir)
    print("Discovered runs:", runs)

    for run in runs:
        if run not in RUN_ORDER:
            continue
        run_dir = os.path.join(base_dir, run)
        data[run] = {}

        for star_id in sorted(os.listdir(run_dir), key=numeric_sort_key):
            star_dir = os.path.join(run_dir, star_id)
            if not os.path.isdir(star_dir):
                continue

            star_name = star_name_from_id(star_id)
            if star_name not in STAR_ORDER:
                continue

            data[run][star_name] = {}
            for scale in sorted(os.listdir(star_dir), key=numeric_sort_key):
                scale_dir = os.path.join(star_dir, scale)
                if not os.path.isdir(scale_dir):
                    continue

                indexed = build_file_index(scale_dir)
                if not indexed:
                    continue

                data[run][star_name][scale] = {}
                for scenario, files in sorted(indexed.items(), key=lambda x: int(x[0])):
                    aerosol = parse_hcaer(files["hcaer_file"]) if files["hcaer_file"] else None
                    data[run][star_name][scale][scenario] = {
                        "aerosol": aerosol,
                        "file": files["hcaer_file"],
                    }
    return data


def preferred_entry_for_scenario(dataset, run, star, scenario):
    scenario = str(scenario)
    preferred_scale = SCENARIO_TO_FSCALE.get(scenario)
    scales = dataset.get(run, {}).get(star, {})

    if preferred_scale in scales and scenario in scales[preferred_scale]:
        return preferred_scale, scales[preferred_scale][scenario]

    for scale in sorted(scales.keys(), key=numeric_sort_key):
        if scenario in scales[scale]:
            return scale, scales[scale][scenario]

    return None, None


# -----------------------------
# METRICS
# -----------------------------
def aerosol_species_list(dataset):
    found = set()
    for run in dataset.values():
        for star in run.values():
            for scale in star.values():
                for entry in scale.values():
                    aero = entry.get("aerosol")
                    if aero:
                        found.update([k for k in aero.keys() if k != "ALT"])
    return sorted(found)


def compute_aerosol_metrics(aerosol_data, species):
    alt = np.asarray(aerosol_data["ALT"], dtype=float)
    x = np.asarray(aerosol_data[species], dtype=float)
    mask = finite_mask(x, alt)
    if not np.any(mask):
        return None

    z = alt[mask]
    y = x[mask]
    pos_mask = np.isfinite(y) & np.isfinite(z) & (y > 0)

    if np.any(pos_mask):
        z_pos = z[pos_mask]
        y_pos = y[pos_mask]
        column_proxy = integrate_curve(y_pos, z_pos)
        peak = np.nanmax(y_pos)
        peak_alt = z_pos[np.nanargmax(y_pos)]
        surface = y_pos[np.nanargmin(z_pos)] if z_pos.size else np.nan
        upper20 = np.nanmean(y_pos[z_pos >= 20]) if np.any(z_pos >= 20) else np.nan
        upper50 = np.nanmean(y_pos[z_pos >= 50]) if np.any(z_pos >= 50) else np.nan
    else:
        column_proxy = peak = peak_alt = surface = upper20 = upper50 = np.nan

    centroid = integrate_curve(y * z, z) / integrate_curve(y, z) if np.any(np.isfinite(y)) and np.nansum(y) != 0 else np.nan

    return {
        "column_proxy": column_proxy,
        "peak": peak,
        "peak_alt": peak_alt,
        "surface": surface,
        "mean_above_20km": upper20,
        "mean_above_50km": upper50,
        "centroid_alt": centroid,
    }


def flatten_metrics(dataset, species_names):
    rows = []
    for run in RUN_ORDER:
        for star in STAR_ORDER:
            for scenario in ["1", "2", "3"]:
                scale, entry = preferred_entry_for_scenario(dataset, run, star, scenario)
                if not entry:
                    continue
                aerosol = entry.get("aerosol")
                if aerosol is None or "ALT" not in aerosol:
                    continue
                for sp in species_names:
                    if sp not in aerosol:
                        continue
                    metrics = compute_aerosol_metrics(aerosol, sp)
                    if metrics is None:
                        continue
                    row = {
                        "run": run,
                        "run_label": RUN_LABELS.get(run, run),
                        "star": star,
                        "scenario": str(scenario),
                        "fscale": SCENARIO_TO_FSCALE.get(str(scenario), np.nan),
                        "scale_dir": scale,
                        "species": sp,
                        "file": entry.get("file"),
                    }
                    row.update(metrics)
                    rows.append(row)
    return pd.DataFrame(rows)


def compute_comparisons(metrics_df):
    rows = []
    metric_names = [
        "column_proxy", "peak", "peak_alt", "surface",
        "mean_above_20km", "mean_above_50km", "centroid_alt",
    ]

    # Compare runs at fixed FSCALE against Run_1
    for star in STAR_ORDER:
        star_df = metrics_df[metrics_df["star"] == star]
        for fscale in FSCALE_ORDER:
            sub = star_df[star_df["fscale"] == fscale]
            for sp in sorted(sub["species"].unique()):
                ref = sub[(sub["species"] == sp) & (sub["run"] == "Run_1")]
                if ref.empty:
                    continue
                ref_row = ref.iloc[0]
                for _, row in sub[sub["species"] == sp].iterrows():
                    for metric in metric_names:
                        rows.append({
                            "comparison_type": "run_at_fixed_fscale",
                            "star": star,
                            "species": sp,
                            "fscale": fscale,
                            "reference": "Run_1",
                            "target": row["run"],
                            "metric": metric,
                            "reference_value": ref_row[metric],
                            "target_value": row[metric],
                            "delta_log10": safe_log10_ratio(row[metric], ref_row[metric]),
                            "delta_linear": row[metric] - ref_row[metric] if np.isfinite(row[metric]) and np.isfinite(ref_row[metric]) else np.nan,
                        })

    # Compare FSCALE at fixed run against 1.00
    for star in STAR_ORDER:
        star_df = metrics_df[metrics_df["star"] == star]
        for run in RUN_ORDER:
            sub = star_df[star_df["run"] == run]
            for sp in sorted(sub["species"].unique()):
                ref = sub[(sub["species"] == sp) & (sub["fscale"] == "1.00")]
                if ref.empty:
                    continue
                ref_row = ref.iloc[0]
                for _, row in sub[sub["species"] == sp].iterrows():
                    for metric in metric_names:
                        rows.append({
                            "comparison_type": "fscale_at_fixed_run",
                            "star": star,
                            "species": sp,
                            "run": run,
                            "reference": "1.00",
                            "target": row["fscale"],
                            "metric": metric,
                            "reference_value": ref_row[metric],
                            "target_value": row[metric],
                            "delta_log10": safe_log10_ratio(row[metric], ref_row[metric]),
                            "delta_linear": row[metric] - ref_row[metric] if np.isfinite(row[metric]) and np.isfinite(ref_row[metric]) else np.nan,
                        })
    return pd.DataFrame(rows)


def interpret_change(delta_log10):
    if not np.isfinite(delta_log10):
        return None
    mag = abs(delta_log10)
    if mag < 0.1:
        strength = "little change"
    elif mag < 0.3:
        strength = "modest change"
    elif mag < 1.0:
        strength = "strong change"
    else:
        strength = "order-of-magnitude change"
    direction = "increase" if delta_log10 > 0 else "decrease"
    return strength, direction


def build_text_summaries(comparisons_df):
    rows = []
    if comparisons_df.empty:
        return pd.DataFrame(rows)

    for star in STAR_ORDER:
        star_df = comparisons_df[comparisons_df["star"] == star]
        for comp_type in ["run_at_fixed_fscale", "fscale_at_fixed_run"]:
            sub = star_df[star_df["comparison_type"] == comp_type].copy()
            if sub.empty:
                continue
            sub = sub[sub["metric"].isin(["column_proxy", "peak", "mean_above_50km", "peak_alt"])].copy()
            sub["abs_delta"] = np.abs(sub["delta_log10"])
            sub = sub.sort_values("abs_delta", ascending=False)
            for _, row in sub.head(12).iterrows():
                interp = interpret_change(row["delta_log10"])
                if interp is None:
                    continue
                strength, direction = interp
                if comp_type == "run_at_fixed_fscale":
                    context = f"at FSCALE={row['fscale']}, relative to {row['reference']}"
                    target = row["target"]
                else:
                    context = f"for {row['run']}, relative to FSCALE={row['reference']}"
                    target = f"FSCALE={row['target']}"
                text = (
                    f"{star}: {row['species']} shows a {strength} {direction} in {row['metric']} "
                    f"for {target} {context} (Δlog10={row['delta_log10']:.2f})."
                )
                rows.append({
                    "star": star,
                    "comparison_type": comp_type,
                    "species": row["species"],
                    "metric": row["metric"],
                    "summary": text,
                })
    return pd.DataFrame(rows)


# -----------------------------
# PLOTTING
# -----------------------------
def plot_case_aerosol(aerosol_data, title, out_base, species_names):
    if aerosol_data is None or "ALT" not in aerosol_data:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_CASE)
    alt = np.asarray(aerosol_data["ALT"], dtype=float)
    plotted = False

    for sp in species_names:
        if sp not in aerosol_data:
            continue
        x = np.asarray(aerosol_data[sp], dtype=float)
        mask = finite_positive_mask(x, alt)
        if np.any(mask):
            ax.plot(x[mask], alt[mask], lw=1.6, label=sp)
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_xlabel("Aerosol density / mixing proxy")
    ax.set_ylabel("Altitude")
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=1, loc="best")

    save_figure(fig, out_base + ".png")
    save_figure(fig, out_base + ".pdf")


def paginate(items, page_size=4):
    for i in range(0, len(items), page_size):
        yield items[i:i + page_size], i // page_size + 1


def plot_overview_pages(dataset, page_cases, title_prefix, out_base, species_names):
    png_paths = []
    pdf_path = out_base + ".pdf"
    ensure_dir(os.path.dirname(out_base))

    with PdfPages(pdf_path) as pdf:
        for page_items, page_num in paginate(page_cases, 4):
            fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_GRID, sharex=True, sharey=True)
            axes = axes.ravel()

            for ax in axes:
                ax.grid(True, which="both", alpha=0.2)

            for ax, case in zip(axes, page_items):
                run, star, scenario = case
                scale, entry = preferred_entry_for_scenario(dataset, run, star, scenario)
                aerosol = entry.get("aerosol") if entry else None
                if aerosol is None or "ALT" not in aerosol:
                    ax.set_axis_off()
                    continue

                alt = np.asarray(aerosol["ALT"], dtype=float)
                plotted = False
                for sp in species_names:
                    if sp not in aerosol:
                        continue
                    x = np.asarray(aerosol[sp], dtype=float)
                    mask = finite_positive_mask(x, alt)
                    if np.any(mask):
                        ax.plot(x[mask], alt[mask], lw=1.2, label=sp)
                        plotted = True

                if plotted:
                    ax.set_xscale("log")
                    ax.set_title(f"{star}\n{RUN_LABELS.get(run, run)} | FSCALE={SCENARIO_TO_FSCALE.get(str(scenario), scenario)}", fontsize=10)
                else:
                    ax.set_axis_off()

            for ax in axes[len(page_items):]:
                ax.set_axis_off()

            handles, labels = [], []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                for hh, ll in zip(h, l):
                    if ll not in labels:
                        handles.append(hh)
                        labels.append(ll)
            if handles:
                fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8, frameon=False)

            fig.suptitle(f"{title_prefix} — page {page_num}", fontsize=13, y=0.98)
            fig.supxlabel("Aerosol density / mixing proxy")
            fig.supylabel("Altitude")
            fig.tight_layout(rect=(0.03, 0.05, 1, 0.95))
            pdf.savefig(fig, dpi=SAVE_DPI, bbox_inches="tight")

            png_path = f"{out_base}_page_{page_num:02d}.png"
            fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
            print(f"Saved: {png_path}")
            png_paths.append(png_path)
            plt.close(fig)

    print(f"Saved: {pdf_path}")
    return png_paths, pdf_path


def plot_heatmap(df, index_col, columns_col, values_col, title, out_base, cmap="coolwarm", center=0):
    if df.empty:
        return
    pivot = df.pivot(index=index_col, columns=columns_col, values=values_col)
    pivot = pivot.replace([np.inf, -np.inf], np.nan)
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
    data = pivot.values.astype(float)
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect="auto", cmap=cmap)

    if center == 0 and np.isfinite(masked).any():
        vmax = np.nanmax(np.abs(data))
        if np.isfinite(vmax) and vmax > 0:
            im.set_clim(-vmax, vmax)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(values_col)
    save_figure(fig, out_base + ".png")
    save_figure(fig, out_base + ".pdf")


def plot_metric_vs_fscale(metrics_df, star, species, metric, out_base):
    sub = metrics_df[(metrics_df["star"] == star) & (metrics_df["species"] == species)].copy()
    if sub.empty:
        return
    sub["fscale_num"] = pd.to_numeric(sub["fscale"], errors="coerce")
    sub = sub.sort_values(["run", "fscale_num"])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plotted = False
    for run in RUN_ORDER:
        rsub = sub[sub["run"] == run]
        if rsub.empty:
            continue
        x = rsub["fscale_num"].values
        y = rsub[metric].values
        mask = np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            ax.plot(x[mask], y[mask], marker="o", lw=1.8, label=RUN_LABELS.get(run, run))
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("FSCALE")
    ax.set_ylabel(metric)
    ax.set_title(f"{star} | {species} | {metric}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    y = sub[metric].values
    y = y[np.isfinite(y) & (y > 0)]
    if y.size > 1 and np.nanmax(y) / np.nanmin(y) > 1e3:
        ax.set_yscale("log")
    save_figure(fig, out_base + ".png")
    save_figure(fig, out_base + ".pdf")


# -----------------------------
# TEXT / TABLE OUTPUT
# -----------------------------
def write_text_summary(text_df, out_path):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        if text_df.empty:
            f.write("No aerosol comparison summaries were generated.\n")
            return
        current_star = None
        for _, row in text_df.iterrows():
            if row["star"] != current_star:
                current_star = row["star"]
                f.write(f"\n{current_star}\n")
                f.write("-" * len(current_star) + "\n")
            f.write(f"- {row['summary']}\n")
    print(f"Saved: {out_path}")


def write_simple_latex_table(df, out_path, caption=None, label=None, max_rows=25):
    ensure_dir(os.path.dirname(out_path))
    table = df.head(max_rows).copy()
    cols = list(table.columns)

    def fmt(v):
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
                return f"{v:.3e}"
            return f"{v:.3f}"
        return str(v)

    with open(out_path, "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        if caption:
            f.write(f"\\caption{{{caption}}}\n")
        if label:
            f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{tabular}{" + "l" * len(cols) + "}\n\\hline\n")
        f.write(" & ".join(cols) + " \\\\ \n\\hline\n")
        for _, row in table.iterrows():
            f.write(" & ".join(fmt(row[c]) for c in cols) + " \\\\ \n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved: {out_path}")


# -----------------------------
# MAIN
# -----------------------------
def main(base_dir=BASE_DIR, out_dir=OUT_DIR):
    ensure_dir(out_dir)
    dataset = build_dataset(base_dir)
    species_names = aerosol_species_list(dataset)

    if not species_names:
        print("No aerosol species found.")
        return

    print(f"Detected aerosol species: {species_names}")

    # Case plots
    for run in RUN_ORDER:
        for star in STAR_ORDER:
            for scenario in ["1", "2", "3"]:
                scale, entry = preferred_entry_for_scenario(dataset, run, star, scenario)
                if not entry:
                    continue
                title = f"{star} | {RUN_LABELS.get(run, run)} | FSCALE={SCENARIO_TO_FSCALE.get(scenario, scenario)}"
                out_base = os.path.join(
                    out_dir,
                    "case_aerosol",
                    f"aerosol_{sanitize_for_filename(star)}_{sanitize_for_filename(run)}_fscale_{sanitize_for_filename(SCENARIO_TO_FSCALE.get(scenario, scenario))}"
                )
                plot_case_aerosol(entry.get("aerosol"), title, out_base, species_names)

    # Overviews
    star_cases = []
    for star in STAR_ORDER:
        for run in RUN_ORDER:
            for scenario in ["1", "2", "3"]:
                if preferred_entry_for_scenario(dataset, run, star, scenario)[1]:
                    star_cases.append((run, star, scenario))
        if star_cases:
            plot_overview_pages(
                dataset,
                [(run, star, scenario) for run in RUN_ORDER for scenario in ["1", "2", "3"] if preferred_entry_for_scenario(dataset, run, star, scenario)[1]],
                f"{star} aerosol overview",
                os.path.join(out_dir, "star_overviews", f"aerosol_overview_{sanitize_for_filename(star)}"),
                species_names,
            )

    for scenario in ["1", "2", "3"]:
        page_cases = [(run, star, scenario) for star in STAR_ORDER for run in RUN_ORDER if preferred_entry_for_scenario(dataset, run, star, scenario)[1]]
        if page_cases:
            plot_overview_pages(
                dataset,
                page_cases,
                f"FSCALE={SCENARIO_TO_FSCALE.get(scenario, scenario)} aerosol overview",
                os.path.join(out_dir, "scenario_overviews", f"aerosol_overview_fscale_{sanitize_for_filename(SCENARIO_TO_FSCALE.get(scenario, scenario))}"),
                species_names,
            )

    for run in RUN_ORDER:
        page_cases = [(run, star, scenario) for star in STAR_ORDER for scenario in ["1", "2", "3"] if preferred_entry_for_scenario(dataset, run, star, scenario)[1]]
        if page_cases:
            plot_overview_pages(
                dataset,
                page_cases,
                f"{RUN_LABELS.get(run, run)} aerosol overview",
                os.path.join(out_dir, "run_overviews", f"aerosol_overview_{sanitize_for_filename(run)}"),
                species_names,
            )

    # Metrics and comparisons
    metrics_df = flatten_metrics(dataset, species_names)
    if metrics_df.empty:
        print("No aerosol metrics could be computed.")
        return

    tables_dir = os.path.join(out_dir, "tables")
    ensure_dir(tables_dir)
    metrics_csv = os.path.join(tables_dir, "aerosol_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved: {metrics_csv}")

    comparisons_df = compute_comparisons(metrics_df)
    comparisons_csv = os.path.join(tables_dir, "aerosol_comparisons.csv")
    comparisons_df.to_csv(comparisons_csv, index=False)
    print(f"Saved: {comparisons_csv}")

    text_df = build_text_summaries(comparisons_df)
    text_csv = os.path.join(tables_dir, "aerosol_text_summaries.csv")
    text_df.to_csv(text_csv, index=False)
    print(f"Saved: {text_csv}")

    write_text_summary(text_df, os.path.join(out_dir, "aerosol_results_summary.txt"))

    # Heatmaps
    heatmap_dir = os.path.join(out_dir, "heatmaps")
    ensure_dir(heatmap_dir)
    for star in STAR_ORDER:
        sub = comparisons_df[
            (comparisons_df["star"] == star)
            & (comparisons_df["comparison_type"] == "run_at_fixed_fscale")
            & (comparisons_df["metric"] == "column_proxy")
            & (comparisons_df["target"].isin(["Run_2", "Run_3"]))
        ].copy()
        if not sub.empty:
            sub["column"] = sub["target"] + " @ FSCALE=" + sub["fscale"].astype(str)
            plot_heatmap(
                sub,
                index_col="species",
                columns_col="column",
                values_col="delta_log10",
                title=f"{star}: aerosol column changes vs Run_1",
                out_base=os.path.join(heatmap_dir, f"{sanitize_for_filename(star)}_run_comparison_column_proxy"),
            )

        sub2 = comparisons_df[
            (comparisons_df["star"] == star)
            & (comparisons_df["comparison_type"] == "fscale_at_fixed_run")
            & (comparisons_df["metric"] == "column_proxy")
        ].copy()
        if not sub2.empty:
            sub2["column"] = sub2["run"] + " @ FSCALE=" + sub2["target"].astype(str)
            plot_heatmap(
                sub2,
                index_col="species",
                columns_col="column",
                values_col="delta_log10",
                title=f"{star}: aerosol column changes vs FSCALE=1.00",
                out_base=os.path.join(heatmap_dir, f"{sanitize_for_filename(star)}_fscale_comparison_column_proxy"),
            )

    # Metric-vs-FSCALE plots
    metric_dir = os.path.join(out_dir, "metric_vs_fscale")
    ensure_dir(metric_dir)
    for star in STAR_ORDER:
        for sp in species_names:
            for metric in ["column_proxy", "peak", "mean_above_50km", "peak_alt"]:
                plot_metric_vs_fscale(
                    metrics_df,
                    star,
                    sp,
                    metric,
                    os.path.join(metric_dir, f"{sanitize_for_filename(star)}_{sanitize_for_filename(sp)}_{metric}")
                )

    # LaTeX tables without jinja2
    latex_dir = os.path.join(out_dir, "latex_tables")
    ensure_dir(latex_dir)
    write_simple_latex_table(
        metrics_df[["star", "run", "fscale", "species", "column_proxy", "peak", "peak_alt", "mean_above_50km"]],
        os.path.join(latex_dir, "aerosol_metrics_sample.tex"),
        caption="Sample aerosol metrics.",
        label="tab:aerosol_metrics_sample",
        max_rows=30,
    )
    write_simple_latex_table(
        comparisons_df[["star", "comparison_type", "species", "metric", "reference", "target", "delta_log10"]],
        os.path.join(latex_dir, "aerosol_comparisons_sample.tex"),
        caption="Sample aerosol comparison metrics.",
        label="tab:aerosol_comparisons_sample",
        max_rows=30,
    )

    print("\nDone.")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
