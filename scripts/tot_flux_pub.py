import os
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.expanduser("~/atmos/OutputStorage")
OUT_DIR = os.path.expanduser("~/atmos/BA/Plots/tot_flux_pub")
SAVE_DPI = 300

RUN_LABELS = {
    "Run_1": "2.7 Gyr star - normal flare activity",
    "Run_2": "0.5 Gyr star - high flare activity",
    "Run_3": "9.9 Gyr star - low flare activity",
}
RUN_ORDER = ["Run_1", "Run_2", "Run_3"]
RUN_HATCHES = {
    "Run_1": "",
    "Run_2": "//",
    "Run_3": "..",
}

SCENARIO_LABELS = {
    "1": "FSCALE = 0.75",
    "2": "FSCALE = 1.0",
    "3": "FSCALE = 1.5",
}

STAR_LABELS = {
    "18": "Epsilon_Eri",
    "24": "HD40307",
    "25": "HD85512",
    "26": "HD97658",
}
STAR_ORDER = ["18", "24", "25", "26"]
STAR_NAME_ORDER = [STAR_LABELS[s] for s in STAR_ORDER]

BANDS = ["UV", "Vis", "IR", "All"]
COMPONENT_BANDS = ["UV", "Vis", "IR"]
BAND_COLORS = {
    "UV": "purple",
    "Vis": "blue",
    "IR": "red",
    "All": "dimgray",
}

SUMMARY_COLUMNS = [
    "WAVLL",
    "WAVLU",
    "FSOL",
    "FSOD",
    "FDNSOL_1",
    "FUPSOL_1",
    "FDNSOL_2",
    "FUPSOL_2",
    "PDNSOL_1",
    "PDNSOL_2",
    "PATTEN",
    "PDNSOD_2",
]

ALL_SPLIT_COLUMNS = ["FSOL", "PATTEN", "PDNSOD_2"]
UV_VIS_IR_COMBINED_METRIC = "FSOL"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# -----------------------------
# HELPERS
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)



def sanitize_for_filename(value):
    value = str(value)
    for old, new in [(" ", "_"), ("/", "_"), ("\\", "_"), ("=", ""), (".", "p")]:
        value = value.replace(old, new)
    return value



def numeric_sort_key(value):
    try:
        return float(value)
    except Exception:
        return str(value)



def star_name_from_id(star_id):
    return STAR_LABELS.get(str(star_id), str(star_id))



def scenario_label(scenario):
    return SCENARIO_LABELS.get(str(scenario), f"Scenario {scenario}")



def discover_run_dirs(base_dir):
    run_dirs = []
    if not os.path.isdir(base_dir):
        print(f"Base directory not found: {base_dir}")
        return run_dirs

    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and re.fullmatch(r"Run_\d+", name) and name in RUN_ORDER:
            run_dirs.append(name)

    return sorted(run_dirs, key=lambda x: int(x.split("_")[1]))



def save_fig_png_pdf(fig, base_path_no_ext):
    ensure_dir(os.path.dirname(base_path_no_ext))
    png_path = f"{base_path_no_ext}.png"
    pdf_path = f"{base_path_no_ext}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    plt.close(fig)



def _read_lines(file_path):
    with open(file_path, "r") as f:
        return [line.rstrip("\n") for line in f]



def _find_last_spectral_header(lines):
    header = "I    WAVLL      WAVLU"
    last_idx = None
    for idx, line in enumerate(lines):
        if header in line:
            last_idx = idx
    return last_idx



def _coerce_numeric(token):
    try:
        return float(token)
    except Exception:
        return np.nan



def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)



def maybe_log_axis(ax, values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals != 0)]
    if vals.size >= 2:
        ratio = np.nanmax(np.abs(vals)) / np.nanmin(np.abs(vals))
        if ratio > 1e3:
            ax.set_yscale("log")



def band_component_legend_handles():
    return [
        Patch(facecolor=BAND_COLORS[band], edgecolor="black", linewidth=0.4, label=band)
        for band in COMPONENT_BANDS
    ]



def run_style_legend_handles():
    return [
        Patch(facecolor="white", edgecolor="black", hatch=RUN_HATCHES[run], linewidth=0.6, label=RUN_LABELS[run])
        for run in RUN_ORDER
    ]


# -----------------------------
# FILE DISCOVERY / PARSING
# -----------------------------
def build_clima_allout_index(scale_dir):
    clima_dir = os.path.join(scale_dir, "CLIMA_OUTPUT")
    scenario_files = {}
    pattern = re.compile(r"^(Run_\d+)_([0-9]+(?:\.[0-9]+)?)_(\d+)_(.+)$")

    if not os.path.isdir(clima_dir):
        return scenario_files

    for fname in os.listdir(clima_dir):
        match = pattern.match(fname)
        if not match:
            continue
        _, _, scenario, suffix = match.groups()
        if suffix != "clima_allout.tab":
            continue
        scenario_files[scenario] = os.path.join(clima_dir, fname)

    return dict(sorted(scenario_files.items(), key=lambda kv: int(kv[0])))



def parse_clima_allout_summary(file_path):
    if file_path is None or not os.path.exists(file_path):
        return None

    try:
        lines = _read_lines(file_path)
    except Exception as exc:
        print(f"Failed to open {file_path}: {exc}")
        return None

    header_idx = _find_last_spectral_header(lines)
    if header_idx is None:
        print(f"No spectral header found in {file_path}")
        return None

    band_rows = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if not parts:
            continue
        band = parts[0]
        if band not in BANDS:
            continue
        values = parts[1:]
        if len(values) < len(SUMMARY_COLUMNS):
            continue
        row = {"band": band}
        for col, token in zip(SUMMARY_COLUMNS, values[: len(SUMMARY_COLUMNS)]):
            row[col] = _coerce_numeric(token)
        band_rows.append(row)
        if len(band_rows) == 4:
            break

    if not band_rows:
        return None

    df = pd.DataFrame(band_rows)
    df["band"] = pd.Categorical(df["band"], categories=BANDS, ordered=True)
    df = df.sort_values("band").reset_index(drop=True)
    return df


# -----------------------------
# DATASET BUILD / FLATTEN
# -----------------------------
def build_dataset(base_dir):
    data = {}
    runs = discover_run_dirs(base_dir)
    print("Discovered runs:", runs)

    for run in runs:
        run_dir = os.path.join(base_dir, run)
        data[run] = {}

        for star_id in sorted(os.listdir(run_dir), key=numeric_sort_key):
            star_dir = os.path.join(run_dir, star_id)
            if not os.path.isdir(star_dir):
                continue
            if str(star_id) not in STAR_LABELS:
                continue

            star_name = star_name_from_id(star_id)
            data[run][star_name] = {}

            for scale in sorted(os.listdir(star_dir), key=numeric_sort_key):
                scale_dir = os.path.join(star_dir, scale)
                if not os.path.isdir(scale_dir):
                    continue
                scenario_files = build_clima_allout_index(scale_dir)
                if not scenario_files:
                    continue

                data[run][star_name][scale] = {}
                for scenario, file_path in scenario_files.items():
                    summary = parse_clima_allout_summary(file_path)
                    data[run][star_name][scale][scenario] = {
                        "summary": summary,
                        "file": file_path,
                    }
    return data



def flatten_dataset(dataset):
    rows = []
    for run, stars in dataset.items():
        for star, scales in stars.items():
            for scale, scenarios in scales.items():
                for scenario, entry in scenarios.items():
                    summary = entry.get("summary")
                    if summary is None or summary.empty:
                        continue
                    for _, row in summary.iterrows():
                        item = {
                            "run": run,
                            "star": star,
                            "scale": str(scale),
                            "scenario": str(scenario),
                            "band": str(row["band"]),
                            "file": entry.get("file"),
                        }
                        for col in SUMMARY_COLUMNS:
                            item[col] = row.get(col, np.nan)
                        rows.append(item)
    return pd.DataFrame(rows)



def aggregate_for_plotting(flat):
    grouped = (
        flat.groupby(["scenario", "run", "star", "band"], as_index=False)[SUMMARY_COLUMNS]
        .mean(numeric_only=True)
    )
    return grouped


# -----------------------------
# PLOTTING
# -----------------------------
def plot_scenario_publication_summary(plot_df, scenario, out_dir):
    scenario_df = plot_df[plot_df["scenario"] == str(scenario)].copy()
    if scenario_df.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    axes = axes.ravel()
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    panel_bands = ["UV", "Vis", "IR", "All"]

    x = np.arange(len(STAR_NAME_ORDER))
    width = 0.23

    for idx, (ax, band) in enumerate(zip(axes, panel_bands)):
        band_df = scenario_df[scenario_df["band"] == band]

        if band != "All":
            metric = UV_VIS_IR_COMBINED_METRIC
            values_all = []
            for run_i, run in enumerate(RUN_ORDER):
                values = []
                for star in STAR_NAME_ORDER:
                    m = band_df[(band_df["run"] == run) & (band_df["star"] == star)][metric]
                    values.append(m.iloc[0] if not m.empty else np.nan)
                values_all.extend([v for v in values if np.isfinite(v)])
                ax.bar(
                    x + (run_i - 1) * width,
                    values,
                    width=width,
                    color=BAND_COLORS[band],
                    edgecolor="black",
                    linewidth=0.4,
                    hatch=RUN_HATCHES[run],
                )
            maybe_log_axis(ax, values_all)
            ax.set_ylabel(metric)
        else:
            values_all = []
            for run_i, run in enumerate(RUN_ORDER):
                bottoms = np.zeros(len(STAR_NAME_ORDER), dtype=float)
                x_pos = x + (run_i - 1) * width
                for comp_band in COMPONENT_BANDS:
                    comp_vals = []
                    for star in STAR_NAME_ORDER:
                        m = scenario_df[
                            (scenario_df["run"] == run)
                            & (scenario_df["star"] == star)
                            & (scenario_df["band"] == comp_band)
                        ][UV_VIS_IR_COMBINED_METRIC]
                        comp_vals.append(m.iloc[0] if not m.empty else np.nan)

                    comp_arr = np.array(comp_vals, dtype=float)
                    bottoms_safe = bottoms.copy()
                    mask = np.isfinite(comp_arr)
                    ax.bar(
                        x_pos[mask],
                        comp_arr[mask],
                        width=width,
                        bottom=bottoms_safe[mask],
                        color=BAND_COLORS[comp_band],
                        edgecolor="black",
                        linewidth=0.4,
                        hatch=RUN_HATCHES[run],
                    )
                    bottoms[mask] += comp_arr[mask]
                    values_all.extend(comp_arr[np.isfinite(comp_arr)].tolist())
                values_all.extend(bottoms[np.isfinite(bottoms)].tolist())
            maybe_log_axis(ax, values_all)
            ax.set_ylabel(f"Stacked {UV_VIS_IR_COMBINED_METRIC}")

        ax.set_xticks(x)
        ax.set_xticklabels(STAR_NAME_ORDER, rotation=20, ha="right")
        title = "Total" if band == "All" else band
        ax.set_title(f"{panel_labels[idx]} {title}")
        clean_axis(ax)

    component_handles = band_component_legend_handles()
    run_handles = run_style_legend_handles()
    fig.legend(
        component_handles,
        [h.get_label() for h in component_handles],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.legend(
        run_handles,
        [h.get_label() for h in run_handles],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
    )

    label = scenario_label(scenario)
    fig.suptitle(f"{label}: spectral summary by star and stellar activity", y=1.10)
    file_stub = sanitize_for_filename(label.lower())
    base = os.path.join(out_dir, "scenario_publication_summaries", f"{file_stub}_publication_summary")
    save_fig_png_pdf(fig, base)
    return f"{base}.png"



def plot_scenario_all_split(plot_df, scenario, out_dir):
    scenario_df = plot_df[(plot_df["scenario"] == str(scenario)) & (plot_df["band"] == "All")].copy()
    if scenario_df.empty:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    x = np.arange(len(STAR_NAME_ORDER))
    width = 0.23

    for ax, metric in zip(axes, ALL_SPLIT_COLUMNS):
        all_vals = []
        for run_i, run in enumerate(RUN_ORDER):
            values = []
            for star in STAR_NAME_ORDER:
                m = scenario_df[(scenario_df["run"] == run) & (scenario_df["star"] == star)][metric]
                values.append(m.iloc[0] if not m.empty else np.nan)
            all_vals.extend([v for v in values if np.isfinite(v)])
            ax.bar(
                x + (run_i - 1) * width,
                values,
                width=width,
                color="white",
                edgecolor="black",
                linewidth=0.6,
                hatch=RUN_HATCHES[run],
            )
        maybe_log_axis(ax, all_vals)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(STAR_NAME_ORDER, rotation=20, ha="right")
        clean_axis(ax)

    axes[0].set_ylabel("All-band value")
    run_handles = run_style_legend_handles()
    fig.legend(
        run_handles,
        [h.get_label() for h in run_handles],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )
    label = scenario_label(scenario)
    fig.suptitle(f"{label}: All-band metrics split by type", y=1.12)
    file_stub = sanitize_for_filename(label.lower())
    base = os.path.join(out_dir, "scenario_all_split", f"{file_stub}_all_split")
    save_fig_png_pdf(fig, base)
    return f"{base}.png"



def plot_scenario_uv_vis_ir_combined(plot_df, scenario, out_dir):
    scenario_df = plot_df[(plot_df["scenario"] == str(scenario)) & (plot_df["band"].isin(COMPONENT_BANDS))].copy()
    if scenario_df.empty:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    x = np.arange(len(STAR_NAME_ORDER))
    width = 0.23

    for ax, run in zip(axes, RUN_ORDER):
        run_df = scenario_df[scenario_df["run"] == run]
        all_vals = []
        for band_idx, band in enumerate(COMPONENT_BANDS):
            values = []
            for star in STAR_NAME_ORDER:
                m = run_df[(run_df["star"] == star) & (run_df["band"] == band)][UV_VIS_IR_COMBINED_METRIC]
                values.append(m.iloc[0] if not m.empty else np.nan)
            offset = (band_idx - 1) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=band,
                color=BAND_COLORS[band],
                edgecolor="black",
                linewidth=0.4,
            )
            all_vals.extend([v for v in values if np.isfinite(v)])
        maybe_log_axis(ax, all_vals)
        ax.set_title(RUN_LABELS[run])
        ax.set_xticks(x)
        ax.set_xticklabels(STAR_NAME_ORDER, rotation=20, ha="right")
        clean_axis(ax)

    axes[0].set_ylabel(UV_VIS_IR_COMBINED_METRIC)
    component_handles = band_component_legend_handles()
    fig.legend(
        component_handles,
        [h.get_label() for h in component_handles],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    label = scenario_label(scenario)
    fig.suptitle(f"{label}: combined UV / Vis / IR ({UV_VIS_IR_COMBINED_METRIC})", y=1.08)
    file_stub = sanitize_for_filename(label.lower())
    base = os.path.join(out_dir, "scenario_uv_vis_ir_combined", f"{file_stub}_uv_vis_ir_combined")
    save_fig_png_pdf(fig, base)
    return f"{base}.png"


# -----------------------------
# MAIN
# -----------------------------
def main(base_dir=BASE_DIR, out_dir=OUT_DIR):
    ensure_dir(out_dir)
    dataset = build_dataset(base_dir)
    flat = flatten_dataset(dataset)

    if flat.empty:
        print("No clima_allout.tab summary data found.")
        return

    plot_df = aggregate_for_plotting(flat)
    csv_path = os.path.join(out_dir, "tot_flux_publication_summary.csv")
    plot_df.to_csv(csv_path, index=False)
    print(f"Saved table: {csv_path}")

    scenarios = sorted(plot_df["scenario"].dropna().unique(), key=lambda x: int(x))
    for scenario in scenarios:
        plot_scenario_publication_summary(plot_df, scenario, out_dir)
        plot_scenario_uv_vis_ir_combined(plot_df, scenario, out_dir)
        plot_scenario_all_split(plot_df, scenario, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
