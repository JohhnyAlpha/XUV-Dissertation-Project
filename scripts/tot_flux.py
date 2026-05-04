import os
import re
from collections import defaultdict
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.expanduser("~/atmos/OutputStorage")
PLOTS_DIR = os.path.expanduser("~/atmos/BA/Plots/tot_flux")
SAVE_DPI = 200
SHOW_PLOTS = False

STARS = {
    "18": "Epsilon_Eri",
    "24": "HD_40307",
    "25": "HD_85512",
    "26": "HD_97658",
}

BANDS = ["UV", "Vis", "IR", "All"]
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
PLOT_COLUMNS = [c for c in SUMMARY_COLUMNS if c not in ("WAVLL", "WAVLU")]

# -----------------------------
# HELPERS
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize_for_filename(value):
    value = str(value)
    for old, new in [(" ", "_"), ("/", "_"), ("\\", "_"), ("=", "_")]:
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
    run_dirs = []
    if not os.path.isdir(base_dir):
        print(f"Base directory not found: {base_dir}")
        return run_dirs

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


# -----------------------------
# FILE DISCOVERY
# -----------------------------
def build_clima_allout_index(scale_dir):
    """
    Find files like:
        Run_4_0.75_1_clima_allout.tab
    under <scale_dir>/CLIMA_OUTPUT.
    """
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


# -----------------------------
# PARSER
# -----------------------------
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


def parse_clima_allout_summary(file_path):
    """
    Parse the final shortwave spectral summary block in clima_allout.tab.

    Expected structure near the end of the block:
        I    WAVLL      WAVLU       FSOL ... PDNSOD(2)
        1  ...
        ...
        UV  ...
        Vis ...
        IR  ...
        All ...

    Returns a DataFrame with one row per band: UV / Vis / IR / All.
    """
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
            print(f"Band row too short in {file_path}: {stripped}")
            continue

        row = {"band": band}
        for col, token in zip(SUMMARY_COLUMNS, values[: len(SUMMARY_COLUMNS)]):
            row[col] = _coerce_numeric(token)
        band_rows.append(row)

        if len(band_rows) == 4:
            break

    if not band_rows:
        print(f"No UV/Vis/IR/All summary rows found in {file_path}")
        return None

    df = pd.DataFrame(band_rows)
    df["band"] = pd.Categorical(df["band"], categories=BANDS, ordered=True)
    df = df.sort_values("band").reset_index(drop=True)
    return df


# -----------------------------
# DATASET BUILD
# -----------------------------
def build_dataset(base_dir):
    """
    data[run][star][scale][scenario] = {
        "summary": DataFrame,
        "file": path,
    }
    """
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
                    print(f"Loading: {run} | {star_name} | scale={scale} | scenario={scenario}")
                    print(f"  clima_allout: {file_path}")
                    summary = parse_clima_allout_summary(file_path)
                    data[run][star_name][scale][scenario] = {
                        "summary": summary,
                        "file": file_path,
                    }

    return data


# -----------------------------
# FLATTEN
# -----------------------------
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
                            "scale": scale,
                            "scenario": str(scenario),
                            "file": entry.get("file"),
                            "band": str(row["band"]),
                        }
                        for col in SUMMARY_COLUMNS:
                            item[col] = row.get(col, np.nan)
                        rows.append(item)
    return pd.DataFrame(rows)


# -----------------------------
# PLOTTING
# -----------------------------
def plot_case_all_metrics(run, star, scale, scenario, summary_df, out_dir):
    if summary_df is None or summary_df.empty:
        return

    plot_df = summary_df.copy()
    plot_df["band"] = plot_df["band"].astype(str)

    ncols = 2
    nrows = int(np.ceil(len(PLOT_COLUMNS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    x = np.arange(len(BANDS))
    for ax, col in zip(axes, PLOT_COLUMNS):
        y = [plot_df.loc[plot_df["band"] == band, col].iloc[0] if np.any(plot_df["band"] == band) else np.nan for band in BANDS]
        ax.bar(x, y)
        ax.set_title(col)
        ax.set_xticks(x)
        ax.set_xticklabels(BANDS)
        ax.grid(True, axis="y", alpha=0.3)
        if np.all(np.isfinite(y)) and np.nanmax(np.abs(y)) > 0 and np.nanmax(np.abs(y)) / max(np.nanmin(np.abs(np.array(y)[np.array(y) != 0])), 1e-300) > 1e3:
            ax.set_yscale("log")

    for ax in axes[len(PLOT_COLUMNS):]:
        ax.axis("off")

    fig.suptitle(f"{run} | {star} | scale={scale} | scenario={scenario} | UV / Vis / IR / All", y=0.995)
    out_path = os.path.join(
        out_dir,
        "cases",
        f"tot_flux_{sanitize_for_filename(run)}_{sanitize_for_filename(star)}_scale_{sanitize_for_filename(scale)}_scenario_{sanitize_for_filename(scenario)}.png",
    )
    save_figure(fig, out_path)


def plot_star_metric_by_case(df, star, metric, out_dir):
    star_df = df[df["star"] == star].copy()
    if star_df.empty:
        return

    case_order = (
        star_df[["run", "scale", "scenario"]]
        .drop_duplicates()
        .sort_values(by=["run", "scale", "scenario"], key=lambda col: col.map(numeric_sort_key))
    )
    case_order["case"] = case_order.apply(lambda r: f"{r['run']}\nscale={r['scale']}\ns={r['scenario']}", axis=1)

    wide = star_df.merge(case_order, on=["run", "scale", "scenario"], how="left")
    wide = wide.pivot_table(index="case", columns="band", values=metric, aggfunc="first")
    wide = wide.reindex(case_order["case"])

    fig, ax = plt.subplots(figsize=(max(8, len(wide) * 0.8), 5))
    x = np.arange(len(wide.index))
    width = 0.2
    for idx, band in enumerate(BANDS):
        if band not in wide.columns:
            continue
        ax.bar(x + (idx - 1.5) * width, wide[band].values, width=width, label=band)

    ax.set_xticks(x)
    ax.set_xticklabels(wide.index, rotation=90)
    ax.set_ylabel(metric)
    ax.set_title(f"{star} | {metric} by scenario")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    vals = wide.to_numpy(dtype=float).ravel()
    vals = vals[np.isfinite(vals) & (vals != 0)]
    if vals.size > 1 and np.nanmax(np.abs(vals)) / np.nanmin(np.abs(vals)) > 1e3:
        ax.set_yscale("log")

    out_path = os.path.join(
        out_dir,
        "stars",
        f"tot_flux_{sanitize_for_filename(star)}_{sanitize_for_filename(metric)}.png",
    )
    save_figure(fig, out_path)


# -----------------------------
# MAIN
# -----------------------------
def main(base_dir=BASE_DIR, out_dir=PLOTS_DIR):
    ensure_dir(out_dir)
    dataset = build_dataset(base_dir)
    flat = flatten_dataset(dataset)

    if flat.empty:
        print("No clima_allout.tab summary data found.")
        return

    csv_path = os.path.join(out_dir, "tot_flux_summary.csv")
    flat.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    for run, stars in dataset.items():
        for star, scales in stars.items():
            for scale, scenarios in scales.items():
                for scenario, entry in scenarios.items():
                    plot_case_all_metrics(run, star, scale, scenario, entry.get("summary"), out_dir)

    for star in sorted(flat["star"].dropna().unique()):
        for metric in PLOT_COLUMNS:
            plot_star_metric_by_case(flat, star, metric, out_dir)

    print("\nDone.")
    print(f"Plots written to: {out_dir}")


if __name__ == "__main__":
    main()
