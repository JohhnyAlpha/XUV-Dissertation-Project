# haze_vs_fscale_publication.py

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.expanduser("~/atmos/OutputStorage")
OUT_DIR = os.path.expanduser("~/atmos/BA/Plots/haze_vs_fscale_publication")

STARS = {
    "18": "Epsilon_Eri",
    "24": "HD_40307",
    "25": "HD_85512",
    "26": "HD_97658",
}

AGE_LABELS = {
    "Run_1": "Young Star",
    "Run_2": "Intermediate Age",
    "Run_3": "Evolved Star",
}

RUN_STYLES = {
    "Run_1": "-",
    "Run_2": "--",
    "Run_3": ":",
}

METRICS = [
    "haze_mean",
    "haze_max",
    "haze_integrated",
]

METRIC_LABELS = {
    "haze_mean": "Mean aerosol signal",
    "haze_max": "Peak aerosol signal",
    "haze_integrated": "Integrated aerosol proxy",
}

METRIC_FILENAMES = {
    "haze_mean": "haze_mean",
    "haze_max": "haze_peak",
    "haze_integrated": "haze_integrated",
}

SAVE_DPI = 300
SHOW_PLOTS = False


# -----------------------------
# HELPERS
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize(value):
    return str(value).replace(" ", "_").replace("/", "_").replace("\\", "_")


def numeric_sort_key(value):
    try:
        return float(value)
    except Exception:
        return str(value)


def star_name_from_id(star_id):
    return STARS.get(str(star_id), str(star_id))


def discover_run_dirs(base_dir):
    runs = []

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)

        if os.path.isdir(full) and re.fullmatch(r"Run_\d+", name):
            runs.append(name)

    return sorted(runs, key=lambda x: int(x.split("_")[1]))


# -----------------------------
# FILE DISCOVERY
# -----------------------------
def find_haze_files(scale_dir):
    photochem_dir = os.path.join(scale_dir, "PHOTOCHEM_OUTPUT")
    files = defaultdict(lambda: None)

    if not os.path.isdir(photochem_dir):
        return {}

    pattern = re.compile(r"^(Run_\d+)_([0-9]+(?:\.[0-9]+)?)_(\d+)_(.+)$")

    for fname in os.listdir(photochem_dir):
        match = pattern.match(fname)

        if not match:
            continue

        _, _, repeat, suffix = match.groups()

        if suffix == "hcaer.out":
            files[repeat] = os.path.join(photochem_dir, fname)

    return dict(files)


# -----------------------------
# HAZE PARSER
# -----------------------------
def parse_haze_file(path):
    """
    Robust parser for hcaer.out.

    This attempts to extract numeric data from the aerosol file even if
    the header is irregular. It then derives three scalar diagnostics:

    haze_mean       = mean aerosol-like value
    haze_max        = peak aerosol-like value
    haze_integrated = sum/integral-like proxy over the file

    If a clean altitude column is found later, this can be refined to
    integrate aerosol signal over altitude.
    """

    numeric_rows = []

    try:
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()

                nums = []
                for part in parts:
                    try:
                        nums.append(float(part.replace("D", "E")))
                    except ValueError:
                        continue

                if nums:
                    numeric_rows.append(nums)

    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None

    if not numeric_rows:
        print(f"No numeric data found in {path}")
        return None

    max_len = max(len(row) for row in numeric_rows)

    arr = np.full((len(numeric_rows), max_len), np.nan)

    for i, row in enumerate(numeric_rows):
        arr[i, :len(row)] = row

    if arr.size == 0:
        return None

    # Remove columns that are entirely NaN
    valid_cols = ~np.all(~np.isfinite(arr), axis=0)
    arr = arr[:, valid_cols]

    if arr.shape[1] == 0:
        return None

    # Heuristic:
    # If multiple numeric columns exist, avoid first column if it looks like altitude/grid.
    # Use the column with the largest finite positive total as aerosol signal.
    candidate_cols = list(range(arr.shape[1]))

    if arr.shape[1] > 1:
        candidate_cols = candidate_cols[1:]

    scores = []

    for col in candidate_cols:
        values = arr[:, col]
        values = values[np.isfinite(values)]

        if len(values) == 0:
            scores.append(-np.inf)
            continue

        positive = values[values > 0]

        if len(positive) == 0:
            scores.append(-np.inf)
            continue

        scores.append(np.nansum(positive))

    if not scores or np.all(~np.isfinite(scores)):
        return None

    chosen_col = candidate_cols[int(np.nanargmax(scores))]

    haze = arr[:, chosen_col]
    haze = haze[np.isfinite(haze) & (haze > 0)]

    if len(haze) == 0:
        return None

    return {
        "haze_mean": np.nanmean(haze),
        "haze_max": np.nanmax(haze),
        "haze_integrated": np.nansum(haze),
        "chosen_numeric_column": chosen_col,
        "source_file": path,
    }


# -----------------------------
# BUILD DATASET
# -----------------------------
def build_haze_metrics(base_dir):
    rows = []

    for run in discover_run_dirs(base_dir):
        run_dir = os.path.join(base_dir, run)

        for star_id in sorted(os.listdir(run_dir), key=numeric_sort_key):
            star_dir = os.path.join(run_dir, star_id)

            if not os.path.isdir(star_dir):
                continue

            star = star_name_from_id(star_id)

            for scale in sorted(os.listdir(star_dir), key=numeric_sort_key):
                scale_dir = os.path.join(star_dir, scale)

                if not os.path.isdir(scale_dir):
                    continue

                repeat_files = find_haze_files(scale_dir)

                for repeat, haze_path in sorted(
                    repeat_files.items(),
                    key=lambda x: int(x[0]),
                ):
                    metrics = parse_haze_file(haze_path)

                    if metrics is None:
                        continue

                    row = {
                        "run": run,
                        "star": star,
                        "fscale": float(scale),
                        "repeat": int(repeat),
                    }

                    row.update(metrics)
                    rows.append(row)

                    print(
                        f"Loaded haze metrics: {run} | {star} | "
                        f"fscale={scale} | repeat={repeat}"
                    )

    return pd.DataFrame(rows)


# -----------------------------
# SUMMARY
# -----------------------------
def summarise_repeats(df, metric):
    summary = (
        df.groupby(["star", "run", "fscale"], as_index=False)
        .agg(
            mean_value=(metric, "mean"),
            std_value=(metric, "std"),
            n=(metric, "count"),
        )
    )

    summary["std_value"] = summary["std_value"].fillna(0.0)

    return summary


def save_summary_tables(df, out_dir):
    ensure_dir(out_dir)

    raw_path = os.path.join(out_dir, "haze_vs_fscale_raw_repeat_metrics.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved {raw_path}")

    for metric in METRICS:
        summary = summarise_repeats(df, metric)

        summary_path = os.path.join(
            out_dir,
            f"haze_vs_fscale_summary_{METRIC_FILENAMES.get(metric, sanitize(metric))}.csv",
        )

        summary.to_csv(summary_path, index=False)
        print(f"Saved {summary_path}")


# -----------------------------
# PLOTTING
# -----------------------------
def plot_metric_publication_panel(df, metric, out_dir):
    summary = summarise_repeats(df, metric)

    stars = sorted(summary["star"].unique())
    runs = sorted(summary["run"].unique(), key=lambda x: int(x.split("_")[1]))

    ncols = 2
    nrows = int(np.ceil(len(stars) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.2, 6.2),
        sharex=True,
        sharey=False,
    )

    axes = np.asarray(axes).flatten()
    all_fscales = sorted(summary["fscale"].unique())

    for ax, star in zip(axes, stars):
        star_summary = summary[summary["star"] == star]

        for run in runs:
            rsub = star_summary[star_summary["run"] == run].sort_values("fscale")

            if rsub.empty:
                continue

            x = rsub["fscale"].to_numpy(dtype=float)
            y = rsub["mean_value"].to_numpy(dtype=float)
            yerr = rsub["std_value"].to_numpy(dtype=float)

            lower = np.clip(y - yerr, a_min=1e-300, a_max=None)
            upper = y + yerr

            ax.plot(
                x,
                y,
                linestyle=RUN_STYLES.get(run, "-"),
                marker="o",
                markersize=4,
                linewidth=1.8,
                label=AGE_LABELS.get(run, run),
            )

            ax.fill_between(
                x,
                lower,
                upper,
                alpha=0.16,
                linewidth=0,
            )

        ax.set_title(star.replace("_", " "), fontsize=10)
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.tick_params(direction="in", top=True, right=True, labelsize=8)
        ax.set_xlabel("FSCALE", fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_xticks(all_fscales)

        values = star_summary["mean_value"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if len(values) > 0 and np.all(values > 0):
            ax.set_yscale("log")

    for ax in axes[len(stars):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        title="Stellar age",
        title_fontsize=9,
    )

    fig.suptitle(
        "Aerosol / haze response to stellar flux scaling",
        fontsize=12,
        y=0.985,
    )

    fig.text(
        0.5,
        0.015,
        r"Points show the mean of three repeat ATMOS runs; shaded regions show $\pm 1\sigma$.",
        ha="center",
        fontsize=8,
    )

    ensure_dir(out_dir)

    base = METRIC_FILENAMES.get(metric, sanitize(metric))

    out_png = os.path.join(out_dir, f"publication_haze_vs_fscale_{base}.png")
    out_pdf = os.path.join(out_dir, f"publication_haze_vs_fscale_{base}.pdf")

    fig.tight_layout(rect=[0, 0.04, 0.84, 0.94])

    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=SAVE_DPI, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


def plot_interpretation_multipane(df, out_dir):
    stars = sorted(df["star"].unique())
    runs = sorted(df["run"].unique(), key=lambda x: int(x.split("_")[1]))

    nrows = len(stars)
    ncols = len(METRICS)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(11, 10),
        sharex=True,
        sharey=False,
    )

    axes = np.asarray(axes)
    all_fscales = sorted(df["fscale"].unique())

    summaries = {
        metric: summarise_repeats(df, metric)
        for metric in METRICS
    }

    for i, star in enumerate(stars):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]

            summary = summaries[metric]
            star_summary = summary[summary["star"] == star]

            for run in runs:
                rsub = star_summary[star_summary["run"] == run].sort_values("fscale")

                if rsub.empty:
                    continue

                x = rsub["fscale"].to_numpy(dtype=float)
                y = rsub["mean_value"].to_numpy(dtype=float)
                yerr = rsub["std_value"].to_numpy(dtype=float)

                lower = np.clip(y - yerr, a_min=1e-300, a_max=None)
                upper = y + yerr

                ax.plot(
                    x,
                    y,
                    linestyle=RUN_STYLES.get(run, "-"),
                    marker="o",
                    markersize=3.5,
                    linewidth=1.6,
                    label=AGE_LABELS.get(run, run),
                )

                ax.fill_between(
                    x,
                    lower,
                    upper,
                    alpha=0.14,
                    linewidth=0,
                )

            ax.grid(True, alpha=0.22, linewidth=0.6)
            ax.tick_params(direction="in", top=True, right=True, labelsize=8)
            ax.set_xticks(all_fscales)

            values = star_summary["mean_value"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]

            if len(values) > 0 and np.all(values > 0):
                ax.set_yscale("log")

            if i == 0:
                ax.set_title(
                    METRIC_LABELS.get(metric, metric),
                    fontsize=9,
                )

            if j == 0:
                ax.set_ylabel(
                    star.replace("_", " "),
                    fontsize=10,
                )

            if i == nrows - 1:
                ax.set_xlabel("FSCALE", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=9,
        title="Stellar age",
        title_fontsize=9,
    )

    fig.suptitle(
        "Aerosol / haze diagnostics as a function of stellar flux scaling",
        fontsize=13,
        y=0.99,
    )

    fig.text(
        0.5,
        0.012,
        r"Each point is the mean of three repeat ATMOS runs; shaded regions show $\pm 1\sigma$ repeat-run variability.",
        ha="center",
        fontsize=8,
    )

    ensure_dir(out_dir)

    out_png = os.path.join(
        out_dir,
        "publication_haze_vs_fscale_interpretation_multipane.png",
    )

    out_pdf = os.path.join(
        out_dir,
        "publication_haze_vs_fscale_interpretation_multipane.pdf",
    )

    fig.tight_layout(rect=[0, 0.035, 0.88, 0.96])

    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=SAVE_DPI, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


def plot_all_metrics(df, out_dir):
    for metric in METRICS:
        plot_metric_publication_panel(df, metric, out_dir)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    ensure_dir(OUT_DIR)

    df = build_haze_metrics(BASE_DIR)

    if df.empty:
        raise RuntimeError(
            "No haze metrics were loaded. Check BASE_DIR and hcaer.out locations."
        )

    save_summary_tables(df, OUT_DIR)
    plot_all_metrics(df, OUT_DIR)
    plot_interpretation_multipane(df, OUT_DIR)

    print("\nDone.")
    print(f"Outputs written to: {OUT_DIR}")
