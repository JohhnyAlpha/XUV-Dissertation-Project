# water_vs_fscale_uncertainty.py

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
OUT_DIR = os.path.expanduser("~/atmos/BA/Plots/water_vs_fscale_uncertainty")

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


def pick_column(df, candidates=None, contains_candidates=None):
    candidates = candidates or []
    contains_candidates = contains_candidates or []

    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for col in df.columns:
        lc = col.lower()
        for cand in contains_candidates:
            if cand.lower() in lc:
                return col

    return None


def read_table(path):
    try:
        return pd.read_csv(
            path,
            sep=r"\s+",
            engine="python",
            comment="#",
        )
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return None


# -----------------------------
# FILE DISCOVERY
# -----------------------------
def find_clima_files(scale_dir):
    clima_dir = os.path.join(scale_dir, "CLIMA_OUTPUT")
    files = defaultdict(lambda: None)

    if not os.path.isdir(clima_dir):
        return {}

    pattern = re.compile(r"^(Run_\d+)_([0-9]+(?:\.[0-9]+)?)_(\d+)_(.+)$")

    for fname in os.listdir(clima_dir):
        match = pattern.match(fname)

        if not match:
            continue

        _, _, scenario, suffix = match.groups()

        if suffix in ("clima_last.tab", "clima_last.out"):
            files[scenario] = os.path.join(clima_dir, fname)

    return dict(files)


# -----------------------------
# PARSING / METRICS
# -----------------------------
def parse_clima_water(path):
    df = read_table(path)

    if df is None or df.empty:
        return None

    alt_col = pick_column(
        df,
        candidates=["ALT", "Z", "Alt", "alt", "z"],
        contains_candidates=["altitude"],
    )

    h2o_col = pick_column(
        df,
        candidates=["H2O", "h2o"],
        contains_candidates=["h2o", "water"],
    )

    if alt_col is None or h2o_col is None:
        print(f"Missing ALT or H2O column in: {path}")
        print(f"Columns found: {list(df.columns)}")
        return None

    alt = pd.to_numeric(df[alt_col], errors="coerce").to_numpy(dtype=float)
    h2o = pd.to_numeric(df[h2o_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(alt) & np.isfinite(h2o) & (h2o > 0)
    alt = alt[mask]
    h2o = h2o[mask]

    if len(alt) < 2:
        return None

    order = np.argsort(alt)
    alt = alt[order]
    h2o = h2o[order]

    above_50 = alt >= 50.0

    surface_h2o = h2o[0]
    mean_h2o = np.nanmean(h2o)
    mean_h2o_above_50km = np.nanmean(h2o[above_50]) if np.any(above_50) else np.nan

    if hasattr(np, "trapezoid"):
        column_proxy_h2o = np.trapezoid(h2o, alt)
    else:
        column_proxy_h2o = np.trapz(h2o, alt)

    return {
        "surface_H2O": surface_h2o,
        "mean_H2O": mean_h2o,
        "mean_H2O_above_50km": mean_h2o_above_50km,
        "column_proxy_H2O": column_proxy_h2o,
        "source_file": path,
        "alt_col": alt_col,
        "h2o_col": h2o_col,
    }


def build_water_metrics(base_dir):
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

                scenario_files = find_clima_files(scale_dir)

                for scenario, clima_path in sorted(
                    scenario_files.items(),
                    key=lambda x: int(x[0]),
                ):
                    metrics = parse_clima_water(clima_path)

                    if metrics is None:
                        continue

                    row = {
                        "run": run,
                        "star": star,
                        "fscale": float(scale),
                        "scenario": int(scenario),
                    }

                    row.update(metrics)
                    rows.append(row)

                    print(
                        f"Loaded water metrics: {run} | {star} | "
                        f"fscale={scale} | repeat={scenario}"
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


# -----------------------------
# PLOTTING
# -----------------------------
def apply_style(ax):
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlabel("FSCALE")


def set_log_if_positive(ax, values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) > 0 and np.all(values > 0):
        ax.set_yscale("log")


def plot_metric_uncertainty_by_star(df, metric, out_dir):
    summary = summarise_repeats(df, metric)

    stars = sorted(summary["star"].unique())
    runs = sorted(summary["run"].unique(), key=lambda x: int(x.split("_")[1]))

    fig, axes = plt.subplots(
        len(stars),
        1,
        figsize=(8, 11),
        sharex=True,
        sharey=False,
    )

    if len(stars) == 1:
        axes = [axes]

    for ax, star in zip(axes, stars):
        star_summary = summary[summary["star"] == star]

        for run in runs:
            rsub = star_summary[star_summary["run"] == run].sort_values("fscale")

            if rsub.empty:
                continue

            x = rsub["fscale"].to_numpy(dtype=float)
            y = rsub["mean_value"].to_numpy(dtype=float)
            yerr = rsub["std_value"].to_numpy(dtype=float)

            ax.plot(
                x,
                y,
                linestyle=RUN_STYLES.get(run, "-"),
                marker="o",
                linewidth=2.2,
                label=AGE_LABELS.get(run, run),
            )

            lower = y - yerr
            upper = y + yerr

            positive_mask = np.isfinite(lower) & np.isfinite(upper)

            if np.any(positive_mask):
                ax.fill_between(
                    x[positive_mask],
                    lower[positive_mask],
                    upper[positive_mask],
                    alpha=0.18,
                )

        apply_style(ax)
        set_log_if_positive(ax, star_summary["mean_value"])
        ax.set_ylabel(f"{star}\n{metric}")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Stellar Age",
        fontsize=10,
    )

    fig.suptitle(
        f"Atmospheric Water vs FSCALE\nMean ± 1σ across repeat runs: {metric}",
        y=0.98,
        fontsize=14,
    )

    ensure_dir(out_dir)

    out_png = os.path.join(
        out_dir,
        f"water_vs_fscale_uncertainty_{sanitize(metric)}.png",
    )

    out_pdf = os.path.join(
        out_dir,
        f"water_vs_fscale_uncertainty_{sanitize(metric)}.pdf",
    )

    fig.tight_layout(rect=[0, 0, 0.84, 0.94])
    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=SAVE_DPI, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


def plot_metric_all_stars_panel(df, metric, out_dir):
    summary = summarise_repeats(df, metric)

    stars = sorted(summary["star"].unique())
    runs = sorted(summary["run"].unique(), key=lambda x: int(x.split("_")[1]))

    ncols = 2
    nrows = int(np.ceil(len(stars) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 9),
        sharex=True,
        sharey=False,
    )

    axes = np.asarray(axes).flatten()

    for ax, star in zip(axes, stars):
        star_summary = summary[summary["star"] == star]

        for run in runs:
            rsub = star_summary[star_summary["run"] == run].sort_values("fscale")

            if rsub.empty:
                continue

            x = rsub["fscale"].to_numpy(dtype=float)
            y = rsub["mean_value"].to_numpy(dtype=float)
            yerr = rsub["std_value"].to_numpy(dtype=float)

            ax.plot(
                x,
                y,
                linestyle=RUN_STYLES.get(run, "-"),
                marker="o",
                linewidth=2.0,
                label=AGE_LABELS.get(run, run),
            )

            lower = y - yerr
            upper = y + yerr

            mask = np.isfinite(lower) & np.isfinite(upper)

            if np.any(mask):
                ax.fill_between(
                    x[mask],
                    lower[mask],
                    upper[mask],
                    alpha=0.18,
                )

        apply_style(ax)
        set_log_if_positive(ax, star_summary["mean_value"])
        ax.set_title(star)
        ax.set_ylabel(metric)

    for ax in axes[len(stars):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Stellar Age",
        fontsize=10,
    )

    fig.suptitle(
        f"Atmospheric Water vs FSCALE: {metric}\nMean ± 1σ across repeat runs",
        y=0.98,
        fontsize=14,
    )

    ensure_dir(out_dir)

    out_png = os.path.join(
        out_dir,
        f"water_vs_fscale_uncertainty_panel_{sanitize(metric)}.png",
    )

    out_pdf = os.path.join(
        out_dir,
        f"water_vs_fscale_uncertainty_panel_{sanitize(metric)}.pdf",
    )

    fig.tight_layout(rect=[0, 0, 0.84, 0.92])
    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=SAVE_DPI, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


def plot_all_metrics(df, out_dir):
    metrics = [
        "surface_H2O",
        "mean_H2O",
        "mean_H2O_above_50km",
        "column_proxy_H2O",
    ]

    for metric in metrics:
        plot_metric_uncertainty_by_star(df, metric, out_dir)
        plot_metric_all_stars_panel(df, metric, out_dir)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    ensure_dir(OUT_DIR)

    df = build_water_metrics(BASE_DIR)

    raw_path = os.path.join(OUT_DIR, "water_vs_fscale_raw_repeat_metrics.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved {raw_path}")

    for metric in [
        "surface_H2O",
        "mean_H2O",
        "mean_H2O_above_50km",
        "column_proxy_H2O",
    ]:
        summary = summarise_repeats(df, metric)
        summary_path = os.path.join(
            OUT_DIR,
            f"water_vs_fscale_summary_{sanitize(metric)}.csv",
        )
        summary.to_csv(summary_path, index=False)
        print(f"Saved {summary_path}")

    plot_all_metrics(df, OUT_DIR)

    print("\nDone.")
    print(f"Outputs written to: {OUT_DIR}")
