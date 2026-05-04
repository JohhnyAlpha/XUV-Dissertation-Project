# haze_synthesis_clean_star_multipane.py
"""
Clean dissertation-ready haze synthesis plots.

Purpose
-------
This script keeps the Arney-style CH4-column versus haze-proxy framing,
but cleans the output set and adds one coherent multi-pane figure per host
star. It expects the already-created species and aerosol metric tables from:

    species_analysis_pub/tables/species_metrics.csv
    aerosol_analysis_pub/tables/aerosol_metrics.csv

Outputs
-------
OUT_DIR/
    tables/
        haze_synthesis.csv
        haze_summary_<metric>.csv
    figures/
        publication_arney_regime_all_stars.png/pdf
    figures/by_star/
        publication_<star>_haze_synthesis_multipane.png/pdf
    haze_summary.txt

The per-star figure is the main dissertation-ready output:
    A: CH4 column proxy vs FSCALE
    B: haze proxy vs FSCALE
    C: redox proxy vs FSCALE
    D: Arney-style CH4 column proxy vs haze proxy regime panel
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE = os.path.expanduser("~/atmos/BA/Plots")

SPECIES_FILE = os.path.join(BASE, "species_analysis_pub", "tables", "species_metrics.csv")
AEROSOL_FILE = os.path.join(BASE, "aerosol_analysis_pub", "tables", "aerosol_metrics.csv")
OUT_DIR = os.path.join(BASE, "haze_synthesis_clean_star_multipane")

RUN_ORDER = ["Run_1", "Run_2", "Run_3"]

RUN_LABELS = {
    "Run_1": "Young star",
    "Run_2": "Intermediate age",
    "Run_3": "Evolved star",
}

RUN_STYLES = {
    "Run_1": "-",
    "Run_2": "--",
    "Run_3": ":",
}

RUN_MARKERS = {
    "Run_1": "o",
    "Run_2": "s",
    "Run_3": "^",
}

STAR_ORDER = ["Epsilon_Eri", "HD40307", "HD85512", "HD97658"]

STAR_LABELS = {
    "Epsilon_Eri": r"$\epsilon$ Eri",
    "HD40307": "HD 40307",
    "HD85512": "HD 85512",
    "HD97658": "HD 97658",
}

STAR_COLORS = {
    "Epsilon_Eri": "tab:blue",
    "HD40307": "tab:orange",
    "HD85512": "tab:green",
    "HD97658": "tab:red",
}

FSCALE_ORDER = [0.75, 1.00, 1.50]

FSCALE_MARKERS = {
    0.75: "o",
    1.00: "s",
    1.50: "^",
}

HAZE_SPECIES = {"AERSOL"}
SPECIES_OF_INTEREST = ["H2O", "CH4", "CO", "CO2", "O2", "O3", "H2"]

SYNTHESIS_METRICS = [
    ("CH4_column", r"CH$_4$ column proxy"),
    ("haze_proxy", "Haze proxy"),
    ("redox_proxy", "Redox proxy"),
]

SAVE_DPI = 300
SAVE_PDF = True

# Keep text editable in vector outputs.
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize(value):
    return str(value).replace(" ", "_").replace("/", "_").replace("\\", "_")


def display_star(star):
    return STAR_LABELS.get(star, str(star).replace("_", " "))


def save_fig(fig, out_png):
    ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved {out_png}")

    if SAVE_PDF:
        out_pdf = out_png.replace(".png", ".pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Saved {out_pdf}")

    plt.close(fig)


def safe_ratio(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = pd.Series(np.nan, index=a.index, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (b > 0)
    out.loc[mask] = a.loc[mask] / b.loc[mask]
    return out


def positive_finite(series):
    s = pd.to_numeric(series, errors="coerce")
    return s[np.isfinite(s) & (s > 0)]


def safe_nanmedian(values):
    values = pd.to_numeric(values, errors="coerce")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return np.nanmedian(values)


def finite_positive_xy(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    return np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)


def classify_haze(x):
    if not np.isfinite(x):
        return "unknown"
    if x >= 1e12:
        return "very_thick_haze"
    if x >= 1e10:
        return "thick_haze"
    if x >= 1e8:
        return "weak_haze"
    return "clear"


def classify_redox(x):
    if not np.isfinite(x):
        return "unknown"
    if x >= 1:
        return "strongly_reducing"
    if x >= 0.1:
        return "moderately_reducing"
    return "weakly_reducing_or_oxidizing"


def style_axis(ax):
    ax.grid(True, which="both", alpha=0.22, linewidth=0.6)
    ax.tick_params(direction="in", top=True, right=True)


def apply_log_if_positive(ax, values):
    vals = positive_finite(values)
    if len(vals) > 0:
        ax.set_yscale("log")


def run_legend_handles():
    return [
        Line2D(
            [0], [0],
            color="black",
            linestyle=RUN_STYLES.get(run, "-"),
            marker=RUN_MARKERS.get(run, "o"),
            markersize=5,
            linewidth=1.6,
            label=RUN_LABELS.get(run, run),
        )
        for run in RUN_ORDER
    ]


def fscale_legend_handles():
    return [
        Line2D(
            [0], [0],
            marker=marker,
            linestyle="",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="black",
            label=f"FSCALE={fscale:.2f}",
        )
        for fscale, marker in FSCALE_MARKERS.items()
    ]


# --------------------------------------------------
# LOAD + RESHAPE
# --------------------------------------------------
def load_species_wide(path):
    df = pd.read_csv(path)

    required = [
        "run", "run_label", "star", "scenario",
        "fscale", "scale_dir", "species", "column_proxy",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"species_metrics.csv missing columns: {missing}")

    case_cols = ["run", "run_label", "star", "scenario", "fscale", "scale_dir"]

    wide = (
        df[case_cols + ["species", "column_proxy"]]
        .pivot_table(
            index=case_cols,
            columns="species",
            values="column_proxy",
            aggfunc="first",
        )
        .reset_index()
    )

    rename_map = {
        sp: f"{sp}_column"
        for sp in SPECIES_OF_INTEREST
        if sp in wide.columns
    }
    wide = wide.rename(columns=rename_map)

    for sp in SPECIES_OF_INTEREST:
        col = f"{sp}_column"
        if col not in wide.columns:
            wide[col] = np.nan

    return wide


def load_aerosol_proxy(path):
    df = pd.read_csv(path)

    required = [
        "run", "run_label", "star", "scenario",
        "fscale", "scale_dir", "species", "column_proxy",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"aerosol_metrics.csv missing columns: {missing}")

    df = df.copy()
    df["species"] = df["species"].astype(str)
    df = df[df["species"].isin(HAZE_SPECIES)].copy()

    if df.empty:
        available = sorted(pd.read_csv(path)["species"].astype(str).unique())
        raise ValueError(
            "No aerosol rows remained after filtering for haze species. "
            f"Available aerosol species were: {available}"
        )

    case_cols = ["run", "run_label", "star", "scenario", "fscale", "scale_dir"]

    return (
        df.groupby(case_cols, as_index=False)["column_proxy"]
        .sum()
        .rename(columns={"column_proxy": "haze_proxy"})
    )


# --------------------------------------------------
# SYNTHESIS
# --------------------------------------------------
def build_synthesis(species_wide, aerosol_sum):
    merge_cols = ["run", "run_label", "star", "scenario", "fscale", "scale_dir"]

    synth = pd.merge(species_wide, aerosol_sum, on=merge_cols, how="inner")
    if synth.empty:
        raise ValueError("Merged synthesis table is empty.")

    synth["fscale"] = pd.to_numeric(synth["fscale"], errors="coerce")
    synth["scenario"] = pd.to_numeric(synth["scenario"], errors="coerce")

    # Arney-style diagnostic: use CH4 column directly, because CO2 is often
    # unavailable or zero in this processed table.
    synth["CH4_CO2"] = safe_ratio(synth["CH4_column"], synth["CO2_column"])

    # Simple reducing/oxidizing proxy from available bulk column proxies.
    synth["redox_proxy"] = safe_ratio(
        synth["H2_column"].fillna(0) + synth["CH4_column"].fillna(0),
        synth["CO2_column"].fillna(0) + synth["O2_column"].fillna(0),
    )

    synth["haze_regime"] = synth["haze_proxy"].apply(classify_haze)
    synth["redox_regime"] = synth["redox_proxy"].apply(classify_redox)

    # Clean ordering for stable output.
    synth["run"] = pd.Categorical(synth["run"], categories=RUN_ORDER, ordered=True)
    synth["star"] = pd.Categorical(synth["star"], categories=STAR_ORDER, ordered=True)

    return synth.sort_values(["star", "run", "fscale"]).reset_index(drop=True)


def print_column_availability(synth):
    print("\nColumn availability check:")
    for col in ["CH4_column", "CO2_column", "CH4_CO2", "haze_proxy", "redox_proxy"]:
        if col not in synth.columns:
            print(f"{col}: MISSING")
            continue
        valid = positive_finite(synth[col])
        print(f"{col}: {len(valid)} positive finite values / {len(synth[col])} rows")


def summarise_by_star_run_fscale(df, metric):
    return (
        df.groupby(["star", "run", "fscale"], as_index=False, observed=False)
        .agg(
            mean_value=(metric, "mean"),
            std_value=(metric, "std"),
            median_value=(metric, "median"),
            n=(metric, "count"),
        )
        .assign(std_value=lambda x: x["std_value"].fillna(0.0))
    )


def save_tables(synth, out_dir):
    table_dir = os.path.join(out_dir, "tables")
    ensure_dir(table_dir)

    synth_csv = os.path.join(table_dir, "haze_synthesis.csv")
    synth.to_csv(synth_csv, index=False)
    print(f"Saved {synth_csv}")

    for metric, _ in SYNTHESIS_METRICS:
        summary = summarise_by_star_run_fscale(synth, metric)
        out_csv = os.path.join(table_dir, f"haze_summary_{sanitize(metric)}.csv")
        summary.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")


# --------------------------------------------------
# PLOTTING HELPERS
# --------------------------------------------------
def plot_metric_vs_fscale(ax, star_df, metric, label):
    for run in RUN_ORDER:
        sub = star_df[star_df["run"] == run]
        if sub.empty:
            continue

        summary = (
            sub.groupby("fscale", as_index=False)
            .agg(mean_value=(metric, "mean"), std_value=(metric, "std"))
            .fillna(0.0)
            .sort_values("fscale")
        )

        x = summary["fscale"].to_numpy(dtype=float)
        y = summary["mean_value"].to_numpy(dtype=float)
        yerr = summary["std_value"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (y > 0)

        if not np.any(mask):
            continue

        ax.plot(
            x[mask],
            y[mask],
            linestyle=RUN_STYLES.get(run, "-"),
            marker=RUN_MARKERS.get(run, "o"),
            markersize=4.5,
            linewidth=1.7,
            color="black",
            label=RUN_LABELS.get(run, run),
        )

        lower = np.clip(y - yerr, 1e-300, None)
        upper = y + yerr
        band_mask = mask & np.isfinite(lower) & np.isfinite(upper)
        if np.any(band_mask):
            ax.fill_between(
                x[band_mask],
                lower[band_mask],
                upper[band_mask],
                color="black",
                alpha=0.10,
                linewidth=0,
            )

    ax.set_xlabel("FSCALE")
    ax.set_ylabel(label)
    ax.set_xticks(FSCALE_ORDER)
    apply_log_if_positive(ax, star_df[metric])
    style_axis(ax)


def plot_arney_panel(ax, star_df, star):
    plotted = False

    for run in RUN_ORDER:
        sub_run = star_df[star_df["run"] == run]
        if sub_run.empty:
            continue

        for fscale in sorted(sub_run["fscale"].dropna().unique()):
            sub = sub_run[np.isclose(sub_run["fscale"], fscale)]
            x = pd.to_numeric(sub["CH4_column"], errors="coerce")
            y = pd.to_numeric(sub["haze_proxy"], errors="coerce")
            mask = finite_positive_xy(x, y)

            if not np.any(mask):
                continue

            ax.scatter(
                x[mask],
                y[mask],
                marker=FSCALE_MARKERS.get(round(float(fscale), 2), "o"),
                s=62,
                color=STAR_COLORS.get(star, "tab:blue"),
                edgecolors="black",
                linewidths=0.5,
                alpha=0.90,
            )
            plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No positive CH4/haze pairs", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"CH$_4$ column proxy")
    ax.set_ylabel("Haze proxy")
    style_axis(ax)


def add_star_figure_legend(fig):
    run_leg = fig.legend(
        handles=run_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.50, 0.045),
        ncol=3,
        frameon=False,
        title="Stellar age case",
        title_fontsize=8,
    )

    fscale_leg = fig.legend(
        handles=fscale_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.50, 0.005),
        ncol=3,
        frameon=False,
        title="Arney-style panel markers",
        title_fontsize=8,
    )
    fig.add_artist(run_leg)
    fig.add_artist(fscale_leg)


# --------------------------------------------------
# FIGURES
# --------------------------------------------------
def save_global_arney_regime(df, out_dir):
    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    plotted = False
    for star in STAR_ORDER:
        sub_star = df[df["star"] == star]
        if sub_star.empty:
            continue

        for fscale in sorted(sub_star["fscale"].dropna().unique()):
            sub = sub_star[np.isclose(sub_star["fscale"], fscale)]
            x = pd.to_numeric(sub["CH4_column"], errors="coerce")
            y = pd.to_numeric(sub["haze_proxy"], errors="coerce")
            mask = finite_positive_xy(x, y)

            if not np.any(mask):
                continue

            ax.scatter(
                x[mask],
                y[mask],
                color=STAR_COLORS.get(star),
                marker=FSCALE_MARKERS.get(round(float(fscale), 2), "o"),
                s=70,
                alpha=0.90,
                edgecolors="black",
                linewidths=0.45,
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        print("Skipped global Arney regime diagram: no positive finite CH4/haze pairs.")
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"CH$_4$ column proxy")
    ax.set_ylabel("Haze proxy (AERSOL column proxy)")
    ax.set_title("Archean haze regime synthesis")
    style_axis(ax)

    star_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=STAR_COLORS[star],
            markeredgecolor="black",
            label=display_star(star),
        )
        for star in STAR_ORDER
    ]

    leg1 = ax.legend(
        handles=star_handles,
        title="Host star",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=fscale_legend_handles(),
        title="Flux scale",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.45),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    save_fig(fig, os.path.join(out_dir, "figures", "publication_arney_regime_all_stars.png"))


def save_star_multipane(df, star, out_dir):
    star_df = df[df["star"] == star].copy()
    if star_df.empty:
        print(f"Skipped {star}: no rows.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.8))
    axes = axes.ravel()

    plot_metric_vs_fscale(axes[0], star_df, "CH4_column", r"CH$_4$ column proxy")
    axes[0].set_title("a) Methane column")

    plot_metric_vs_fscale(axes[1], star_df, "haze_proxy", "Haze proxy")
    axes[1].set_title("b) Aerosol/haze column")

    plot_metric_vs_fscale(axes[2], star_df, "redox_proxy", "Redox proxy")
    axes[2].set_title("c) Redox state proxy")

    plot_arney_panel(axes[3], star_df, star)
    axes[3].set_title("d) Arney-style haze regime")

    fig.suptitle(f"{display_star(star)}: haze synthesis across stellar flux scaling", y=0.985)
    fig.text(
        0.5,
        0.075,
        r"Line panels show means across matching cases; shaded regions show $\pm 1\sigma$ where repeats are available.",
        ha="center",
        fontsize=8,
    )
    add_star_figure_legend(fig)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])

    out_png = os.path.join(
        out_dir,
        "figures",
        "by_star",
        f"publication_{sanitize(star)}_haze_synthesis_multipane.png",
    )
    save_fig(fig, out_png)


def save_all_star_multipanes(df, out_dir):
    for star in STAR_ORDER:
        save_star_multipane(df, star, out_dir)


# --------------------------------------------------
# SUMMARY TEXT
# --------------------------------------------------
def build_summary_text(df):
    lines = []
    lines.append("Archean haze synthesis summary")
    lines.append("==============================")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Haze proxy uses AERSOL column_proxy only.")
    lines.append("- The Arney-style regime panel uses CH4 column proxy against haze proxy.")
    lines.append("- CH4/CO2 is retained in the table where available, but CH4 is used for plotting because CO2 can be unavailable or zero.")
    lines.append("- Redox proxy = (H2 + CH4) / (CO2 + O2), using available species column proxies.")
    lines.append("- Values are proxy diagnostics, not optical-depth retrievals.")
    lines.append("")

    for star in STAR_ORDER:
        sub = df[df["star"] == star]
        if sub.empty:
            continue

        lines.append(display_star(star))
        lines.append("-" * len(display_star(star)))
        for fscale in sorted(sub["fscale"].dropna().unique()):
            fsub = sub[np.isclose(sub["fscale"], fscale)]
            lines.append(
                f"FSCALE={fscale:.2f}: "
                f"median CH4={safe_nanmedian(fsub['CH4_column']):.3e}, "
                f"median haze={safe_nanmedian(fsub['haze_proxy']):.3e}, "
                f"median redox={safe_nanmedian(fsub['redox_proxy']):.3e}"
            )
        lines.append("")

    return "\n".join(lines)


def save_summary_text(df, out_dir):
    path = os.path.join(out_dir, "haze_summary.txt")
    with open(path, "w") as f:
        f.write(build_summary_text(df))
    print(f"Saved {path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "tables"))
    ensure_dir(os.path.join(OUT_DIR, "figures"))
    ensure_dir(os.path.join(OUT_DIR, "figures", "by_star"))

    if not os.path.exists(SPECIES_FILE):
        raise FileNotFoundError(f"Missing file: {SPECIES_FILE}")

    if not os.path.exists(AEROSOL_FILE):
        raise FileNotFoundError(f"Missing file: {AEROSOL_FILE}")

    print("Loading species metrics...")
    species_wide = load_species_wide(SPECIES_FILE)

    print("Loading aerosol metrics...")
    aerosol_proxy = load_aerosol_proxy(AEROSOL_FILE)

    print("Building synthesis table...")
    synth = build_synthesis(species_wide, aerosol_proxy)
    print_column_availability(synth)

    print("Saving clean tables...")
    save_tables(synth, OUT_DIR)

    print("Saving global Arney-style regime diagram...")
    save_global_arney_regime(synth, OUT_DIR)

    print("Saving coherent per-star multipane figures...")
    save_all_star_multipanes(synth, OUT_DIR)

    save_summary_text(synth, OUT_DIR)

    print("\nDone.")
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
