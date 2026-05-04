# haze_syth_pub.py

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
OUT_DIR = os.path.join(BASE, "haze_synthesis_pub_updated")

RUN_ORDER = ["Run_1", "Run_2", "Run_3"]

RUN_LABELS = {
    "Run_1": "Young star",
    "Run_2": "Intermediate age",
    "Run_3": "Evolved star",
}

STAR_ORDER = ["Epsilon_Eri", "HD40307", "HD85512", "HD97658"]

STAR_COLORS = {
    "Epsilon_Eri": "tab:blue",
    "HD40307": "tab:orange",
    "HD85512": "tab:green",
    "HD97658": "tab:red",
}

FSCALE_MARKERS = {
    0.75: "o",
    1.00: "s",
    1.50: "^",
}

RUN_STYLES = {
    "Run_1": "-",
    "Run_2": "--",
    "Run_3": ":",
}

HAZE_SPECIES = {"AERSOL"}
SPECIES_OF_INTEREST = ["H2O", "CH4", "CO", "CO2", "O2", "O3", "H2"]

SAVE_DPI = 300


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize(value):
    return str(value).replace(" ", "_").replace("/", "_").replace("\\", "_")


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


# --------------------------------------------------
# LOAD + RESHAPE
# --------------------------------------------------
def load_species_wide(path):
    df = pd.read_csv(path)

    required = [
        "run", "run_label", "star", "scenario",
        "fscale", "scale_dir", "species", "column_proxy"
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
        "fscale", "scale_dir", "species", "column_proxy"
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

    summed = (
        df.groupby(case_cols, as_index=False)["column_proxy"]
        .sum()
        .rename(columns={"column_proxy": "haze_proxy"})
    )

    return summed


# --------------------------------------------------
# SYNTHESIS
# --------------------------------------------------
def build_synthesis(species_wide, aerosol_sum):
    merge_cols = ["run", "run_label", "star", "scenario", "fscale", "scale_dir"]

    synth = pd.merge(
        species_wide,
        aerosol_sum,
        on=merge_cols,
        how="inner",
    )

    if synth.empty:
        raise ValueError("Merged synthesis table is empty.")

    synth["CH4_CO2"] = safe_ratio(
        synth["CH4_column"],
        synth["CO2_column"],
    )

    synth["redox_proxy"] = safe_ratio(
        synth["H2_column"].fillna(0) + synth["CH4_column"].fillna(0),
        synth["CO2_column"].fillna(0) + synth["O2_column"].fillna(0),
    )

    synth["haze_regime"] = synth["haze_proxy"].apply(classify_haze)
    synth["redox_regime"] = synth["redox_proxy"].apply(classify_redox)

    synth["fscale"] = pd.to_numeric(synth["fscale"], errors="coerce")
    synth["scenario"] = pd.to_numeric(synth["scenario"], errors="coerce")

    return synth


def print_column_availability(synth):
    print("\nColumn availability check:")

    for col in [
        "CH4_column",
        "CO2_column",
        "CH4_CO2",
        "haze_proxy",
        "redox_proxy",
    ]:
        if col not in synth.columns:
            print(f"{col}: MISSING")
            continue

        valid = positive_finite(synth[col])
        total = len(synth[col])
        print(f"{col}: {len(valid)} positive finite values / {total} rows")


def summarise_by_star_run_fscale(df, metric):
    return (
        df.groupby(["star", "run", "fscale"], as_index=False)
        .agg(
            mean_value=(metric, "mean"),
            std_value=(metric, "std"),
            median_value=(metric, "median"),
            n=(metric, "count"),
        )
        .assign(std_value=lambda x: x["std_value"].fillna(0.0))
    )


def save_tables(synth, out_dir):
    ensure_dir(out_dir)

    synth_csv = os.path.join(out_dir, "haze_synthesis.csv")
    synth.to_csv(synth_csv, index=False)

    for metric in ["CH4_column", "haze_proxy", "redox_proxy"]:
        summary = summarise_by_star_run_fscale(synth, metric)
        summary.to_csv(
            os.path.join(out_dir, f"haze_summary_{sanitize(metric)}.csv"),
            index=False,
        )

    print(f"Saved {synth_csv}")


# --------------------------------------------------
# FIGURE STYLE
# --------------------------------------------------
def style_axis(ax):
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.tick_params(direction="in", top=True, right=True)


def add_split_legend(ax):
    star_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=STAR_COLORS[star],
            markeredgecolor="black",
            label=star.replace("_", " "),
        )
        for star in STAR_ORDER
    ]

    fscale_handles = [
        Line2D(
            [0], [0],
            marker=marker,
            linestyle="",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="black",
            label=f"FSCALE={fscale:.2f}",
        )
        for fscale, marker in FSCALE_MARKERS.items()
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
        handles=fscale_handles,
        title="Flux scale",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.45),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )


# --------------------------------------------------
# FIGURES
# --------------------------------------------------
def save_regime_diagram(df, out_dir):
    """
    CH4-column vs haze proxy regime diagram.

    Uses CH4_column directly because CO2_column is unavailable/zero
    in the current species_metrics table.
    """

    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    plotted = False

    for star in STAR_ORDER:
        sub_star = df[df["star"] == star]

        if sub_star.empty:
            continue

        for fscale in sorted(sub_star["fscale"].dropna().unique()):
            sub = sub_star[np.isclose(sub_star["fscale"], fscale)]

            x = pd.to_numeric(sub["CH4_column"], errors="coerce")
            y = pd.to_numeric(sub["haze_proxy"], errors="coerce")

            mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

            if not np.any(mask):
                continue

            ax.scatter(
                x[mask],
                y[mask],
                color=STAR_COLORS.get(star),
                marker=FSCALE_MARKERS.get(round(float(fscale), 2), "o"),
                s=70,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.45,
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        print("Skipped regime diagram: no positive finite CH4_column and haze_proxy pairs.")
        return

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"CH$_4$ column proxy", fontsize=10)
    ax.set_ylabel("Haze proxy (AERSOL column proxy)", fontsize=10)
    ax.set_title("Archean haze regime synthesis", fontsize=12)

    style_axis(ax)
    add_split_legend(ax)

    fig.tight_layout(rect=[0, 0, 0.78, 1])

    fig.savefig(
        os.path.join(out_dir, "publication_haze_regime_diagram.png"),
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        os.path.join(out_dir, "publication_haze_regime_diagram.pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def save_star_comparison_updated(df, out_dir):
    metrics = [
        ("CH4_column", r"CH$_4$ column proxy"),
        ("haze_proxy", "Haze proxy"),
        ("redox_proxy", "Redox proxy"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))

    for ax, (metric, label) in zip(axes, metrics):
        x = np.arange(len(STAR_ORDER))
        width = 0.22

        for i, fscale in enumerate(sorted(df["fscale"].dropna().unique())):
            vals = []

            for star in STAR_ORDER:
                sub = df[
                    (df["star"] == star)
                    & np.isclose(df["fscale"], fscale)
                ]

                vals.append(safe_nanmedian(sub[metric]) if not sub.empty else np.nan)

            vals = np.asarray(vals, dtype=float)

            ax.bar(
                x + (i - 1) * width,
                vals,
                width=width,
                label=f"FSCALE={fscale:.2f}",
                alpha=0.88,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [s.replace("_", " ") for s in STAR_ORDER],
            rotation=25,
            ha="right",
            fontsize=8,
        )

        ax.set_title(label, fontsize=10)
        style_axis(ax)

        vals_all = positive_finite(df[metric])

        if len(vals_all) > 0:
            ax.set_yscale("log")

    axes[0].set_ylabel("Median value", fontsize=9)

    axes[-1].legend(
        fontsize=8,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Flux scale",
        title_fontsize=9,
    )

    fig.suptitle("Median star-by-star haze synthesis", y=1.02, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.88, 1])

    fig.savefig(
        os.path.join(out_dir, "publication_star_comparison.png"),
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        os.path.join(out_dir, "publication_star_comparison.pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def save_fscale_sensitivity_updated(df, out_dir):
    metrics = [
        ("CH4_column", r"CH$_4$ column proxy"),
        ("haze_proxy", "Haze proxy"),
        ("redox_proxy", "Redox proxy"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharex=True)

    for ax, (metric, label) in zip(axes, metrics):
        for star in STAR_ORDER:
            star_df = df[df["star"] == star]

            if star_df.empty:
                continue

            grouped = (
                star_df.groupby("fscale", as_index=False)
                .agg(
                    median_value=(metric, safe_nanmedian),
                )
                .sort_values("fscale")
            )

            x = grouped["fscale"].to_numpy(dtype=float)
            y = grouped["median_value"].to_numpy(dtype=float)

            mask = np.isfinite(x) & np.isfinite(y) & (y > 0)

            if not np.any(mask):
                continue

            ax.plot(
                x[mask],
                y[mask],
                marker="o",
                linewidth=1.8,
                color=STAR_COLORS.get(star),
                label=star.replace("_", " "),
            )

        ax.set_xlabel("FSCALE", fontsize=9)
        ax.set_title(label, fontsize=10)
        style_axis(ax)

        vals_all = positive_finite(df[metric])

        if len(vals_all) > 0:
            ax.set_yscale("log")

    axes[0].set_ylabel("Median value", fontsize=9)

    axes[-1].legend(
        fontsize=8,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Host star",
        title_fontsize=9,
    )

    fig.suptitle("FSCALE sensitivity of haze-related diagnostics", y=1.02, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.88, 1])

    fig.savefig(
        os.path.join(out_dir, "publication_fscale_sensitivity.png"),
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        os.path.join(out_dir, "publication_fscale_sensitivity.pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def save_multipane_interpretation(df, out_dir):
    metrics = [
        ("CH4_column", r"CH$_4$ column proxy"),
        ("haze_proxy", "Haze proxy"),
        ("redox_proxy", "Redox proxy"),
    ]

    fig, axes = plt.subplots(
        len(STAR_ORDER),
        len(metrics),
        figsize=(10.8, 9.2),
        sharex=True,
        sharey=False,
    )

    for i, star in enumerate(STAR_ORDER):
        for j, (metric, label) in enumerate(metrics):
            ax = axes[i, j]

            star_df = df[df["star"] == star]

            for run in RUN_ORDER:
                sub = star_df[star_df["run"] == run]

                if sub.empty:
                    continue

                summary = (
                    sub.groupby("fscale", as_index=False)
                    .agg(
                        mean_value=(metric, "mean"),
                        std_value=(metric, "std"),
                    )
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
                    marker="o",
                    markersize=3.5,
                    linewidth=1.5,
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
                        alpha=0.12,
                        linewidth=0,
                    )

            style_axis(ax)

            vals_all = positive_finite(star_df[metric])
            if len(vals_all) > 0:
                ax.set_yscale("log")

            if i == 0:
                ax.set_title(label, fontsize=9)

            if j == 0:
                ax.set_ylabel(star.replace("_", " "), fontsize=9)

            if i == len(STAR_ORDER) - 1:
                ax.set_xlabel("FSCALE", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
        title="Stellar age",
        title_fontsize=9,
    )

    fig.suptitle(
        "Archean haze diagnostics as a function of stellar flux scaling",
        fontsize=12,
        y=0.99,
    )

    fig.text(
        0.5,
        0.012,
        r"Points show means across repeat cases; shaded regions show $\pm 1\sigma$ variability.",
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0.035, 0.88, 0.96])

    fig.savefig(
        os.path.join(out_dir, "publication_haze_interpretation_multipane.png"),
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        os.path.join(out_dir, "publication_haze_interpretation_multipane.pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


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
    lines.append("- CH4 uses CH4 column_proxy directly.")
    lines.append("- CH4/CO2 is not used because CO2 column_proxy is unavailable/zero in this dataset.")
    lines.append("- Redox is calculated from species column_proxy values.")
    lines.append("- Values are proxy diagnostics, not optical-depth retrievals.")
    lines.append("")

    for star in STAR_ORDER:
        sub = df[df["star"] == star]

        if sub.empty:
            continue

        lines.append(star)
        lines.append("-" * len(star))

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


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    ensure_dir(OUT_DIR)

    if not os.path.exists(SPECIES_FILE):
        raise FileNotFoundError(f"Missing file: {SPECIES_FILE}")

    if not os.path.exists(AEROSOL_FILE):
        raise FileNotFoundError(f"Missing file: {AEROSOL_FILE}")

    species_wide = load_species_wide(SPECIES_FILE)
    aerosol_proxy = load_aerosol_proxy(AEROSOL_FILE)

    synth = build_synthesis(species_wide, aerosol_proxy)
    synth = synth.sort_values(["star", "run", "fscale"]).reset_index(drop=True)

    print_column_availability(synth)

    save_tables(synth, OUT_DIR)

    save_regime_diagram(synth, OUT_DIR)
    save_star_comparison_updated(synth, OUT_DIR)
    save_fscale_sensitivity_updated(synth, OUT_DIR)
    save_multipane_interpretation(synth, OUT_DIR)

    summary_txt = os.path.join(OUT_DIR, "haze_summary.txt")

    with open(summary_txt, "w") as f:
        f.write(build_summary_text(synth))

    print("\nDone.")
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
