#!/usr/bin/env python3
"""
Create an improved "Figure C: Altitude of Changes" plot.

This script reads species_difference_highlights.csv and creates a scatter plot of:
    x = maximum |Δ log10 mixing ratio|
    y = altitude of maximum change

Each species is colour coded and a legend is added.

Usage:
    python plot_species_altitude_changes.py

Optional:
    python plot_species_altitude_changes.py \
        --input species_difference_highlights.csv \
        --outdir figures
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


SPECIES_COLORS = {
    "H2O": "#1f77b4",
    "CH4": "#2ca02c",
    "CO":  "#8c564b",
    "CO2": "#d62728",
    "O2":  "#7f7f7f",
    "O3":  "#9467bd",
    "H2":  "#ff7f0e",
}

SPECIES_ORDER = ["H2O", "CH4", "CO", "CO2", "O2", "O3", "H2"]


def find_column(df, candidates):
    """Find a column using case-insensitive candidate names."""
    lowered = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def normalise_altitude_km(values):
    """
    Return altitude in km.

    If values look unrealistically large, assume they are in metres and convert to km.
    """
    alt = pd.to_numeric(values, errors="coerce")

    # Conservative auto-detection:
    # atmospheric profiles should usually be <= a few hundred km.
    # If max is > 1e5, assume metres.
    if alt.max(skipna=True) > 1e5:
        alt = alt / 1000.0

    return alt


def make_plot(input_csv, outdir):
    input_csv = Path(input_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    species_col = find_column(df, ["species"])
    delta_col = find_column(
        df,
        [
            "max_abs_delta_log10",
            "max_delta_log10",
            "delta_log10",
            "delta",
        ],
    )
    alt_col = find_column(
        df,
        [
            "altitude_of_max_abs_delta_km",
            "altitude_of_max_difference_km",
            "alt_km",
            "altitude_km",
            "altitude",
        ],
    )
    star_col = find_column(df, ["star"])
    scenario_col = find_column(df, ["scenario"])
    run_col = find_column(df, ["run"])
    fscale_col = find_column(df, ["fscale", "scale"])

    missing = [
        name
        for name, col in {
            "species": species_col,
            "delta metric": delta_col,
            "altitude": alt_col,
        }.items()
        if col is None
    ]
    if missing:
        raise ValueError(
            f"Could not find required column(s): {', '.join(missing)}. "
            f"Columns found: {list(df.columns)}"
        )

    df = df.copy()
    df["_delta"] = pd.to_numeric(df[delta_col], errors="coerce").abs()
    df["_alt_km"] = normalise_altitude_km(df[alt_col])
    df = df.dropna(subset=[species_col, "_delta", "_alt_km"])

    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    plotted_any = False
    species_to_plot = [sp for sp in SPECIES_ORDER if sp in set(df[species_col].astype(str))]
    remaining = [
        sp for sp in sorted(set(df[species_col].astype(str)))
        if sp not in species_to_plot
    ]
    species_to_plot += remaining

    for sp in species_to_plot:
        sdf = df[df[species_col].astype(str) == sp]
        if sdf.empty:
            continue

        ax.scatter(
            sdf["_delta"],
            sdf["_alt_km"],
            s=52,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.35,
            label=sp,
            color=SPECIES_COLORS.get(sp),
        )
        plotted_any = True

    if not plotted_any:
        raise ValueError("No valid data points found after cleaning.")

    ax.set_xlabel(r"Maximum $|\Delta \log_{10}$ mixing ratio|")
    ax.set_ylabel("Altitude of maximum change (km)")
    ax.set_title("Altitude of Maximum Chemical Change by Species")

    ax.grid(True, alpha=0.25)
    ax.legend(title="Species", frameon=False, loc="best")

    # Optional: mark upper atmosphere region
    ax.axhspan(60, 100, alpha=0.08)
    ax.text(
        0.98,
        0.96,
        "Upper atmosphere",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    fig.tight_layout()

    png_path = outdir / "figure_c_altitude_changes_by_species.png"
    pdf_path = outdir / "figure_c_altitude_changes_by_species.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # Also save a compact table of the strongest species responses.
    summary_cols = [species_col, "_delta", "_alt_km"]
    for optional_col in [star_col, scenario_col, run_col, fscale_col]:
        if optional_col is not None:
            summary_cols.insert(-2, optional_col)

    summary = (
        df[summary_cols]
        .sort_values("_delta", ascending=False)
        .rename(
            columns={
                species_col: "species",
                "_delta": "max_abs_delta_log10",
                "_alt_km": "altitude_km",
            }
        )
    )
    summary_path = outdir / "figure_c_altitude_changes_ranked_points.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create colour-coded Figure C altitude-of-change scatter plot."
    )
    parser.add_argument(
        "--input",
        default="species_difference_highlights.csv",
        help="Input CSV file. Default: species_difference_highlights.csv",
    )
    parser.add_argument(
        "--outdir",
        default="figure_c_outputs",
        help="Output directory. Default: figure_c_outputs",
    )
    args = parser.parse_args()

    make_plot(args.input, args.outdir)


if __name__ == "__main__":
    main()
