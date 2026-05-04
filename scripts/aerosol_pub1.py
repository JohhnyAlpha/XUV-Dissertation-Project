import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_DIR = os.path.expanduser("~/atmos/BA/Plots/aerosol_analysis_pub")
TABLES_DIR = os.path.join(BASE_DIR, "tables")
OUT_DIR = os.path.join(BASE_DIR, "interpretation")
SAVE_DPI = 300
SHOW_PLOTS = False

RUN_LABELS = {
    "Run_1": "2.7 Gy star",
    "Run_2": "0.5 Gy star",
    "Run_3": "9.9 Gy star",
}
RUN_ORDER = ["Run_1", "Run_2", "Run_3"]
FSCALE_ORDER = ["0.75", "1.00", "1.50"]
STAR_ORDER = ["Epsilon_Eri", "HD40307", "HD85512", "HD97658"]

# Used only if an aerosol_metrics.csv contains a scenario column. Most aerosol
# tables use FSCALE directly, unlike the species script where scenario and FSCALE
# are independent axes.
SCENARIO_ORDER = ["1", "2", "3"]

AEROSOL_COLORS = {
    "H2SO4": "#1f77b4",
    "S8": "#ff7f0e",
    "SULFUR": "#ff7f0e",
    "SO4": "#2ca02c",
    "HAZE": "#9467bd",
    "AER": "#7f7f7f",
}

INTERPRETATION_METRICS = [
    "column_proxy",
    "peak",
    "mean_above_50km",
    "peak_alt",
    "centroid_alt",
]
ABUNDANCE_METRICS = {"column_proxy", "peak", "surface", "mean_above_20km", "mean_above_50km"}
ALTITUDE_METRICS = {"peak_alt", "centroid_alt"}

FIGSIZE_HEATMAP = (11.0, 6.5)
FIGSIZE_SUMMARY = (10.5, 7.0)
FIGSIZE_BARS = (11.0, 4.8)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize_for_filename(value):
    value = str(value)
    for old, new in [(" ", "_"), ("/", "_"), ("\\", "_"), ("=", "_"), (":", "_")]:
        value = value.replace(old, new)
    return value


def save_current_figure_both(fig, stem):
    ensure_dir(os.path.dirname(stem))
    fig.tight_layout()
    fig.savefig(stem + ".png", dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    print(f"Saved: {stem}.png")
    print(f"Saved: {stem}.pdf")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def fmt_num(x):
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(x):
        return "nan"
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax < 1e-2 or ax >= 1e3:
        return f"{x:.2e}"
    return f"{x:.3f}"


def safe_log10_ratio(num, den):
    if np.isfinite(num) and np.isfinite(den) and num > 0 and den > 0:
        return np.log10(num / den)
    return np.nan


def magnitude_phrase(delta_log10):
    if not np.isfinite(delta_log10):
        return None
    absd = abs(delta_log10)
    if absd >= 2.0:
        return "~2+ orders of magnitude"
    if absd >= 1.0:
        return "~1 order of magnitude"
    if absd >= 0.5:
        return f"~{10 ** absd:.1f}x"
    if absd >= 0.1:
        return f"~{10 ** absd:.2f}x"
    return None


def ordered_unique(values, preferred_order=None):
    values = [str(v) for v in pd.Series(values).dropna().unique()]
    if preferred_order is None:
        return sorted(values)
    ordered = [v for v in preferred_order if v in values]
    ordered.extend(sorted(v for v in values if v not in ordered))
    return ordered


def metric_delta(row, metric, ref_val, target_val):
    if metric in ALTITUDE_METRICS:
        return np.nan, target_val - ref_val if np.isfinite(target_val) and np.isfinite(ref_val) else np.nan
    return (
        safe_log10_ratio(target_val, ref_val),
        target_val - ref_val if np.isfinite(target_val) and np.isfinite(ref_val) else np.nan,
    )


def interpret_strength(metric, delta_log10, delta_linear):
    if metric in ALTITUDE_METRICS:
        if not np.isfinite(delta_linear):
            return None
        mag = abs(delta_linear)
        if mag < 2:
            return "little altitude shift"
        if mag < 5:
            return "minor altitude shift"
        if mag < 10:
            return "moderate altitude shift"
        return "large altitude shift"
    if not np.isfinite(delta_log10):
        return None
    mag = abs(delta_log10)
    if mag < 0.1:
        return "little change"
    if mag < 0.3:
        return "modest change"
    if mag < 1.0:
        return "strong change"
    return "order-of-magnitude change"


def interpret_direction(metric, delta_log10, delta_linear):
    if metric in ALTITUDE_METRICS:
        if not np.isfinite(delta_linear):
            return None
        return "upward" if delta_linear > 0 else "downward"
    if not np.isfinite(delta_log10):
        return None
    return "increase" if delta_log10 > 0 else "decrease"


def infer_physical_signal(metric, delta_log10=None, delta_linear=None):
    if metric == "column_proxy":
        if not np.isfinite(delta_log10):
            return None
        return "greater total aerosol loading" if delta_log10 > 0 else "reduced total aerosol loading"
    if metric == "peak":
        if not np.isfinite(delta_log10):
            return None
        return "a denser peak aerosol layer" if delta_log10 > 0 else "a weaker peak aerosol layer"
    if metric == "mean_above_50km":
        if not np.isfinite(delta_log10):
            return None
        return "stronger upper-atmosphere aerosol loading" if delta_log10 > 0 else "weaker upper-atmosphere aerosol loading"
    if metric == "peak_alt":
        if not np.isfinite(delta_linear):
            return None
        return "an upward shift of the peak aerosol layer" if delta_linear > 0 else "a downward shift of the peak aerosol layer"
    if metric == "centroid_alt":
        if not np.isfinite(delta_linear):
            return None
        return "a broader or lofted aerosol distribution" if delta_linear > 0 else "a more compact or lower aerosol distribution"
    return None


def classify_significance(metric, delta_log10, delta_linear):
    if metric in ALTITUDE_METRICS:
        if not np.isfinite(delta_linear):
            return "unclassified"
        mag = abs(delta_linear)
        if mag >= 10:
            return "major"
        if mag >= 5:
            return "moderate"
        if mag >= 2:
            return "minor"
        return "small"
    if not np.isfinite(delta_log10):
        return "unclassified"
    mag = abs(delta_log10)
    if mag >= 1.0:
        return "major"
    if mag >= 0.3:
        return "moderate"
    if mag >= 0.1:
        return "minor"
    return "small"


def sort_score(metric, delta_log10, delta_linear):
    if metric in ALTITUDE_METRICS:
        return abs(delta_linear) if np.isfinite(delta_linear) else np.nan
    return abs(delta_log10) if np.isfinite(delta_log10) else np.nan


def color_for_species(species):
    sp = str(species)
    key = sp.upper()
    for token, color in AEROSOL_COLORS.items():
        if token.upper() in key:
            return color
    return None


# --------------------------------------------------
# I/O
# --------------------------------------------------
def load_metrics(metrics_csv=None):
    metrics_csv = metrics_csv or os.path.join(TABLES_DIR, "aerosol_metrics.csv")
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(
            f"Could not find metrics table: {metrics_csv}\n"
            "Run aerosol_analysis_pub.py first to generate aerosol_metrics.csv."
        )
    df = pd.read_csv(metrics_csv)
    required = {"star", "run", "fscale", "species", "column_proxy", "peak", "peak_alt", "mean_above_50km", "centroid_alt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metrics table is missing required columns: {sorted(missing)}")

    for col in ["run", "fscale", "star", "species"]:
        df[col] = df[col].astype(str)
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str)
    else:
        df["scenario"] = "all"
    return df


def case_sort_categories(df):
    out = df.copy()
    if "run" in out:
        out["run"] = pd.Categorical(out["run"], categories=RUN_ORDER, ordered=True)
    if "fscale" in out:
        out["fscale"] = pd.Categorical(out["fscale"], categories=FSCALE_ORDER, ordered=True)
    if "star" in out:
        out["star"] = pd.Categorical(out["star"], categories=STAR_ORDER, ordered=True)
    if "scenario" in out:
        cats = SCENARIO_ORDER if set(out["scenario"].astype(str).unique()).issubset(set(SCENARIO_ORDER)) else ordered_unique(out["scenario"])
        out["scenario"] = pd.Categorical(out["scenario"], categories=cats, ordered=True)
    return out


# --------------------------------------------------
# COMPARISONS
# --------------------------------------------------
def compute_comparisons(metrics_df):
    rows = []
    metric_names = [
        "column_proxy", "peak", "peak_alt", "surface",
        "mean_above_20km", "mean_above_50km", "centroid_alt",
    ]
    available_metrics = [m for m in metric_names if m in metrics_df.columns]

    stars = ordered_unique(metrics_df["star"], STAR_ORDER)
    scenarios = ordered_unique(metrics_df["scenario"], SCENARIO_ORDER if "all" not in set(metrics_df["scenario"]) else None)

    for star in stars:
        for scenario in scenarios:
            star_df = metrics_df[(metrics_df["star"] == star) & (metrics_df["scenario"] == scenario)]
            if star_df.empty:
                continue

            # Across runs at fixed FSCALE, relative to Run_1.
            for fscale in FSCALE_ORDER:
                sub = star_df[star_df["fscale"] == fscale]
                if sub.empty:
                    continue
                for sp in sorted(sub["species"].dropna().unique()):
                    ref = sub[(sub["species"] == sp) & (sub["run"] == "Run_1")]
                    if ref.empty:
                        continue
                    ref_row = ref.iloc[0]
                    for _, row in sub[sub["species"] == sp].iterrows():
                        for metric in available_metrics:
                            ref_val = float(ref_row.get(metric, np.nan))
                            tar_val = float(row.get(metric, np.nan))
                            dl10, dlin = metric_delta(row, metric, ref_val, tar_val)
                            rows.append({
                                "comparison_type": "vs_Run1_same_fscale",
                                "star": star,
                                "scenario": scenario,
                                "species": sp,
                                "fscale": fscale,
                                "run": row["run"],
                                "reference": "Run_1",
                                "target": row["run"],
                                "metric": metric,
                                "reference_value": ref_val,
                                "target_value": tar_val,
                                "delta_log10": dl10,
                                "delta_linear": dlin,
                            })

            # Across FSCALE at fixed run, relative to FSCALE = 1.00.
            for run in RUN_ORDER:
                sub = star_df[star_df["run"] == run]
                if sub.empty:
                    continue
                for sp in sorted(sub["species"].dropna().unique()):
                    ref = sub[(sub["species"] == sp) & (sub["fscale"] == "1.00")]
                    if ref.empty:
                        continue
                    ref_row = ref.iloc[0]
                    for _, row in sub[sub["species"] == sp].iterrows():
                        for metric in available_metrics:
                            ref_val = float(ref_row.get(metric, np.nan))
                            tar_val = float(row.get(metric, np.nan))
                            dl10, dlin = metric_delta(row, metric, ref_val, tar_val)
                            rows.append({
                                "comparison_type": "vs_Fscale1.00_same_run",
                                "star": star,
                                "scenario": scenario,
                                "species": sp,
                                "fscale": row["fscale"],
                                "run": run,
                                "reference": "1.00",
                                "target": row["fscale"],
                                "metric": metric,
                                "reference_value": ref_val,
                                "target_value": tar_val,
                                "delta_log10": dl10,
                                "delta_linear": dlin,
                            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = case_sort_categories(out)
    return out.sort_values(["comparison_type", "star", "scenario", "species", "run", "fscale", "metric"]).reset_index(drop=True)


def build_significance_table(comparisons_df):
    if comparisons_df.empty:
        return comparisons_df.copy()
    out = comparisons_df.copy()
    out["abs_delta_log10"] = np.abs(out["delta_log10"])
    out["abs_delta_linear"] = np.abs(out["delta_linear"])
    out["strength"] = [interpret_strength(m, dl10, dlin) for m, dl10, dlin in zip(out["metric"], out["delta_log10"], out["delta_linear"])]
    out["direction"] = [interpret_direction(m, dl10, dlin) for m, dl10, dlin in zip(out["metric"], out["delta_log10"], out["delta_linear"])]
    out["physical_signal"] = [infer_physical_signal(m, dl10, dlin) for m, dl10, dlin in zip(out["metric"], out["delta_log10"], out["delta_linear"])]
    out["significance"] = [classify_significance(m, dl10, dlin) for m, dl10, dlin in zip(out["metric"], out["delta_log10"], out["delta_linear"])]
    out["sort_score"] = [sort_score(m, dl10, dlin) for m, dl10, dlin in zip(out["metric"], out["delta_log10"], out["delta_linear"])]
    return out


# --------------------------------------------------
# TEXT SUMMARIES
# --------------------------------------------------
def build_text_summaries(sig_df):
    rows = []
    if sig_df.empty:
        return pd.DataFrame(rows)

    focus = sig_df[
        sig_df["metric"].isin(INTERPRETATION_METRICS)
        & sig_df["significance"].isin(["major", "moderate"])
    ].copy()
    if focus.empty:
        return pd.DataFrame(rows)

    for star in ordered_unique(focus["star"], STAR_ORDER):
        star_df = focus[focus["star"] == star].copy()
        lines = []

        run_comp = star_df[
            (star_df["comparison_type"] == "vs_Run1_same_fscale")
            & (star_df["run"].isin(["Run_2", "Run_3"]))
        ].sort_values("sort_score", ascending=False, na_position="last")
        for _, r in run_comp.head(5).iterrows():
            if r["metric"] in ALTITUDE_METRICS:
                detail = f"{abs(float(r['delta_linear'])):.1f} km {r['direction']}"
            else:
                phrase = magnitude_phrase(float(r["delta_log10"])) or f"Δlog10={float(r['delta_log10']):.2f}"
                detail = f"{phrase} {'higher' if r['delta_log10'] > 0 else 'lower'}"
            line = (
                f"At FSCALE={r['fscale']}, {r['species']} {r['metric']} is {detail} in "
                f"{RUN_LABELS.get(str(r['run']), r['run'])} relative to {RUN_LABELS['Run_1']}."
            )
            if isinstance(r.get("physical_signal"), str):
                line += f" This suggests {r['physical_signal']}."
            if line not in lines:
                lines.append(line)

        fs_comp = star_df[
            (star_df["comparison_type"] == "vs_Fscale1.00_same_run")
            & (star_df["fscale"].isin(["0.75", "1.50"]))
        ].sort_values("sort_score", ascending=False, na_position="last")
        for _, r in fs_comp.head(5).iterrows():
            if r["metric"] in ALTITUDE_METRICS:
                detail = f"{abs(float(r['delta_linear'])):.1f} km {r['direction']}"
            else:
                phrase = magnitude_phrase(float(r["delta_log10"])) or f"Δlog10={float(r['delta_log10']):.2f}"
                detail = f"{phrase} {'higher' if r['delta_log10'] > 0 else 'lower'}"
            line = (
                f"Within {RUN_LABELS.get(str(r['run']), r['run'])}, {r['species']} {r['metric']} is {detail} "
                f"at FSCALE={r['fscale']} relative to FSCALE=1.00."
            )
            if isinstance(r.get("physical_signal"), str):
                line += f" This suggests {r['physical_signal']}."
            if line not in lines:
                lines.append(line)

        combo = star_df.groupby("species", observed=False)["sort_score"].max().sort_values(ascending=False)
        if not combo.empty:
            top_species = ", ".join(combo.head(3).index.astype(str).tolist())
            lines.insert(0, f"The aerosol species showing the strongest overall sensitivity are: {top_species}.")

        rows.append({"star": star, "summary_text": " ".join(lines) if lines else "No large aerosol differences exceeded the automatic reporting thresholds."})
    return pd.DataFrame(rows)


def write_text_summary_file(summary_df, sig_df, out_path):
    lines = [
        "Aerosol comparison summary",
        "==========================",
        "",
        "Run mapping: Run_1 = 2.7 Gy star; Run_2 = 0.5 Gy star; Run_3 = 9.9 Gy star",
        "Reference cases: run comparisons use Run_1 at fixed FSCALE; FSCALE comparisons use FSCALE=1.00 at fixed run.",
        "Altitude metrics are reported as linear km shifts, not log-ratios.",
        "",
    ]
    if summary_df.empty:
        lines.append("No interpretable differences were identified above the selected thresholds.")
        lines.append("")
    else:
        for _, row in summary_df.iterrows():
            lines.append(str(row["star"]))
            lines.append("-" * len(str(row["star"])))
            lines.append(str(row["summary_text"]).strip())
            lines.append("")

    lines.extend(["Most significant individual changes", "-------------------------------"])
    focus = sig_df[
        sig_df["metric"].isin(INTERPRETATION_METRICS)
        & sig_df["significance"].isin(["major", "moderate"])
    ].copy().sort_values("sort_score", ascending=False, na_position="last")
    for _, row in focus.head(50).iterrows():
        if row["comparison_type"] == "vs_Run1_same_fscale":
            context = f"{row['star']} | {row['species']} | FSCALE={row['fscale']} | {row['target']} vs {row['reference']}"
        else:
            context = f"{row['star']} | {row['species']} | {row['run']} | FSCALE={row['target']} vs {row['reference']}"
        if row["metric"] in ALTITUDE_METRICS:
            delta_text = f"Δz={fmt_num(row['delta_linear'])} km"
        else:
            delta_text = f"Δlog10={fmt_num(row['delta_log10'])}; factor={fmt_num(10 ** row['delta_log10']) if np.isfinite(row['delta_log10']) else 'nan'}"
        extra = f" | signal={row['physical_signal']}" if isinstance(row.get("physical_signal"), str) else ""
        lines.append(f"- {context}: {row['metric']} | {delta_text} | {row['significance']}{extra}")

    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


# --------------------------------------------------
# PLOTS
# --------------------------------------------------
def heatmap_matrix(sig_df, star, comparison_type, metric):
    sub = sig_df[(sig_df["star"] == star) & (sig_df["comparison_type"] == comparison_type) & (sig_df["metric"] == metric)].copy()
    if sub.empty:
        return None
    value_col = "delta_linear" if metric in ALTITUDE_METRICS else "delta_log10"
    if comparison_type == "vs_Run1_same_fscale":
        sub["comparison"] = sub["fscale"].astype(str) + "\n" + sub["run"].astype(str)
        desired = [f"{f}\n{r}" for f in FSCALE_ORDER for r in RUN_ORDER]
    else:
        sub["comparison"] = sub["run"].astype(str) + "\n" + sub["fscale"].astype(str)
        desired = [f"{r}\n{f}" for r in RUN_ORDER for f in FSCALE_ORDER]
    piv = sub.pivot_table(index="species", columns="comparison", values=value_col, aggfunc="first")
    species_order = sorted(piv.index.astype(str).tolist())
    existing = [c for c in desired if c in piv.columns]
    return piv.reindex(index=species_order, columns=existing)


def draw_heatmap(ax, mat, metric, title):
    if mat is None or mat.empty:
        ax.axis("off")
        return None
    vals = mat.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(vals)
    if masked.count() == 0:
        ax.axis("off")
        return None
    if metric in ALTITUDE_METRICS:
        im = ax.imshow(masked, aspect="auto")
        cbar_label = "Δ altitude (km)"
    else:
        im = ax.imshow(masked, aspect="auto", vmin=-2, vmax=2)
        cbar_label = "Δ log10(metric)"
    ax.set_title(title)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_ylabel("Aerosol species")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6)
    return im, cbar_label


def plot_heatmaps(sig_df, out_dir):
    if sig_df.empty:
        return
    ensure_dir(out_dir)
    pdf_path = os.path.join(out_dir, "aerosol_interpretation_heatmaps.pdf")
    metrics = ["column_proxy", "peak", "mean_above_50km", "centroid_alt"]
    with PdfPages(pdf_path) as pdf:
        for star in ordered_unique(sig_df["star"], STAR_ORDER):
            fig, axes = plt.subplots(len(metrics), 2, figsize=(12.5, 13.5))
            for i, metric in enumerate(metrics):
                for j, (ctype, label) in enumerate([
                    ("vs_Run1_same_fscale", "run effect vs Run_1"),
                    ("vs_Fscale1.00_same_run", "FSCALE effect vs 1.00"),
                ]):
                    ax = axes[i, j]
                    mat = heatmap_matrix(sig_df, star, ctype, metric)
                    drawn = draw_heatmap(ax, mat, metric, f"{metric} | {label}")
                    if drawn is not None:
                        im, cbar_label = drawn
                        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label(cbar_label)
            fig.suptitle(f"{star} | aerosol interpretation heatmaps", y=0.995)
            fig.tight_layout(rect=[0, 0, 1, 0.985])
            stem = os.path.join(out_dir, f"{sanitize_for_filename(star)}_aerosol_interpretation_heatmaps")
            fig.savefig(stem + ".png", dpi=SAVE_DPI, bbox_inches="tight")
            fig.savefig(stem + ".pdf", bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            print(f"Saved: {stem}.png")
            print(f"Saved: {stem}.pdf")
            plt.close(fig)
    print(f"Saved: {pdf_path}")


def plot_significance_bars(sig_df, out_dir):
    ensure_dir(out_dir)
    if sig_df.empty:
        return
    for star in ordered_unique(sig_df["star"], STAR_ORDER):
        sub = sig_df[
            (sig_df["star"] == star)
            & (sig_df["comparison_type"] == "vs_Fscale1.00_same_run")
            & (sig_df["metric"].isin(["column_proxy", "peak", "mean_above_50km"]))
        ].copy()
        if sub.empty:
            continue
        grouped = sub.groupby(["species", "run"], observed=False)["abs_delta_log10"].max().reset_index()
        species = grouped.groupby("species", observed=False)["abs_delta_log10"].max().sort_values(ascending=False).index.astype(str).tolist()
        x = np.arange(len(species))
        width = 0.25
        fig, ax = plt.subplots(figsize=FIGSIZE_BARS)
        for i, run in enumerate(RUN_ORDER):
            vals = []
            for sp in species:
                row = grouped[(grouped["species"].astype(str) == sp) & (grouped["run"].astype(str) == run)]
                vals.append(float(row["abs_delta_log10"].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (i - 1) * width, vals, width=width, label=RUN_LABELS.get(run, run))
        ax.set_xticks(x)
        ax.set_xticklabels(species, rotation=35, ha="right")
        ax.set_ylabel(r"max |Δlog$_{10}$| vs FSCALE=1.00")
        ax.set_title(f"{star} | strongest FSCALE sensitivity by aerosol species")
        ax.legend(frameon=False)
        ax.grid(True, axis="y", alpha=0.25)
        save_current_figure_both(fig, os.path.join(out_dir, f"fscale_sensitivity_{sanitize_for_filename(star)}"))


def plot_metric_scatter(metrics_df, out_dir):
    ensure_dir(out_dir)
    available = [m for m in ["column_proxy", "peak", "mean_above_50km", "centroid_alt"] if m in metrics_df.columns]
    if not available:
        return
    for star in ordered_unique(metrics_df["star"], STAR_ORDER):
        sdf = metrics_df[metrics_df["star"] == star]
        if sdf.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_SUMMARY)
        axes = axes.ravel()
        for ax in axes:
            ax.axis("off")
        for ax, metric in zip(axes, available):
            ax.axis("on")
            for sp in sorted(sdf["species"].dropna().unique()):
                spdf = sdf[sdf["species"] == sp]
                for run, marker in zip(RUN_ORDER, ["o", "s", "^"]):
                    g = spdf[spdf["run"] == run].copy().sort_values("fscale")
                    if g.empty:
                        continue
                    xf = pd.to_numeric(g["fscale"], errors="coerce").to_numpy()
                    y = pd.to_numeric(g[metric], errors="coerce").to_numpy()
                    ax.plot(xf, y, marker=marker, linewidth=1.2, alpha=0.85, color=color_for_species(sp), label=f"{sp} | {run}")
            if metric not in ALTITUDE_METRICS:
                yall = pd.to_numeric(sdf[metric], errors="coerce").to_numpy()
                ypos = yall[np.isfinite(yall) & (yall > 0)]
                if ypos.size > 1 and np.nanmax(ypos) / np.nanmin(ypos) > 100:
                    ax.set_yscale("log")
            ax.set_title(metric.replace("_", " "))
            ax.set_xlabel("FSCALE")
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_xticks([float(f) for f in FSCALE_ORDER])
            ax.grid(True, alpha=0.25)
        fig.suptitle(f"{star} | aerosol metrics vs FSCALE", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_current_figure_both(fig, os.path.join(out_dir, f"{sanitize_for_filename(star)}_aerosol_metric_scatter"))


# --------------------------------------------------
# TABLES
# --------------------------------------------------
def save_simple_latex(df, out_path, caption, label):
    ensure_dir(os.path.dirname(out_path))
    if df.empty:
        return
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_float_dtype(safe[col]) or pd.api.types.is_integer_dtype(safe[col]):
            safe[col] = safe[col].apply(fmt_num)
    latex = safe.to_latex(index=False, escape=True, caption=caption, label=label)
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"Saved: {out_path}")


def save_latex_metric_tables(metrics_df, out_dir):
    ensure_dir(out_dir)
    cols = [c for c in ["run", "fscale", "species", "column_proxy", "peak", "peak_alt", "mean_above_50km", "centroid_alt"] if c in metrics_df.columns]
    for star in ordered_unique(metrics_df["star"], STAR_ORDER):
        sdf = metrics_df[metrics_df["star"] == star][cols].copy()
        if sdf.empty:
            continue
        save_simple_latex(
            sdf,
            os.path.join(out_dir, f"{sanitize_for_filename(star)}_aerosol_metrics.tex"),
            caption=f"Aerosol metrics for {star.replace('_', ' ')}.",
            label=f"tab:{sanitize_for_filename(star).lower()}_aerosol_metrics",
        )


def build_case_index(metrics_df):
    cols = [c for c in ["star", "scenario", "run", "fscale", "species"] if c in metrics_df.columns]
    return metrics_df[cols].drop_duplicates().sort_values(cols).reset_index(drop=True)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main(metrics_csv=None, out_dir=OUT_DIR):
    ensure_dir(out_dir)
    for sub in ["tables", "figures", "figures/heatmaps", "figures/bars", "figures/scatter", "latex_tables"]:
        ensure_dir(os.path.join(out_dir, sub))

    metrics_df = load_metrics(metrics_csv)
    metrics_df = case_sort_categories(metrics_df).sort_values(["star", "scenario", "species", "run", "fscale"]).reset_index(drop=True)

    metrics_out = os.path.join(out_dir, "tables", "aerosol_metrics_used.csv")
    metrics_df.to_csv(metrics_out, index=False)
    print(f"Saved: {metrics_out}")

    index_df = build_case_index(metrics_df)
    index_path = os.path.join(out_dir, "tables", "aerosol_case_index.csv")
    index_df.to_csv(index_path, index=False)
    print(f"Saved: {index_path}")

    comparisons_df = compute_comparisons(metrics_df)
    if comparisons_df.empty:
        print("No comparisons could be generated from aerosol_metrics.csv")
        return

    sig_df = build_significance_table(comparisons_df)

    comparisons_csv = os.path.join(out_dir, "tables", "aerosol_comparisons.csv")
    sig_csv = os.path.join(out_dir, "tables", "aerosol_significance.csv")
    comparisons_df.to_csv(comparisons_csv, index=False)
    sig_df.to_csv(sig_csv, index=False)
    print(f"Saved: {comparisons_csv}")
    print(f"Saved: {sig_csv}")

    summaries_df = build_text_summaries(sig_df)
    summaries_csv = os.path.join(out_dir, "tables", "aerosol_text_summaries.csv")
    summaries_df.to_csv(summaries_csv, index=False)
    print(f"Saved: {summaries_csv}")
    write_text_summary_file(summaries_df, sig_df, os.path.join(out_dir, "aerosol_results_summary.txt"))

    top_changes = sig_df[
        sig_df["metric"].isin(INTERPRETATION_METRICS)
        & sig_df["significance"].isin(["major", "moderate"])
    ].copy().sort_values("sort_score", ascending=False, na_position="last").head(40)
    top_csv = os.path.join(out_dir, "tables", "aerosol_top_changes.csv")
    top_changes.to_csv(top_csv, index=False)
    print(f"Saved: {top_csv}")

    latex_cols = ["star", "species", "comparison_type", "metric", "target", "delta_log10", "delta_linear", "physical_signal", "significance"]
    save_simple_latex(
        top_changes[[c for c in latex_cols if c in top_changes.columns]].copy(),
        os.path.join(out_dir, "latex_tables", "aerosol_top_changes.tex"),
        caption="Most significant aerosol changes identified by the interpretation script.",
        label="tab:aerosol_top_changes",
    )
    save_latex_metric_tables(metrics_df, os.path.join(out_dir, "latex_tables"))

    plot_heatmaps(sig_df, os.path.join(out_dir, "figures", "heatmaps"))
    plot_significance_bars(sig_df, os.path.join(out_dir, "figures", "bars"))
    plot_metric_scatter(metrics_df, os.path.join(out_dir, "figures", "scatter"))

    print("\nDone.")
    print(f"Interpretation outputs written to: {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create publication-ready aerosol interpretation tables, summaries and plots.")
    parser.add_argument("--metrics-csv", default=None, help="Path to aerosol_metrics.csv. Defaults to TABLES_DIR/aerosol_metrics.csv.")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Directory where interpretation outputs are written.")
    args = parser.parse_args()

    main(metrics_csv=args.metrics_csv, out_dir=args.out_dir)
