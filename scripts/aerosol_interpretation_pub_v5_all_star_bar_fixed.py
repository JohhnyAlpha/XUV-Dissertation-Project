import os
import re
import math
from io import StringIO
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_DIR = os.path.expanduser("~/atmos/BA/Plots/aerosol_analysis_pub")
RAW_BASE_DIR = os.path.expanduser("~/atmos/OutputStorage")
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
FIGSIZE_ALL_STAR_HEATMAP = (18.0, 10.0)
FIGSIZE_ALL_STAR_SCATTER = (17.0, 10.0)
ALTITUDE_AUTO_CONVERT = True
PROFILE_ALTITUDE_LIMIT_KM = (0.0, 100.0)

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
    # Do not call tight_layout here: several publication figures use external
    # legends/colorbars/manual axes, which are intentionally incompatible with
    # tight_layout and trigger warnings. Individual plotting functions set their
    # own spacing with subplots_adjust or constrained axes; bbox_inches="tight"
    # keeps exported PNG/PDF boundaries clean.
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
    """Return a valid Matplotlib colour for any aerosol species name.

    Known aerosol families use the fixed publication palette above. Unknown
    species fall back to Matplotlib's tab10 cycle instead of returning None;
    returning None causes bar/barh to fail when a sequence of colours is passed.
    """
    sp = str(species)
    key = sp.upper()
    for token, color in AEROSOL_COLORS.items():
        if token.upper() in key:
            return color

    fallback_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["0.35"])
    # Stable hash independent of Python's randomized hash seed.
    idx = sum(ord(ch) for ch in key) % len(fallback_cycle)
    return fallback_cycle[idx]


# --------------------------------------------------
# OPTIONAL RAW AEROSOL ANALYSIS FUNCTIONS
# --------------------------------------------------
# These functions incorporate the core functionality of aerosol_analysis_pub.py.
# If aerosol_metrics.csv is not present, the interpretation script can now build
# the metrics table directly from PHOTOCHEM_OUTPUT/*_hcaer.out files.
STARS = {
    "18": "Epsilon_Eri",
    "24": "HD40307",
    "25": "HD85512",
    "26": "HD97658",
}
SCENARIO_TO_FSCALE = {"1": "0.75", "2": "1.00", "3": "1.50"}


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


def pick_altitude_column(df):
    for candidate in ["ALT", "Z", "Alt", "alt", "z"]:
        if candidate in df.columns:
            return candidate
    return None


def to_numeric_array(series):
    return pd.to_numeric(series, errors="coerce").values


def altitude_to_km(alt_raw):
    """Return altitude in km using conservative auto-detection."""
    alt = np.asarray(alt_raw, dtype=float)
    finite = alt[np.isfinite(alt)]
    if finite.size == 0 or not ALTITUDE_AUTO_CONVERT:
        return alt
    max_abs = float(np.nanmax(np.abs(finite)))
    if max_abs > 1.0e6:
        return alt / 1.0e5
    if max_abs > 1.0e3:
        return alt / 1.0e3
    return alt


def normalize_altitude_metric_columns(df):
    """Convert loaded peak/centroid altitude columns to km if they are still in cm/m."""
    out = df.copy()
    for col in ["peak_alt", "centroid_alt"]:
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        max_abs = float(np.nanmax(np.abs(finite)))
        if max_abs > 1.0e6:
            out[col] = vals / 1.0e5
            print(f"Converted {col} from cm to km in loaded metrics table.")
        elif max_abs > 1.0e3:
            out[col] = vals / 1.0e3
            print(f"Converted {col} from m to km in loaded metrics table.")
    return out


def finite_mask(x, y):
    return np.isfinite(x) & np.isfinite(y)


def finite_positive_mask(x, y):
    return np.isfinite(x) & np.isfinite(y) & (x > 0)


def integrate_curve(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


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
            if suffix == "hcaer.out":
                scenario_files[scenario]["hcaer_file"] = os.path.join(photochem_dir, fname)

    return dict(scenario_files)


def parse_hcaer(file_path):
    if file_path is None or not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except Exception as exc:
        print(f"Failed to open {file_path}: {exc}")
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
        df = pd.read_csv(StringIO("\n".join(table_lines)), sep=r"\s+", engine="python")
    except Exception as exc:
        print(f"Failed to parse aerosol table in {file_path}: {exc}")
        return None

    alt_col = pick_altitude_column(df)
    if alt_col is None:
        print(f"No altitude column found in {file_path}")
        return None

    data = {"ALT": altitude_to_km(to_numeric_array(df[alt_col]))}
    for col in df.columns:
        if col != alt_col:
            data[col] = to_numeric_array(df[col])
    return data


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
                    aerosol = parse_hcaer(files.get("hcaer_file")) if files.get("hcaer_file") else None
                    data[run][star_name][scale][scenario] = {
                        "aerosol": aerosol,
                        "file": files.get("hcaer_file"),
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


def aerosol_species_list(dataset):
    found = set()
    for run in dataset.values():
        for star in run.values():
            for scale in star.values():
                for entry in scale.values():
                    aero = entry.get("aerosol")
                    if aero:
                        found.update(k for k in aero.keys() if k != "ALT")
    return sorted(found)


def compute_aerosol_metrics(aerosol_data, species):
    if aerosol_data is None or "ALT" not in aerosol_data or species not in aerosol_data:
        return None
    alt = np.asarray(aerosol_data["ALT"], dtype=float)
    x = np.asarray(aerosol_data[species], dtype=float)
    mask = finite_mask(x, alt)
    if not np.any(mask):
        return None
    z = alt[mask]
    y = x[mask]
    order = np.argsort(z)
    z = z[order]
    y = y[order]
    pos_mask = np.isfinite(y) & np.isfinite(z) & (y > 0)

    if np.any(pos_mask):
        z_pos = z[pos_mask]
        y_pos = y[pos_mask]
        column_proxy = integrate_curve(y_pos, z_pos) if len(z_pos) >= 2 else np.nan
        peak = np.nanmax(y_pos)
        peak_alt = z_pos[np.nanargmax(y_pos)]
        surface = y_pos[np.nanargmin(z_pos)] if z_pos.size else np.nan
        upper20 = np.nanmean(y_pos[z_pos >= 20]) if np.any(z_pos >= 20) else np.nan
        upper50 = np.nanmean(y_pos[z_pos >= 50]) if np.any(z_pos >= 50) else np.nan
    else:
        column_proxy = peak = peak_alt = surface = upper20 = upper50 = np.nan

    denom = integrate_curve(y, z) if len(z) >= 2 else np.nan
    centroid = integrate_curve(y * z, z) / denom if np.isfinite(denom) and denom != 0 else np.nan
    return {
        "column_proxy": column_proxy,
        "peak": peak,
        "peak_alt": peak_alt,
        "surface": surface,
        "mean_above_20km": upper20,
        "mean_above_50km": upper50,
        "centroid_alt": centroid,
    }


def flatten_raw_metrics(dataset, species_names):
    """Flatten raw hcaer output into one row per run/star/FSCALE/scenario/species.

    The scale directory (FSCALE) and filename scenario are treated as independent
    axes, matching the final species workflow. This prevents all-star FSCALE
    scatter plots from collapsing onto a single x value.
    """
    rows = []
    for run in RUN_ORDER:
        if run not in dataset:
            continue
        for star in STAR_ORDER:
            star_dict = dataset.get(run, {}).get(star, {})
            if not star_dict:
                continue
            for fscale in FSCALE_ORDER:
                if fscale not in star_dict:
                    continue
                for scenario in SCENARIO_ORDER:
                    entry = star_dict[fscale].get(scenario)
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
                            "fscale": str(fscale),
                            "scale_dir": str(fscale),
                            "species": sp,
                            "file": entry.get("file"),
                        }
                        row.update(metrics)
                        rows.append(row)
    return pd.DataFrame(rows)

def build_metrics_from_raw(raw_base_dir=RAW_BASE_DIR, output_csv=None):
    dataset = build_dataset(raw_base_dir)
    species_names = aerosol_species_list(dataset)
    if not species_names:
        raise RuntimeError(f"No aerosol species found under raw base directory: {raw_base_dir}")
    print(f"Detected aerosol species: {species_names}")
    metrics_df = flatten_raw_metrics(dataset, species_names)
    if metrics_df.empty:
        raise RuntimeError("No aerosol metrics could be computed from raw hcaer outputs.")
    if output_csv is not None:
        ensure_dir(os.path.dirname(output_csv))
        metrics_df.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}")
    return metrics_df


# --------------------------------------------------
# I/O
# --------------------------------------------------
def load_metrics(metrics_csv=None, raw_base_dir=RAW_BASE_DIR, rebuild_from_raw=False):
    """Load aerosol metrics, or build them from raw hcaer.out files if needed.

    If the existing CSV contains only one FSCALE but raw outputs are available,
    the script rebuilds from raw output by default because that usually means the
    older aerosol helper collapsed the independent FSCALE/scenario axes.
    """
    metrics_csv = metrics_csv or os.path.join(TABLES_DIR, "aerosol_metrics.csv")
    should_build = rebuild_from_raw or not os.path.exists(metrics_csv)

    if should_build:
        if not os.path.exists(metrics_csv):
            print(
                f"Could not find metrics table: {metrics_csv}\n"
                f"Building aerosol metrics directly from raw outputs in: {raw_base_dir}"
            )
        else:
            print(f"Rebuilding aerosol metrics directly from raw outputs in: {raw_base_dir}")
        df = build_metrics_from_raw(raw_base_dir=raw_base_dir, output_csv=metrics_csv)
    else:
        df = pd.read_csv(metrics_csv)
        fscales = set(df.get("fscale", pd.Series(dtype=str)).astype(str).dropna().unique())
        if len(fscales.intersection(set(FSCALE_ORDER))) < 2 and os.path.isdir(raw_base_dir):
            print(
                "Loaded metrics table has fewer than two recognised FSCALE values; "
                "rebuilding from raw hcaer outputs to preserve the independent FSCALE axis."
            )
            df = build_metrics_from_raw(raw_base_dir=raw_base_dir, output_csv=metrics_csv)

    df = normalize_altitude_metric_columns(df)
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


def format_heatmap_value(value, metric):
    if not np.isfinite(value):
        return ""
    if abs(value) < 5e-3:
        return "0"
    if metric in ALTITUDE_METRICS:
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw_heatmap(ax, mat, metric, title, *, annotate=True, max_annotated_cells=120, ytick_fontsize=8):
    if mat is None or mat.empty:
        ax.axis("off")
        return None
    vals = mat.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(vals)
    if masked.count() == 0:
        ax.axis("off")
        return None

    finite_vals = vals[np.isfinite(vals)]
    vmax = float(np.nanmax(np.abs(finite_vals))) if finite_vals.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    if metric in ALTITUDE_METRICS:
        im = ax.imshow(masked, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        cbar_label = "Delta altitude (km)"
    else:
        vlim = min(max(vmax, 0.3), 2.0)
        im = ax.imshow(masked, aspect="auto", cmap="coolwarm", vmin=-vlim, vmax=vlim)
        cbar_label = "Delta log10(metric)"

    ax.set_title(title, pad=8)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=ytick_fontsize)
    ax.set_ylabel("Aerosol | star" if any(" | " in str(v) for v in mat.index) else "Aerosol species")
    ax.tick_params(axis="both", length=0)

    if annotate and mat.size <= max_annotated_cells:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.iloc[i, j]
                if np.isfinite(val) and abs(val) >= 5e-3:
                    ax.text(j, i, format_heatmap_value(val, metric), ha="center", va="center", fontsize=6)
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
            fig.subplots_adjust(left=0.08, right=0.95, bottom=0.06, top=0.94, hspace=0.40, wspace=0.28)
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
        fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.92, hspace=0.35, wspace=0.28)
        save_current_figure_both(fig, os.path.join(out_dir, f"{sanitize_for_filename(star)}_aerosol_metric_scatter"))



def all_star_heatmap_matrix(sig_df, comparison_type, metric):
    """Heatmap rows are aerosol/star combinations; columns are run/FSCALE comparisons."""
    sub = sig_df[(sig_df["comparison_type"] == comparison_type) & (sig_df["metric"] == metric)].copy()
    if sub.empty:
        return None
    value_col = "delta_linear" if metric in ALTITUDE_METRICS else "delta_log10"
    sub["row_label"] = sub["species"].astype(str) + " | " + sub["star"].astype(str)
    if comparison_type == "vs_Run1_same_fscale":
        sub["comparison"] = sub["fscale"].astype(str) + "\n" + sub["run"].astype(str)
        desired = [f"{f}\n{r}" for f in FSCALE_ORDER for r in RUN_ORDER]
    else:
        sub["comparison"] = sub["run"].astype(str) + "\n" + sub["fscale"].astype(str)
        desired = [f"{r}\n{f}" for r in RUN_ORDER for f in FSCALE_ORDER]
    piv = sub.pivot_table(index="row_label", columns="comparison", values=value_col, aggfunc="first")
    species_order = sorted(sub["species"].astype(str).unique())
    row_order = [f"{sp} | {star}" for sp in species_order for star in STAR_ORDER if f"{sp} | {star}" in piv.index]
    existing = [c for c in desired if c in piv.columns]
    return piv.reindex(index=row_order, columns=existing)


def plot_all_star_heatmaps(sig_df, out_dir):
    """Combined heatmaps across all stars for each core metric and comparison mode."""
    ensure_dir(out_dir)
    if sig_df.empty:
        return
    metrics = [m for m in ["column_proxy", "peak", "mean_above_50km", "centroid_alt"] if m in set(sig_df["metric"].astype(str))]
    if not metrics:
        return
    pdf_path = os.path.join(out_dir, "all_stars_aerosol_heatmaps.pdf")
    with PdfPages(pdf_path) as pdf:
        for metric in metrics:
            mats = [
                all_star_heatmap_matrix(sig_df, "vs_Run1_same_fscale", metric),
                all_star_heatmap_matrix(sig_df, "vs_Fscale1.00_same_run", metric),
            ]
            nrows = max([m.shape[0] for m in mats if m is not None and not m.empty] or [1])
            fig_h = max(8.5, min(18.0, 0.31 * nrows + 2.3))
            fig, axes = plt.subplots(1, 2, figsize=(18.5, fig_h), gridspec_kw={"wspace": 0.35})
            for ax, mat, title in zip(axes, mats, ["run effect vs Run_1", "FSCALE effect vs 1.00"]):
                drawn = draw_heatmap(
                    ax, mat, metric, title,
                    annotate=(nrows <= 18),
                    max_annotated_cells=140,
                    ytick_fontsize=7 if nrows > 24 else 8,
                )
                if drawn is not None:
                    im, cbar_label = drawn
                    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.018)
                    cbar.set_label(cbar_label)
            fig.suptitle(f"All stars | aerosol {metric.replace('_', ' ')} heatmaps", y=0.985)
            fig.subplots_adjust(left=0.18, right=0.96, bottom=0.12, top=0.92, wspace=0.35)
            stem = os.path.join(out_dir, f"all_stars_aerosol_{sanitize_for_filename(metric)}_heatmaps")
            fig.savefig(stem + ".png", dpi=SAVE_DPI, bbox_inches="tight")
            fig.savefig(stem + ".pdf", bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            print(f"Saved: {stem}.png")
            print(f"Saved: {stem}.pdf")
            plt.close(fig)
    print(f"Saved: {pdf_path}")


def plot_all_star_metric_scatter(metrics_df, out_dir):
    """All-star metric scatter with uncluttered legends outside the panels.

    Encoding is separated: aerosol = colour, star = marker, run = line style.
    This keeps every aerosol and every star in the legend without a 30+ item
    legend covering the data.
    """
    ensure_dir(out_dir)
    available = [m for m in ["column_proxy", "peak", "mean_above_50km", "centroid_alt"] if m in metrics_df.columns]
    if not available:
        return
    species_order = sorted(metrics_df["species"].dropna().astype(str).unique())
    star_markers = {"Epsilon_Eri": "o", "HD40307": "s", "HD85512": "^", "HD97658": "D"}
    run_linestyles = {"Run_1": "-", "Run_2": "--", "Run_3": ":"}

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_ALL_STAR_SCATTER)
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")

    for ax, metric in zip(axes, available):
        ax.axis("on")
        for sp in species_order:
            for star in STAR_ORDER:
                combo = metrics_df[(metrics_df["species"].astype(str) == sp) & (metrics_df["star"].astype(str) == star)]
                if combo.empty:
                    continue
                for run in RUN_ORDER:
                    g = combo[combo["run"].astype(str) == run].copy()
                    if g.empty:
                        continue
                    g["fscale_num"] = pd.to_numeric(g["fscale"], errors="coerce")
                    g = g.sort_values("fscale_num")
                    x = g["fscale_num"].to_numpy(dtype=float)
                    y = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
                    mask = np.isfinite(x) & np.isfinite(y)
                    if not np.any(mask):
                        continue
                    ax.plot(
                        x[mask], y[mask],
                        marker=star_markers.get(star, "o"),
                        linestyle=run_linestyles.get(run, "-"),
                        linewidth=1.15,
                        markersize=4.4,
                        alpha=0.78,
                        color=color_for_species(sp),
                    )
        if metric not in ALTITUDE_METRICS:
            yall = pd.to_numeric(metrics_df[metric], errors="coerce").to_numpy(dtype=float)
            ypos = yall[np.isfinite(yall) & (yall > 0)]
            if ypos.size > 1 and np.nanmax(ypos) / np.nanmin(ypos) > 100:
                ax.set_yscale("log")
        else:
            if PROFILE_ALTITUDE_LIMIT_KM is not None:
                yall = pd.to_numeric(metrics_df[metric], errors="coerce").to_numpy(dtype=float)
                finite = yall[np.isfinite(yall)]
                if finite.size and np.nanmax(finite) <= PROFILE_ALTITUDE_LIMIT_KM[1] * 1.5:
                    ax.set_ylim(*PROFILE_ALTITUDE_LIMIT_KM)
        ax.set_title(metric.replace("_", " "), pad=8)
        ax.set_xlabel("FSCALE")
        ax.set_ylabel(metric.replace("_", " ") + (" (km)" if metric in ALTITUDE_METRICS else ""))
        ax.set_xticks([float(f) for f in FSCALE_ORDER])
        ax.set_xlim(min(float(f) for f in FSCALE_ORDER) - 0.06, max(float(f) for f in FSCALE_ORDER) + 0.06)
        ax.grid(True, alpha=0.22, linewidth=0.7)

    aerosol_handles = [plt.Line2D([0], [0], color=color_for_species(sp), lw=2.2) for sp in species_order]
    star_handles = [plt.Line2D([0], [0], color="0.25", marker=star_markers.get(star, "o"), linestyle="", markersize=6) for star in STAR_ORDER]
    run_handles = [plt.Line2D([0], [0], color="0.25", linestyle=run_linestyles[r], lw=1.8) for r in RUN_ORDER]

    legend_ax = fig.add_axes([0.815, 0.13, 0.17, 0.74])
    legend_ax.axis("off")
    leg1 = legend_ax.legend(aerosol_handles, species_order, title="Aerosol", loc="upper left", frameon=False, fontsize=8, title_fontsize=9)
    legend_ax.add_artist(leg1)
    leg2 = legend_ax.legend(star_handles, [s.replace("_", " ") for s in STAR_ORDER], title="Star", loc="center left", frameon=False, fontsize=8, title_fontsize=9)
    legend_ax.add_artist(leg2)
    legend_ax.legend(run_handles, [RUN_LABELS.get(r, r) for r in RUN_ORDER], title="Run", loc="lower left", frameon=False, fontsize=8, title_fontsize=9)

    fig.suptitle("All stars | aerosol metrics vs FSCALE", y=0.975)
    fig.subplots_adjust(left=0.07, right=0.78, bottom=0.08, top=0.92, hspace=0.36, wspace=0.24)
    save_current_figure_both(fig, os.path.join(out_dir, "all_stars_aerosol_metric_scatter"))


def plot_all_star_sensitivity_bars(sig_df, out_dir):
    """All-star dissertation-style bar summaries of aerosol sensitivity.

    The main panel ranks each aerosol/star pair by the strongest abundance
    response across the core interpretation metrics. A companion panel ranks
    aerosol species across all stars. Altitude-only metrics are excluded from
    this score so the bar length has one clear meaning: maximum absolute
    abundance response in dex.
    """
    ensure_dir(out_dir)
    if sig_df.empty:
        return

    abundance_metrics = ["column_proxy", "peak", "mean_above_50km"]
    sub = sig_df[sig_df["metric"].isin(abundance_metrics)].copy()
    if sub.empty or "abs_delta_log10" not in sub.columns:
        return
    sub["abs_delta_log10"] = pd.to_numeric(sub["abs_delta_log10"], errors="coerce")
    sub = sub[np.isfinite(sub["abs_delta_log10"])]
    if sub.empty:
        return

    idx = sub.groupby(["species", "star"], observed=False)["abs_delta_log10"].idxmax()
    best = sub.loc[idx].copy().dropna(subset=["abs_delta_log10"])
    if best.empty:
        return

    best["row_label"] = best["species"].astype(str) + " | " + best["star"].astype(str).str.replace("_", " ", regex=False)
    best = best.sort_values("abs_delta_log10", ascending=True).reset_index(drop=True)

    fig_h = max(7.5, min(16.0, 0.34 * len(best) + 2.0))
    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    y = np.arange(len(best))
    bar_colors = [color_for_species(sp) for sp in best["species"].astype(str)]
    ax.barh(y, best["abs_delta_log10"].to_numpy(dtype=float), color=bar_colors, alpha=0.86)

    ax.set_yticks(y)
    ax.set_yticklabels(best["row_label"].tolist(), fontsize=8)
    ax.set_xlabel(r"strongest abundance response, max |Δlog$_{10}$| (dex)")
    ax.set_title("All stars | strongest aerosol abundance sensitivity", pad=12)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)

    xmax = float(np.nanmax(best["abs_delta_log10"].to_numpy(dtype=float)))
    xpad = 0.015 * xmax if xmax > 0 else 0.05
    ax.set_xlim(0, xmax * 1.28 if xmax > 0 else 1.0)
    for yi, (_, row) in enumerate(best.iterrows()):
        val = float(row["abs_delta_log10"])
        metric = str(row.get("metric", ""))
        comp = "FSCALE" if row.get("comparison_type") == "vs_Fscale1.00_same_run" else "run"
        target = str(row.get("target", "")) if "target" in row else ""
        ax.text(val + xpad, yi, f"{metric}; {comp} {target}", va="center", ha="left", fontsize=7.4, color="0.25")

    species_order = best.groupby("species", observed=False)["abs_delta_log10"].max().sort_values(ascending=False).index.astype(str).tolist()
    handles = [plt.Line2D([0], [0], color=color_for_species(sp), lw=6) for sp in species_order]
    ax.legend(handles, species_order, title="Aerosol", loc="lower right", frameon=False, fontsize=8, title_fontsize=9)
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.08, top=0.93)
    save_current_figure_both(fig, os.path.join(out_dir, "all_stars_aerosol_sensitivity_bar"))

    agg = sub.groupby("species", observed=False)["abs_delta_log10"].max().sort_values(ascending=True).reset_index()
    if agg.empty:
        return
    fig_h2 = max(4.8, 0.42 * len(agg) + 1.8)
    fig2, ax2 = plt.subplots(figsize=(9.5, fig_h2))
    y2 = np.arange(len(agg))
    colors2 = [color_for_species(sp) for sp in agg["species"].astype(str)]
    ax2.barh(y2, agg["abs_delta_log10"].to_numpy(dtype=float), color=colors2, alpha=0.88)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(agg["species"].astype(str).tolist())
    ax2.set_xlabel(r"max |Δlog$_{10}$| across all stars/runs/FSCALE")
    ax2.set_title("All stars | aerosol sensitivity ranking", pad=12)
    ax2.grid(True, axis="x", alpha=0.25, linewidth=0.7)
    ax2.set_axisbelow(True)
    xmax2 = float(np.nanmax(agg["abs_delta_log10"].to_numpy(dtype=float)))
    ax2.set_xlim(0, xmax2 * 1.12 if xmax2 > 0 else 1.0)
    for yi, val in enumerate(agg["abs_delta_log10"].to_numpy(dtype=float)):
        ax2.text(val + (0.01 * xmax2 if xmax2 > 0 else 0.02), yi, f"{val:.2f}", va="center", ha="left", fontsize=8)
    fig2.subplots_adjust(left=0.20, right=0.96, bottom=0.12, top=0.90)
    save_current_figure_both(fig2, os.path.join(out_dir, "all_stars_aerosol_species_sensitivity_ranking"))


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
def main(metrics_csv=None, out_dir=OUT_DIR, raw_base_dir=RAW_BASE_DIR, rebuild_from_raw=False):
    ensure_dir(out_dir)
    for sub in ["tables", "figures", "figures/heatmaps", "figures/bars", "figures/scatter", "figures/all_star_heatmaps", "figures/all_star_scatter", "figures/all_star_bars", "latex_tables"]:
        ensure_dir(os.path.join(out_dir, sub))

    metrics_df = load_metrics(metrics_csv, raw_base_dir=raw_base_dir, rebuild_from_raw=rebuild_from_raw)
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
    plot_all_star_heatmaps(sig_df, os.path.join(out_dir, "figures", "all_star_heatmaps"))
    plot_all_star_metric_scatter(metrics_df, os.path.join(out_dir, "figures", "all_star_scatter"))
    plot_all_star_sensitivity_bars(sig_df, os.path.join(out_dir, "figures", "all_star_bars"))

    print("\nDone.")
    print(f"Interpretation outputs written to: {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create publication-ready aerosol interpretation tables, summaries and plots.")
    parser.add_argument("--metrics-csv", default=None, help="Path to aerosol_metrics.csv. Defaults to TABLES_DIR/aerosol_metrics.csv.")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Directory where interpretation outputs are written.")
    parser.add_argument("--raw-base-dir", default=RAW_BASE_DIR, help="Raw OutputStorage directory used if aerosol_metrics.csv is missing.")
    parser.add_argument("--rebuild-from-raw", action="store_true", help="Rebuild aerosol_metrics.csv from raw hcaer.out files even if a metrics CSV already exists.")
    args = parser.parse_args()

    main(metrics_csv=args.metrics_csv, out_dir=args.out_dir, raw_base_dir=args.raw_base_dir, rebuild_from_raw=args.rebuild_from_raw)
