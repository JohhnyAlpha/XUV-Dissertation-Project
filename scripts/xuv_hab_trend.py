import os
import re
from io import StringIO
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.expanduser("~/atmos/OutputStorage")
OUT_DIR = os.path.expanduser("~/atmos/BA/Plots/xuv_trends")

STARS = {
    "18": "Epsilon_Eri",
    "24": "HD_40307",
    "25": "HD_85512",
    "26": "HD_97658",
}

SAVE_DPI = 220
SHOW_PLOTS = False

FIGSIZE = (8, 6)
WIDE_FIGSIZE = (10, 6)

# These are only for annotation/reference lines, not hard physics cutoffs.
FREEZING_TEMPERATURE_K = 273.15
DESICCATION_H2O_THRESHOLD = 1e-5
ABIOTIC_O2_THRESHOLD = 1e-3
ABIOTIC_O3_THRESHOLD = 1e-8

USE_LOG_Y_FOR_MIXING_RATIOS = True
ANNOTATE_POINTS = False


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
    fig.savefig(out_path, format="jpeg", dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def read_table(file_path, skiprows=None):
    try:
        return pd.read_csv(
            file_path,
            sep=r"\s+",
            engine="python",
            skiprows=skiprows,
            comment="#",
        )
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None


def to_numeric_array(series):
    return pd.to_numeric(series, errors="coerce").values


def pick_altitude_column(df):
    for candidate in ["ALT", "Z", "Alt", "alt", "z"]:
        if candidate in df.columns:
            return candidate
    return None


def find_matching_column(df, target, exact=True, contains=False):
    for col in df.columns:
        if exact and col.lower() == target.lower():
            return col
        if contains and target.lower() in col.lower():
            return col
    return None


def find_first_matching_column(df, candidates_exact=None, candidates_contains=None):
    candidates_exact = candidates_exact or []
    candidates_contains = candidates_contains or []

    for target in candidates_exact:
        for col in df.columns:
            if col.lower() == target.lower():
                return col

    for target in candidates_contains:
        for col in df.columns:
            if target.lower() in col.lower():
                return col

    return None


def finite_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def maybe_set_log_y(ax, y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return
    if USE_LOG_Y_FOR_MIXING_RATIOS and np.all(y > 0) and (np.nanmax(y) / np.nanmin(y) > 50):
        ax.set_yscale("log")


# -----------------------------
# FILE DISCOVERY
# -----------------------------
def build_file_index(scale_dir):
    photochem_dir = os.path.join(scale_dir, "PHOTOCHEM_OUTPUT")
    clima_dir = os.path.join(scale_dir, "CLIMA_OUTPUT")

    scenario_files = defaultdict(
        lambda: {
            "out_file": None,
            "clima_file": None,
        }
    )

    pattern = re.compile(r"^(Run_\d+)_([0-9]+(?:\.[0-9]+)?)_(\d+)_(.+)$")

    if os.path.isdir(photochem_dir):
        for fname in os.listdir(photochem_dir):
            match = pattern.match(fname)
            if not match:
                continue

            _, _, scenario, suffix = match.groups()
            if suffix == "out.out":
                scenario_files[scenario]["out_file"] = os.path.join(photochem_dir, fname)

    if os.path.isdir(clima_dir):
        for fname in os.listdir(clima_dir):
            match = pattern.match(fname)
            if not match:
                continue

            _, _, scenario, suffix = match.groups()
            if suffix in ("clima_last.tab", "clima_last.out"):
                scenario_files[scenario]["clima_file"] = os.path.join(clima_dir, fname)

    return dict(scenario_files)


# -----------------------------
# PARSERS
# -----------------------------
def parse_out_species_table(file_path):
    """
    Parse the 'MIXING RATIOS OF LONG-LIVED SPECIES' section from out.out.
    """
    if file_path is None or not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Failed to open {file_path}: {e}")
        return None

    start_idx = None
    for i, line in enumerate(lines):
        if "MIXING RATIOS OF LONG-LIVED SPECIES" in line:
            start_idx = i
            break

    if start_idx is None:
        print(f"No species section found in {file_path}")
        return None

    header_idx = None
    for i in range(start_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("Z") or stripped.startswith("ALT"):
            header_idx = i
            break

    if header_idx is None:
        print(f"No species header found in {file_path}")
        return None

    table_lines = []
    data_started = False

    for i in range(header_idx, len(lines)):
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        if not stripped:
            if data_started:
                break
            continue

        table_lines.append(line)
        if len(table_lines) > 1:
            data_started = True

    if len(table_lines) < 2:
        print(f"No species data found in {file_path}")
        return None

    try:
        df = pd.read_csv(
            StringIO("\n".join(table_lines)),
            sep=r"\s+",
            engine="python",
        )
        return df
    except Exception as e:
        print(f"Failed to parse species table in {file_path}: {e}")
        return None


def parse_clima_table(file_path):
    if file_path is None or not os.path.exists(file_path):
        return None

    df = read_table(file_path)
    if df is None or df.empty:
        return None
    return df


# -----------------------------
# PROFILE EXTRACTION
# -----------------------------
def get_surface_row_index(df):
    alt_col = pick_altitude_column(df)
    if alt_col is None:
        return 0

    alt = pd.to_numeric(df[alt_col], errors="coerce").values
    finite = np.isfinite(alt)
    if not finite.any():
        return 0

    return int(np.nanargmin(alt))


def get_top_row_index(df):
    alt_col = pick_altitude_column(df)
    if alt_col is None:
        return len(df) - 1

    alt = pd.to_numeric(df[alt_col], errors="coerce").values
    finite = np.isfinite(alt)
    if not finite.any():
        return len(df) - 1

    return int(np.nanargmax(alt))


def get_species_value(df, species_name, where="surface"):
    if df is None or df.empty:
        return np.nan

    col = find_matching_column(df, species_name, exact=True)
    if col is None:
        return np.nan

    vals = pd.to_numeric(df[col], errors="coerce").values
    idx = get_surface_row_index(df) if where == "surface" else get_top_row_index(df)

    if idx < 0 or idx >= len(vals):
        return np.nan
    return vals[idx]


def get_clima_value(df, target_name, where="surface"):
    if df is None or df.empty:
        return np.nan

    col = find_matching_column(df, target_name, exact=False, contains=True)
    if col is None:
        return np.nan

    vals = pd.to_numeric(df[col], errors="coerce").values
    idx = get_surface_row_index(df) if where == "surface" else get_top_row_index(df)

    if idx < 0 or idx >= len(vals):
        return np.nan
    return vals[idx]


def get_surface_temperature(clima_df):
    if clima_df is None or clima_df.empty:
        return np.nan

    temp_col = find_first_matching_column(
        clima_df,
        candidates_exact=["T", "TEMP", "TEMPK", "TEMPERATURE"],
        candidates_contains=["temp", "temperature"],
    )
    if temp_col is None:
        return np.nan

    vals = pd.to_numeric(clima_df[temp_col], errors="coerce").values
    idx = get_surface_row_index(clima_df)
    if idx < 0 or idx >= len(vals):
        return np.nan
    return vals[idx]


# -----------------------------
# BUILD SUMMARY DATAFRAME
# -----------------------------
def build_summary_dataframe(base_dir):
    rows = []
    runs = discover_run_dirs(base_dir)
    print("Discovered runs:", runs)

    for run in runs:
        run_dir = os.path.join(base_dir, run)

        for star_id in sorted(os.listdir(run_dir), key=numeric_sort_key):
            star_dir = os.path.join(run_dir, star_id)
            if not os.path.isdir(star_dir):
                continue

            star_name = star_name_from_id(star_id)

            for scale in sorted(os.listdir(star_dir), key=numeric_sort_key):
                scale_dir = os.path.join(star_dir, scale)
                if not os.path.isdir(scale_dir):
                    continue

                indexed = build_file_index(scale_dir)
                if not indexed:
                    continue

                for scenario, files in sorted(indexed.items(), key=lambda x: int(x[0])):
                    print(f"\nLoading: {run} | {star_name} | {scale} | scenario {scenario}")
                    print("  out_file   :", files["out_file"])
                    print("  clima_file :", files["clima_file"])

                    species_df = parse_out_species_table(files["out_file"]) if files["out_file"] else None
                    clima_df = parse_clima_table(files["clima_file"]) if files["clima_file"] else None

                    row = {
                        "run": run,
                        "star": star_name,
                        "star_id": str(star_id),
                        "scale": float(scale),
                        "scenario": int(scenario),

                        "surface_temperature": get_surface_temperature(clima_df),

                        "surface_h2o": (
                            get_species_value(species_df, "H2O", "surface")
                            if np.isfinite(get_species_value(species_df, "H2O", "surface"))
                            else get_clima_value(clima_df, "H2O", "surface")
                        ),
                        "surface_o2": get_species_value(species_df, "O2", "surface"),
                        "surface_o3": (
                            get_species_value(species_df, "O3", "surface")
                            if np.isfinite(get_species_value(species_df, "O3", "surface"))
                            else get_clima_value(clima_df, "O3", "surface")
                        ),
                        "surface_ch4": get_species_value(species_df, "CH4", "surface"),
                        "surface_co2": get_species_value(species_df, "CO2", "surface"),

                        "upper_h2o": (
                            get_species_value(species_df, "H2O", "top")
                            if np.isfinite(get_species_value(species_df, "H2O", "top"))
                            else get_clima_value(clima_df, "H2O", "top")
                        ),
                        "upper_h2": get_species_value(species_df, "H2", "top"),
                        "upper_o2": get_species_value(species_df, "O2", "top"),
                        "upper_o3": (
                            get_species_value(species_df, "O3", "top")
                            if np.isfinite(get_species_value(species_df, "O3", "top"))
                            else get_clima_value(clima_df, "O3", "top")
                        ),
                    }

                    rows.append(row)

    df = pd.DataFrame(rows)
    return df


# -----------------------------
# PLOTTING CORE
# -----------------------------
def marker_map():
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    return markers


def plot_trend_by_star(df, x_col, y_col, title_prefix, y_label, out_stub,
                       y_log_candidate=False, add_hline=None, add_vline=None):
    stars = sorted(df["star"].dropna().unique())
    markers = marker_map()

    for star in stars:
        sub = df[(df["star"] == star)].copy()
        x, y = finite_xy(sub[x_col], sub[y_col])

        if len(x) == 0:
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE)

        scenarios = sorted(sub["scenario"].dropna().unique())
        for i, scenario in enumerate(scenarios):
            ssub = sub[sub["scenario"] == scenario]
            xx, yy = finite_xy(ssub[x_col], ssub[y_col])
            if len(xx) == 0:
                continue

            ax.scatter(xx, yy, marker=markers[i % len(markers)], alpha=0.9, label=f"Scenario {scenario}")

            if ANNOTATE_POINTS:
                for _, row in ssub.iterrows():
                    if np.isfinite(row[x_col]) and np.isfinite(row[y_col]):
                        ax.annotate(
                            f'{row["run"]}, {row["scale"]}',
                            (row[x_col], row[y_col]),
                            fontsize=7,
                        )

        if add_hline is not None:
            ax.axhline(add_hline, linestyle="--")
        if add_vline is not None:
            ax.axvline(add_vline, linestyle="--")

        ax.set_xlabel("XUV Proxy (scale)")
        ax.set_ylabel(y_label)
        ax.set_title(f"{title_prefix} | {star}")
        ax.grid(True, alpha=0.3)

        if y_log_candidate:
            maybe_set_log_y(ax, sub[y_col].values)

        ax.legend(fontsize=8, loc="best")

        out_path = os.path.join(
            OUT_DIR,
            sanitize_for_filename(star),
            f"{out_stub}_{sanitize_for_filename(star)}.jpeg",
        )
        save_figure(fig, out_path)


def plot_cross_by_star(df, x_col, y_col, title_prefix, x_label, y_label, out_stub,
                       x_log_candidate=False, y_log_candidate=False,
                       add_hline=None, add_vline=None):
    stars = sorted(df["star"].dropna().unique())
    markers = marker_map()

    for star in stars:
        sub = df[(df["star"] == star)].copy()
        x, y = finite_xy(sub[x_col], sub[y_col])

        if len(x) == 0:
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE)

        scenarios = sorted(sub["scenario"].dropna().unique())
        for i, scenario in enumerate(scenarios):
            ssub = sub[sub["scenario"] == scenario]
            xx, yy = finite_xy(ssub[x_col], ssub[y_col])
            if len(xx) == 0:
                continue

            ax.scatter(xx, yy, marker=markers[i % len(markers)], alpha=0.9, label=f"Scenario {scenario}")

            if ANNOTATE_POINTS:
                for _, row in ssub.iterrows():
                    if np.isfinite(row[x_col]) and np.isfinite(row[y_col]):
                        ax.annotate(
                            f'{row["run"]}, {row["scale"]}',
                            (row[x_col], row[y_col]),
                            fontsize=7,
                        )

        if add_hline is not None:
            ax.axhline(add_hline, linestyle="--")
        if add_vline is not None:
            ax.axvline(add_vline, linestyle="--")

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{title_prefix} | {star}")
        ax.grid(True, alpha=0.3)

        if x_log_candidate:
            xx = np.asarray(sub[x_col], dtype=float)
            xx = xx[np.isfinite(xx)]
            if xx.size > 0 and np.all(xx > 0) and (np.nanmax(xx) / np.nanmin(xx) > 50):
                ax.set_xscale("log")

        if y_log_candidate:
            maybe_set_log_y(ax, sub[y_col].values)

        ax.legend(fontsize=8, loc="best")

        out_path = os.path.join(
            OUT_DIR,
            sanitize_for_filename(star),
            f"{out_stub}_{sanitize_for_filename(star)}.jpeg",
        )
        save_figure(fig, out_path)


def plot_o2_o3_dual_axis_by_star(df):
    stars = sorted(df["star"].dropna().unique())
    markers = marker_map()

    for star in stars:
        sub = df[df["star"] == star].copy()
        if sub.empty:
            continue

        fig, ax1 = plt.subplots(figsize=WIDE_FIGSIZE)
        ax2 = ax1.twinx()

        scenarios = sorted(sub["scenario"].dropna().unique())
        plotted_any = False

        for i, scenario in enumerate(scenarios):
            ssub = sub[sub["scenario"] == scenario]

            x1, y1 = finite_xy(ssub["scale"], ssub["surface_o2"])
            if len(x1) > 0:
                ax1.scatter(x1, y1, marker=markers[i % len(markers)], alpha=0.9, label=f"O2 s={scenario}")
                plotted_any = True

            x2, y2 = finite_xy(ssub["scale"], ssub["surface_o3"])
            if len(x2) > 0:
                ax2.scatter(x2, y2, marker=markers[i % len(markers)], alpha=0.6, label=f"O3 s={scenario}")
                plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        ax1.axhline(ABIOTIC_O2_THRESHOLD, linestyle="--")
        ax2.axhline(ABIOTIC_O3_THRESHOLD, linestyle="--")

        ax1.set_xlabel("XUV Proxy (scale)")
        ax1.set_ylabel("Surface O2 Mixing Ratio")
        ax2.set_ylabel("Surface O3 Mixing Ratio")
        ax1.set_title(f"O2 and O3 vs XUV | {star}")
        ax1.grid(True, alpha=0.3)

        maybe_set_log_y(ax1, sub["surface_o2"].values)
        maybe_set_log_y(ax2, sub["surface_o3"].values)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")

        out_path = os.path.join(
            OUT_DIR,
            sanitize_for_filename(star),
            f"o2_o3_dual_axis_{sanitize_for_filename(star)}.jpeg",
        )
        save_figure(fig, out_path)


# -----------------------------
# SPECIAL SUMMARY PLOTS
# -----------------------------
def plot_desiccation_by_star(df):
    plot_trend_by_star(
        df=df,
        x_col="scale",
        y_col="surface_h2o",
        title_prefix="Desiccation Indicator (Surface H2O vs XUV)",
        y_label="Surface H2O Mixing Ratio",
        out_stub="desiccation_indicator",
        y_log_candidate=True,
        add_hline=DESICCATION_H2O_THRESHOLD,
    )


def plot_false_biosignature_by_star(df):
    stars = sorted(df["star"].dropna().unique())
    markers = marker_map()

    for star in stars:
        sub = df[df["star"] == star].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)
        scenarios = sorted(sub["scenario"].dropna().unique())
        plotted_any = False

        for i, scenario in enumerate(scenarios):
            ssub = sub[sub["scenario"] == scenario]
            xx, yy = finite_xy(ssub["surface_o2"], ssub["surface_o3"])
            if len(xx) == 0:
                continue

            ax.scatter(xx, yy, marker=markers[i % len(markers)], alpha=0.9, label=f"Scenario {scenario}")
            plotted_any = True

            if ANNOTATE_POINTS:
                for _, row in ssub.iterrows():
                    if np.isfinite(row["surface_o2"]) and np.isfinite(row["surface_o3"]):
                        ax.annotate(
                            f'{row["run"]}, {row["scale"]}',
                            (row["surface_o2"], row["surface_o3"]),
                            fontsize=7,
                        )

        if not plotted_any:
            plt.close(fig)
            continue

        ax.axvline(ABIOTIC_O2_THRESHOLD, linestyle="--")
        ax.axhline(ABIOTIC_O3_THRESHOLD, linestyle="--")

        ax.set_xlabel("Surface O2 Mixing Ratio")
        ax.set_ylabel("Surface O3 Mixing Ratio")
        ax.set_title(f"False Biosignature Space (O2 vs O3) | {star}")
        ax.grid(True, alpha=0.3)

        xx = np.asarray(sub["surface_o2"], dtype=float)
        yy = np.asarray(sub["surface_o3"], dtype=float)

        if np.all(xx[np.isfinite(xx)] > 0) and np.nanmax(xx[np.isfinite(xx)]) / np.nanmin(xx[np.isfinite(xx)]) > 50:
            ax.set_xscale("log")
        if np.all(yy[np.isfinite(yy)] > 0) and np.nanmax(yy[np.isfinite(yy)]) / np.nanmin(yy[np.isfinite(yy)]) > 50:
            ax.set_yscale("log")

        ax.legend(fontsize=8, loc="best")

        out_path = os.path.join(
            OUT_DIR,
            sanitize_for_filename(star),
            f"false_biosignature_space_{sanitize_for_filename(star)}.jpeg",
        )
        save_figure(fig, out_path)


# -----------------------------
# CSV OUTPUT
# -----------------------------
def save_summary_tables(df):
    ensure_dir(OUT_DIR)

    full_csv = os.path.join(OUT_DIR, "xuv_trend_summary.csv")
    df.to_csv(full_csv, index=False)
    print(f"Saved: {full_csv}")

    grouped = (
        df.groupby(["star", "scenario"])
        .agg(
            mean_surface_temperature=("surface_temperature", "mean"),
            mean_surface_h2o=("surface_h2o", "mean"),
            mean_surface_o2=("surface_o2", "mean"),
            mean_surface_o3=("surface_o3", "mean"),
            mean_surface_ch4=("surface_ch4", "mean"),
            mean_surface_co2=("surface_co2", "mean"),
            mean_upper_h2o=("upper_h2o", "mean"),
            mean_upper_h2=("upper_h2", "mean"),
            mean_upper_o2=("upper_o2", "mean"),
        )
        .reset_index()
    )

    grouped_csv = os.path.join(OUT_DIR, "xuv_trend_grouped_summary.csv")
    grouped.to_csv(grouped_csv, index=False)
    print(f"Saved: {grouped_csv}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    ensure_dir(OUT_DIR)

    df = build_summary_dataframe(BASE_DIR)

    print("\nSummary dataframe:")
    print(df)

    save_summary_tables(df)

    # XUV proxy trend plots
    plot_trend_by_star(df, "scale", "surface_h2o", "Surface H2O vs XUV", "Surface H2O Mixing Ratio", "surface_h2o_vs_xuv", y_log_candidate=True)
    plot_trend_by_star(df, "scale", "surface_o2", "Surface O2 vs XUV", "Surface O2 Mixing Ratio", "surface_o2_vs_xuv", y_log_candidate=True)
    plot_trend_by_star(df, "scale", "surface_o3", "Surface O3 vs XUV", "Surface O3 Mixing Ratio", "surface_o3_vs_xuv", y_log_candidate=True)
    plot_trend_by_star(df, "scale", "surface_ch4", "Surface CH4 vs XUV", "Surface CH4 Mixing Ratio", "surface_ch4_vs_xuv", y_log_candidate=True)
    plot_trend_by_star(df, "scale", "surface_co2", "Surface CO2 vs XUV", "Surface CO2 Mixing Ratio", "surface_co2_vs_xuv", y_log_candidate=True)
    plot_trend_by_star(df, "scale", "surface_temperature", "Surface Temperature vs XUV", "Surface Temperature [K]", "surface_temperature_vs_xuv", y_log_candidate=False, add_hline=FREEZING_TEMPERATURE_K)

    # Upper atmosphere / escape proxy plots
    plot_trend_by_star(df, "scale", "upper_h2o", "Upper Atmosphere H2O vs XUV", "Upper Atmosphere H2O Mixing Ratio", "upper_h2o_vs_xuv", y_log_candidate=True)
    plot_trend_by_star(df, "scale", "upper_h2", "Upper Atmosphere H2 vs XUV", "Upper Atmosphere H2 Mixing Ratio", "upper_h2_vs_xuv", y_log_candidate=True)
    plot_trend_by_star(df, "scale", "upper_o2", "Upper Atmosphere O2 vs XUV", "Upper Atmosphere O2 Mixing Ratio", "upper_o2_vs_xuv", y_log_candidate=True)

    # Chemistry regime plots
    plot_cross_by_star(df, "surface_o2", "surface_ch4",
                       "CH4 vs O2", "Surface O2 Mixing Ratio", "Surface CH4 Mixing Ratio",
                       "surface_ch4_vs_surface_o2", x_log_candidate=True, y_log_candidate=True)

    plot_cross_by_star(df, "surface_o2", "surface_o3",
                       "O3 vs O2", "Surface O2 Mixing Ratio", "Surface O3 Mixing Ratio",
                       "surface_o3_vs_surface_o2", x_log_candidate=True, y_log_candidate=True,
                       add_hline=ABIOTIC_O3_THRESHOLD, add_vline=ABIOTIC_O2_THRESHOLD)

    plot_cross_by_star(df, "surface_temperature", "surface_h2o",
                       "Surface H2O vs Surface Temperature",
                       "Surface Temperature [K]", "Surface H2O Mixing Ratio",
                       "surface_h2o_vs_surface_temperature",
                       x_log_candidate=False, y_log_candidate=True,
                       add_hline=DESICCATION_H2O_THRESHOLD, add_vline=FREEZING_TEMPERATURE_K)

    # Dual-axis oxygen/ozone summary
    plot_o2_o3_dual_axis_by_star(df)

    # Special thesis-targeted summary plots
    plot_desiccation_by_star(df)
    plot_false_biosignature_by_star(df)

    print("\nDone.")
    print(f"Trend plots written to: {OUT_DIR}")
