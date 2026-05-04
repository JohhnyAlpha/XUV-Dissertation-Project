import os
import re
from io import StringIO
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.expanduser("~/atmos/OutputStorage")
OUT_DIR = os.path.expanduser("~/atmos/BA/Plots/habitability_matrix")

STARS = {
    "18": "Epsilon_Eri",
    "24": "HD_40307",
    "25": "HD_85512",
    "26": "HD_97658",
}

FIGSIZE = (9, 7)
PUB_MATRIX_FIGSIZE = (10, 7.5)
PUB_SUMMARY_FIGSIZE = (12, 10)
SAVE_DPI = 220
PUB_DPI = 300
SHOW_PLOTS = False

# -----------------------------
# MATRIX SETTINGS
# -----------------------------
# Main default:
#   x = surface temperature
#   y = surface H2O
X_MODE = "surface_temperature"
Y_MODE = "surface_h2o"

# Thresholds dividing the 4 quadrants
X_THRESHOLD = 273.15
Y_THRESHOLD = 1e-3

USE_LOG_Y = True

# Label each point with run/scale/scenario
ANNOTATE_POINTS = False
ANNOTATION_FONT = 7

# Slight text offset as a fraction of data span
ANNOTATE_X_FRAC = 0.01
ANNOTATE_Y_FRAC = 0.03

# Publication styling
STAR_COLORS = {
    "Epsilon_Eri": "tab:blue",
    "HD_40307": "tab:orange",
    "HD_85512": "tab:green",
    "HD_97658": "tab:red",
}

SCENARIO_MARKERS = {
    "1": "o",
    "2": "s",
    "3": "^",
    "4": "D",
    "5": "v",
    "6": "P",
    "7": "X",
    "8": "*",
    "9": "<",
    "10": ">",
}

QUADRANT_FILLS = {
    "cold_wet": "#cfe8ff",
    "warm_wet": "#d9f2d9",
    "cold_dry": "#eeeeee",
    "warm_dry": "#ffe0cc",
}


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


def save_figure(fig, out_path, dpi=SAVE_DPI, fmt=None, use_tight=True):
    ensure_dir(os.path.dirname(out_path))
    if fmt is None:
        fmt = os.path.splitext(out_path)[1].replace(".", "").lower()

    if use_tight:
        fig.savefig(out_path, format=fmt, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(out_path, format=fmt, dpi=dpi)

    print(f"Saved: {out_path}")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def to_numeric_array(series):
    return pd.to_numeric(series, errors="coerce").values


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


def pick_altitude_column(df):
    for candidate in ["ALT", "Z", "Alt", "alt", "z"]:
        if candidate in df.columns:
            return candidate
    return None


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


def axis_label_for_mode(mode):
    labels = {
        "surface_temperature": "Surface Temperature [K]",
        "surface_h2o": "Surface H2O Mixing Ratio",
        "surface_o2": "Surface O2 Mixing Ratio",
        "surface_o3": "Surface O3 Mixing Ratio",
        "surface_ch4": "Surface CH4 Mixing Ratio",
        "surface_co2": "Surface CO2 Mixing Ratio",
    }
    return labels.get(mode, mode)


def nice_mode_name(mode):
    names = {
        "surface_temperature": "Surface temperature",
        "surface_h2o": "Surface H2O",
        "surface_o2": "Surface O2",
        "surface_o3": "Surface O3",
        "surface_ch4": "Surface CH4",
        "surface_co2": "Surface CO2",
    }
    return names.get(mode, mode)


def scenario_marker(scenario):
    scenario = str(scenario)
    return SCENARIO_MARKERS.get(scenario, "o")


def star_color(star):
    return STAR_COLORS.get(star, "tab:blue")


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
def parse_out_species(file_path):
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
        return None

    header_idx = None
    for i in range(start_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("Z") or stripped.startswith("ALT"):
            header_idx = i
            break

    if header_idx is None:
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
        return None

    try:
        df = pd.read_csv(
            StringIO("\n".join(table_lines)),
            sep=r"\s+",
            engine="python",
        )
    except Exception as e:
        print(f"Failed to parse species table in {file_path}: {e}")
        return None

    return df


def parse_clima(file_path):
    if file_path is None or not os.path.exists(file_path):
        return None

    df = read_table(file_path)
    if df is None or df.empty:
        return None

    return df


# -----------------------------
# SURFACE EXTRACTION
# -----------------------------
def get_surface_row_index_from_altitude(df):
    alt_col = pick_altitude_column(df)
    if alt_col is None:
        return 0

    alt = pd.to_numeric(df[alt_col], errors="coerce")
    if np.isfinite(alt).sum() == 0:
        return 0

    return int(np.nanargmin(alt.values))


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

    idx = get_surface_row_index_from_altitude(clima_df)
    vals = pd.to_numeric(clima_df[temp_col], errors="coerce").values
    if idx >= len(vals):
        return np.nan
    return vals[idx]


def get_surface_species_value(species_df, species_name):
    if species_df is None or species_df.empty:
        return np.nan

    col = find_matching_column(species_df, species_name, exact=True)
    if col is None:
        return np.nan

    idx = get_surface_row_index_from_altitude(species_df)
    vals = pd.to_numeric(species_df[col], errors="coerce").values
    if idx >= len(vals):
        return np.nan
    return vals[idx]


def get_surface_clima_value(clima_df, target_name):
    if clima_df is None or clima_df.empty:
        return np.nan

    col = find_matching_column(clima_df, target_name, exact=False, contains=True)
    if col is None:
        return np.nan

    idx = get_surface_row_index_from_altitude(clima_df)
    vals = pd.to_numeric(clima_df[col], errors="coerce").values
    if idx >= len(vals):
        return np.nan
    return vals[idx]


def extract_metric_value(mode, species_df, clima_df):
    if mode == "surface_temperature":
        return get_surface_temperature(clima_df)

    if mode == "surface_h2o":
        val = get_surface_species_value(species_df, "H2O")
        if np.isfinite(val):
            return val
        return get_surface_clima_value(clima_df, "H2O")

    if mode == "surface_o2":
        return get_surface_species_value(species_df, "O2")

    if mode == "surface_o3":
        val = get_surface_species_value(species_df, "O3")
        if np.isfinite(val):
            return val
        return get_surface_clima_value(clima_df, "O3")

    if mode == "surface_ch4":
        return get_surface_species_value(species_df, "CH4")

    if mode == "surface_co2":
        return get_surface_species_value(species_df, "CO2")

    return np.nan


# -----------------------------
# BUILD MATRIX DATAFRAME
# -----------------------------
def build_habitability_dataframe(base_dir):
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

                    species_df = parse_out_species(files["out_file"]) if files["out_file"] else None
                    clima_df = parse_clima(files["clima_file"]) if files["clima_file"] else None

                    x_val = extract_metric_value(X_MODE, species_df, clima_df)
                    y_val = extract_metric_value(Y_MODE, species_df, clima_df)

                    rows.append(
                        {
                            "run": run,
                            "star": star_name,
                            "star_id": star_id,
                            "scale": str(scale),
                            "scenario": str(scenario),
                            "x": x_val,
                            "y": y_val,
                        }
                    )

    return pd.DataFrame(rows)


# -----------------------------
# QUADRANT CLASSIFICATION
# -----------------------------
def classify_quadrant(x, y, x_threshold, y_threshold):
    if not np.isfinite(x) or not np.isfinite(y):
        return "missing"

    if x >= x_threshold and y >= y_threshold:
        return "warm_wet"
    if x >= x_threshold and y < y_threshold:
        return "warm_dry"
    if x < x_threshold and y >= y_threshold:
        return "cold_wet"
    return "cold_dry"


def add_quadrant_labels(df):
    out = df.copy()
    out["quadrant"] = [
        classify_quadrant(x, y, X_THRESHOLD, Y_THRESHOLD)
        for x, y in zip(out["x"], out["y"])
    ]
    return out


# -----------------------------
# SUMMARY TABLES
# -----------------------------
def build_quadrant_count_table(df):
    counts = (
        df.groupby(["star", "quadrant"])
        .size()
        .reset_index(name="count")
        .sort_values(["star", "quadrant"])
    )
    return counts


def build_quadrant_pivot(df):
    counts = build_quadrant_count_table(df)
    if counts.empty:
        return pd.DataFrame()

    pivot = (
        counts.pivot(index="star", columns="quadrant", values="count")
        .fillna(0)
        .astype(int)
    )

    desired_cols = ["cold_dry", "cold_wet", "warm_dry", "warm_wet", "missing"]
    for col in desired_cols:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot = pivot[desired_cols]
    pivot["total"] = pivot.sum(axis=1)
    return pivot.reset_index()


# -----------------------------
# MATRIX PLOTTING
# -----------------------------
def add_quadrant_background(ax, xlim, ylim, x_threshold, y_threshold, log_y=False):
    x0, x1 = xlim
    y0, y1 = ylim

    rect_specs = [
        ("cold_wet", x0, y_threshold, x_threshold - x0, y1 - y_threshold),
        ("warm_wet", x_threshold, y_threshold, x1 - x_threshold, y1 - y_threshold),
        ("cold_dry", x0, y0, x_threshold - x0, y_threshold - y0),
        ("warm_dry", x_threshold, y0, x1 - x_threshold, y_threshold - y0),
    ]

    for quadrant, x, y, w, h in rect_specs:
        if not np.isfinite(w) or not np.isfinite(h) or w <= 0 or h <= 0:
            continue
        ax.axvspan(
            x,
            x + w,
            ymin=(np.log10(y) - np.log10(y0)) / (np.log10(y1) - np.log10(y0)) if (log_y and y0 > 0 and y1 > 0 and y > 0) else 0,
            ymax=(np.log10(y + h) - np.log10(y0)) / (np.log10(y1) - np.log10(y0)) if (log_y and y0 > 0 and y1 > 0 and (y + h) > 0) else 1,
            color=QUADRANT_FILLS[quadrant],
            alpha=0.45,
            zorder=0,
        )

    # Redraw a cleaner approach on top using fill_between in data coords
    if log_y:
        # In log space, patch objects in data coords are acceptable
        ax.fill_between([x0, x_threshold], y_threshold, y1, color=QUADRANT_FILLS["cold_wet"], alpha=0.45, zorder=0)
        ax.fill_between([x_threshold, x1], y_threshold, y1, color=QUADRANT_FILLS["warm_wet"], alpha=0.45, zorder=0)
        ax.fill_between([x0, x_threshold], y0, y_threshold, color=QUADRANT_FILLS["cold_dry"], alpha=0.45, zorder=0)
        ax.fill_between([x_threshold, x1], y0, y_threshold, color=QUADRANT_FILLS["warm_dry"], alpha=0.45, zorder=0)
    else:
        ax.fill_between([x0, x_threshold], y_threshold, y1, color=QUADRANT_FILLS["cold_wet"], alpha=0.45, zorder=0)
        ax.fill_between([x_threshold, x1], y_threshold, y1, color=QUADRANT_FILLS["warm_wet"], alpha=0.45, zorder=0)
        ax.fill_between([x0, x_threshold], y0, y_threshold, color=QUADRANT_FILLS["cold_dry"], alpha=0.45, zorder=0)
        ax.fill_between([x_threshold, x1], y0, y_threshold, color=QUADRANT_FILLS["warm_dry"], alpha=0.45, zorder=0)


def add_quadrant_text(ax, xlim, ylim, log_y=False):
    x0, x1 = xlim
    y0, y1 = ylim

    x_mid_left = 0.5 * (x0 + X_THRESHOLD)
    x_mid_right = 0.5 * (X_THRESHOLD + x1)

    if log_y and y0 > 0 and Y_THRESHOLD > 0 and y1 > 0:
        y_mid_low = 10 ** (0.5 * (np.log10(y0) + np.log10(Y_THRESHOLD)))
        y_mid_high = 10 ** (0.5 * (np.log10(Y_THRESHOLD) + np.log10(y1)))
    else:
        y_mid_low = 0.5 * (y0 + Y_THRESHOLD)
        y_mid_high = 0.5 * (Y_THRESHOLD + y1)

    box = dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none")
    ax.text(x_mid_right, y_mid_high, "Warm / Wet", ha="center", va="center", fontsize=10, bbox=box)
    ax.text(x_mid_right, y_mid_low, "Warm / Dry", ha="center", va="center", fontsize=10, bbox=box)
    ax.text(x_mid_left, y_mid_high, "Cold / Wet", ha="center", va="center", fontsize=10, bbox=box)
    ax.text(x_mid_left, y_mid_low, "Cold / Dry", ha="center", va="center", fontsize=10, bbox=box)


def make_point_label(row):
    return f'{row["run"]}, {row["scale"]}, s{row["scenario"]}'


def compute_plot_limits(valid_df):
    x = valid_df["x"].to_numpy(dtype=float)
    y = valid_df["y"].to_numpy(dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    x0, x1 = np.nanmin(x), np.nanmax(x)
    if x0 == x1:
        x0 -= 1
        x1 += 1
    xpad = 0.06 * (x1 - x0)
    xlim = (x0 - xpad, x1 + xpad)

    if USE_LOG_Y:
        y = y[y > 0]
        if y.size == 0:
            return xlim, (0, 1)
        y0, y1 = np.nanmin(y), np.nanmax(y)
        if y0 == y1:
            y0 /= 1.5
            y1 *= 1.5
        ylim = (y0 / 1.4, y1 * 1.4)
    else:
        y0, y1 = np.nanmin(y), np.nanmax(y)
        if y0 == y1:
            y0 -= 1
            y1 += 1
        ypad = 0.06 * (y1 - y0)
        ylim = (y0 - ypad, y1 + ypad)

    return xlim, ylim


def draw_habitability_matrix(ax, df, title):
    valid = df[np.isfinite(df["x"]) & np.isfinite(df["y"])].copy()
    if USE_LOG_Y:
        valid = valid[valid["y"] > 0].copy()

    if valid.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(axis_label_for_mode(X_MODE))
        ax.set_ylabel(axis_label_for_mode(Y_MODE))
        return

    stars = sorted(valid["star"].unique())
    scenarios = sorted(valid["scenario"].unique(), key=lambda x: int(x))

    if USE_LOG_Y:
        ax.set_yscale("log")

    xlim, ylim = compute_plot_limits(valid)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    add_quadrant_background(
        ax,
        xlim,
        ylim,
        X_THRESHOLD,
        Y_THRESHOLD,
        log_y=(ax.get_yscale() == "log"),
    )

    ax.axvline(X_THRESHOLD, linestyle="--", linewidth=1.2, color="black", zorder=2)
    ax.axhline(Y_THRESHOLD, linestyle="--", linewidth=1.2, color="black", zorder=2)

    for star in stars:
        sub_star = valid[valid["star"] == star]
        color = star_color(star)

        for scenario in scenarios:
            sub = sub_star[sub_star["scenario"] == scenario]
            if sub.empty:
                continue

            ax.scatter(
                sub["x"],
                sub["y"],
                marker=scenario_marker(scenario),
                s=65,
                alpha=0.92,
                edgecolor="black",
                linewidth=0.5,
                color=color,
                zorder=4,
            )

            if ANNOTATE_POINTS:
                dx = (xlim[1] - xlim[0]) * ANNOTATE_X_FRAC
                for _, row in sub.iterrows():
                    if ax.get_yscale() == "log" and row["y"] > 0:
                        y2 = row["y"] * (10 ** ANNOTATE_Y_FRAC)
                    else:
                        dy = (ylim[1] - ylim[0]) * ANNOTATE_Y_FRAC
                        y2 = row["y"] + dy

                    ax.annotate(
                        make_point_label(row),
                        (row["x"] + dx, y2),
                        fontsize=ANNOTATION_FONT,
                        zorder=5,
                    )

    ax.set_xlabel(axis_label_for_mode(X_MODE))
    ax.set_ylabel(axis_label_for_mode(Y_MODE))
    ax.set_title(title)
    ax.grid(True, alpha=0.22, zorder=1)

    add_quadrant_text(ax, xlim, ylim, log_y=(ax.get_yscale() == "log"))


def add_dual_legend(ax, df):
    valid = df[np.isfinite(df["x"]) & np.isfinite(df["y"])].copy()
    if USE_LOG_Y:
        valid = valid[valid["y"] > 0].copy()

    if valid.empty:
        return

    star_handles = []
    for star in sorted(valid["star"].unique()):
        star_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=star_color(star),
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=8,
                label=star,
            )
        )

    scenario_handles = []
    for scenario in sorted(valid["scenario"].unique(), key=lambda x: int(x)):
        scenario_handles.append(
            Line2D(
                [0], [0],
                marker=scenario_marker(scenario),
                linestyle="",
                color="black",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=8,
                label=f"Scenario {scenario}",
            )
        )

    legend1 = ax.legend(
        handles=star_handles,
        title="Star",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.00),
        frameon=True,
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=scenario_handles,
        title="Scenario",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.52),
        frameon=True,
    )


def plot_habitability_matrix(df, title, out_path, publication=False):
    figsize = PUB_MATRIX_FIGSIZE if publication else FIGSIZE
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=publication)

    draw_habitability_matrix(ax, df, title)
    add_dual_legend(ax, df)

    if publication:
        save_figure(fig, out_path, dpi=PUB_DPI, fmt=os.path.splitext(out_path)[1].replace(".", ""), use_tight=False)
    else:
        save_figure(fig, out_path, dpi=SAVE_DPI)


def plot_all_matrices(df, out_dir):
    ensure_dir(out_dir)

    plot_habitability_matrix(
        df,
        title="Habitability Matrix | All Stars and Scenarios",
        out_path=os.path.join(out_dir, "habitability_matrix_all.jpeg"),
        publication=False,
    )

    for star in sorted(df["star"].dropna().unique()):
        sub = df[df["star"] == star].copy()
        plot_habitability_matrix(
            sub,
            title=f"Habitability Matrix | {star}",
            out_path=os.path.join(
                out_dir,
                f"habitability_matrix_{sanitize_for_filename(star)}.jpeg",
            ),
            publication=False,
        )


def save_publication_matrices(df, out_dir):
    pub_dir = os.path.join(out_dir, "publication")
    ensure_dir(pub_dir)

    plot_habitability_matrix(
        df,
        title="Habitability Matrix",
        out_path=os.path.join(pub_dir, "habitability_matrix_all.png"),
        publication=True,
    )

    for star in sorted(df["star"].dropna().unique()):
        sub = df[df["star"] == star].copy()
        plot_habitability_matrix(
            sub,
            title=f"Habitability Matrix | {star}",
            out_path=os.path.join(pub_dir, f"habitability_matrix_{sanitize_for_filename(star)}.png"),
            publication=True,
        )


# -----------------------------
# PUBLICATION SUMMARY FIGURE
# -----------------------------
def add_summary_text_panel(ax, df):
    ax.axis("off")

    valid = df[np.isfinite(df["x"]) & np.isfinite(df["y"])].copy()
    if USE_LOG_Y:
        valid = valid[valid["y"] > 0].copy()

    total_cases = len(df)
    valid_cases = len(valid)
    missing_cases = int((df["quadrant"] == "missing").sum())

    pivot = build_quadrant_pivot(df)

    lines = [
        "Summary",
        "",
        f"X metric: {nice_mode_name(X_MODE)}",
        f"Y metric: {nice_mode_name(Y_MODE)}",
        f"X threshold: {X_THRESHOLD:.2f}",
        f"Y threshold: {Y_THRESHOLD:.2e}",
        "",
        f"Total cases: {total_cases}",
        f"Valid plotted cases: {valid_cases}",
        f"Missing / unclassified: {missing_cases}",
        "",
    ]

    if not pivot.empty:
        lines.append("Per-star counts:")
        lines.append("")
        for _, row in pivot.iterrows():
            lines.append(
                f"{row['star']}: "
                f"CD={row['cold_dry']}, "
                f"CW={row['cold_wet']}, "
                f"WD={row['warm_dry']}, "
                f"WW={row['warm_wet']}, "
                f"M={row['missing']}"
            )

    ax.text(
        0.0, 1.0,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        transform=ax.transAxes,
    )


def add_quadrant_table(ax, df):
    ax.axis("off")
    pivot = build_quadrant_pivot(df)

    if pivot.empty:
        ax.text(0.5, 0.5, "No summary data", ha="center", va="center", transform=ax.transAxes)
        return

    col_labels = ["Star", "Cold/Dry", "Cold/Wet", "Warm/Dry", "Warm/Wet", "Missing", "Total"]
    cell_text = []

    for _, row in pivot.iterrows():
        cell_text.append([
            row["star"],
            int(row["cold_dry"]),
            int(row["cold_wet"]),
            int(row["warm_dry"]),
            int(row["warm_wet"]),
            int(row["missing"]),
            int(row["total"]),
        ])

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)
    ax.set_title("Quadrant counts by star", fontsize=11, pad=10)


def build_publication_summary_figure(df):
    fig = plt.figure(figsize=PUB_SUMMARY_FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.2], width_ratios=[2.2, 1.0])

    ax_matrix = fig.add_subplot(gs[0, :])
    draw_habitability_matrix(ax_matrix, df, "Habitability Matrix")
    add_dual_legend(ax_matrix, df)

    ax_table = fig.add_subplot(gs[1, 0])
    add_quadrant_table(ax_table, df)

    ax_text = fig.add_subplot(gs[1, 1])
    add_summary_text_panel(ax_text, df)

    fig.suptitle("Publication Summary | Habitability Matrix", fontsize=15)
    return fig


def save_publication_summary(df, out_dir):
    pub_dir = os.path.join(out_dir, "publication")
    ensure_dir(pub_dir)

    fig = build_publication_summary_figure(df)
    save_figure(
        fig,
        os.path.join(pub_dir, "habitability_publication_summary.png"),
        dpi=PUB_DPI,
        fmt="png",
        use_tight=False,
    )

    pdf_path = os.path.join(pub_dir, "habitability_publication_summary.pdf")
    fig = build_publication_summary_figure(df)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, dpi=PUB_DPI)
        plt.close(fig)
    print(f"Saved: {pdf_path}")


# -----------------------------
# SAVE TABLES
# -----------------------------
def save_summary_outputs(df, out_dir):
    ensure_dir(out_dir)

    summary_csv = os.path.join(out_dir, "habitability_matrix_summary.csv")
    df.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    counts = build_quadrant_count_table(df)
    counts_csv = os.path.join(out_dir, "habitability_matrix_quadrant_counts.csv")
    counts.to_csv(counts_csv, index=False)
    print(f"Saved: {counts_csv}")

    pivot = build_quadrant_pivot(df)
    pivot_csv = os.path.join(out_dir, "habitability_matrix_quadrant_pivot.csv")
    pivot.to_csv(pivot_csv, index=False)
    print(f"Saved: {pivot_csv}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    ensure_dir(OUT_DIR)

    df = build_habitability_dataframe(BASE_DIR)
    df = add_quadrant_labels(df)

    print("\nExtracted points:")
    print(df)

    save_summary_outputs(df, OUT_DIR)
    plot_all_matrices(df, OUT_DIR)

    # Publication-ready outputs
    save_publication_matrices(df, OUT_DIR)
    save_publication_summary(df, OUT_DIR)

    print("\nDone.")
    print(f"Habitability matrix outputs written to: {OUT_DIR}")
    print(f"Publication outputs written to: {os.path.join(OUT_DIR, 'publication')}")
