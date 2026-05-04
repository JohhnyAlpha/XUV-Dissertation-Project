import os
import re
from io import StringIO
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

# Physical ages represented by each Run directory
RUN_AGES = {
    "Run_1": "2.7 Gy",
    "Run_2": "0.5 Gy",
    "Run_3": "9.9 Gy",
}

# File suffix index is treated as iteration, not scenario
ITERATION_LABELS = {
    "1": "Iteration 1",
    "2": "Iteration 2",
    "3": "Iteration 3",
}

# Expected FSCALE values. Used for marker-size legend.
FSCALE_VALUES = ["0.75", "1.0", "1.5"]

FIGSIZE = (10, 7)
SAVE_DPI = 220
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

# Label each point with run/scale/iteration
ANNOTATE_POINTS = False
ANNOTATION_FONT = 7

# Slight text offset as a fraction of data span
ANNOTATE_X_FRAC = 0.01
ANNOTATE_Y_FRAC = 0.03


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


# -----------------------------
# FILE DISCOVERY
# -----------------------------
def build_file_index(scale_dir):
    photochem_dir = os.path.join(scale_dir, "PHOTOCHEM_OUTPUT")
    clima_dir = os.path.join(scale_dir, "CLIMA_OUTPUT")

    iteration_files = defaultdict(
        lambda: {
            "out_file": None,
            "clima_file": None,
        }
    )

    # Filename structure: Run_N_FSCALE_ITERATION_suffix
    pattern = re.compile(r"^(Run_\d+)_([0-9]+(?:\.[0-9]+)?)_(\d+)_(.+)$")

    if os.path.isdir(photochem_dir):
        for fname in os.listdir(photochem_dir):
            match = pattern.match(fname)
            if not match:
                continue

            _, _, iteration, suffix = match.groups()
            if suffix == "out.out":
                iteration_files[iteration]["out_file"] = os.path.join(photochem_dir, fname)

    if os.path.isdir(clima_dir):
        for fname in os.listdir(clima_dir):
            match = pattern.match(fname)
            if not match:
                continue

            _, _, iteration, suffix = match.groups()
            if suffix in ("clima_last.tab", "clima_last.out"):
                iteration_files[iteration]["clima_file"] = os.path.join(clima_dir, fname)

    return dict(iteration_files)

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

                for iteration, files in sorted(indexed.items(), key=lambda x: int(x[0])):
                    print(f"\nLoading: {run} ({RUN_AGES.get(run, run)}) | {star_name} | FSCALE {scale} | iteration {iteration}")
                    print("  out_file   :", files["out_file"])
                    print("  clima_file :", files["clima_file"])

                    species_df = parse_out_species(files["out_file"]) if files["out_file"] else None
                    clima_df = parse_clima(files["clima_file"]) if files["clima_file"] else None

                    x_val = extract_metric_value(X_MODE, species_df, clima_df)
                    y_val = extract_metric_value(Y_MODE, species_df, clima_df)

                    rows.append(
                        {
                            "run": run,
                            "run_age": RUN_AGES.get(run, run),
                            "star": star_name,
                            "star_id": star_id,
                            "scale": str(scale),
                            "iteration": str(iteration),
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
# MATRIX PLOTTING
# -----------------------------
def add_quadrant_background(ax, xlim, ylim, x_threshold, y_threshold, log_y=False):
    x0, x1 = xlim
    y0, y1 = ylim

    if log_y:
        if y0 <= 0 or y_threshold <= 0 or y1 <= 0:
            return

        # Use standard rectangles in data space; good enough for log axes
        rects = [
            (x0, y_threshold, x_threshold - x0, y1 - y_threshold, "Cold / Wet"),
            (x_threshold, y_threshold, x1 - x_threshold, y1 - y_threshold, "Warm / Wet"),
            (x0, y0, x_threshold - x0, y_threshold - y0, "Cold / Dry"),
            (x_threshold, y0, x1 - x_threshold, y_threshold - y0, "Warm / Dry"),
        ]
    else:
        rects = [
            (x0, y_threshold, x_threshold - x0, y1 - y_threshold, "Cold / Wet"),
            (x_threshold, y_threshold, x1 - x_threshold, y1 - y_threshold, "Warm / Wet"),
            (x0, y0, x_threshold - x0, y_threshold - y0, "Cold / Dry"),
            (x_threshold, y0, x1 - x_threshold, y_threshold - y0, "Warm / Dry"),
        ]

    for x, y, w, h, label in rects:
        if w <= 0 or h <= 0:
            continue
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                alpha=0.10,
                zorder=0,
            )
        )


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

    ax.text(x_mid_right, y_mid_high, "Warm / Wet", ha="center", va="center", fontsize=10)
    ax.text(x_mid_right, y_mid_low, "Warm / Dry", ha="center", va="center", fontsize=10)
    ax.text(x_mid_left, y_mid_high, "Cold / Wet", ha="center", va="center", fontsize=10)
    ax.text(x_mid_left, y_mid_low, "Cold / Dry", ha="center", va="center", fontsize=10)


def make_point_label(row):
    iteration_label = ITERATION_LABELS.get(str(row["iteration"]), f'Iteration {row["iteration"]}')
    return f'{RUN_AGES.get(row["run"], row["run"])}, F={row["scale"]}, {iteration_label}'


def normalized_fscale(value):
    """Return a consistent string for FSCALE values such as 1, 1.0, 1.50."""
    try:
        return f"{float(value):.2f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def get_visual_mappings(valid):
    stars = sorted(valid["star"].unique())
    iterations = sorted(valid["iteration"].unique(), key=lambda x: int(x))
    runs = sorted(valid["run"].unique(), key=lambda x: int(str(x).split("_")[-1]))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    star_to_color = {star: color_cycle[i % len(color_cycle)] for i, star in enumerate(stars)}

    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    iteration_to_marker = {
        iteration: marker_cycle[i % len(marker_cycle)]
        for i, iteration in enumerate(iterations)
    }

    # FSCALE is encoded by marker size.
    fscales_present = sorted(
        {normalized_fscale(v) for v in valid["scale"].dropna().unique()},
        key=numeric_sort_key,
    )
    preferred = [normalized_fscale(v) for v in FSCALE_VALUES]
    fscales = [v for v in preferred if v in fscales_present] + [
        v for v in fscales_present if v not in preferred
    ]
    size_levels = [55, 90, 135, 180, 230]
    fscale_to_size = {
        fscale: size_levels[i % len(size_levels)]
        for i, fscale in enumerate(fscales)
    }

    # Run age is encoded by marker edge width.
    linewidth_levels = [0.7, 1.6, 2.6, 3.5]
    run_to_linewidth = {
        run: linewidth_levels[i % len(linewidth_levels)]
        for i, run in enumerate(runs)
    }

    return star_to_color, iteration_to_marker, fscale_to_size, run_to_linewidth


def add_publication_legends(ax, valid, star_to_color, iteration_to_marker, fscale_to_size, run_to_linewidth):
    """Add separate, readable legends for Star, Iteration, FSCALE, and Run age."""
    stars = sorted(valid["star"].unique())
    iterations = sorted(valid["iteration"].unique(), key=lambda x: int(x))
    runs = sorted(valid["run"].unique(), key=lambda x: int(str(x).split("_")[-1]))
    fscales = sorted(fscale_to_size.keys(), key=numeric_sort_key)

    star_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=star_to_color[star],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=8,
            label=star,
        )
        for star in stars
    ]

    iteration_handles = [
        plt.Line2D(
            [0], [0],
            marker=iteration_to_marker[iteration],
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=8,
            label=ITERATION_LABELS.get(iteration, f"Iteration {iteration}"),
        )
        for iteration in iterations
    ]

    fscale_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=(fscale_to_size[fscale] ** 0.5),
            label=f"F={fscale}",
        )
        for fscale in fscales
    ]

    run_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=run_to_linewidth[run],
            markersize=9,
            label=f"{run}: {RUN_AGES.get(run, run)}",
        )
        for run in runs
    ]

    legend1 = ax.legend(
        handles=star_handles,
        title="Star",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
    )
    ax.add_artist(legend1)

    legend2 = ax.legend(
        handles=iteration_handles,
        title="Iteration",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.72),
        borderaxespad=0.0,
    )
    ax.add_artist(legend2)

    legend3 = ax.legend(
        handles=fscale_handles,
        title="FSCALE",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.50),
        borderaxespad=0.0,
    )
    ax.add_artist(legend3)

    ax.legend(
        handles=run_handles,
        title="Run age",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.28),
        borderaxespad=0.0,
    )


def plot_habitability_matrix(df, title, out_path):
    valid = df[np.isfinite(df["x"]) & np.isfinite(df["y"])].copy()
    if valid.empty:
        print(f"No valid data for {title}")
        return

    # Backward compatibility for older summary CSVs or partially processed dataframes.
    if "iteration" not in valid.columns and "scenario" in valid.columns:
        valid["iteration"] = valid["scenario"].astype(str)

    star_to_color, iteration_to_marker, fscale_to_size, run_to_linewidth = get_visual_mappings(valid)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Decide scale first so the text/background positions work well.
    if USE_LOG_Y:
        positive = valid["y"] > 0
        if positive.any():
            ax.set_yscale("log")

    # First pass, to establish data limits before adding quadrant backgrounds.
    for _, row in valid.iterrows():
        fscale_key = normalized_fscale(row["scale"])
        ax.scatter(
            row["x"],
            row["y"],
            s=fscale_to_size.get(fscale_key, 90),
            marker=iteration_to_marker[row["iteration"]],
            facecolor=star_to_color[row["star"]],
            edgecolor="black",
            linewidth=run_to_linewidth.get(row["run"], 1.0),
            alpha=0.85,
            zorder=3,
        )

    ax.relim()
    ax.autoscale_view()

    # Expand slightly so markers and quadrant labels fit better.
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xspan = xlim[1] - xlim[0]
    xlim = (xlim[0] - 0.05 * xspan, xlim[1] + 0.05 * xspan)

    if ax.get_yscale() == "log":
        if ylim[0] > 0 and ylim[1] > 0:
            ylim = (ylim[0] / 1.2, ylim[1] * 1.2)
    else:
        yspan = ylim[1] - ylim[0]
        ylim = (ylim[0] - 0.05 * yspan, ylim[1] + 0.05 * yspan)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Colored quadrant background.
    add_quadrant_background(
        ax,
        ax.get_xlim(),
        ax.get_ylim(),
        X_THRESHOLD,
        Y_THRESHOLD,
        log_y=(ax.get_yscale() == "log"),
    )

    # Threshold lines.
    ax.axvline(X_THRESHOLD, linestyle="--", linewidth=1.2, color="black", zorder=2)
    ax.axhline(Y_THRESHOLD, linestyle="--", linewidth=1.2, color="black", zorder=2)

    # Re-plot points above backgrounds/lines for visibility.
    for _, row in valid.iterrows():
        fscale_key = normalized_fscale(row["scale"])
        ax.scatter(
            row["x"],
            row["y"],
            s=fscale_to_size.get(fscale_key, 90),
            marker=iteration_to_marker[row["iteration"]],
            facecolor=star_to_color[row["star"]],
            edgecolor="black",
            linewidth=run_to_linewidth.get(row["run"], 1.0),
            alpha=0.9,
            zorder=4,
        )

        if ANNOTATE_POINTS:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            dx = (xlim[1] - xlim[0]) * ANNOTATE_X_FRAC

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
    ax.grid(True, alpha=0.25, zorder=1)

    add_quadrant_text(ax, ax.get_xlim(), ax.get_ylim(), log_y=(ax.get_yscale() == "log"))
    add_publication_legends(ax, valid, star_to_color, iteration_to_marker, fscale_to_size, run_to_linewidth)

    save_figure(fig, out_path)

def plot_all_matrices(df, out_dir):
    ensure_dir(out_dir)

    plot_habitability_matrix(
        df,
        title="Habitability Matrix | All Stars and Iterations",
        out_path=os.path.join(out_dir, "habitability_matrix_all.jpeg"),
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
        )


# -----------------------------
# SAVE TABLES
# -----------------------------
def save_summary_outputs(df, out_dir):
    ensure_dir(out_dir)

    summary_csv = os.path.join(out_dir, "habitability_matrix_summary.csv")
    df.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    counts = (
        df.groupby(["star", "quadrant"])
        .size()
        .reset_index(name="count")
        .sort_values(["star", "quadrant"])
    )
    counts_csv = os.path.join(out_dir, "habitability_matrix_quadrant_counts.csv")
    counts.to_csv(counts_csv, index=False)
    print(f"Saved: {counts_csv}")


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

    print("\nDone.")
    print(f"Habitability matrix outputs written to: {OUT_DIR}")
