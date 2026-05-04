import os
import re
from io import StringIO
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Optional smoothing
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOWESS = True
except Exception:
    HAS_LOWESS = False


# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.expanduser("~/atmos/OutputStorage")
OUT_DIR = os.path.expanduser("~/atmos/BA/Plots/publication_figures")

STARS = {
    "18": "Epsilon_Eri",
    "24": "HD_40307",
    "25": "HD_85512",
    "26": "HD_97658",
}

SAVE_DPI = 400
EXPORT_PDF = True
EXPORT_PNG = True

FIGSIZE = (12.5, 10.8)
ESCAPE_FIGSIZE = (12.5, 10.8)

FREEZING_TEMPERATURE = 273.15
DESICCATION_THRESHOLD = 1e-5
ABIOTIC_O2_THRESHOLD = 1e-3
ABIOTIC_O3_THRESHOLD = 1e-8

ANNOTATE_POINTS = False
ANNOTATION_FONT = 7

LOWESS_FRAC = 0.6
SHOW_LOWESS_IN_SUMMARY = False

# -----------------------------
# MATPLOTLIB STYLE
# -----------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": SAVE_DPI,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titlepad": 8,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.45,
    "grid.alpha": 0.22,
    "lines.linewidth": 1.4,
})


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


def star_name(star_id):
    return STARS.get(str(star_id), str(star_id))


def discover_runs(base_dir):
    runs = []
    if not os.path.isdir(base_dir):
        return runs

    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and re.fullmatch(r"Run_\d+", name):
            runs.append(name)

    return sorted(runs, key=lambda x: int(x.split("_")[1]))


def save_figure(fig, out_base):
    ensure_dir(os.path.dirname(out_base))

    if EXPORT_PNG:
        png_path = f"{out_base}.png"
        fig.savefig(png_path, bbox_inches="tight")
        print(f"Saved: {png_path}")

    if EXPORT_PDF:
        pdf_path = f"{out_base}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {pdf_path}")

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


def finite_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def get_surface_index(df):
    alt_col = pick_altitude_column(df)
    if alt_col is None:
        return 0

    alt = pd.to_numeric(df[alt_col], errors="coerce").values
    finite = np.isfinite(alt)
    if not finite.any():
        return 0

    return int(np.nanargmin(alt))


def get_top_index(df):
    alt_col = pick_altitude_column(df)
    if alt_col is None:
        return len(df) - 1

    alt = pd.to_numeric(df[alt_col], errors="coerce").values
    finite = np.isfinite(alt)
    if not finite.any():
        return len(df) - 1

    return int(np.nanargmax(alt))


def maybe_set_log_axis(ax, axis, values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    positive = vals[vals > 0]
    if positive.size == 0:
        return

    ratio = np.nanmax(positive) / np.nanmin(positive)
    if ratio >= 100:
        if axis == "x":
            ax.set_xscale("log")
        elif axis == "y":
            ax.set_yscale("log")


def panel_label(ax, label):
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12,
        fontweight="bold",
    )


def add_threshold_label(ax, x=None, y=None, text=""):
    if x is not None:
        ax.axvline(x, linestyle="--", linewidth=1.0, color="black")
        ymin, ymax = ax.get_ylim()
        yy = ymax / 1.15 if ax.get_yscale() == "log" and ymax > 0 else ymax - 0.05 * (ymax - ymin)
        ax.text(x, yy, text, rotation=90, va="top", ha="right", fontsize=9)

    if y is not None:
        ax.axhline(y, linestyle="--", linewidth=1.0, color="black")
        xmin, xmax = ax.get_xlim()
        xx = xmax / 1.1 if ax.get_xscale() == "log" and xmax > 0 else xmax - 0.02 * (xmax - xmin)
        ax.text(xx, y, text, va="bottom", ha="right", fontsize=9)


def add_lowess(ax, x, y, frac=LOWESS_FRAC):
    if not HAS_LOWESS:
        return

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return

    xs = x[mask]
    ys = y[mask]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    try:
        smoothed = lowess(ys, xs, frac=frac, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", linewidth=2.0, alpha=0.75, zorder=2)
    except Exception:
        pass


def detect_spectral_units(wavelength):
    w = np.asarray(wavelength, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]

    if w.size == 0:
        return "unknown"

    med = np.nanmedian(w)
    if med > 50:
        return "nm"
    return "um"


def get_band_edges(units):
    if units == "nm":
        return 400.0, 700.0
    if units == "um":
        return 0.4, 0.7
    return None, None


def safe_band_mask(wavelength, lo=None, hi=None):
    mask = np.isfinite(wavelength)
    if lo is not None:
        mask &= wavelength >= lo
    if hi is not None:
        mask &= wavelength < hi
    return mask


def integrate_curve(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    if hasattr(np, "trapz"):
        return np.trapz(y, x)
    return np.nan


def integrate_band(x, y, mask):
    xm = np.asarray(x[mask], dtype=float)
    ym = np.asarray(y[mask], dtype=float)

    if xm.size < 2:
        return np.nan

    order = np.argsort(xm)
    xm = xm[order]
    ym = ym[order]

    return integrate_curve(ym, xm)


# -----------------------------
# FILE DISCOVERY
# -----------------------------
def build_index(scale_dir):
    photochem_dir = os.path.join(scale_dir, "PHOTOCHEM_OUTPUT")
    clima_dir = os.path.join(scale_dir, "CLIMA_OUTPUT")

    out = defaultdict(lambda: {
        "out": None,
        "clima": None,
        "ftir": None,
        "ftso": None,
    })

    pattern = re.compile(r"^(Run_\d+)_([0-9]+(?:\.[0-9]+)?)_(\d+)_(.+)$")

    if os.path.isdir(photochem_dir):
        for fname in os.listdir(photochem_dir):
            match = pattern.match(fname)
            if not match:
                continue

            _, _, scenario, suffix = match.groups()
            full = os.path.join(photochem_dir, fname)

            if suffix == "out.out":
                out[scenario]["out"] = full

    if os.path.isdir(clima_dir):
        for fname in os.listdir(clima_dir):
            match = pattern.match(fname)
            if not match:
                continue

            _, _, scenario, suffix = match.groups()
            full = os.path.join(clima_dir, fname)

            if suffix in ("clima_last.tab", "clima_last.out"):
                out[scenario]["clima"] = full
            elif suffix == "FTIR.dat":
                out[scenario]["ftir"] = full
            elif suffix == "FTSO.dat":
                out[scenario]["ftso"] = full

    return dict(out)


# -----------------------------
# PARSERS
# -----------------------------
def parse_species(file_path):
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
        if lines[i].strip().startswith(("Z", "ALT")):
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
        return pd.read_csv(StringIO("\n".join(table_lines)), sep=r"\s+", engine="python")
    except Exception as e:
        print(f"Failed to parse species table in {file_path}: {e}")
        return None


def parse_clima(file_path):
    if file_path is None or not os.path.exists(file_path):
        return None

    df = read_table(file_path)
    if df is None or df.empty:
        return None
    return df


def parse_two_column_spectrum(file_path):
    if file_path is None or not os.path.exists(file_path):
        return None

    try:
        arr = np.loadtxt(file_path)
    except Exception as e:
        print(f"Failed to load spectrum {file_path}: {e}")
        return None

    if arr.ndim == 1:
        if arr.size >= 2:
            arr = arr.reshape(1, -1)
        else:
            return None

    if arr.ndim != 2 or arr.shape[1] < 2:
        return None

    wavelength = np.asarray(arr[:, 0], dtype=float)
    flux = np.asarray(arr[:, 1], dtype=float)

    mask = np.isfinite(wavelength) & np.isfinite(flux)
    if not np.any(mask):
        return None

    wavelength = wavelength[mask]
    flux = flux[mask]

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    flux = flux[order]

    units = detect_spectral_units(wavelength)
    uv_max, vis_max = get_band_edges(units)

    if units == "unknown":
        uv_mask = np.zeros_like(wavelength, dtype=bool)
        vis_mask = np.zeros_like(wavelength, dtype=bool)
        ir_mask = np.zeros_like(wavelength, dtype=bool)
    else:
        uv_mask = safe_band_mask(wavelength, None, uv_max)
        vis_mask = safe_band_mask(wavelength, uv_max, vis_max)
        ir_mask = safe_band_mask(wavelength, vis_max, None)

    all_mask = np.isfinite(wavelength) & np.isfinite(flux)

    return {
        "wavelength": wavelength,
        "flux": flux,
        "units": units,
        "uv_max": uv_max,
        "vis_max": vis_max,
        "integrated": {
            "UV": integrate_band(wavelength, flux, uv_mask),
            "VIS": integrate_band(wavelength, flux, vis_mask),
            "IR": integrate_band(wavelength, flux, ir_mask),
            "TOTAL": integrate_band(wavelength, flux, all_mask),
        },
    }


# -----------------------------
# VALUE EXTRACTION
# -----------------------------
def get_surface(df, column_name):
    if df is None or df.empty or column_name not in df.columns:
        return np.nan

    idx = get_surface_index(df)
    vals = pd.to_numeric(df[column_name], errors="coerce").values

    if idx < 0 or idx >= len(vals):
        return np.nan
    return vals[idx]


def get_top(df, column_name):
    if df is None or df.empty or column_name not in df.columns:
        return np.nan

    idx = get_top_index(df)
    vals = pd.to_numeric(df[column_name], errors="coerce").values

    if idx < 0 or idx >= len(vals):
        return np.nan
    return vals[idx]


def get_surface_species(species_df, name):
    if species_df is None:
        return np.nan

    col = find_matching_column(species_df, name, exact=True)
    if col is None:
        return np.nan
    return get_surface(species_df, col)


def get_top_species(species_df, name):
    if species_df is None:
        return np.nan

    col = find_matching_column(species_df, name, exact=True)
    if col is None:
        return np.nan
    return get_top(species_df, col)


# -----------------------------
# BUILD DATAFRAME + SPECTRAL LOOKUP
# -----------------------------
def build_dataframe(base_dir):
    rows = []
    spectra = defaultdict(list)

    for run in discover_runs(base_dir):
        run_dir = os.path.join(base_dir, run)

        for star_id in sorted(os.listdir(run_dir), key=numeric_sort_key):
            star_dir = os.path.join(run_dir, star_id)
            if not os.path.isdir(star_dir):
                continue

            star_label = star_name(star_id)

            for scale in sorted(os.listdir(star_dir), key=numeric_sort_key):
                scale_dir = os.path.join(star_dir, scale)
                if not os.path.isdir(scale_dir):
                    continue

                indexed = build_index(scale_dir)

                for scenario, files in sorted(indexed.items(), key=lambda x: int(x[0])):
                    species_df = parse_species(files["out"]) if files["out"] else None
                    clima_df = parse_clima(files["clima"]) if files["clima"] else None

                    row = {
                        "run": run,
                        "star": star_label,
                        "star_id": str(star_id),
                        "scale": float(scale),
                        "scenario": int(scenario),

                        # clima columns
                        "surface_temperature": get_surface(clima_df, "T"),
                        "surface_h2o": get_surface(clima_df, "FH2O"),
                        "surface_o3": get_surface(clima_df, "O3"),
                        "surface_co2": get_surface(clima_df, "FCO2"),

                        # species from out.out
                        "surface_o2": get_surface_species(species_df, "O2"),
                        "surface_ch4": get_surface_species(species_df, "CH4"),

                        "upper_h2o": (
                            get_top(clima_df, "FH2O")
                            if np.isfinite(get_top(clima_df, "FH2O"))
                            else get_top_species(species_df, "H2O")
                        ),
                        "upper_h2": get_top_species(species_df, "H2"),
                        "upper_o2": get_top_species(species_df, "O2"),
                        "upper_o3": (
                            get_top(clima_df, "O3")
                            if np.isfinite(get_top(clima_df, "O3"))
                            else get_top_species(species_df, "O3")
                        ),
                    }
                    rows.append(row)

                    ftir = parse_two_column_spectrum(files["ftir"]) if files["ftir"] else None
                    ftso = parse_two_column_spectrum(files["ftso"]) if files["ftso"] else None

                    spectra[star_label].append({
                        "run": run,
                        "scale": float(scale),
                        "scenario": int(scenario),
                        "ftir": ftir,
                        "ftso": ftso,
                    })

    return pd.DataFrame(rows), spectra


# -----------------------------
# STYLE MAPS
# -----------------------------
def build_style_maps(sub):
    scenarios = sorted(sub["scenario"].dropna().unique())
    runs = sorted(sub["run"].dropna().unique())

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(runs))))

    scenario_to_marker = {s: markers[i % len(markers)] for i, s in enumerate(scenarios)}
    run_to_color = {r: colors[i % len(colors)] for i, r in enumerate(runs)}

    return scenario_to_marker, run_to_color


def scatter_by_run_and_scenario(ax, sub, x_col, y_col):
    scenario_to_marker, run_to_color = build_style_maps(sub)

    for _, row in sub.iterrows():
        x = row[x_col]
        y = row[y_col]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        ax.scatter(
            x, y,
            marker=scenario_to_marker[row["scenario"]],
            color=run_to_color[row["run"]],
            alpha=0.9,
            s=60,
            linewidths=0.4,
            edgecolors="black",
            zorder=3,
        )

        if ANNOTATE_POINTS:
            ax.annotate(
                f'{row["run"]}, {row["scale"]}, s{row["scenario"]}',
                (x, y),
                fontsize=ANNOTATION_FONT,
            )

    return scenario_to_marker, run_to_color


def make_combined_legend(fig, sub):
    scenario_to_marker, run_to_color = build_style_maps(sub)

    run_handles = [
        Line2D([0], [0], marker="o", color=run_to_color[r], linestyle="",
               markeredgecolor="black", markeredgewidth=0.5, label=r)
        for r in sorted(run_to_color.keys())
    ]

    scenario_handles = [
        Line2D([0], [0], marker=scenario_to_marker[s], color="gray", linestyle="",
               markeredgecolor="black", markeredgewidth=0.5, label=f"Scenario {s}")
        for s in sorted(scenario_to_marker.keys())
    ]

    fig.legend(
        handles=run_handles,
        loc="upper center",
        bbox_to_anchor=(0.33, 0.99),
        ncol=max(1, len(run_handles)),
        frameon=False,
        title="Run",
    )

    fig.legend(
        handles=scenario_handles,
        loc="upper center",
        bbox_to_anchor=(0.76, 0.99),
        ncol=max(1, len(scenario_handles)),
        frameon=False,
        title="Scenario",
    )


# -----------------------------
# REPRESENTATIVE SPECTRUM CHOICE
# -----------------------------
def choose_representative_spectrum(entries):
    valid = [e for e in entries if e.get("ftir") is not None or e.get("ftso") is not None]
    if not valid:
        return None

    scales = np.array([e["scale"] for e in valid], dtype=float)
    target = np.nanmedian(scales)

    valid_sorted = sorted(
        valid,
        key=lambda e: (
            abs(e["scale"] - target),
            int(e["run"].split("_")[1]),
            e["scenario"],
        )
    )
    return valid_sorted[0]


# -----------------------------
# PUBLICATION FIGURE 1
# -----------------------------
def plot_publication_figure(df, star):
    sub = df[df["star"] == star].copy()
    if sub.empty:
        return

    fig, axs = plt.subplots(3, 2, figsize=FIGSIZE)
    axs = axs.flatten()

    ax = axs[0]
    scatter_by_run_and_scenario(ax, sub, "scale", "surface_temperature")
    add_lowess(ax, sub["scale"], sub["surface_temperature"])
    ax.set_xlabel("XUV Proxy (scale)")
    ax.set_ylabel("Surface Temperature [K]")
    ax.set_title("Surface temperature vs XUV")
    ax.grid(True)
    panel_label(ax, "(a)")

    ax = axs[1]
    scatter_by_run_and_scenario(ax, sub, "scale", "surface_h2o")
    add_lowess(ax, sub["scale"], sub["surface_h2o"])
    ax.set_xlabel("XUV Proxy (scale)")
    ax.set_ylabel("Surface H$_2$O mixing ratio")
    ax.set_title("Surface H$_2$O vs XUV")
    maybe_set_log_axis(ax, "y", sub["surface_h2o"].values)
    ax.grid(True)
    panel_label(ax, "(b)")

    ax = axs[2]
    scatter_by_run_and_scenario(ax, sub, "scale", "surface_o2")
    add_lowess(ax, sub["scale"], sub["surface_o2"])
    ax.set_xlabel("XUV Proxy (scale)")
    ax.set_ylabel("Surface O$_2$ mixing ratio")
    ax.set_title("Surface O$_2$ vs XUV")
    maybe_set_log_axis(ax, "y", sub["surface_o2"].values)
    ax.grid(True)
    panel_label(ax, "(c)")

    ax = axs[3]
    scatter_by_run_and_scenario(ax, sub, "scale", "surface_o3")
    add_lowess(ax, sub["scale"], sub["surface_o3"])
    ax.set_xlabel("XUV Proxy (scale)")
    ax.set_ylabel("Surface O$_3$ mixing ratio")
    ax.set_title("Surface O$_3$ vs XUV")
    maybe_set_log_axis(ax, "y", sub["surface_o3"].values)
    ax.grid(True)
    panel_label(ax, "(d)")

    ax = axs[4]
    scatter_by_run_and_scenario(ax, sub, "surface_o2", "surface_ch4")
    ax.set_xlabel("Surface O$_2$ mixing ratio")
    ax.set_ylabel("Surface CH$_4$ mixing ratio")
    ax.set_title("CH$_4$ vs O$_2$")
    maybe_set_log_axis(ax, "x", sub["surface_o2"].values)
    maybe_set_log_axis(ax, "y", sub["surface_ch4"].values)
    ax.grid(True)
    panel_label(ax, "(e)")

    ax = axs[5]
    scatter_by_run_and_scenario(ax, sub, "surface_o2", "surface_o3")
    ax.set_xlabel("Surface O$_2$ mixing ratio")
    ax.set_ylabel("Surface O$_3$ mixing ratio")
    ax.set_title("O$_3$ vs O$_2$")
    maybe_set_log_axis(ax, "x", sub["surface_o2"].values)
    maybe_set_log_axis(ax, "y", sub["surface_o3"].values)
    ax.grid(True)
    panel_label(ax, "(f)")

    add_threshold_label(axs[0], y=FREEZING_TEMPERATURE, text="273 K")
    add_threshold_label(axs[1], y=DESICCATION_THRESHOLD, text="desiccation proxy")
    add_threshold_label(axs[2], y=ABIOTIC_O2_THRESHOLD, text="abiotic O$_2$ proxy")
    add_threshold_label(axs[3], y=ABIOTIC_O3_THRESHOLD, text="abiotic O$_3$ proxy")
    add_threshold_label(axs[5], x=ABIOTIC_O2_THRESHOLD, text="abiotic O$_2$")
    add_threshold_label(axs[5], y=ABIOTIC_O3_THRESHOLD, text="abiotic O$_3$")

    fig.suptitle(f"XUV-driven atmospheric and habitability diagnostics: {star}", fontsize=14, y=1.02)
    make_combined_legend(fig, sub)

    out_base = os.path.join(OUT_DIR, f"publication_summary_{sanitize_for_filename(star)}")
    save_figure(fig, out_base)


# -----------------------------
# PUBLICATION FIGURE 2
# -----------------------------
def plot_spectrum_panel(ax, spec, title, y_label):
    if spec is None:
        ax.text(0.5, 0.5, "No spectrum available", ha="center", va="center")
        ax.set_axis_off()
        return

    wavelength = np.asarray(spec["wavelength"], dtype=float)
    flux = np.asarray(spec["flux"], dtype=float)
    mask = np.isfinite(wavelength) & np.isfinite(flux)

    if not np.any(mask):
        ax.text(0.5, 0.5, "No spectrum available", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.plot(wavelength[mask], flux[mask], color="black")
    ax.set_xlabel(f"Wavelength ({spec['units']})")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)

    x = wavelength[mask]
    if np.all(x > 0) and np.nanmax(x) / np.nanmin(x) > 20:
        ax.set_xscale("log")


def plot_band_totals_panel(ax, spec):
    if spec is None:
        ax.text(0.5, 0.5, "No band data available", ha="center", va="center")
        ax.set_axis_off()
        return

    vals = spec.get("integrated", {})
    labels = ["UV", "VIS", "IR"]
    heights = [vals.get(k, np.nan) for k in labels]

    if not np.any(np.isfinite(heights)):
        ax.text(0.5, 0.5, "No band data available", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.bar(labels, heights)
    ax.set_ylabel("Integrated flux")
    ax.set_title("Representative UV / VIS / IR band totals")
    ax.grid(True, axis="y")


def plot_escape_observables_figure(df, spectra, star):
    sub = df[df["star"] == star].copy()
    if sub.empty:
        return

    rep = choose_representative_spectrum(spectra.get(star, []))
    rep_ftir = rep.get("ftir") if rep else None
    rep_ftso = rep.get("ftso") if rep else None

    fig, axs = plt.subplots(3, 2, figsize=ESCAPE_FIGSIZE, constrained_layout=True)
    axs = axs.flatten()

    ax = axs[0]
    scatter_by_run_and_scenario(ax, sub, "scale", "upper_h2o")
    add_lowess(ax, sub["scale"], sub["upper_h2o"])
    ax.set_title("Upper H$_2$O vs XUV")
    ax.set_xlabel("XUV Proxy (scale)")
    ax.set_ylabel("Upper H$_2$O mixing ratio")
    maybe_set_log_axis(ax, "y", sub["upper_h2o"].values)
    ax.grid(True)
    panel_label(ax, "(a)")

    ax = axs[1]
    scatter_by_run_and_scenario(ax, sub, "scale", "upper_h2")
    add_lowess(ax, sub["scale"], sub["upper_h2"])
    ax.set_title("Upper H$_2$ vs XUV")
    ax.set_xlabel("XUV Proxy (scale)")
    ax.set_ylabel("Upper H$_2$ mixing ratio")
    maybe_set_log_axis(ax, "y", sub["upper_h2"].values)
    ax.grid(True)
    panel_label(ax, "(b)")

    ax = axs[2]
    scatter_by_run_and_scenario(ax, sub, "surface_o2", "surface_o3")
    ax.set_title("O$_3$ vs O$_2$")
    ax.set_xlabel("Surface O$_2$ mixing ratio")
    ax.set_ylabel("Surface O$_3$ mixing ratio")
    maybe_set_log_axis(ax, "x", sub["surface_o2"].values)
    maybe_set_log_axis(ax, "y", sub["surface_o3"].values)
    ax.grid(True)
    add_threshold_label(ax, x=ABIOTIC_O2_THRESHOLD, text="abiotic O$_2$")
    add_threshold_label(ax, y=ABIOTIC_O3_THRESHOLD, text="abiotic O$_3$")
    panel_label(ax, "(c)")

    ax = axs[3]
    plot_spectrum_panel(ax, rep_ftir, "Representative FTIR spectrum", "FTIR flux")
    panel_label(ax, "(d)")

    ax = axs[4]
    plot_spectrum_panel(ax, rep_ftso, "Representative FTSO spectrum", "FTSO flux")
    panel_label(ax, "(e)")

    ax = axs[5]
    plot_band_totals_panel(ax, rep_ftso if rep_ftso is not None else rep_ftir)
    panel_label(ax, "(f)")

    rep_text = ""
    if rep is not None:
        rep_text = f" | representative case: {rep['run']}, scale={rep['scale']}, scenario={rep['scenario']}"

    fig.suptitle(f"Escape proxies and observables: {star}{rep_text}", fontsize=15)
    make_combined_legend(fig, sub)

    out_base = os.path.join(OUT_DIR, f"escape_observables_{sanitize_for_filename(star)}")
    save_figure(fig, out_base)


# -----------------------------
# TABLES
# -----------------------------
def save_tables(df):
    ensure_dir(OUT_DIR)

    full_csv = os.path.join(OUT_DIR, "publication_summary_table.csv")
    df.to_csv(full_csv, index=False)
    print(f"Saved: {full_csv}")

    grouped = (
        df.groupby(["star", "scenario"])
        .agg(
            mean_temperature=("surface_temperature", "mean"),
            mean_h2o=("surface_h2o", "mean"),
            mean_o2=("surface_o2", "mean"),
            mean_o3=("surface_o3", "mean"),
            mean_ch4=("surface_ch4", "mean"),
            mean_co2=("surface_co2", "mean"),
            mean_upper_h2o=("upper_h2o", "mean"),
            mean_upper_h2=("upper_h2", "mean"),
        )
        .reset_index()
    )

    grouped_csv = os.path.join(OUT_DIR, "publication_grouped_table.csv")
    grouped.to_csv(grouped_csv, index=False)
    print(f"Saved: {grouped_csv}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    ensure_dir(OUT_DIR)

    df, spectra = build_dataframe(BASE_DIR)

    print("\nSummary dataframe preview:")
    print(df.head())

    save_tables(df)

    for star in sorted(df["star"].dropna().unique()):
        plot_publication_figure(df, star)
        plot_escape_observables_figure(df, spectra, star)

    print("\nDone.")
    print(f"Publication-ready figures written to: {OUT_DIR}")
