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
PLOTS_DIR = os.path.expanduser("~/atmos/BA/Plots")

STARS = {
    "18": "Epsilon_Eri",
    "24": "HD_40307",
    "25": "HD_85512",
    "26": "HD_97658",
}

SPECIES = ["CO2", "H2", "CH4", "CO", "O2", "O3", "H2O"]
PLOT_SPECIES = ["H2O", "CH4", "CO", "CO2", "O2", "O3", "H2"]

FIGSIZE = (6, 8)
SUMMARY_FIGSIZE = (10, 16)
SPECTRAL_FIGSIZE = (8, 5)
BAND_BAR_FIGSIZE = (6, 4)
COMBINED_SPECTRAL_FIGSIZE = (9, 6)
COMBINED_BAND_FIGSIZE = (8, 5)

SAVE_DPI = 200
SHOW_PLOTS = False


# -----------------------------
# HELPERS
# -----------------------------
def star_name_from_id(star_id):
    return STARS.get(str(star_id), str(star_id))


def numeric_sort_key(value):
    try:
        return float(value)
    except Exception:
        return str(value)


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


def read_table(file_path, skiprows=None):
    try:
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            engine="python",
            skiprows=skiprows,
            comment="#",
        )
        return df
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None


def sanitize_for_filename(value):
    value = str(value)
    for old, new in [(" ", "_"), ("/", "_"), ("\\", "_"), ("=", "_")]:
        value = value.replace(old, new)
    return value


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_figure(fig, out_path):
    ensure_dir(os.path.dirname(out_path))
    fig.tight_layout()
    fig.savefig(out_path, format="jpeg", dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


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


def to_numeric_array(series):
    return pd.to_numeric(series, errors="coerce").values


def finite_positive_mask(x, y):
    return np.isfinite(x) & np.isfinite(y) & (x > 0)


def finite_mask(x, y):
    return np.isfinite(x) & np.isfinite(y)


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


def detect_spectral_units(wavelength):
    """
    Best-effort auto-detection of spectral x-axis units.

    Heuristic:
    - If the finite positive median x is > 50, treat as nm
    - Else treat as um

    This works well for the uploaded examples, which begin with
    values like ~11, 20, 70, 130 and are therefore much more
    naturally interpreted as nm than um.
    """
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


# -----------------------------
# FILE DISCOVERY
# -----------------------------
def build_file_index(scale_dir):
    """
    Scan PHOTOCHEM_OUTPUT and CLIMA_OUTPUT under a scale directory and
    group files by scenario number.

    Expected filename examples:
        Run_1_0.75_1_out.out
        Run_1_0.75_1_hcaer.out
        Run_1_0.75_1_clima_last.tab
        Run_1_0.75_1_FTIR.dat
        Run_1_0.75_1_FTSO.dat
    """
    photochem_dir = os.path.join(scale_dir, "PHOTOCHEM_OUTPUT")
    clima_dir = os.path.join(scale_dir, "CLIMA_OUTPUT")

    scenario_files = defaultdict(
        lambda: {
            "out_file": None,
            "hcaer_file": None,
            "clima_file": None,
            "ftir_file": None,
            "ftso_file": None,
        }
    )

    pattern = re.compile(r"^(Run_\d+)_([0-9]+(?:\.[0-9]+)?)_(\d+)_(.+)$")

    if os.path.isdir(photochem_dir):
        for fname in os.listdir(photochem_dir):
            match = pattern.match(fname)
            if not match:
                continue

            _, _, scenario, suffix = match.groups()
            full_path = os.path.join(photochem_dir, fname)

            if suffix == "out.out":
                scenario_files[scenario]["out_file"] = full_path
            elif suffix == "hcaer.out":
                scenario_files[scenario]["hcaer_file"] = full_path

    if os.path.isdir(clima_dir):
        for fname in os.listdir(clima_dir):
            match = pattern.match(fname)
            if not match:
                continue

            _, _, scenario, suffix = match.groups()
            full_path = os.path.join(clima_dir, fname)

            if suffix in ("clima_last.tab", "clima_last.out"):
                scenario_files[scenario]["clima_file"] = full_path
            elif suffix == "FTIR.dat":
                scenario_files[scenario]["ftir_file"] = full_path
            elif suffix == "FTSO.dat":
                scenario_files[scenario]["ftso_file"] = full_path

    return dict(scenario_files)


# -----------------------------
# PARSERS
# -----------------------------
def parse_out_file(file_path):
    """
    Parse the 'MIXING RATIOS OF LONG-LIVED SPECIES' section
    from the main out.out file.
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
        print(f"No mixing-ratio section found in {file_path}")
        return None

    header_idx = None
    for i in range(start_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("Z") or stripped.startswith("ALT"):
            header_idx = i
            break

    if header_idx is None:
        print(f"No header found in mixing-ratio section: {file_path}")
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
        print(f"No table data found in {file_path}")
        return None

    try:
        df = pd.read_csv(
            StringIO("\n".join(table_lines)),
            sep=r"\s+",
            engine="python",
        )
    except Exception as e:
        print(f"Failed to parse mixing-ratio table in {file_path}: {e}")
        return None

    data = {}

    alt_col = pick_altitude_column(df)
    if alt_col is not None:
        data["ALT"] = to_numeric_array(df[alt_col])

    for sp in SPECIES:
        col = find_matching_column(df, sp, exact=True)
        if col is not None:
            data[sp] = to_numeric_array(df[col])

    return data if data else None


def parse_clima(file_path):
    """
    Parse clima_last.out or clima_last.tab.
    """
    if file_path is None or not os.path.exists(file_path):
        return None

    df = read_table(file_path)
    if df is None:
        return None

    data = {}

    alt_col = pick_altitude_column(df)
    if alt_col is not None:
        data["ALT"] = to_numeric_array(df[alt_col])

    for target in ["P", "H2O", "O3", "FCO2"]:
        col = find_matching_column(df, target, exact=False, contains=True)
        if col is not None:
            data[target] = to_numeric_array(df[col])

    return data if data else None


def parse_hcaer(file_path):
    """
    Parse aerosol file after skipping title text until the header line.
    """
    if file_path is None or not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Failed to open {file_path}: {e}")
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
        df = pd.read_csv(
            StringIO("\n".join(table_lines)),
            sep=r"\s+",
            engine="python",
        )
    except Exception as e:
        print(f"Failed to parse aerosol table in {file_path}: {e}")
        return None

    data = {}

    alt_col = pick_altitude_column(df)
    if alt_col is not None:
        data["ALT"] = to_numeric_array(df[alt_col])

    for col in df.columns:
        if col == alt_col:
            continue
        data[col] = to_numeric_array(df[col])

    return data if data else None


def parse_two_column_spectrum(file_path):
    """
    Parse FTIR.dat / FTSO.dat as plain 2-column numeric files:
        column 0 = wavelength
        column 1 = flux-like quantity
    """
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
            print(f"Spectrum file too short: {file_path}")
            return None

    if arr.ndim != 2 or arr.shape[1] < 2:
        print(f"Spectrum file does not look like 2-column data: {file_path}")
        return None

    wavelength = np.asarray(arr[:, 0], dtype=float)
    flux = np.asarray(arr[:, 1], dtype=float)

    mask = np.isfinite(wavelength) & np.isfinite(flux)
    if not np.any(mask):
        print(f"No finite spectral data in {file_path}")
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
# BUILD DATA STRUCTURE
# -----------------------------
def build_dataset(base_dir):
    """
    data[run][star][scale][scenario] = {
        "species": ...,
        "clima": ...,
        "aerosol": ...,
        "ftir": ...,
        "ftso": ...,
        "files": ...
    }
    """
    data = {}
    runs = discover_run_dirs(base_dir)
    print("Discovered runs:", runs)

    for run in runs:
        run_dir = os.path.join(base_dir, run)
        data[run] = {}

        for star_id in sorted(os.listdir(run_dir), key=numeric_sort_key):
            star_dir = os.path.join(run_dir, star_id)
            if not os.path.isdir(star_dir):
                continue

            star_name = star_name_from_id(star_id)
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
                    print(f"\nLoading: {run} | {star_name} | {scale} | scenario {scenario}")
                    print("  out_file   :", files["out_file"])
                    print("  hcaer_file :", files["hcaer_file"])
                    print("  clima_file :", files["clima_file"])
                    print("  ftir_file  :", files["ftir_file"])
                    print("  ftso_file  :", files["ftso_file"])

                    ftir = parse_two_column_spectrum(files["ftir_file"]) if files["ftir_file"] else None
                    ftso = parse_two_column_spectrum(files["ftso_file"]) if files["ftso_file"] else None

                    if ftir is not None:
                        print(f"  ftir_units : {ftir['units']}")
                    if ftso is not None:
                        print(f"  ftso_units : {ftso['units']}")

                    data[run][star_name][scale][scenario] = {
                        "species": parse_out_file(files["out_file"]) if files["out_file"] else None,
                        "clima": parse_clima(files["clima_file"]) if files["clima_file"] else None,
                        "aerosol": parse_hcaer(files["hcaer_file"]) if files["hcaer_file"] else None,
                        "ftir": ftir,
                        "ftso": ftso,
                        "files": files.copy(),
                    }

    return data


# -----------------------------
# ACCESSORS
# -----------------------------
def safe_get(dataset, run, star, scale, scenario):
    return dataset.get(run, {}).get(star, {}).get(scale, {}).get(str(scenario))


def list_available(dataset):
    print("\nAvailable dataset entries:")
    for run in dataset:
        for star in dataset[run]:
            for scale in dataset[run][star]:
                scenarios = sorted(dataset[run][star][scale].keys(), key=lambda x: int(x))
                print(f"  {run} | {star} | {scale} -> scenarios {scenarios}")


# -----------------------------
# INDIVIDUAL PROFILE PLOTS
# -----------------------------
def plot_vertical_species(dataset, run, star, scale, scenario, out_dir, species_to_plot=None):
    entry = safe_get(dataset, run, star, scale, scenario)
    if entry is None:
        return

    species_data = entry["species"]
    if species_data is None or "ALT" not in species_data:
        return

    alt = species_data["ALT"]
    species_to_plot = species_to_plot or PLOT_SPECIES

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = False

    for sp in species_to_plot:
        if sp in species_data:
            x = np.asarray(species_data[sp], dtype=float)
            y = np.asarray(alt, dtype=float)
            mask = finite_positive_mask(x, y)
            if np.any(mask):
                ax.plot(x[mask], y[mask], label=sp)
                plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_xlabel("Mixing Ratio")
    ax.set_ylabel("Altitude")
    ax.set_title(f"{run} | {star} | scale={scale} | scenario={scenario}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    filename = (
        f"species_{sanitize_for_filename(run)}_"
        f"{sanitize_for_filename(star)}_scale_{sanitize_for_filename(scale)}_"
        f"scenario_{sanitize_for_filename(scenario)}.jpeg"
    )
    save_figure(fig, os.path.join(out_dir, "species", filename))


def plot_clima(dataset, run, star, scale, scenario, out_dir):
    entry = safe_get(dataset, run, star, scale, scenario)
    if entry is None:
        return

    clima = entry["clima"]
    if clima is None or "ALT" not in clima:
        return

    alt = clima["ALT"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = False

    for key in ["H2O", "O3", "FCO2"]:
        if key in clima:
            x = np.asarray(clima[key], dtype=float)
            y = np.asarray(alt, dtype=float)
            mask = finite_positive_mask(x, y)
            if np.any(mask):
                ax.plot(x[mask], y[mask], label=key)
                plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_xlabel("Mixing Ratio")
    ax.set_ylabel("Altitude")
    ax.set_title(f"Clima: {run} | {star} | scale={scale} | scenario={scenario}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    filename = (
        f"clima_{sanitize_for_filename(run)}_"
        f"{sanitize_for_filename(star)}_scale_{sanitize_for_filename(scale)}_"
        f"scenario_{sanitize_for_filename(scenario)}.jpeg"
    )
    save_figure(fig, os.path.join(out_dir, "clima", filename))


def plot_aerosol(dataset, run, star, scale, scenario, out_dir):
    entry = safe_get(dataset, run, star, scale, scenario)
    if entry is None:
        return

    aero = entry["aerosol"]
    if aero is None or "ALT" not in aero:
        return

    alt = aero["ALT"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = False

    for key in aero:
        if key == "ALT":
            continue

        x = np.asarray(aero[key], dtype=float)
        y = np.asarray(alt, dtype=float)
        mask = finite_mask(x, y)
        if np.any(mask):
            ax.plot(x[mask], y[mask], label=key)
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Density")
    ax.set_ylabel("Altitude")
    ax.set_title(f"Aerosols: {run} | {star} | scale={scale} | scenario={scenario}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    filename = (
        f"aerosol_{sanitize_for_filename(run)}_"
        f"{sanitize_for_filename(star)}_scale_{sanitize_for_filename(scale)}_"
        f"scenario_{sanitize_for_filename(scenario)}.jpeg"
    )
    save_figure(fig, os.path.join(out_dir, "aerosol", filename))


# -----------------------------
# INDIVIDUAL SPECTRAL PLOTS
# -----------------------------
def plot_spectrum(spec, title, out_path, y_label="Flux"):
    if spec is None:
        return

    wavelength = np.asarray(spec["wavelength"], dtype=float)
    flux = np.asarray(spec["flux"], dtype=float)
    mask = finite_mask(wavelength, flux)

    if not np.any(mask):
        return

    fig, ax = plt.subplots(figsize=SPECTRAL_FIGSIZE)
    ax.plot(wavelength[mask], flux[mask])
    ax.set_xlabel(f"Wavelength ({spec['units']})")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    w = wavelength[mask]
    if np.all(w > 0) and (np.nanmax(w) / np.nanmin(w) > 20):
        ax.set_xscale("log")

    save_figure(fig, out_path)


def plot_spectrum_bands(spec, title, out_path, y_label="Flux"):
    if spec is None or spec["units"] == "unknown":
        return

    wavelength = np.asarray(spec["wavelength"], dtype=float)
    flux = np.asarray(spec["flux"], dtype=float)
    base_mask = finite_mask(wavelength, flux)

    if not np.any(base_mask):
        return

    uv_max = spec["uv_max"]
    vis_max = spec["vis_max"]

    uv_mask = base_mask & safe_band_mask(wavelength, None, uv_max)
    vis_mask = base_mask & safe_band_mask(wavelength, uv_max, vis_max)
    ir_mask = base_mask & safe_band_mask(wavelength, vis_max, None)

    fig, ax = plt.subplots(figsize=SPECTRAL_FIGSIZE)
    plotted = False

    if np.any(uv_mask):
        ax.plot(wavelength[uv_mask], flux[uv_mask], label=f"UV < {uv_max:g} {spec['units']}")
        plotted = True
    if np.any(vis_mask):
        ax.plot(wavelength[vis_mask], flux[vis_mask], label=f"VIS {uv_max:g}-{vis_max:g} {spec['units']}")
        plotted = True
    if np.any(ir_mask):
        ax.plot(wavelength[ir_mask], flux[ir_mask], label=f"IR > {vis_max:g} {spec['units']}")
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel(f"Wavelength ({spec['units']})")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    w = wavelength[base_mask]
    if np.all(w > 0) and (np.nanmax(w) / np.nanmin(w) > 20):
        ax.set_xscale("log")

    save_figure(fig, out_path)


def plot_band_totals(spec, title, out_path, y_label="Integrated flux"):
    if spec is None:
        return

    vals = spec.get("integrated", {})
    labels = ["UV", "VIS", "IR"]
    heights = [vals.get(k, np.nan) for k in labels]

    if not np.any(np.isfinite(heights)):
        return

    fig, ax = plt.subplots(figsize=BAND_BAR_FIGSIZE)
    ax.bar(labels, heights)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    save_figure(fig, out_path)


def plot_case_spectral_outputs(dataset, run, star, scale, scenario, out_dir):
    entry = safe_get(dataset, run, star, scale, scenario)
    if entry is None:
        return

    prefix = (
        f"{sanitize_for_filename(run)}_"
        f"{sanitize_for_filename(star)}_scale_{sanitize_for_filename(scale)}_"
        f"scenario_{sanitize_for_filename(scenario)}"
    )

    ftir = entry.get("ftir")
    ftso = entry.get("ftso")

    if ftir is not None:
        plot_spectrum(
            ftir,
            title=f"FTIR: {run} | {star} | scale={scale} | scenario={scenario}",
            out_path=os.path.join(out_dir, "spectra", f"ftir_{prefix}.jpeg"),
            y_label="FTIR Flux",
        )
        plot_spectrum_bands(
            ftir,
            title=f"FTIR UV/VIS/IR: {run} | {star} | scale={scale} | scenario={scenario}",
            out_path=os.path.join(out_dir, "spectra_bands", f"ftir_bands_{prefix}.jpeg"),
            y_label="FTIR Flux",
        )
        plot_band_totals(
            ftir,
            title=f"FTIR Band Totals: {run} | {star} | scale={scale} | scenario={scenario}",
            out_path=os.path.join(out_dir, "band_totals", f"ftir_band_totals_{prefix}.jpeg"),
            y_label="Integrated FTIR Flux",
        )

    if ftso is not None:
        plot_spectrum(
            ftso,
            title=f"FTSO: {run} | {star} | scale={scale} | scenario={scenario}",
            out_path=os.path.join(out_dir, "spectra", f"ftso_{prefix}.jpeg"),
            y_label="FTSO Flux",
        )
        plot_spectrum_bands(
            ftso,
            title=f"FTSO UV/VIS/IR: {run} | {star} | scale={scale} | scenario={scenario}",
            out_path=os.path.join(out_dir, "spectra_bands", f"ftso_bands_{prefix}.jpeg"),
            y_label="FTSO Flux",
        )
        plot_band_totals(
            ftso,
            title=f"FTSO Band Totals / FSOL: {run} | {star} | scale={scale} | scenario={scenario}",
            out_path=os.path.join(out_dir, "fsol", f"ftso_fsol_{prefix}.jpeg"),
            y_label="Integrated FTSO Flux",
        )


# -----------------------------
# COMBINED STAR SPECTRAL SUMMARIES
# -----------------------------
def collect_star_entries(dataset, star):
    rows = []
    for run in dataset:
        if star not in dataset[run]:
            continue

        for scale in dataset[run][star]:
            for scenario, entry in dataset[run][star][scale].items():
                rows.append(
                    {
                        "run": run,
                        "star": star,
                        "scale": scale,
                        "scenario": scenario,
                        "entry": entry,
                    }
                )
    return rows


def plot_combined_star_spectrum(dataset, star, spectrum_key, out_dir):
    rows = collect_star_entries(dataset, star)
    if not rows:
        return

    fig, ax = plt.subplots(figsize=COMBINED_SPECTRAL_FIGSIZE)
    plotted = False
    units_seen = set()

    for row in rows:
        spec = row["entry"].get(spectrum_key)
        if spec is None:
            continue

        wavelength = np.asarray(spec["wavelength"], dtype=float)
        flux = np.asarray(spec["flux"], dtype=float)
        mask = finite_mask(wavelength, flux)

        if not np.any(mask):
            continue

        label = f'{row["run"]} | scale={row["scale"]} | s={row["scenario"]}'
        ax.plot(wavelength[mask], flux[mask], label=label, alpha=0.8)
        plotted = True
        units_seen.add(spec["units"])

    if not plotted:
        plt.close(fig)
        return

    units_label = ",".join(sorted(units_seen)) if units_seen else "unknown"
    ax.set_xlabel(f"Wavelength ({units_label})")
    ax.set_ylabel(f"{spectrum_key.upper()} Flux")
    ax.set_title(f"{star} | Combined {spectrum_key.upper()} Spectra")
    ax.grid(True, alpha=0.3)

    if ax.get_lines():
        x_all = np.concatenate([line.get_xdata() for line in ax.get_lines()])
        x_all = x_all[np.isfinite(x_all) & (x_all > 0)]
        if x_all.size > 0 and np.nanmax(x_all) / np.nanmin(x_all) > 20:
            ax.set_xscale("log")

    ax.legend(fontsize=7, loc="best")

    filename = f"combined_{spectrum_key}_{sanitize_for_filename(star)}.jpeg"
    save_figure(fig, os.path.join(out_dir, "combined_spectra_by_star", filename))


def plot_combined_star_band_totals(dataset, star, spectrum_key, out_dir):
    rows = collect_star_entries(dataset, star)
    if not rows:
        return

    labels = []
    uv_vals = []
    vis_vals = []
    ir_vals = []

    for row in rows:
        spec = row["entry"].get(spectrum_key)
        if spec is None:
            continue

        vals = spec.get("integrated", {})
        uv = vals.get("UV", np.nan)
        vis = vals.get("VIS", np.nan)
        ir = vals.get("IR", np.nan)

        if not (np.isfinite(uv) or np.isfinite(vis) or np.isfinite(ir)):
            continue

        labels.append(f'{row["run"]}\n{row["scale"]}\ns{row["scenario"]}')
        uv_vals.append(uv)
        vis_vals.append(vis)
        ir_vals.append(ir)

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=COMBINED_BAND_FIGSIZE)
    ax.bar(x - width, uv_vals, width=width, label="UV")
    ax.bar(x, vis_vals, width=width, label="VIS")
    ax.bar(x + width, ir_vals, width=width, label="IR")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel(f"Integrated {spectrum_key.upper()} Flux")
    ax.set_title(f"{star} | Combined {spectrum_key.upper()} UV/VIS/IR Totals")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    filename = f"combined_{spectrum_key}_band_totals_{sanitize_for_filename(star)}.jpeg"
    save_figure(fig, os.path.join(out_dir, "combined_band_totals_by_star", filename))


def save_all_star_spectral_summaries(dataset, out_dir):
    all_stars = sorted({star for run in dataset for star in dataset[run]})

    for star in all_stars:
        print(f"\nSaving combined spectral summaries for star: {star}")
        plot_combined_star_spectrum(dataset, star, "ftir", out_dir)
        plot_combined_star_spectrum(dataset, star, "ftso", out_dir)
        plot_combined_star_band_totals(dataset, star, "ftir", out_dir)
        plot_combined_star_band_totals(dataset, star, "ftso", out_dir)


# -----------------------------
# COMBINED STAR SPECIES SUMMARY
# -----------------------------
def plot_combined_star_summary(dataset, star, out_dir, species_to_plot=None):
    rows = collect_star_entries(dataset, star)
    if not rows:
        return

    species_to_plot = species_to_plot or PLOT_SPECIES
    n_species = len(species_to_plot)

    fig, axes = plt.subplots(n_species, 1, figsize=SUMMARY_FIGSIZE, sharey=True)
    if n_species == 1:
        axes = [axes]

    anything_plotted = False

    for ax, sp in zip(axes, species_to_plot):
        plotted_here = False

        for row in rows:
            entry = row["entry"]
            species_data = entry.get("species")

            if species_data is None or "ALT" not in species_data or sp not in species_data:
                continue

            x = np.asarray(species_data[sp], dtype=float)
            y = np.asarray(species_data["ALT"], dtype=float)
            mask = finite_positive_mask(x, y)

            if not np.any(mask):
                continue

            label = f'{row["run"]} | scale={row["scale"]} | s={row["scenario"]}'
            ax.plot(x[mask], y[mask], label=label, alpha=0.8)
            plotted_here = True
            anything_plotted = True

        ax.set_xscale("log")
        ax.set_xlabel(sp)
        ax.set_ylabel("Altitude")
        ax.set_title(f"{star} | {sp}")
        ax.grid(True, which="both", alpha=0.3)

        if plotted_here:
            ax.legend(fontsize=7, loc="best")

    if not anything_plotted:
        plt.close(fig)
        return

    fig.suptitle(f"Combined Runs Summary | {star}", y=0.995)
    filename = f"combined_species_summary_{sanitize_for_filename(star)}.jpeg"
    save_figure(fig, os.path.join(out_dir, "combined_by_star", filename))


def save_all_star_species_summaries(dataset, out_dir):
    all_stars = sorted({star for run in dataset for star in dataset[run]})
    for star in all_stars:
        print(f"\nSaving combined species summary for star: {star}")
        plot_combined_star_summary(dataset, star, out_dir)


# -----------------------------
# SAVE ALL CASE PLOTS
# -----------------------------
def save_all_case_plots(dataset, out_dir):
    for run in dataset:
        for star in dataset[run]:
            for scale in dataset[run][star]:
                for scenario in dataset[run][star][scale]:
                    print(f"\nSaving case plots: {run} | {star} | {scale} | scenario {scenario}")
                    plot_vertical_species(dataset, run, star, scale, scenario, out_dir)
                    plot_clima(dataset, run, star, scale, scenario, out_dir)
                    plot_aerosol(dataset, run, star, scale, scenario, out_dir)
                    plot_case_spectral_outputs(dataset, run, star, scale, scenario, out_dir)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    ensure_dir(PLOTS_DIR)

    data = build_dataset(BASE_DIR)
    list_available(data)

    save_all_case_plots(data, PLOTS_DIR)
    save_all_star_species_summaries(data, PLOTS_DIR)
    save_all_star_spectral_summaries(data, PLOTS_DIR)

    print("\nDone.")
    print(f"JPEG plots written to: {PLOTS_DIR}")
