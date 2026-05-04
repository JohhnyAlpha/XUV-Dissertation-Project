import os
import re
import math
from io import StringIO
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_DIR = os.path.expanduser("~/atmos/OutputStorage")
OUT_DIR = os.path.expanduser("~/atmos/BA/Plots/species_analysis_pub")
SAVE_DPI = 300
SHOW_PLOTS = False

STAR_ID_TO_NAME = {
    "18": "Epsilon_Eri",
    "24": "HD40307",
    "25": "HD85512",
    "26": "HD97658",
}
STAR_ORDER = ["Epsilon_Eri", "HD40307", "HD85512", "HD97658"]

RUN_LABELS = {
    "Run_1": "2.7 Gy star",
    "Run_2": "0.5 Gy star",
    "Run_3": "9.9 Gy star",
}
RUN_ORDER = ["Run_1", "Run_2", "Run_3"]

SCENARIO_TO_FSCALE = {
    "1": "0.75",
    "2": "1.00",
    "3": "1.50",
}
SCENARIO_ORDER = ["1", "2", "3"]
FSCALE_ORDER = ["0.75", "1.00", "1.50"]
FSCALE_TO_SCENARIO = {v: k for k, v in SCENARIO_TO_FSCALE.items()}

SPECIES = ["H2O", "CH4", "CO", "CO2", "O2", "O3", "H2"]
PLOT_SPECIES = ["H2O", "CH4", "CO", "CO2", "O2", "O3", "H2"]

SPECIES_COLORS = {
    "H2O": "#1f77b4",
    "CH4": "#2ca02c",
    "CO": "#8c564b",
    "CO2": "#d62728",
    "O2": "#7f7f7f",
    "O3": "#9467bd",
    "H2": "#ff7f0e",
}

UPPER_ATM_KM = 50.0
MIDDLE_ATM_KM = 20.0
FIGSIZE_CASE = (7.0, 8.2)
FIGSIZE_PAGE = (11.0, 8.5)
FIGSIZE_HEATMAP = (10.5, 6.5)
FIGSIZE_SUMMARY = (10.0, 12.0)
FIGSIZE_SCATTER = (9.5, 6.5)
FIGSIZE_STAR_MULTIPANE = (13.5, 11.0)  # publication multipane with right-side legend
MIXING_RATIO_BLOCK = "last"  # photochem output has repeated blocks; "last" is final/converged state
ALTITUDE_AUTO_CONVERT = True
PROFILE_YLIM_KM = (0.0, 100.0)
DIFF_FULL_YLIM_KM = (0.0, 100.0)
DIFF_UPPER_YLIM_KM = (50.0, 100.0)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
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


def numeric_sort_key(value):
    try:
        return float(value)
    except Exception:
        return str(value)


def star_name_from_id(star_id):
    return STAR_ID_TO_NAME.get(str(star_id), str(star_id))


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
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def save_dual(fig, stem):
    save_figure(fig, stem + ".png")
    fig2 = None
    # fig already closed by save_figure, so use caller for separate saves when needed.


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


def pick_altitude_column(df):
    for candidate in ["ALT", "Z", "Alt", "alt", "z"]:
        if candidate in df.columns:
            return candidate
    return None


def find_matching_column(df, target):
    for col in df.columns:
        if col.lower() == target.lower():
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
    return np.trapz(y, x)


def safe_log10_ratio(v_new, v_ref):
    if np.isfinite(v_new) and np.isfinite(v_ref) and v_new > 0 and v_ref > 0:
        return np.log10(v_new / v_ref)
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
        return f"~{10**absd:.1f}x"
    if absd >= 0.1:
        return f"~{10**absd:.2f}x"
    return None


# --------------------------------------------------
# FILE DISCOVERY / PARSING
# --------------------------------------------------
def build_file_index(scale_dir):
    photochem_dir = os.path.join(scale_dir, "PHOTOCHEM_OUTPUT")
    scenario_files = defaultdict(lambda: {"out_file": None})
    pattern = re.compile(r"^(Run_\d+)_([0-9]+(?:\.[0-9]+)?)_(\d+)_(.+)$")

    if os.path.isdir(photochem_dir):
        for fname in os.listdir(photochem_dir):
            match = pattern.match(fname)
            if not match:
                continue
            _, _, scenario, suffix = match.groups()
            if suffix == "out.out":
                scenario_files[scenario]["out_file"] = os.path.join(photochem_dir, fname)

    return dict(scenario_files)


def altitude_to_km(alt_raw):
    """Return altitude in km with conservative auto-detection.

    PHOTOCHEM Z columns are often in cm (e.g. 1e7 cm = 100 km).
    Some outputs may already be km, or occasionally metres. This keeps plots
    and diagnostics on a sane 0--100 km scale without hard-coding one unit.
    """
    alt = np.asarray(alt_raw, dtype=float)
    finite = alt[np.isfinite(alt)]
    if finite.size == 0 or not ALTITUDE_AUTO_CONVERT:
        return alt

    max_abs = float(np.nanmax(np.abs(finite)))
    if max_abs > 1e6:
        # Very likely centimetres. 1 km = 1e5 cm.
        return alt / 1.0e5
    if max_abs > 1.0e3:
        # Likely metres. 1 km = 1000 m.
        return alt / 1.0e3
    return alt


def parse_out_file(file_path):
    if file_path is None or not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Failed to open {file_path}: {e}")
        return None

    block_indices = [
        i for i, line in enumerate(lines)
        if "MIXING RATIOS OF LONG-LIVED SPECIES" in line
    ]
    if not block_indices:
        print(f"No mixing-ratio section found in {file_path}")
        return None

    # PHOTOCHEM out.out files can contain several repeated mixing-ratio tables.
    # The first table is commonly an initial/intermediate state and can be
    # nearly identical across runs. For analysis/plots, use the final table.
    if MIXING_RATIO_BLOCK == "first":
        start_idx = block_indices[0]
    elif MIXING_RATIO_BLOCK == "last":
        start_idx = block_indices[-1]
    else:
        raise ValueError("MIXING_RATIO_BLOCK must be 'first' or 'last'")

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
        df = pd.read_csv(StringIO("\n".join(table_lines)), sep=r"\s+", engine="python")
    except Exception as e:
        print(f"Failed to parse mixing-ratio table in {file_path}: {e}")
        return None

    alt_col = pick_altitude_column(df)
    if alt_col is None:
        print(f"No altitude column in {file_path}")
        return None

    data = {"ALT": altitude_to_km(to_numeric_array(df[alt_col]))}
    for sp in SPECIES:
        col = find_matching_column(df, sp)
        if col is not None:
            data[sp] = to_numeric_array(df[col])
    return data


def build_dataset(base_dir):
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
                    out_file = files.get("out_file")
                    species = parse_out_file(out_file) if out_file else None
                    data[run][star_name][scale][scenario] = {
                        "species": species,
                        "out_file": out_file,
                    }
                    print(f"Loaded: {run} | {star_name} | scale={scale} | scenario={scenario} | file={out_file}")
    return data


# --------------------------------------------------
# CASE INDEXING
# --------------------------------------------------
# Important: in this dataset the directory path defines the physical case:
#   Run folder = stellar-age case, scale folder = FSCALE, filename suffix = scenario.
# The Run_* token inside the filename is not trusted.  Scenario and FSCALE are
# independent axes, so do not map scenario -> FSCALE.
def build_case_index(dataset):
    cases = []
    for run in RUN_ORDER:
        if run not in dataset:
            continue
        for star in STAR_ORDER:
            if star not in dataset[run]:
                continue
            star_dict = dataset[run][star]
            for fscale in FSCALE_ORDER:
                if fscale not in star_dict:
                    continue
                for scenario in SCENARIO_ORDER:
                    entry = star_dict[fscale].get(scenario)
                    if entry is None or entry.get("species") is None:
                        continue
                    cases.append({
                        "run": run,
                        "run_label": RUN_LABELS.get(run, run),
                        "star": star,
                        "scenario": scenario,
                        "fscale": fscale,
                        "scale_dir": fscale,
                        "entry": entry,
                    })
    return cases


# --------------------------------------------------
# METRICS
# --------------------------------------------------
def compute_species_metrics(species_data):
    if species_data is None or "ALT" not in species_data:
        return {}

    alt = np.asarray(species_data["ALT"], dtype=float)
    metrics = {}

    for sp in SPECIES:
        if sp not in species_data:
            continue
        x = np.asarray(species_data[sp], dtype=float)
        mask = np.isfinite(x) & np.isfinite(alt)
        if not np.any(mask):
            continue

        z = alt[mask]
        y = x[mask]
        order = np.argsort(z)
        z = z[order]
        y = y[order]

        if z.size == 0:
            continue

        positive = y[np.isfinite(y) & (y > 0)]
        peak_idx = np.nanargmax(y)
        upper_mask = z >= UPPER_ATM_KM
        middle_mask = z >= MIDDLE_ATM_KM

        metrics[sp] = {
            "column_proxy": integrate_curve(y, z) if z.size >= 2 else np.nan,
            "peak_mixing_ratio": np.nanmax(y),
            "peak_alt_km": z[peak_idx],
            "surface_mixing_ratio": y[0],
            "upper_mean": np.nanmean(y[upper_mask]) if np.any(upper_mask) else np.nan,
            "middle_mean": np.nanmean(y[middle_mask]) if np.any(middle_mask) else np.nan,
            "min_positive": np.nanmin(positive) if positive.size else np.nan,
            "max_positive": np.nanmax(positive) if positive.size else np.nan,
        }
    return metrics


def flatten_metrics(cases):
    rows = []
    for case in cases:
        metrics = compute_species_metrics(case["entry"].get("species"))
        for sp, vals in metrics.items():
            row = {
                "run": case["run"],
                "run_label": case["run_label"],
                "star": case["star"],
                "scenario": case["scenario"],
                "fscale": case["fscale"],
                "scale_dir": case["scale_dir"],
                "species": sp,
                "out_file": case["entry"].get("out_file"),
            }
            row.update(vals)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["run"] = pd.Categorical(df["run"], categories=RUN_ORDER, ordered=True)
    df["star"] = pd.Categorical(df["star"], categories=STAR_ORDER, ordered=True)
    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    df["fscale"] = pd.Categorical(df["fscale"], categories=FSCALE_ORDER, ordered=True)
    df["species"] = pd.Categorical(df["species"], categories=SPECIES, ordered=True)
    return df.sort_values(["star", "species", "run", "scenario"]).reset_index(drop=True)


# --------------------------------------------------
# COMPARISONS
# --------------------------------------------------
def build_comparison_tables(metrics_df):
    comp_rows = []
    if metrics_df.empty:
        return pd.DataFrame()

    metric_cols = [
        "column_proxy", "peak_mixing_ratio", "peak_alt_km",
        "surface_mixing_ratio", "upper_mean", "middle_mean",
    ]

    for star in STAR_ORDER:
        sdf = metrics_df[metrics_df["star"] == star]
        if sdf.empty:
            continue
        for scenario in SCENARIO_ORDER:
            scen_df = sdf[sdf["scenario"] == scenario]
            if scen_df.empty:
                continue
            for sp in SPECIES:
                spdf = scen_df[scen_df["species"] == sp]
                if spdf.empty:
                    continue

                # Across runs at fixed scenario and fixed FSCALE, relative to Run_1.
                for fscale in FSCALE_ORDER:
                    g = spdf[spdf["fscale"] == fscale].copy()
                    if g.empty:
                        continue
                    ref = g[g["run"] == "Run_1"]
                    if ref.empty:
                        continue
                    ref = ref.iloc[0]
                    for _, row in g.iterrows():
                        for metric in metric_cols:
                            comp_rows.append({
                                "comparison_type": "vs_Run1_same_scenario_fscale",
                                "star": star,
                                "scenario": scenario,
                                "species": sp,
                                "fscale": fscale,
                                "run": row["run"],
                                "metric": metric,
                                "ref_value": ref.get(metric, np.nan),
                                "test_value": row.get(metric, np.nan),
                                "delta_log10": safe_log10_ratio(row.get(metric, np.nan), ref.get(metric, np.nan)) if metric != "peak_alt_km" else row.get(metric, np.nan) - ref.get(metric, np.nan),
                            })

                # Across FSCALE at fixed scenario and run, relative to FSCALE = 1.00.
                for run in RUN_ORDER:
                    g = spdf[spdf["run"] == run].copy()
                    if g.empty:
                        continue
                    ref = g[g["fscale"] == "1.00"]
                    if ref.empty:
                        continue
                    ref = ref.iloc[0]
                    for _, row in g.iterrows():
                        for metric in metric_cols:
                            comp_rows.append({
                                "comparison_type": "vs_Fscale1.00_same_scenario_run",
                                "star": star,
                                "scenario": scenario,
                                "species": sp,
                                "fscale": row["fscale"],
                                "run": run,
                                "metric": metric,
                                "ref_value": ref.get(metric, np.nan),
                                "test_value": row.get(metric, np.nan),
                                "delta_log10": safe_log10_ratio(row.get(metric, np.nan), ref.get(metric, np.nan)) if metric != "peak_alt_km" else row.get(metric, np.nan) - ref.get(metric, np.nan),
                            })

    out = pd.DataFrame(comp_rows)
    if not out.empty:
        out["run"] = pd.Categorical(out["run"], categories=RUN_ORDER, ordered=True)
        out["star"] = pd.Categorical(out["star"], categories=STAR_ORDER, ordered=True)
        out["scenario"] = pd.Categorical(out["scenario"], categories=SCENARIO_ORDER, ordered=True)
        out["species"] = pd.Categorical(out["species"], categories=SPECIES, ordered=True)
        out["fscale"] = pd.Categorical(out["fscale"], categories=FSCALE_ORDER, ordered=True)
    return out.sort_values(["comparison_type", "star", "scenario", "species", "run", "fscale", "metric"]).reset_index(drop=True)


# --------------------------------------------------
# PLOTS: CASE / OVERVIEW
# --------------------------------------------------
def plot_case_species(case, out_dir):
    species_data = case["entry"].get("species")
    if species_data is None or "ALT" not in species_data:
        return

    alt = np.asarray(species_data["ALT"], dtype=float)
    fig, ax = plt.subplots(figsize=FIGSIZE_CASE)
    plotted = False
    for sp in PLOT_SPECIES:
        if sp not in species_data:
            continue
        x = np.asarray(species_data[sp], dtype=float)
        mask = finite_positive_mask(x, alt)
        if np.any(mask):
            ax.plot(x[mask], alt[mask], label=sp, linewidth=2.0, color=SPECIES_COLORS.get(sp))
            plotted = True
    if not plotted:
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_xlabel("Mixing ratio")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"{case['star']} | {case['run_label']} | FSCALE = {case['fscale']} | scenario {case['scenario']}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", ncol=1, frameon=False)

    stem = os.path.join(
        out_dir,
        "case_species",
        f"species_{sanitize_for_filename(case['star'])}_{sanitize_for_filename(case['run'])}_fscale_{sanitize_for_filename(case['fscale'])}_scenario_{sanitize_for_filename(case['scenario'])}",
    )
    save_current_figure_both(fig, stem)


def paginate(items, page_size=4):
    for i in range(0, len(items), page_size):
        yield items[i:i + page_size], i // page_size + 1


def plot_overview_pages(cases, group_name, group_value, out_dir):
    if not cases:
        return
    png_dir = os.path.join(out_dir, f"{group_name}_overviews")
    ensure_dir(png_dir)
    pdf_path = os.path.join(png_dir, f"{sanitize_for_filename(group_value)}_overview_pages.pdf")

    with PdfPages(pdf_path) as pdf:
        for page_cases, page_no in paginate(cases, page_size=4):
            fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_PAGE, sharex=True, sharey=True)
            axes = axes.ravel()
            for ax in axes:
                ax.set_visible(False)

            for ax, case in zip(axes, page_cases):
                ax.set_visible(True)
                species_data = case["entry"].get("species")
                if species_data is None or "ALT" not in species_data:
                    continue
                alt = np.asarray(species_data["ALT"], dtype=float)
                for sp in PLOT_SPECIES:
                    if sp not in species_data:
                        continue
                    x = np.asarray(species_data[sp], dtype=float)
                    mask = finite_positive_mask(x, alt)
                    if np.any(mask):
                        ax.plot(x[mask], alt[mask], linewidth=1.8, color=SPECIES_COLORS.get(sp), label=sp)
                ax.set_xscale("log")
                ax.grid(True, which="both", alpha=0.2)
                ax.set_title(f"{case['star']} | {case['run_label']} | FSCALE = {case['fscale']} | scenario {case['scenario']}")
                ax.set_xlabel("Mixing ratio")
                ax.set_ylabel("Altitude (km)")

            handles = []
            labels = []
            for sp in PLOT_SPECIES:
                h, = axes[0].plot([], [], color=SPECIES_COLORS.get(sp), linewidth=2.0)
                handles.append(h)
                labels.append(sp)
            fig.legend(handles, labels, loc="lower center", ncol=len(PLOT_SPECIES), frameon=False, bbox_to_anchor=(0.5, -0.01))
            fig.suptitle(f"{group_value} | species overview | page {page_no}", y=0.98)
            fig.tight_layout(rect=[0, 0.04, 1, 0.96])
            pdf.savefig(fig, bbox_inches="tight")
            png_path = os.path.join(png_dir, f"{sanitize_for_filename(group_value)}_page_{page_no}.png")
            fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
            print(f"Saved: {png_path}")
            plt.close(fig)
    print(f"Saved: {pdf_path}")


# --------------------------------------------------
# PLOTS: HEATMAPS / COMPARISON
# --------------------------------------------------
def heatmap_matrix(comparison_df, star, scenario, comparison_type, metric):
    sdf = comparison_df[
        (comparison_df["star"] == star)
        & (comparison_df["scenario"] == scenario)
        & (comparison_df["comparison_type"] == comparison_type)
        & (comparison_df["metric"] == metric)
    ].copy()
    if sdf.empty:
        return None

    if comparison_type == "vs_Run1_same_scenario_fscale":
        piv = sdf.pivot_table(index="species", columns=["fscale", "run"], values="delta_log10", aggfunc="first")
        desired_cols = [(f, r) for f in FSCALE_ORDER for r in RUN_ORDER]
    else:
        piv = sdf.pivot_table(index="species", columns=["run", "fscale"], values="delta_log10", aggfunc="first")
        desired_cols = [(r, f) for r in RUN_ORDER for f in FSCALE_ORDER]

    piv = piv.reindex(index=SPECIES)
    existing = [c for c in desired_cols if c in piv.columns]
    piv = piv.reindex(columns=existing)
    return piv


def plot_heatmaps(comparison_df, out_dir):
    if comparison_df.empty:
        return

    for star in STAR_ORDER:
        for scenario in SCENARIO_ORDER:
            for comparison_type, title_stub in [
                ("vs_Run1_same_scenario_fscale", "relative to Run_1 at fixed scenario and FSCALE"),
                ("vs_Fscale1.00_same_scenario_run", "relative to FSCALE = 1.00 at fixed scenario and run"),
            ]:
                fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_SUMMARY)
                axes = axes.ravel()
                metrics = ["column_proxy", "peak_mixing_ratio", "upper_mean", "peak_alt_km"]

                for ax, metric in zip(axes, metrics):
                    mat = heatmap_matrix(comparison_df, star, scenario, comparison_type, metric)
                    if mat is None or mat.empty:
                        ax.axis("off")
                        continue
                    vals = mat.to_numpy(dtype=float)
                    if metric == "peak_alt_km":
                        im = ax.imshow(vals, aspect="auto")
                        cbar_label = "Δ peak altitude (km)"
                    else:
                        im = ax.imshow(vals, aspect="auto", vmin=-2, vmax=2)
                        cbar_label = "Δ log10(metric)"
                    ax.set_title(metric.replace("_", " "))
                    ax.set_yticks(np.arange(len(mat.index)))
                    ax.set_yticklabels(mat.index)
                    ax.set_xticks(np.arange(len(mat.columns)))
                    ax.set_xticklabels([
                        f"{a}\n{b}" if isinstance(col, tuple) and len(col) == 2 else str(col)
                        for col in mat.columns
                        for a, b in [col if isinstance(col, tuple) else (str(col), "")]
                    ], rotation=45, ha="right")
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label(cbar_label)

                fig.suptitle(f"{star} | scenario {scenario} | {title_stub}", y=0.98)
                stem = os.path.join(out_dir, "heatmaps", f"{sanitize_for_filename(star)}_scenario_{scenario}_{sanitize_for_filename(comparison_type)}")
                save_current_figure_both(fig, stem)


def plot_metric_scatter(metrics_df, out_dir):
    if metrics_df.empty:
        return

    for star in STAR_ORDER:
        for scenario in SCENARIO_ORDER:
            sdf = metrics_df[(metrics_df["star"] == star) & (metrics_df["scenario"] == scenario)]
            if sdf.empty:
                continue

            fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_SCATTER)
            axes = axes.ravel()
            plot_specs = [
                ("column_proxy", "Column proxy"),
                ("peak_mixing_ratio", "Peak mixing ratio"),
                ("upper_mean", f"Mean above {UPPER_ATM_KM:g} km"),
                ("peak_alt_km", "Peak altitude (km)"),
            ]

            xvals = np.array([float(f) for f in FSCALE_ORDER])
            run_markers = {"Run_1": "o", "Run_2": "s", "Run_3": "^"}

            for ax, (metric, ylabel) in zip(axes, plot_specs):
                for sp in SPECIES:
                    spdf = sdf[sdf["species"] == sp].copy()
                    if spdf.empty:
                        continue
                    for run in RUN_ORDER:
                        g = spdf[spdf["run"] == run].sort_values("fscale")
                        if g.empty:
                            continue
                        y = g[metric].astype(float).to_numpy()
                        xf = np.array([float(v) for v in g["fscale"].astype(str)])
                        ax.plot(xf, y, marker=run_markers[run], linewidth=1.2, alpha=0.9, color=SPECIES_COLORS.get(sp), label=f"{sp} | {run}")
                if metric != "peak_alt_km":
                    finite_positive = sdf[metric].to_numpy(dtype=float)
                    finite_positive = finite_positive[np.isfinite(finite_positive) & (finite_positive > 0)]
                    if finite_positive.size > 1 and np.nanmax(finite_positive) / np.nanmin(finite_positive) > 100:
                        ax.set_yscale("log")
                ax.set_xlabel("FSCALE")
                ax.set_ylabel(ylabel)
                ax.set_title(ylabel)
                ax.grid(True, alpha=0.25)
                ax.set_xticks(xvals)

            handles_species = [plt.Line2D([0], [0], color=SPECIES_COLORS[s], lw=2) for s in SPECIES]
            labels_species = SPECIES.copy()
            handles_runs = [plt.Line2D([0], [0], color="black", marker={"Run_1": "o", "Run_2": "s", "Run_3": "^"}[r], lw=0) for r in RUN_ORDER]
            labels_runs = [RUN_LABELS[r] for r in RUN_ORDER]
            fig.legend(handles_species + handles_runs, labels_species + labels_runs, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
            fig.suptitle(f"{star} | scenario {scenario} | species metrics vs FSCALE", y=0.98)
            fig.tight_layout(rect=[0, 0.05, 1, 0.96])
            stem = os.path.join(out_dir, "metric_scatter", f"{sanitize_for_filename(star)}_scenario_{scenario}_metric_scatter")
            fig.savefig(stem + ".png", dpi=SAVE_DPI, bbox_inches="tight")
            fig.savefig(stem + ".pdf", bbox_inches="tight")
            print(f"Saved: {stem}.png")
            print(f"Saved: {stem}.pdf")
            plt.close(fig)


# --------------------------------------------------
# PUBLICATION-READY MULTI-PANE PLOTS
# --------------------------------------------------
SPECIES_LABELS = {
    "H2O": r"H$_2$O",
    "CH4": r"CH$_4$",
    "CO": r"CO",
    "CO2": r"CO$_2$",
    "O2": r"O$_2$",
    "O3": r"O$_3$",
    "H2": r"H$_2$",
}

PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def species_label(sp):
    return SPECIES_LABELS.get(sp, sp)


def apply_publication_style():
    """Matplotlib settings aimed at clean vector/PDF output for papers."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": SAVE_DPI,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 12,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def clean_axis(ax, *, show_xlabel=True, show_ylabel=True):
    ax.set_xscale("log")
    ax.set_xlim(1e-14, 1.5)
    # y-limits are applied after plotting in the multipane routine; setting them here
    # can lock shared axes to the default 0--1 range and produce blank panels.
    ax.grid(True, which="major", alpha=0.22, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.4)
    ax.set_xlabel("Mixing ratio" if show_xlabel else "")
    ax.set_ylabel("Altitude (km)" if show_ylabel else "")


def plot_species_profiles_on_axis(ax, species_data, linewidth=1.5):
    """Draw all configured species on one axis; returns labels actually plotted."""
    plotted = []
    if species_data is None or "ALT" not in species_data:
        return plotted
    alt = np.asarray(species_data["ALT"], dtype=float)
    for sp in PLOT_SPECIES:
        if sp not in species_data:
            continue
        x = np.asarray(species_data[sp], dtype=float)
        mask = finite_positive_mask(x, alt)
        if np.any(mask):
            order = np.argsort(alt[mask])
            ax.plot(
                x[mask][order],
                alt[mask][order],
                label=species_label(sp),
                linewidth=linewidth,
                color=SPECIES_COLORS.get(sp),
            )
            plotted.append(sp)
    return plotted


def plot_star_multipane(cases, star, scenario, out_dir):
    """One publication-ready 3x3 grid per star and scenario.

    Rows are stellar-age run folders; columns are FSCALE scale folders.
    Scenario is held fixed because scenario and FSCALE are independent axes
    in the current output tree.
    """
    star_cases = [c for c in cases if c["star"] == star and str(c["scenario"]) == str(scenario)]
    if not star_cases:
        return
    by_key = {(c["run"], c["fscale"]): c for c in star_cases}

    apply_publication_style()
    fig, axes = plt.subplots(
        len(RUN_ORDER),
        len(FSCALE_ORDER),
        figsize=FIGSIZE_STAR_MULTIPANE,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    plotted_any = set()
    for i, run in enumerate(RUN_ORDER):
        for j, fscale in enumerate(FSCALE_ORDER):
            ax = axes[i, j]
            case = by_key.get((run, fscale))
            clean_axis(ax, show_xlabel=(i == len(RUN_ORDER) - 1), show_ylabel=(j == 0))

            if case is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            else:
                plotted = plot_species_profiles_on_axis(ax, case["entry"].get("species"), linewidth=1.35)
                plotted_any.update(plotted)

            letter = PANEL_LETTERS[i * len(FSCALE_ORDER) + j]
            ax.text(0.03, 0.95, f"({letter})", transform=ax.transAxes, ha="left", va="top", fontweight="bold")
            if i == 0:
                ax.set_title(f"FSCALE = {fscale}")
            if j == 0:
                ax.text(
                    -0.30, 0.5, RUN_LABELS.get(run, run), transform=ax.transAxes,
                    rotation=90, ha="center", va="center", fontsize=9
                )

    # Apply shared y-limits only after all profiles are drawn. This prevents the
    # recurring blank-multipane bug caused by pre-plot y-limit locking on shared axes.
    finite_alts = []
    for c in star_cases:
        spdata = c.get("entry", {}).get("species")
        if spdata is not None and "ALT" in spdata:
            alt = np.asarray(spdata["ALT"], dtype=float)
            finite_alts.extend(alt[np.isfinite(alt)].tolist())
    if PROFILE_YLIM_KM is not None:
        for ax in np.ravel(axes):
            ax.set_ylim(*PROFILE_YLIM_KM)
    elif finite_alts:
        ymin = min(0.0, float(np.nanmin(finite_alts)))
        ymax = float(np.nanmax(finite_alts))
        if ymax > ymin:
            pad = 0.02 * (ymax - ymin)
            for ax in np.ravel(axes):
                ax.set_ylim(ymin, ymax + pad)

    handles = [plt.Line2D([0], [0], color=SPECIES_COLORS[sp], lw=2.0) for sp in PLOT_SPECIES if sp in plotted_any]
    labels = [species_label(sp) for sp in PLOT_SPECIES if sp in plotted_any]
    if handles:
        fig.legend(
            handles, labels,
            title="Species",
            loc="center left",
            bbox_to_anchor=(0.86, 0.5),
            frameon=False,
            borderaxespad=0.0,
        )

    fig.suptitle(f"{star.replace('_', ' ')} | scenario {scenario}", y=0.985)
    fig.tight_layout(rect=[0.06, 0.04, 0.84, 0.955])

    stem = os.path.join(
        out_dir,
        "star_multipane",
        f"{sanitize_for_filename(star)}_scenario_{sanitize_for_filename(scenario)}_species_profiles_multipane",
    )
    save_current_figure_both(fig, stem)


def plot_all_star_multipanes(cases, out_dir):
    for star in STAR_ORDER:
        for scenario in SCENARIO_ORDER:
            plot_star_multipane(cases, star, scenario, out_dir)


# --------------------------------------------------

# --------------------------------------------------
# DIFFERENCE DIAGNOSTICS: PANELS, HIGHLIGHTS, METRICS
# --------------------------------------------------
DIFF_REFERENCE_RUN = "Run_1"
DIFF_REFERENCE_FSCALE = "1.00"
DELTA_CLIP = 3.0  # plot +/- dex; larger values are clipped visually only


def get_case_map(cases):
    """Map (star, scenario, run, fscale) -> case."""
    return {
        (str(c["star"]), str(c["scenario"]), str(c["run"]), str(c["fscale"])): c
        for c in cases
    }


def profile_for_species(species_data, sp):
    if species_data is None or "ALT" not in species_data or sp not in species_data:
        return None, None
    z = np.asarray(species_data["ALT"], dtype=float)
    y = np.asarray(species_data[sp], dtype=float)
    mask = np.isfinite(z) & np.isfinite(y) & (y > 0)
    if not np.any(mask):
        return None, None
    z = z[mask]
    y = y[mask]
    order = np.argsort(z)
    return z[order], y[order]


def delta_log_profile(case, ref_case, sp):
    """Return altitude grid and log10(case/ref) for one species.

    The reference profile is interpolated onto the case altitude grid in log10-space.
    """
    z, y = profile_for_species(case["entry"].get("species"), sp)
    zref, yref = profile_for_species(ref_case["entry"].get("species"), sp)
    if z is None or zref is None or len(zref) < 2:
        return None, None

    zmin = max(float(np.nanmin(z)), float(np.nanmin(zref)))
    zmax = min(float(np.nanmax(z)), float(np.nanmax(zref)))
    common = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    if np.count_nonzero(common) < 2:
        return None, None

    zc = z[common]
    logy = np.log10(y[common])
    logref = np.interp(zc, zref, np.log10(yref))
    delta = logy - logref
    finite = np.isfinite(zc) & np.isfinite(delta)
    if not np.any(finite):
        return None, None
    return zc[finite], delta[finite]


def clean_delta_axis(ax, *, show_xlabel=True, show_ylabel=True):
    ax.axvline(0.0, color="0.25", linewidth=0.8, alpha=0.7)
    ax.axvline(1.0, color="0.70", linewidth=0.5, alpha=0.7)
    ax.axvline(-1.0, color="0.70", linewidth=0.5, alpha=0.7)
    ax.set_xlim(-DELTA_CLIP, DELTA_CLIP)
    ax.grid(True, which="major", alpha=0.22, linewidth=0.6)
    ax.set_xlabel(r"$\Delta\log_{10}$(mixing ratio)" if show_xlabel else "")
    ax.set_ylabel("Altitude (km)" if show_ylabel else "")


def plot_star_difference_multipane(cases, star, scenario, out_dir,
                                   ref_run=DIFF_REFERENCE_RUN,
                                   ref_fscale=DIFF_REFERENCE_FSCALE,
                                   ylim_km=DIFF_FULL_YLIM_KM,
                                   suffix="full") :
    """3x3 delta-log profile panel for one star+scenario.

    Each panel shows log10(profile/reference), where the reference is
    Run_1, FSCALE=1.00 by default for the same star and scenario.
    """
    case_map = get_case_map(cases)
    ref = case_map.get((str(star), str(scenario), str(ref_run), str(ref_fscale)))
    if ref is None:
        print(f"Skipping difference panel for {star} scenario {scenario}: missing reference {ref_run}, FSCALE={ref_fscale}")
        return

    apply_publication_style()
    fig, axes = plt.subplots(
        len(RUN_ORDER),
        len(FSCALE_ORDER),
        figsize=FIGSIZE_STAR_MULTIPANE,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    plotted_any = set()
    finite_alts = []
    for i, run in enumerate(RUN_ORDER):
        for j, fscale in enumerate(FSCALE_ORDER):
            ax = axes[i, j]
            clean_delta_axis(ax, show_xlabel=(i == len(RUN_ORDER) - 1), show_ylabel=(j == 0))
            case = case_map.get((str(star), str(scenario), str(run), str(fscale)))
            if case is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            else:
                for sp in PLOT_SPECIES:
                    z, d = delta_log_profile(case, ref, sp)
                    if z is None:
                        continue
                    finite_alts.extend(z[np.isfinite(z)].tolist())
                    ax.plot(np.clip(d, -DELTA_CLIP, DELTA_CLIP), z,
                            color=SPECIES_COLORS.get(sp), linewidth=1.3,
                            label=species_label(sp))
                    plotted_any.add(sp)

            letter = PANEL_LETTERS[i * len(FSCALE_ORDER) + j]
            ax.text(0.03, 0.95, f"({letter})", transform=ax.transAxes, ha="left", va="top", fontweight="bold")
            if i == 0:
                ax.set_title(f"FSCALE = {fscale}")
            if j == 0:
                ax.text(-0.30, 0.5, RUN_LABELS.get(run, run), transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontsize=9)

    if ylim_km is not None:
        for ax in np.ravel(axes):
            ax.set_ylim(*ylim_km)
    elif finite_alts:
        ymin = min(0.0, float(np.nanmin(finite_alts)))
        ymax = float(np.nanmax(finite_alts))
        if ymax > ymin:
            pad = 0.02 * (ymax - ymin)
            for ax in np.ravel(axes):
                ax.set_ylim(ymin, ymax + pad)

    handles = [plt.Line2D([0], [0], color=SPECIES_COLORS[sp], lw=2.0) for sp in PLOT_SPECIES if sp in plotted_any]
    labels = [species_label(sp) for sp in PLOT_SPECIES if sp in plotted_any]
    if handles:
        fig.legend(handles, labels, title="Species", loc="center left",
                   bbox_to_anchor=(0.86, 0.5), frameon=False, borderaxespad=0.0)

    ylim_text = "" if ylim_km is None else f" | {ylim_km[0]:g}-{ylim_km[1]:g} km"
    fig.suptitle(
        f"{star.replace('_', ' ')} | scenario {scenario} | Δlog10 vs {RUN_LABELS.get(ref_run, ref_run)}, FSCALE={ref_fscale}{ylim_text}",
        y=0.985,
    )
    fig.tight_layout(rect=[0.06, 0.04, 0.84, 0.955])

    stem = os.path.join(
        out_dir,
        "star_difference",
        f"{sanitize_for_filename(star)}_scenario_{sanitize_for_filename(scenario)}_{sanitize_for_filename(suffix)}_delta_profiles_vs_{sanitize_for_filename(ref_run)}_fscale_{sanitize_for_filename(ref_fscale)}",
    )
    save_current_figure_both(fig, stem)


def plot_all_star_difference_multipanes(cases, out_dir):
    for star in STAR_ORDER:
        for scenario in SCENARIO_ORDER:
            plot_star_difference_multipane(cases, star, scenario, out_dir,
                                           ylim_km=DIFF_FULL_YLIM_KM, suffix="0_100km")
            plot_star_difference_multipane(cases, star, scenario, out_dir,
                                           ylim_km=DIFF_UPPER_YLIM_KM, suffix="50_100km")


def build_profile_difference_metrics(cases,
                                     ref_run=DIFF_REFERENCE_RUN,
                                     ref_fscale=DIFF_REFERENCE_FSCALE):
    """Quantify profile differences against the reference case for each star/scenario/species."""
    case_map = get_case_map(cases)
    rows = []
    for star in STAR_ORDER:
        for scenario in SCENARIO_ORDER:
            ref = case_map.get((str(star), str(scenario), str(ref_run), str(ref_fscale)))
            if ref is None:
                continue
            for run in RUN_ORDER:
                for fscale in FSCALE_ORDER:
                    case = case_map.get((str(star), str(scenario), str(run), str(fscale)))
                    if case is None:
                        continue
                    for sp in PLOT_SPECIES:
                        z, d = delta_log_profile(case, ref, sp)
                        if z is None:
                            continue
                        absd = np.abs(d)
                        imax = int(np.nanargmax(absd)) if np.any(np.isfinite(absd)) else None
                        rows.append({
                            "star": star,
                            "scenario": scenario,
                            "run": run,
                            "run_label": RUN_LABELS.get(run, run),
                            "fscale": fscale,
                            "species": sp,
                            "reference_run": ref_run,
                            "reference_fscale": ref_fscale,
                            "mean_delta_log10": float(np.nanmean(d)),
                            "mean_abs_delta_log10": float(np.nanmean(absd)),
                            "median_abs_delta_log10": float(np.nanmedian(absd)),
                            "max_abs_delta_log10": float(np.nanmax(absd)),
                            "altitude_of_max_abs_delta_km": float(z[imax]) if imax is not None else np.nan,
                            "delta_log10_at_max_abs_delta": float(d[imax]) if imax is not None else np.nan,
                            "integrated_abs_delta_log10_km": float(integrate_curve(absd, z)) if len(z) >= 2 else np.nan,
                            "n_altitude_points_compared": int(len(z)),
                            "out_file": case["entry"].get("out_file"),
                            "reference_out_file": ref["entry"].get("out_file"),
                        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["star"] = pd.Categorical(df["star"], categories=STAR_ORDER, ordered=True)
        df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER, ordered=True)
        df["run"] = pd.Categorical(df["run"], categories=RUN_ORDER, ordered=True)
        df["fscale"] = pd.Categorical(df["fscale"], categories=FSCALE_ORDER, ordered=True)
        df["species"] = pd.Categorical(df["species"], categories=PLOT_SPECIES, ordered=True)
        df = df.sort_values(["star", "scenario", "run", "fscale", "species"]).reset_index(drop=True)
    return df


def build_difference_highlights(diff_metrics_df, top_n=8):
    if diff_metrics_df.empty:
        return pd.DataFrame(), "No profile-difference metrics were generated.\n"

    work = diff_metrics_df.copy()
    is_ref = (work["run"].astype(str) == DIFF_REFERENCE_RUN) & (work["fscale"].astype(str) == DIFF_REFERENCE_FSCALE)
    work = work[~is_ref].copy()
    work = work[np.isfinite(work["max_abs_delta_log10"].astype(float))]
    if work.empty:
        return pd.DataFrame(), "Only reference panels were available; no highlights to report.\n"

    highlights = []
    for (star, scenario), g in work.groupby(["star", "scenario"], observed=True):
        g = g.sort_values("max_abs_delta_log10", ascending=False).head(top_n)
        highlights.append(g)
    out = pd.concat(highlights, ignore_index=True) if highlights else pd.DataFrame()

    lines = []
    lines.append("Profile-difference highlights")
    lines.append("=============================\n")
    lines.append(f"Reference for all differences: {RUN_LABELS.get(DIFF_REFERENCE_RUN, DIFF_REFERENCE_RUN)}, FSCALE={DIFF_REFERENCE_FSCALE}.\n")
    for (star, scenario), g in out.groupby(["star", "scenario"], observed=True):
        lines.append(f"{star} | scenario {scenario}")
        lines.append("-" * (len(str(star)) + len(str(scenario)) + 12))
        for _, row in g.head(top_n).iterrows():
            factor = 10 ** float(row["max_abs_delta_log10"])
            direction = "higher" if float(row["delta_log10_at_max_abs_delta"]) > 0 else "lower"
            lines.append(
                f"{row['species']} in {RUN_LABELS.get(str(row['run']), row['run'])}, FSCALE={row['fscale']}: "
                f"max |Δlog10|={float(row['max_abs_delta_log10']):.2f} dex "
                f"(~{factor:.1f}x), {direction} at {float(row['altitude_of_max_abs_delta_km']):.1f} km; "
                f"mean |Δlog10|={float(row['mean_abs_delta_log10']):.2f} dex."
            )
        lines.append("")
    return out, "\n".join(lines)


def save_difference_diagnostics(cases, out_dir):
    diff_df = build_profile_difference_metrics(cases)
    diff_csv = os.path.join(out_dir, "tables", "species_profile_difference_metrics.csv")
    diff_df.to_csv(diff_csv, index=False)
    print(f"Saved: {diff_csv}")

    highlights_df, highlight_text = build_difference_highlights(diff_df)
    highlights_csv = os.path.join(out_dir, "tables", "species_difference_highlights.csv")
    highlights_df.to_csv(highlights_csv, index=False)
    print(f"Saved: {highlights_csv}")

    highlights_txt = os.path.join(out_dir, "species_difference_highlights.txt")
    with open(highlights_txt, "w") as f:
        f.write(highlight_text)
    print(f"Saved: {highlights_txt}")

    plot_all_star_difference_multipanes(cases, out_dir)
    return diff_df, highlights_df
# TEXT SUMMARIES
# --------------------------------------------------
def build_text_summaries(metrics_df, comparison_df):
    summaries = []
    if metrics_df.empty or comparison_df.empty:
        return pd.DataFrame()

    for star in STAR_ORDER:
        star_lines = []
        star_comp = comparison_df[comparison_df["star"] == star]
        if star_comp.empty:
            continue

        # strongest run-driven changes: compare Run_2 and Run_3 against Run_1 in column_proxy
        run_comp = star_comp[
            (star_comp["comparison_type"] == "vs_Run1_same_fscale")
            & (star_comp["metric"] == "column_proxy")
            & (star_comp["run"].isin(["Run_2", "Run_3"]))
        ].copy()
        run_comp["abs_delta"] = run_comp["delta_log10"].abs()
        run_comp = run_comp.sort_values("abs_delta", ascending=False)

        added = 0
        for _, row in run_comp.iterrows():
            phrase = magnitude_phrase(row["delta_log10"])
            if phrase is None:
                continue
            direction = "higher" if row["delta_log10"] > 0 else "lower"
            line = (
                f"At FSCALE = {row['fscale']}, {row['species']} column proxy is {phrase} {direction} in "
                f"{RUN_LABELS.get(str(row['run']), row['run'])} relative to {RUN_LABELS['Run_1']}."
            )
            if line not in star_lines:
                star_lines.append(line)
                added += 1
            if added >= 4:
                break

        # strongest fscale-driven changes: compare 0.75/1.50 to 1.00 within same run
        fs_comp = star_comp[
            (star_comp["comparison_type"] == "vs_Fscale1.00_same_run")
            & (star_comp["metric"] == "column_proxy")
            & (star_comp["fscale"].isin(["0.75", "1.50"]))
        ].copy()
        fs_comp["abs_delta"] = fs_comp["delta_log10"].abs()
        fs_comp = fs_comp.sort_values("abs_delta", ascending=False)

        added = 0
        for _, row in fs_comp.iterrows():
            phrase = magnitude_phrase(row["delta_log10"])
            if phrase is None:
                continue
            direction = "higher" if row["delta_log10"] > 0 else "lower"
            line = (
                f"Within {RUN_LABELS.get(str(row['run']), row['run'])}, {row['species']} column proxy is {phrase} {direction} "
                f"at FSCALE = {row['fscale']} relative to FSCALE = 1.00."
            )
            if line not in star_lines:
                star_lines.append(line)
                added += 1
            if added >= 4:
                break

        # altitude shifts
        alt_comp = star_comp[
            (star_comp["comparison_type"] == "vs_Run1_same_fscale")
            & (star_comp["metric"] == "peak_alt_km")
            & (star_comp["run"].isin(["Run_2", "Run_3"]))
        ].copy()
        alt_comp["abs_delta"] = alt_comp["delta_log10"].abs()
        alt_comp = alt_comp.sort_values("abs_delta", ascending=False)
        added = 0
        for _, row in alt_comp.iterrows():
            if not np.isfinite(row["delta_log10"]) or abs(row["delta_log10"]) < 5:
                continue
            direction = "higher" if row["delta_log10"] > 0 else "lower"
            line = (
                f"{row['species']} peak altitude is {abs(row['delta_log10']):.1f} km {direction} in {RUN_LABELS.get(str(row['run']), row['run'])} "
                f"than in {RUN_LABELS['Run_1']} at FSCALE = {row['fscale']}."
            )
            if line not in star_lines:
                star_lines.append(line)
                added += 1
            if added >= 2:
                break

        if not star_lines:
            star_lines.append("No large species differences exceeded the automatic reporting thresholds.")

        summaries.append({
            "star": star,
            "summary_text": " ".join(star_lines),
        })
    return pd.DataFrame(summaries)


def write_text_summary_file(summary_df, out_dir):
    path = os.path.join(out_dir, "species_results_summary.txt")
    ensure_dir(out_dir)
    with open(path, "w") as f:
        f.write("Species comparison summary\n")
        f.write("==========================\n\n")
        for _, row in summary_df.iterrows():
            f.write(f"{row['star']}\n")
            f.write(f"{'-' * len(str(row['star']))}\n")
            f.write(str(row['summary_text']).strip() + "\n\n")
    print(f"Saved: {path}")


# --------------------------------------------------
# TABLES
# --------------------------------------------------
def save_latex_tables(metrics_df, out_dir):
    latex_dir = os.path.join(out_dir, "latex_tables")
    ensure_dir(latex_dir)

    if metrics_df.empty:
        return

    for star in STAR_ORDER:
        sdf = metrics_df[metrics_df["star"] == star].copy()
        if sdf.empty:
            continue
        table = sdf[["run_label", "fscale", "species", "column_proxy", "peak_mixing_ratio", "peak_alt_km", "upper_mean"]].copy()
        table = table.rename(columns={
            "run_label": "Run",
            "fscale": "FSCALE",
            "species": "Species",
            "column_proxy": "ColumnProxy",
            "peak_mixing_ratio": "PeakMixingRatio",
            "peak_alt_km": "PeakAltKm",
            "upper_mean": f"MeanAbove{int(UPPER_ATM_KM)}km",
        })
        latex = table.to_latex(index=False, float_format=lambda x: f"{x:.3e}" if np.isfinite(x) and abs(x) < 1e4 else f"{x:.3f}")
        path = os.path.join(latex_dir, f"{sanitize_for_filename(star)}_species_metrics.tex")
        with open(path, "w") as f:
            f.write(latex)
        print(f"Saved: {path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main(base_dir=BASE_DIR, out_dir=OUT_DIR):
    ensure_dir(out_dir)
    for sub in [
        "case_species",
        "star_overviews",
        "scenario_overviews",
        "run_overviews",
        "heatmaps",
        "metric_scatter",
        "star_multipane",
        "star_difference",
        "tables",
        "latex_tables",
    ]:
        ensure_dir(os.path.join(out_dir, sub))

    dataset = build_dataset(base_dir)
    cases = build_case_index(dataset)
    if not cases:
        print("No valid cases found.")
        return

    # Case plots
    for case in cases:
        plot_case_species(case, out_dir)

    # Overview pages
    for star in STAR_ORDER:
        plot_overview_pages([c for c in cases if c["star"] == star], "star", star, out_dir)
    for scenario in SCENARIO_ORDER:
        fscale_label = f"FSCALE_{SCENARIO_TO_FSCALE[scenario]}"
        plot_overview_pages([c for c in cases if c["scenario"] == scenario], "scenario", fscale_label, out_dir)
    for run in RUN_ORDER:
        plot_overview_pages([c for c in cases if c["run"] == run], "run", RUN_LABELS[run], out_dir)

    # Publication-ready multi-panel summary figures, one per star
    plot_all_star_multipanes(cases, out_dir)

    # Difference panels and profile-difference diagnostics
    save_difference_diagnostics(cases, out_dir)

    # Metrics and comparisons
    metrics_df = flatten_metrics(cases)
    metrics_csv = os.path.join(out_dir, "tables", "species_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved: {metrics_csv}")

    comparison_df = build_comparison_tables(metrics_df)
    comp_csv = os.path.join(out_dir, "tables", "species_comparisons.csv")
    comparison_df.to_csv(comp_csv, index=False)
    print(f"Saved: {comp_csv}")

    plot_heatmaps(comparison_df, out_dir)
    plot_metric_scatter(metrics_df, out_dir)

    summary_df = build_text_summaries(metrics_df, comparison_df)
    summary_csv = os.path.join(out_dir, "tables", "species_text_summaries.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")
    write_text_summary_file(summary_df, out_dir)

    save_latex_tables(metrics_df, out_dir)

    index_rows = [{
        "run": c["run"],
        "run_label": c["run_label"],
        "star": c["star"],
        "scenario": c["scenario"],
        "fscale": c["fscale"],
        "scale_dir": c["scale_dir"],
        "out_file": c["entry"].get("out_file"),
    } for c in cases]
    index_df = pd.DataFrame(index_rows)
    index_path = os.path.join(out_dir, "tables", "species_case_index.csv")
    index_df.to_csv(index_path, index=False)
    print(f"Saved: {index_path}")

    print("\nDone.")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create publication-ready species profile plots and comparison tables.")
    parser.add_argument("--base-dir", default=BASE_DIR, help="Root directory containing Run_* outputs.")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Directory where plots and tables are written.")
    args = parser.parse_args()

    main(base_dir=args.base_dir, out_dir=args.out_dir)
