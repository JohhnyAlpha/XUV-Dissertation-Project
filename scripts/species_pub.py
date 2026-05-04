import os
import re
from io import StringIO
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.expanduser("~/atmos/OutputStorage")
OUT_DIR = os.path.expanduser("~/atmos/BA/Plots/species_pub")
SAVE_DPI = 300
SHOW_PLOTS = False

STARS = {
    "18": "Epsilon_Eri",
    "24": "HD40307",
    "25": "HD85512",
    "26": "HD97658",
}
STAR_ORDER = ["18", "24", "25", "26"]
STAR_NAME_ORDER = [STARS[s] for s in STAR_ORDER]

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

SPECIES = ["CO2", "H2", "CH4", "CO", "O2", "O3", "H2O"]
PLOT_SPECIES = ["H2O", "CH4", "CO", "CO2", "O2", "O3", "H2"]

SPECIES_COLORS = {
    "H2O": "#1f77b4",
    "CH4": "#2ca02c",
    "CO": "#ff7f0e",
    "CO2": "#d62728",
    "O2": "#9467bd",
    "O3": "#17becf",
    "H2": "#8c564b",
}

CASE_FIGSIZE = (6.0, 7.0)
PAGE_FIGSIZE = (11.0, 8.5)
PANELS_PER_PAGE = 4
GRID_NROWS = 2
GRID_NCOLS = 2

TITLE_FONTSIZE = 14
SUBTITLE_FONTSIZE = 11
LABEL_FONTSIZE = 11
TICK_FONTSIZE = 9
LEGEND_FONTSIZE = 8
LINEWIDTH = 2.0

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

    return sorted([r for r in run_dirs if r in RUN_ORDER], key=lambda x: int(x.split("_")[1]))


def save_figure(fig, out_path):
    ensure_dir(os.path.dirname(out_path))
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {out_path}")
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


def style_axis(ax):
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def run_sort_key(run):
    return RUN_ORDER.index(run) if run in RUN_ORDER else 999


def scenario_sort_key(scenario):
    return SCENARIO_ORDER.index(str(scenario)) if str(scenario) in SCENARIO_ORDER else 999


def star_sort_key(star_name):
    return STAR_NAME_ORDER.index(star_name) if star_name in STAR_NAME_ORDER else 999


def panel_title(run, star, scenario, scale):
    fscale = SCENARIO_TO_FSCALE.get(str(scenario), str(scale))
    return f"{RUN_LABELS.get(run, run)} | {star} | FSCALE {fscale}"


# -----------------------------
# FILE DISCOVERY
# -----------------------------
def build_file_index(scale_dir):
    photochem_dir = os.path.join(scale_dir, "PHOTOCHEM_OUTPUT")
    scenario_files = defaultdict(lambda: {"out_file": None})
    pattern = re.compile(r"^(Run_\d+)_([0-9]+(?:\.[0-9]+)?)_(\d+)_(.+)$")

    if not os.path.isdir(photochem_dir):
        return {}

    for fname in os.listdir(photochem_dir):
        match = pattern.match(fname)
        if not match:
            continue
        _, _, scenario, suffix = match.groups()
        if suffix != "out.out":
            continue
        scenario_files[scenario]["out_file"] = os.path.join(photochem_dir, fname)

    return dict(sorted(scenario_files.items(), key=lambda kv: scenario_sort_key(kv[0])))


# -----------------------------
# PARSER
# -----------------------------
def parse_out_file(file_path):
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
        df = pd.read_csv(StringIO("\n".join(table_lines)), sep=r"\s+", engine="python")
    except Exception as e:
        print(f"Failed to parse mixing-ratio table in {file_path}: {e}")
        return None

    data = {}
    alt_col = pick_altitude_column(df)
    if alt_col is not None:
        data["ALT"] = to_numeric_array(df[alt_col])

    for sp in SPECIES:
        col = find_matching_column(df, sp)
        if col is not None:
            data[sp] = to_numeric_array(df[col])

    return data if data else None


# -----------------------------
# DATASET BUILD
# -----------------------------
def choose_scale_dirs(star_dir):
    choices = {}
    available = [d for d in os.listdir(star_dir) if os.path.isdir(os.path.join(star_dir, d))]
    for scenario, fscale in SCENARIO_TO_FSCALE.items():
        exact = None
        for d in available:
            try:
                if abs(float(d) - float(fscale)) < 1e-9:
                    exact = d
                    break
            except Exception:
                continue
        if exact is not None:
            choices[scenario] = exact
    return choices


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
            if str(star_id) not in STARS:
                continue

            star_name = star_name_from_id(star_id)
            data[run][star_name] = {}
            preferred_scale_dirs = choose_scale_dirs(star_dir)

            for scenario in SCENARIO_ORDER:
                scale = preferred_scale_dirs.get(scenario)
                if scale is None:
                    continue

                scale_dir = os.path.join(star_dir, scale)
                scenario_files = build_file_index(scale_dir)
                if scenario not in scenario_files:
                    continue

                out_file = scenario_files[scenario].get("out_file")
                print(f"Loading: {run} | {star_name} | scale={scale} | scenario={scenario}")
                print(f"  out_file: {out_file}")
                species = parse_out_file(out_file) if out_file else None

                data[run][star_name][scenario] = {
                    "scale": scale,
                    "scenario": scenario,
                    "fscale": SCENARIO_TO_FSCALE.get(scenario, scale),
                    "species": species,
                    "file": out_file,
                }

    return data


# -----------------------------
# FLATTEN / INDEX
# -----------------------------
def build_index_rows(dataset):
    rows = []
    for run in RUN_ORDER:
        if run not in dataset:
            continue
        for star in STAR_NAME_ORDER:
            if star not in dataset[run]:
                continue
            for scenario in SCENARIO_ORDER:
                entry = dataset[run][star].get(scenario)
                if entry is None:
                    continue
                rows.append({
                    "run": run,
                    "run_label": RUN_LABELS.get(run, run),
                    "star": star,
                    "scenario": scenario,
                    "fscale": entry.get("fscale"),
                    "scale_dir": entry.get("scale"),
                    "file": entry.get("file"),
                    "has_species": entry.get("species") is not None,
                })
    return pd.DataFrame(rows)


# -----------------------------
# PLOTTING PRIMITIVES
# -----------------------------
def plot_species_on_axis(ax, species_data, show_legend=True, subtitle=None):
    if species_data is None or "ALT" not in species_data:
        ax.text(0.5, 0.5, "No species data", ha="center", va="center", transform=ax.transAxes)
        style_axis(ax)
        return False

    alt = np.asarray(species_data["ALT"], dtype=float)
    plotted = False

    for sp in PLOT_SPECIES:
        if sp not in species_data:
            continue
        x = np.asarray(species_data[sp], dtype=float)
        mask = finite_positive_mask(x, alt)
        if not np.any(mask):
            continue
        ax.plot(
            x[mask],
            alt[mask],
            label=sp,
            linewidth=LINEWIDTH,
            color=SPECIES_COLORS.get(sp),
        )
        plotted = True

    ax.set_xscale("log")
    ax.set_xlabel("Mixing ratio", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Altitude", fontsize=LABEL_FONTSIZE)
    if subtitle:
        ax.set_title(subtitle, fontsize=SUBTITLE_FONTSIZE)
    style_axis(ax)

    if show_legend and plotted:
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="best", frameon=False)
    return plotted


# -----------------------------
# CASE PLOTS
# -----------------------------
def save_case_plots(dataset, out_dir):
    case_dir = os.path.join(out_dir, "case_species")
    ensure_dir(case_dir)

    for run in RUN_ORDER:
        if run not in dataset:
            continue
        for star in STAR_NAME_ORDER:
            if star not in dataset[run]:
                continue
            for scenario in SCENARIO_ORDER:
                entry = dataset[run][star].get(scenario)
                if entry is None:
                    continue

                fig, ax = plt.subplots(figsize=CASE_FIGSIZE)
                plot_species_on_axis(ax, entry.get("species"), show_legend=True)
                fig.suptitle(panel_title(run, star, scenario, entry.get("scale")), fontsize=TITLE_FONTSIZE)

                stem = (
                    f"species_{sanitize_for_filename(run)}_"
                    f"{sanitize_for_filename(star)}_"
                    f"scenario_{sanitize_for_filename(scenario)}_"
                    f"fscale_{sanitize_for_filename(entry.get('fscale'))}"
                )
                save_figure(fig, os.path.join(case_dir, f"{stem}.png"))

                fig, ax = plt.subplots(figsize=CASE_FIGSIZE)
                plot_species_on_axis(ax, entry.get("species"), show_legend=True)
                fig.suptitle(panel_title(run, star, scenario, entry.get("scale")), fontsize=TITLE_FONTSIZE)
                save_figure(fig, os.path.join(case_dir, f"{stem}.pdf"))


# -----------------------------
# MULTIPAGE OVERVIEWS
# -----------------------------
def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def save_multipage_overview(pages, pdf_path):
    ensure_dir(os.path.dirname(pdf_path))
    with PdfPages(pdf_path) as pdf:
        for fig in pages:
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, dpi=SAVE_DPI, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved: {pdf_path}")


def create_overview_pages(cases, document_title):
    pages = []

    for page_num, page_cases in enumerate(chunked(cases, PANELS_PER_PAGE), start=1):
        fig, axes = plt.subplots(GRID_NROWS, GRID_NCOLS, figsize=PAGE_FIGSIZE, sharex=False, sharey=False)
        axes = np.atleast_1d(axes).ravel()

        for ax, case in zip(axes, page_cases):
            plot_species_on_axis(ax, case["species"], show_legend=True, subtitle=case["subtitle"])

        for ax in axes[len(page_cases):]:
            ax.axis("off")

        suffix = f" (page {page_num})" if len(cases) > PANELS_PER_PAGE else ""
        fig.suptitle(document_title + suffix, fontsize=TITLE_FONTSIZE)
        pages.append(fig)

    return pages


def export_pages_as_png(pages, stem):
    for idx, fig in enumerate(pages, start=1):
        out_path = f"{stem}_page_{idx}.png"
        ensure_dir(os.path.dirname(out_path))
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        print(f"Saved: {out_path}")


def export_pages_as_pdf_copy(pages, stem):
    pdf_path = f"{stem}.pdf"
    ensure_dir(os.path.dirname(pdf_path))
    with PdfPages(pdf_path) as pdf:
        for fig in pages:
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {pdf_path}")


def save_star_overviews(dataset, out_dir):
    out_base = os.path.join(out_dir, "star_overviews")
    ensure_dir(out_base)

    for star in STAR_NAME_ORDER:
        cases = []
        for run in RUN_ORDER:
            if run not in dataset or star not in dataset[run]:
                continue
            for scenario in SCENARIO_ORDER:
                entry = dataset[run][star].get(scenario)
                if entry is None:
                    continue
                cases.append({
                    "species": entry.get("species"),
                    "subtitle": f"{RUN_LABELS.get(run, run)} | FSCALE {entry.get('fscale')}",
                })

        if not cases:
            continue

        pages = create_overview_pages(cases, f"Species overview | {star}")
        stem = os.path.join(out_base, f"species_overview_{sanitize_for_filename(star)}")
        export_pages_as_png(pages, stem)
        export_pages_as_pdf_copy(pages, stem)
        for fig in pages:
            plt.close(fig)


def save_scenario_overviews(dataset, out_dir):
    out_base = os.path.join(out_dir, "scenario_overviews")
    ensure_dir(out_base)

    for scenario in SCENARIO_ORDER:
        fscale = SCENARIO_TO_FSCALE.get(scenario, scenario)
        cases = []
        for run in RUN_ORDER:
            if run not in dataset:
                continue
            for star in STAR_NAME_ORDER:
                if star not in dataset[run]:
                    continue
                entry = dataset[run][star].get(scenario)
                if entry is None:
                    continue
                cases.append({
                    "species": entry.get("species"),
                    "subtitle": f"{RUN_LABELS.get(run, run)} | {star}",
                })

        if not cases:
            continue

        pages = create_overview_pages(cases, f"Species overview | Scenario {scenario} | FSCALE {fscale}")
        stem = os.path.join(out_base, f"species_overview_scenario_{scenario}_fscale_{sanitize_for_filename(fscale)}")
        export_pages_as_png(pages, stem)
        export_pages_as_pdf_copy(pages, stem)
        for fig in pages:
            plt.close(fig)


def save_run_overviews(dataset, out_dir):
    out_base = os.path.join(out_dir, "run_overviews")
    ensure_dir(out_base)

    for run in RUN_ORDER:
        if run not in dataset:
            continue
        cases = []
        for star in STAR_NAME_ORDER:
            if star not in dataset[run]:
                continue
            for scenario in SCENARIO_ORDER:
                entry = dataset[run][star].get(scenario)
                if entry is None:
                    continue
                cases.append({
                    "species": entry.get("species"),
                    "subtitle": f"{star} | FSCALE {entry.get('fscale')}",
                })

        if not cases:
            continue

        pages = create_overview_pages(cases, f"Species overview | {RUN_LABELS.get(run, run)}")
        stem = os.path.join(out_base, f"species_overview_{sanitize_for_filename(run)}")
        export_pages_as_png(pages, stem)
        export_pages_as_pdf_copy(pages, stem)
        for fig in pages:
            plt.close(fig)


# -----------------------------
# MAIN
# -----------------------------
def main(base_dir=BASE_DIR, out_dir=OUT_DIR):
    ensure_dir(out_dir)
    dataset = build_dataset(base_dir)

    index_df = build_index_rows(dataset)
    csv_path = os.path.join(out_dir, "species_overview_index.csv")
    index_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    save_case_plots(dataset, out_dir)
    save_star_overviews(dataset, out_dir)
    save_scenario_overviews(dataset, out_dir)
    save_run_overviews(dataset, out_dir)

    print("\nDone.")
    print(f"Species plots written to: {out_dir}")


if __name__ == "__main__":
    main()
