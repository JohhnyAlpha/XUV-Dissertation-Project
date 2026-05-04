#!/usr/bin/env python3
"""
Extract UV, Vis, IR and All emission summary rows from ATMOS clima_allout files
stored in the standard OutputStorage directory tree, then create publication-style
multi-frame charts for each FSCALE.

Each FSCALE figure contains four panels:
  (a) UV
  (b) Vis
  (c) IR
  (d) Total stacked UV + Vis + IR

Within each panel, all stars are plotted side by side. For each star, Run_1,
Run_2 and Run_3 are shown as adjacent bars using the same run-age hatch legend
style used in the existing publication examples.

Expected directory structure:

  ~/atmos/OutputStorage/
    Run_1/
      18/
        0.75/
          CLIMA_OUTPUT/
            Run_1_0.75_1_clima_allout.tab
            Run_1_0.75_2_clima_allout.tab
            Run_1_0.75_3_clima_allout.tab

Default plotted value is fsol_erg_cm2_s. Use --value-column to plot another
available numeric column.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(os.path.expanduser("~/atmos/OutputStorage"))
OUT_DIR = Path(os.path.expanduser("~/atmos/BA/Plots/stellar_emissions"))

STARS = {
    "18": "Epsilon_Eri",
    "24": "HD_40307",
    "25": "HD_85512",
    "26": "HD_97658",
}

# Cleaner axis labels matching the publication-style example.
STAR_PLOT_LABELS = {
    "Epsilon_Eri": "Epsilon_Eri",
    "HD_40307": "HD40307",
    "HD_85512": "HD85512",
    "HD_97658": "HD97658",
}

RUN_AGES = {
    "Run_1": "2.7 Gyr star - normal flare activity",
    "Run_2": "0.5 Gyr star - high flare activity",
    "Run_3": "9.9 Gyr star - low flare activity",
}

RUN_HATCHES = {
    "Run_1": "",
    "Run_2": "///",
    "Run_3": "...",
}

RUN_ORDER = ["Run_1", "Run_2", "Run_3"]
FSCALE_ORDER = ["0.75", "1.0", "1.5"]
BANDS = ["UV", "Vis", "IR"]

# Earlier user colour request: UV purple, Vis green, IR red.
BAND_COLOURS = {"UV": "purple", "Vis": "green", "IR": "red"}

SAVE_DPI = 300
SHOW_PLOTS = False

COLUMNS = [
    "band",
    "wavll_um",
    "wavlu_um",
    "fsol_erg_cm2_s",
    "fsod_w_m2_um",
    "fdnsol_1_erg_cm2_s",
    "fupsol_1_erg_cm2_s",
    "fdnsol_2_erg_cm2_s",
    "fupsol_2_erg_cm2_s",
    "pdnsol_1_umol_m2_s",
    "pdnsol_2_umol_m2_s",
    "patten_fraction",
    "pdnsod_2_mol_m2_s_m",
]
VALUE_COLUMNS = COLUMNS[3:]

# -----------------------------
# HELPERS
# -----------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def numeric_sort_key(value: object):
    try:
        return float(value)
    except Exception:
        return str(value)


def run_sort_key(run: str) -> int:
    m = re.search(r"(\d+)$", str(run))
    return int(m.group(1)) if m else 999999


def fscale_key(value: object) -> str:
    try:
        return f"{float(value):.2f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def sanitize_for_filename(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def star_name_from_id(star_id: object) -> str:
    return STARS.get(str(star_id), str(star_id))


def plot_star_label(star: object) -> str:
    return STAR_PLOT_LABELS.get(str(star), str(star).replace("_", ""))


def normalize_band(raw: str) -> str:
    lower = raw.strip().lower()
    if lower == "uv":
        return "UV"
    if lower == "vis":
        return "Vis"
    if lower == "ir":
        return "IR"
    if lower == "all":
        return "All"
    return raw.strip()


def discover_run_dirs(base_dir: Path) -> list[Path]:
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    return sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and re.fullmatch(r"Run_\d+", p.name)],
        key=lambda p: run_sort_key(p.name),
    )


def discover_allout_files(base_dir: Path) -> list[dict[str, object]]:
    """Find all standard clima_allout files using DIRECTORY metadata as truth.

    The OutputStorage tree encodes the actual scenario:
        Run_N / star_id / fscale / CLIMA_OUTPUT / file

    Some copied/restarted ATMOS files can have stale run/fscale tokens in the
    filename, so run, star, and fscale are taken from the directory structure.
    The filename is used only to recover the iteration number when possible.
    """
    found: list[dict[str, object]] = []
    pattern = re.compile(
        r"^(Run_\d+)_([0-9]+(?:\.[0-9]+)?)_(\d+)_clima_allout\.tab$",
        flags=re.IGNORECASE,
    )

    for run_dir in discover_run_dirs(base_dir):
        run_name = run_dir.name

        for star_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()], key=lambda p: numeric_sort_key(p.name)):
            star_id = star_dir.name
            star_name = star_name_from_id(star_id)

            for scale_dir in sorted([p for p in star_dir.iterdir() if p.is_dir()], key=lambda p: numeric_sort_key(p.name)):
                dir_fscale = fscale_key(scale_dir.name)

                candidate_dirs = []
                clima_output = scale_dir / "CLIMA_OUTPUT"
                if clima_output.is_dir():
                    candidate_dirs.append(clima_output)

                # Fallback for slightly different layouts: search the scale folder too.
                candidate_dirs.append(scale_dir)

                files = []
                for candidate_dir in candidate_dirs:
                    files.extend(candidate_dir.rglob("*_clima_allout.tab"))
                files = sorted(set(files))

                for file_path in files:
                    match = pattern.match(file_path.name)
                    iteration = match.group(3) if match else "unknown"

                    found.append(
                        {
                            "path": file_path,
                            # DIRECTORY METADATA IS CANONICAL.
                            "run": run_name,
                            "run_age": RUN_AGES.get(run_name, run_name),
                            "star_id": star_id,
                            "star": star_name,
                            "fscale": dir_fscale,
                            "iteration": str(iteration),
                            # Filename metadata is retained only for diagnostics.
                            "filename_run": match.group(1) if match else "",
                            "filename_fscale": fscale_key(match.group(2)) if match else "",
                        }
                    )

    return found

def parse_star_from_file(text: str, fallback: str) -> str:
    match = re.search(r"OUTPUT\s+FILES\s+FOR\s+THE\s+(\S+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else fallback

# -----------------------------
# PARSING
# -----------------------------
def parse_summary_rows(file_record: dict[str, object]) -> list[dict[str, object]]:
    path = Path(file_record["path"])
    text = path.read_text(errors="ignore")

    # IMPORTANT: group/plot stars by the directory star_id, not only by the
    # star name printed inside the allout file. Some ATMOS allout files can
    # repeat the same internal atmosphere label, which would collapse multiple
    # stars into one x-axis category.
    star_id = str(file_record["star_id"])
    star = str(file_record["star"])
    if star == star_id:
        star = parse_star_from_file(text, star)
    star_key = star_id

    # Keep the final occurrence of each band, because the final UV/Vis/IR/All
    # block near the bottom is the requested reporting block.
    rows_by_band: dict[str, dict[str, object]] = {}
    summary_re = re.compile(r"^\s*(UV|Vis|IR|All)\s+(.+?)\s*$", flags=re.IGNORECASE)

    for line in text.splitlines():
        match = summary_re.match(line)
        if not match:
            continue

        band = normalize_band(match.group(1))
        numbers = match.group(2).split()
        if len(numbers) != 12:
            continue

        row = {
            "star": star,
            "star_id": star_id,
            "star_key": star_key,
            "run": str(file_record["run"]),
            "run_age": str(file_record["run_age"]),
            "fscale": fscale_key(file_record["fscale"]),
            "iteration": str(file_record["iteration"]),
            "source_file": path.name,
            "source_path": str(path),
            "filename_run": str(file_record.get("filename_run", "")),
            "filename_fscale": str(file_record.get("filename_fscale", "")),
            "band": band,
        }

        for column, value in zip(COLUMNS[1:], numbers):
            row[column] = float(value)

        rows_by_band[band] = row

    return [rows_by_band[band] for band in ["UV", "Vis", "IR", "All"] if band in rows_by_band]


def load_emission_dataframe(base_dir: Path) -> pd.DataFrame:
    records = discover_allout_files(base_dir)
    if not records:
        raise FileNotFoundError(f"No *_clima_allout.tab files found under: {base_dir}")

    rows: list[dict[str, object]] = []
    for record in records:
        rows.extend(parse_summary_rows(record))

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No UV/Vis/IR/All summary rows were found in the discovered files.")

    df["run_number"] = df["run"].map(run_sort_key)
    df["fscale_number"] = pd.to_numeric(df["fscale"], errors="coerce")
    df["iteration_number"] = pd.to_numeric(df["iteration"], errors="coerce")

    return df.sort_values(
        ["star_key", "run_number", "fscale_number", "iteration_number", "band"],
        na_position="last",
    ).reset_index(drop=True)


def select_iteration(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return one iteration per star/run/fscale/band for publication plots."""
    out = df.copy()
    if mode.lower() in {"all", "none"}:
        return out

    if mode.lower() in {"latest", "final", "max"}:
        idx = (
            out.sort_values("iteration_number")
            .groupby(["star_key", "run", "fscale", "band"], dropna=False)["iteration_number"]
            .idxmax()
        )
        return out.loc[idx].reset_index(drop=True)

    # Otherwise treat the mode as an explicit iteration number, for example --iteration 3.
    return out[out["iteration"].astype(str) == str(mode)].reset_index(drop=True)

# -----------------------------
# DIAGNOSTICS
# -----------------------------

def sha256_file(path_value: object) -> str:
    try:
        path = Path(str(path_value))
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def write_star_pair_diagnostic(
    df: pd.DataFrame,
    output_dir: Path,
    iteration_label: str,
    left_star_key: str = "18",
    right_star_key: str = "24",
    value_column: str = "fsol_erg_cm2_s",
) -> Path | None:
    """Compare two plotted stars across FSCALE, Run and band.

    This is useful for checking whether two visually similar stars are genuinely
    similar in the extracted ATMOS outputs, or whether the same source file has
    accidentally been read for both stars.
    """
    left = df[(df["star_key"].astype(str) == str(left_star_key)) & (df["band"].isin(BANDS))].copy()
    right = df[(df["star_key"].astype(str) == str(right_star_key)) & (df["band"].isin(BANDS))].copy()

    if left.empty or right.empty:
        print(
            f"Pair diagnostic skipped: missing star_key {left_star_key} or {right_star_key} "
            "in selected plotting dataframe."
        )
        return None

    keys = ["fscale", "run", "band"]
    lcols = keys + ["star", value_column, "source_file", "source_path"]
    rcols = keys + ["star", value_column, "source_file", "source_path"]

    merged = left[lcols].merge(
        right[rcols],
        on=keys,
        how="outer",
        suffixes=("_left", "_right"),
    )

    merged["abs_diff"] = merged[f"{value_column}_right"] - merged[f"{value_column}_left"]
    denom = merged[f"{value_column}_left"].replace(0, np.nan)
    merged["pct_diff_right_vs_left"] = 100.0 * merged["abs_diff"] / denom
    merged["source_path_same"] = merged["source_path_left"].astype(str) == merged["source_path_right"].astype(str)

    left_hashes = {p: sha256_file(p) for p in merged["source_path_left"].dropna().unique()}
    right_hashes = {p: sha256_file(p) for p in merged["source_path_right"].dropna().unique()}
    merged["source_sha256_left"] = merged["source_path_left"].map(left_hashes)
    merged["source_sha256_right"] = merged["source_path_right"].map(right_hashes)
    merged["source_file_content_identical"] = (
        (merged["source_sha256_left"].astype(str) != "")
        & (merged["source_sha256_left"].astype(str) == merged["source_sha256_right"].astype(str))
    )

    merged = merged.sort_values(["fscale", "run", "band"], key=lambda col: col.map(numeric_sort_key) if col.name == "fscale" else col)

    out_path = output_dir / f"epsilon_eri_vs_hd40307_diagnostic_{sanitize_for_filename(iteration_label)}.csv"
    merged.to_csv(out_path, index=False)

    value_close = merged["pct_diff_right_vs_left"].abs().dropna()
    min_pct = value_close.min() if not value_close.empty else np.nan
    max_pct = value_close.max() if not value_close.empty else np.nan
    identical_file_rows = int(merged["source_file_content_identical"].sum())
    same_path_rows = int(merged["source_path_same"].sum())

    print(f"Saved Epsilon_Eri vs HD40307 diagnostic: {out_path}")
    print(f"  Rows compared: {len(merged)}")
    print(f"  Absolute percent-difference range: {min_pct:.6g}% to {max_pct:.6g}%")
    print(f"  Same source path rows: {same_path_rows}")
    print(f"  Identical source-file content rows: {identical_file_rows}")
    if identical_file_rows or same_path_rows:
        print("  WARNING: at least one compared row uses the same/identical source file content.")
    else:
        print("  Source files are distinct by path and SHA-256 content for the compared rows.")

    return out_path


# -----------------------------
# PLOTTING
# -----------------------------
def make_legend_handles() -> tuple[list[Patch], list[Patch]]:
    band_handles = [
        Patch(facecolor=BAND_COLOURS[band], edgecolor="black", label=band)
        for band in BANDS
    ]
    run_handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=RUN_HATCHES.get(run, ""),
            label=RUN_AGES.get(run, run),
        )
        for run in RUN_ORDER
    ]
    return band_handles, run_handles


def get_value(panel_data: pd.DataFrame, star_key: str, run: str, fscale: str, band: str, value_column: str) -> float:
    match = panel_data[
        (panel_data["star_key"] == star_key)
        & (panel_data["run"] == run)
        & (panel_data["fscale"].map(fscale_key) == fscale_key(fscale))
        & (panel_data["band"] == band)
    ]
    if match.empty:
        return np.nan
    return float(match.iloc[0][value_column])


def draw_grouped_band_panel(ax, data: pd.DataFrame, star_keys: list[str], fscale: str, band: str, value_column: str) -> None:
    x = np.arange(len(star_keys))
    width = 0.22
    offsets = np.linspace(-width, width, len(RUN_ORDER))

    for run, offset in zip(RUN_ORDER, offsets):
        values = [get_value(data, star_key, run, fscale, band, value_column) for star_key in star_keys]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=BAND_COLOURS[band],
            edgecolor="black",
            linewidth=0.7,
            hatch=RUN_HATCHES.get(run, ""),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([plot_star_label(data.loc[data["star_key"] == star_key, "star"].iloc[0]) for star_key in star_keys], rotation=20, ha="right")
    ax.set_ylabel("FSOL" if value_column == "fsol_erg_cm2_s" else value_column)
    ax.grid(axis="y", alpha=0.20)


def draw_stacked_total_panel(ax, data: pd.DataFrame, star_keys: list[str], fscale: str, value_column: str) -> None:
    x = np.arange(len(star_keys))
    width = 0.22
    offsets = np.linspace(-width, width, len(RUN_ORDER))

    for run, offset in zip(RUN_ORDER, offsets):
        bottoms = np.zeros(len(star_keys), dtype=float)
        for band in BANDS:
            values = np.array([get_value(data, star_key, run, fscale, band, value_column) for star_key in star_keys], dtype=float)
            values = np.nan_to_num(values, nan=0.0)
            ax.bar(
                x + offset,
                values,
                width=width,
                bottom=bottoms,
                color=BAND_COLOURS[band],
                edgecolor="black",
                linewidth=0.7,
                hatch=RUN_HATCHES.get(run, ""),
            )
            bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels([plot_star_label(data.loc[data["star_key"] == star_key, "star"].iloc[0]) for star_key in star_keys], rotation=20, ha="right")
    ax.set_ylabel("Stacked FSOL" if value_column == "fsol_erg_cm2_s" else f"Stacked {value_column}")
    ax.grid(axis="y", alpha=0.20)


def set_common_ylim_for_band_axes(axes: list, data: pd.DataFrame, fscale: str, value_column: str) -> None:
    # Use a separate sensible scale per panel, matching the example layout.
    for ax, band in zip(axes[:3], BANDS):
        vals = data[
            (data["fscale"].map(fscale_key) == fscale_key(fscale))
            & (data["band"] == band)
        ][value_column]
        max_val = pd.to_numeric(vals, errors="coerce").max()
        if pd.notna(max_val) and max_val > 0:
            ax.set_ylim(0, float(max_val) * 1.12)

    total_by_case = (
        data[(data["fscale"].map(fscale_key) == fscale_key(fscale)) & (data["band"].isin(BANDS))]
        .groupby(["star_key", "run"], dropna=False)[value_column]
        .sum()
    )
    max_total = pd.to_numeric(total_by_case, errors="coerce").max()
    if pd.notna(max_total) and max_total > 0:
        axes[3].set_ylim(0, float(max_total) * 1.12)


def plot_fscale_publication_summary(df: pd.DataFrame, fscale: str, value_column: str, output_dir: Path) -> Path | None:
    data = df[(df["fscale"].map(fscale_key) == fscale_key(fscale)) & (df["band"].isin(BANDS))].copy()
    if data.empty:
        print(f"No data for FSCALE {fscale}; skipping.")
        return None

    # Keep stars in configured ID order. Plot/group by star_key so duplicate
    # internal allout labels do not collapse separate stars into one category.
    configured_star_keys = sorted(STARS.keys(), key=numeric_sort_key)
    present_star_keys = list(data["star_key"].drop_duplicates())
    star_keys = [s for s in configured_star_keys if s in present_star_keys] + [s for s in present_star_keys if s not in configured_star_keys]

    if len(star_keys) < 4:
        print(f"Warning: FSCALE {fscale} has {len(star_keys)} plotted star categories: {star_keys}")
        print("  Check discovered counts in uv_vis_ir_selected_iteration_*.csv if this is unexpected.")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), squeeze=False)
    ax_uv, ax_vis, ax_ir, ax_total = axes.ravel()

    draw_grouped_band_panel(ax_uv, data, star_keys, fscale, "UV", value_column)
    draw_grouped_band_panel(ax_vis, data, star_keys, fscale, "Vis", value_column)
    draw_grouped_band_panel(ax_ir, data, star_keys, fscale, "IR", value_column)
    draw_stacked_total_panel(ax_total, data, star_keys, fscale, value_column)

    titles = ["(a) UV", "(b) Vis", "(c) IR", "(d) Total"]
    for ax, title in zip([ax_uv, ax_vis, ax_ir, ax_total], titles):
        ax.set_title(title)

    set_common_ylim_for_band_axes([ax_uv, ax_vis, ax_ir, ax_total], data, fscale, value_column)

    band_handles, run_handles = make_legend_handles()
    fig.legend(
        handles=band_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
    )
    fig.legend(
        handles=run_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.890),
        ncol=3,
        frameon=False,
    )

    fig.suptitle(f"FSCALE = {fscale}: spectral summary by star and stellar activity", y=0.985, fontsize=14)
    fig.tight_layout(rect=[0.03, 0.03, 1.00, 0.84])

    out_file = output_dir / "fscale_publication_panels" / f"fscale_{sanitize_for_filename(fscale).replace('.', 'p')}_publication_summary.jpeg"
    ensure_dir(out_file.parent)
    fig.savefig(out_file, dpi=SAVE_DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return out_file


def plot_all_fscale_publication_summaries(df: pd.DataFrame, value_column: str, output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    present = {fscale_key(v) for v in df["fscale"].dropna().unique()}
    ordered_fscales = [fscale_key(v) for v in FSCALE_ORDER if fscale_key(v) in present]
    ordered_fscales += sorted([v for v in present if v not in ordered_fscales], key=numeric_sort_key)

    for fscale in ordered_fscales:
        out = plot_fscale_publication_summary(df, fscale, value_column, output_dir)
        if out is not None:
            outputs.append(out)
    return outputs


# -----------------------------
# EXTRA ANALYSIS PLOTS: DIFFERENCES, NORMALISED SPECTRA, LOG RESIDUALS
# -----------------------------
def pivot_bands_for_cases(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """One row per star/run/fscale with UV, Vis, IR, total and normalised fractions."""
    band_df = df[df["band"].isin(BANDS)].copy()
    wide = (
        band_df.pivot_table(
            index=["star_key", "star", "run", "run_age", "fscale"],
            columns="band",
            values=value_column,
            aggfunc="first",
        )
        .reset_index()
    )
    for band in BANDS:
        if band not in wide.columns:
            wide[band] = np.nan
    wide["Total"] = wide[BANDS].sum(axis=1, min_count=1)
    for band in BANDS:
        wide[f"{band}_fraction"] = wide[band] / wide["Total"].replace(0, np.nan)
    return wide


def get_case_value(wide: pd.DataFrame, star_key: str, run: str, fscale: str, column: str) -> float:
    match = wide[
        (wide["star_key"].astype(str) == str(star_key))
        & (wide["run"].astype(str) == str(run))
        & (wide["fscale"].map(fscale_key) == fscale_key(fscale))
    ]
    if match.empty or column not in match.columns:
        return np.nan
    return float(match.iloc[0][column])


def plot_normalised_spectral_fractions(df: pd.DataFrame, value_column: str, output_dir: Path) -> list[Path]:
    """Stacked percentage composition plots: UV/Vis/IR as fractions of total."""
    outputs = []
    wide = pivot_bands_for_cases(df, value_column)
    present_fscales = sorted({fscale_key(v) for v in wide["fscale"].dropna().unique()}, key=numeric_sort_key)
    configured_star_keys = sorted(STARS.keys(), key=numeric_sort_key)

    for fs in present_fscales:
        data = wide[wide["fscale"].map(fscale_key) == fscale_key(fs)].copy()
        if data.empty:
            continue

        present_star_keys = list(data["star_key"].astype(str).drop_duplicates())
        star_keys = [s for s in configured_star_keys if s in present_star_keys] + [s for s in present_star_keys if s not in configured_star_keys]

        fig, ax = plt.subplots(figsize=(12.0, 6.6))
        x = np.arange(len(star_keys))
        width = 0.22
        offsets = np.linspace(-width, width, len(RUN_ORDER))

        for run, offset in zip(RUN_ORDER, offsets):
            bottoms = np.zeros(len(star_keys), dtype=float)
            for band in BANDS:
                vals = np.array([get_case_value(data, sk, run, fs, f"{band}_fraction") for sk in star_keys], dtype=float)
                vals = np.nan_to_num(vals, nan=0.0)
                ax.bar(
                    x + offset,
                    vals,
                    width=width,
                    bottom=bottoms,
                    color=BAND_COLOURS[band],
                    edgecolor="black",
                    linewidth=0.7,
                    hatch=RUN_HATCHES.get(run, ""),
                    label=band if run == RUN_ORDER[0] else None,
                )
                bottoms += vals

        ax.set_title(f"FSCALE = {fs}: normalised spectral composition")
        ax.set_ylabel("Fraction of UV + Vis + IR total")
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([plot_star_label(data.loc[data["star_key"].astype(str) == sk, "star"].iloc[0]) for sk in star_keys], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.20)

        band_handles, run_handles = make_legend_handles()
        fig.legend(handles=band_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False)
        fig.legend(handles=run_handles, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=3, frameon=False)
        fig.tight_layout(rect=[0.03, 0.03, 1.00, 0.84])

        out = output_dir / "diagnostic_delta_plots" / f"fscale_{sanitize_for_filename(fs).replace('.', 'p')}_normalised_spectral_fractions.jpeg"
        ensure_dir(out.parent)
        fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
        if SHOW_PLOTS:
            plt.show()
        plt.close(fig)
        outputs.append(out)
    return outputs


def compute_reference_comparison(df: pd.DataFrame, value_column: str, reference_star_key: str = "18") -> pd.DataFrame:
    """Compare every star to the reference star for each fscale/run/band."""
    data = df[df["band"].isin(BANDS)].copy()
    keys = ["fscale", "run", "band"]
    ref = data[data["star_key"].astype(str) == str(reference_star_key)][keys + [value_column]].rename(columns={value_column: "reference_value"})
    comp = data.merge(ref, on=keys, how="left")
    comp["absolute_difference"] = comp[value_column] - comp["reference_value"]
    comp["delta_percent_vs_reference"] = 100.0 * comp["absolute_difference"] / comp["reference_value"].replace(0, np.nan)
    comp["log10_residual_vs_reference"] = np.log10(comp[value_column] / comp["reference_value"].replace(0, np.nan))
    return comp


def get_comparison_value(comp: pd.DataFrame, star_key: str, run: str, fscale: str, band: str, column: str) -> float:
    match = comp[
        (comp["star_key"].astype(str) == str(star_key))
        & (comp["run"].astype(str) == str(run))
        & (comp["fscale"].map(fscale_key) == fscale_key(fscale))
        & (comp["band"].astype(str) == str(band))
    ]
    if match.empty:
        return np.nan
    return float(match.iloc[0][column])


def plot_delta_or_log_residual_panels(
    comp: pd.DataFrame,
    output_dir: Path,
    column: str,
    ylabel: str,
    filename_suffix: str,
    reference_star_key: str = "18",
) -> list[Path]:
    """2x2 panels: UV, Vis, IR, and mean absolute diagnostic across bands."""
    outputs = []
    present_fscales = sorted({fscale_key(v) for v in comp["fscale"].dropna().unique()}, key=numeric_sort_key)
    configured_star_keys = sorted(STARS.keys(), key=numeric_sort_key)

    for fs in present_fscales:
        data = comp[comp["fscale"].map(fscale_key) == fscale_key(fs)].copy()
        if data.empty:
            continue

        present_star_keys = list(data["star_key"].astype(str).drop_duplicates())
        star_keys = [s for s in configured_star_keys if s in present_star_keys] + [s for s in present_star_keys if s not in configured_star_keys]

        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), squeeze=False)
        axes_flat = axes.ravel()
        x = np.arange(len(star_keys))
        width = 0.22
        offsets = np.linspace(-width, width, len(RUN_ORDER))

        for ax, band, title in zip(axes_flat[:3], BANDS, ["(a) UV", "(b) Vis", "(c) IR"]):
            for run, offset in zip(RUN_ORDER, offsets):
                vals = [get_comparison_value(data, sk, run, fs, band, column) for sk in star_keys]
                ax.bar(
                    x + offset,
                    vals,
                    width=width,
                    color=BAND_COLOURS[band],
                    edgecolor="black",
                    linewidth=0.7,
                    hatch=RUN_HATCHES.get(run, ""),
                )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels([plot_star_label(data.loc[data["star_key"].astype(str) == sk, "star"].iloc[0]) for sk in star_keys], rotation=20, ha="right")
            ax.grid(axis="y", alpha=0.20)

        ax = axes_flat[3]
        # Mean absolute magnitude across UV/Vis/IR per star/run. This highlights small deviations compactly.
        for run, offset in zip(RUN_ORDER, offsets):
            vals = []
            for sk in star_keys:
                subset = data[(data["star_key"].astype(str) == str(sk)) & (data["run"].astype(str) == str(run)) & (data["band"].isin(BANDS))]
                vals.append(float(np.nanmean(np.abs(pd.to_numeric(subset[column], errors="coerce")))) if not subset.empty else np.nan)
            ax.bar(
                x + offset,
                vals,
                width=width,
                color="white",
                edgecolor="black",
                linewidth=0.7,
                hatch=RUN_HATCHES.get(run, ""),
            )
        ax.set_title("(d) Mean |deviation|")
        ax.set_ylabel(f"Mean |{ylabel}|")
        ax.set_xticks(x)
        ax.set_xticklabels([plot_star_label(data.loc[data["star_key"].astype(str) == sk, "star"].iloc[0]) for sk in star_keys], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.20)

        band_handles, run_handles = make_legend_handles()
        fig.legend(handles=band_handles, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3, frameon=False)
        fig.legend(handles=run_handles, loc="upper center", bbox_to_anchor=(0.5, 0.890), ncol=3, frameon=False)
        ref_name = STARS.get(str(reference_star_key), str(reference_star_key))
        fig.suptitle(f"FSCALE = {fs}: {ylabel} relative to {ref_name}", y=0.985, fontsize=14)
        fig.tight_layout(rect=[0.03, 0.03, 1.00, 0.84])

        out = output_dir / "diagnostic_delta_plots" / f"fscale_{sanitize_for_filename(fs).replace('.', 'p')}_{filename_suffix}.jpeg"
        ensure_dir(out.parent)
        fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
        if SHOW_PLOTS:
            plt.show()
        plt.close(fig)
        outputs.append(out)
    return outputs


def write_comparison_tables(df: pd.DataFrame, value_column: str, output_dir: Path, reference_star_key: str = "18") -> tuple[Path, Path]:
    comp = compute_reference_comparison(df, value_column, reference_star_key=reference_star_key)
    out_dir = output_dir / "diagnostic_delta_plots"
    ensure_dir(out_dir)
    full_csv = out_dir / f"star_deltas_vs_{STARS.get(reference_star_key, reference_star_key)}.csv"
    comp.to_csv(full_csv, index=False)

    summary = (
        comp[comp["star_key"].astype(str) != str(reference_star_key)]
        .groupby(["star_key", "star", "fscale", "run"], dropna=False)
        .agg(
            max_abs_delta_percent=("delta_percent_vs_reference", lambda s: float(np.nanmax(np.abs(pd.to_numeric(s, errors="coerce"))))),
            mean_abs_delta_percent=("delta_percent_vs_reference", lambda s: float(np.nanmean(np.abs(pd.to_numeric(s, errors="coerce"))))),
            max_abs_log10_residual=("log10_residual_vs_reference", lambda s: float(np.nanmax(np.abs(pd.to_numeric(s, errors="coerce"))))),
            mean_abs_log10_residual=("log10_residual_vs_reference", lambda s: float(np.nanmean(np.abs(pd.to_numeric(s, errors="coerce"))))),
        )
        .reset_index()
        .sort_values(["fscale", "star_key", "run"], key=lambda col: col.map(numeric_sort_key) if col.name in {"fscale", "star_key"} else col)
    )
    summary_csv = out_dir / f"star_delta_summary_vs_{STARS.get(reference_star_key, reference_star_key)}.csv"
    summary.to_csv(summary_csv, index=False)
    return full_csv, summary_csv


def make_extra_diagnostic_outputs(df_plot: pd.DataFrame, value_column: str, output_dir: Path, reference_star_key: str = "18") -> list[Path]:
    outputs: list[Path] = []
    full_csv, summary_csv = write_comparison_tables(df_plot, value_column, output_dir, reference_star_key=reference_star_key)
    outputs.extend([full_csv, summary_csv])

    comp = compute_reference_comparison(df_plot, value_column, reference_star_key=reference_star_key)
    outputs.extend(plot_delta_or_log_residual_panels(comp, output_dir, "delta_percent_vs_reference", "Δ%", "delta_percent_vs_reference", reference_star_key))
    outputs.extend(plot_delta_or_log_residual_panels(comp, output_dir, "log10_residual_vs_reference", "log10 ratio", "log10_residual_vs_reference", reference_star_key))
    outputs.extend(plot_normalised_spectral_fractions(df_plot, value_column, output_dir))
    return outputs

# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create FSCALE multi-panel UV/Vis/IR/Total summaries from ATMOS OutputStorage data."
    )
    parser.add_argument("--base-dir", default=str(BASE_DIR), help=f"ATMOS OutputStorage base directory. Default: {BASE_DIR}")
    parser.add_argument("--output-dir", default=str(OUT_DIR), help=f"Output folder. Default: {OUT_DIR}")
    parser.add_argument("--value-column", default="fsol_erg_cm2_s", choices=VALUE_COLUMNS, help="Column to plot. Default: fsol_erg_cm2_s")
    parser.add_argument(
        "--iteration",
        default="latest",
        help="Iteration to plot. Use 'latest' for the highest available iteration per case, or specify a number such as 3. Default: latest",
    )
    args = parser.parse_args()

    base_dir = Path(os.path.expanduser(args.base_dir))
    output_dir = Path(os.path.expanduser(args.output_dir))
    ensure_dir(output_dir)

    df_all = load_emission_dataframe(base_dir)

    mismatches = df_all[
        (df_all["filename_run"].astype(str) != "")
        & (
            (df_all["run"].astype(str) != df_all["filename_run"].astype(str))
            | (df_all["fscale"].map(fscale_key) != df_all["filename_fscale"].map(fscale_key))
        )
    ]
    if not mismatches.empty:
        mismatch_csv = output_dir / "uv_vis_ir_filename_directory_mismatches.csv"
        mismatches[["run", "fscale", "filename_run", "filename_fscale", "source_path"]].drop_duplicates().to_csv(mismatch_csv, index=False)
        print(f"Note: some filenames disagree with their directory metadata; directory metadata is being used. Details: {mismatch_csv}")

    summary_csv = output_dir / "uv_vis_ir_all_summary.csv"
    df_all.to_csv(summary_csv, index=False)
    print(f"Saved full summary CSV: {summary_csv}")

    df_plot = select_iteration(df_all, args.iteration)
    selected_csv = output_dir / f"uv_vis_ir_selected_iteration_{sanitize_for_filename(args.iteration)}.csv"
    df_plot.to_csv(selected_csv, index=False)
    print(f"Saved selected-iteration CSV: {selected_csv}")

    counts = (
        df_plot[df_plot["band"].isin(BANDS)]
        .drop_duplicates(["star_key", "run", "fscale", "band"])
        .groupby(["fscale", "star_key", "star"], dropna=False)
        .size()
        .reset_index(name="band_run_records")
        .sort_values(["fscale", "star_key"])
    )
    counts_csv = output_dir / f"uv_vis_ir_selected_iteration_{sanitize_for_filename(args.iteration)}_counts_by_star.csv"
    counts.to_csv(counts_csv, index=False)
    print(f"Saved selected-iteration counts by star: {counts_csv}")
    print(counts.to_string(index=False))

    write_star_pair_diagnostic(
        df_plot,
        output_dir,
        args.iteration,
        left_star_key="18",
        right_star_key="24",
        value_column=args.value_column,
    )

    outputs = plot_all_fscale_publication_summaries(df_plot, args.value_column, output_dir)
    print(f"Saved FSCALE publication panel charts: {len(outputs)}")
    for out in outputs:
        print(f"  {out}")

    extra_outputs = make_extra_diagnostic_outputs(df_plot, args.value_column, output_dir, reference_star_key="18")
    print(f"Saved extra diagnostic delta/normalised/residual outputs: {len(extra_outputs)}")
    for out in extra_outputs:
        print(f"  {out}")

    print("\nDone.")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
