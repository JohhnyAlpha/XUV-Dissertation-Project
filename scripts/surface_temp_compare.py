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
OUT_DIR = os.path.expanduser("~/atmos/BA/Plots/surface_temp_compare")
SAVE_DPI = 300
SHOW_PLOTS = False

RUN_LABELS = {
    "Run_1": "2.7 Gy star",
    "Run_2": "0.5 Gy star",
    "Run_3": "9.9 Gy star",
}

STAR_MAP = {
    "18": "Epsilon_Eri",
    "24": "HD40307",
    "25": "HD85512",
    "26": "HD97658",
}

SCENARIO_TO_FSCALE = {
    "1": 0.75,
    "2": 1.00,
    "3": 1.50,
}

STAR_ORDER = ["Epsilon_Eri", "HD40307", "HD85512", "HD97658"]
RUN_ORDER = ["Run_1", "Run_2", "Run_3"]
SCENARIO_ORDER = ["1", "2", "3"]


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
    return STAR_MAP.get(str(star_id), str(star_id))


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
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def find_matching_column(columns, candidates):
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    for cand in candidates:
        for c in columns:
            if cand.lower() == c.lower().replace("_", ""):
                return c
    for cand in candidates:
        for c in columns:
            if cand.lower() in c.lower():
                return c
    return None


def pick_altitude_column(df):
    return find_matching_column(df.columns, ["ALT", "Z", "Alt", "alt", "z"])


def pick_temperature_column(df):
    return find_matching_column(df.columns, ["T", "TEMP", "Temperature", "temp", "temperature"])


# -----------------------------
# FILE DISCOVERY
# -----------------------------
def build_clima_index(scale_dir):
    clima_dir = os.path.join(scale_dir, "CLIMA_OUTPUT")
    scenario_files = {}
    pattern = re.compile(r"^(Run_\d+)_([0-9]+(?:\.[0-9]+)?)_(\d+)_(.+)$")

    if not os.path.isdir(clima_dir):
        return scenario_files

    for fname in os.listdir(clima_dir):
        match = pattern.match(fname)
        if not match:
            continue
        _, _, scenario, suffix = match.groups()
        if suffix not in ("clima_last.tab", "clima_last.out"):
            continue
        scenario_files[scenario] = os.path.join(clima_dir, fname)

    return dict(sorted(scenario_files.items(), key=lambda kv: int(kv[0])))


# -----------------------------
# PARSING
# -----------------------------
def read_table(file_path):
    try:
        return pd.read_csv(file_path, sep=r"\s+", engine="python", comment="#")
    except Exception as exc:
        print(f"Failed to read {file_path}: {exc}")
        return None


def extract_surface_temperature(file_path):
    if file_path is None or not os.path.exists(file_path):
        return None

    df = read_table(file_path)
    if df is None or df.empty:
        return None

    alt_col = pick_altitude_column(df)
    temp_col = pick_temperature_column(df)
    if temp_col is None:
        print(f"No temperature column found in {file_path}. Columns: {list(df.columns)}")
        return None

    temp = pd.to_numeric(df[temp_col], errors="coerce")
    if alt_col is not None:
        alt = pd.to_numeric(df[alt_col], errors="coerce")
        mask = np.isfinite(alt) & np.isfinite(temp)
        if np.any(mask):
            idx = np.nanargmin(alt[mask].to_numpy())
            surface_temp = temp[mask].to_numpy()[idx]
        else:
            surface_temp = np.nan
    else:
        finite = temp[np.isfinite(temp)]
        surface_temp = finite.iloc[0] if len(finite) else np.nan

    if not np.isfinite(surface_temp):
        return None
    return float(surface_temp)


# -----------------------------
# BUILD DATASET
# -----------------------------
def collect_surface_temperatures(base_dir):
    rows = []
    runs = discover_run_dirs(base_dir)
    print("Discovered runs:", runs)

    for run in runs:
        run_dir = os.path.join(base_dir, run)
        for star_id in sorted(os.listdir(run_dir), key=numeric_sort_key):
            star_dir = os.path.join(run_dir, star_id)
            if not os.path.isdir(star_dir):
                continue
            star = star_name_from_id(star_id)

            for scale in sorted(os.listdir(star_dir), key=numeric_sort_key):
                scale_dir = os.path.join(star_dir, scale)
                if not os.path.isdir(scale_dir):
                    continue

                clima_files = build_clima_index(scale_dir)
                if not clima_files:
                    continue

                for scenario, file_path in clima_files.items():
                    temp = extract_surface_temperature(file_path)
                    row = {
                        "run": run,
                        "run_label": RUN_LABELS.get(run, run),
                        "star": star,
                        "star_id": str(star_id),
                        "scenario": str(scenario),
                        "fscale": SCENARIO_TO_FSCALE.get(str(scenario), np.nan),
                        "scale_dir": str(scale),
                        "clima_file": file_path,
                        "surface_temp_K": temp,
                    }
                    rows.append(row)
                    print(
                        f"Loaded: {run} | {star} | scenario {scenario} | "
                        f"scale_dir={scale} | Tsurf={temp}"
                    )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["scenario"] = out["scenario"].astype(str)
    out["fscale"] = pd.to_numeric(out["fscale"], errors="coerce")
    out["surface_temp_K"] = pd.to_numeric(out["surface_temp_K"], errors="coerce")
    out["surface_temp_C"] = out["surface_temp_K"] - 273.15
    return out


# -----------------------------
# COMPARISON TABLES
# -----------------------------
def build_comparison_tables(df):
    tables = {}

    # relative to Run_1 at fixed star + scenario
    rows = []
    for star in sorted(df["star"].dropna().unique()):
        for scenario in sorted(df["scenario"].dropna().unique(), key=lambda x: float(x)):
            sub = df[(df["star"] == star) & (df["scenario"] == scenario)].copy()
            ref = sub[sub["run"] == "Run_1"]
            if sub.empty or ref.empty:
                continue
            ref_temp = ref["surface_temp_K"].iloc[0]
            for _, row in sub.iterrows():
                rows.append({
                    "comparison_type": "run_at_fixed_fscale",
                    "star": star,
                    "scenario": scenario,
                    "fscale": row["fscale"],
                    "run": row["run"],
                    "run_label": row["run_label"],
                    "surface_temp_K": row["surface_temp_K"],
                    "delta_vs_Run1_K": row["surface_temp_K"] - ref_temp if pd.notna(row["surface_temp_K"]) and pd.notna(ref_temp) else np.nan,
                })
    tables["vs_run1"] = pd.DataFrame(rows)

    # relative to scenario 2 at fixed star + run
    rows = []
    for star in sorted(df["star"].dropna().unique()):
        for run in sorted(df["run"].dropna().unique()):
            sub = df[(df["star"] == star) & (df["run"] == run)].copy()
            ref = sub[sub["scenario"] == "2"]
            if sub.empty or ref.empty:
                continue
            ref_temp = ref["surface_temp_K"].iloc[0]
            for _, row in sub.iterrows():
                rows.append({
                    "comparison_type": "fscale_at_fixed_run",
                    "star": star,
                    "scenario": row["scenario"],
                    "fscale": row["fscale"],
                    "run": row["run"],
                    "run_label": row["run_label"],
                    "surface_temp_K": row["surface_temp_K"],
                    "delta_vs_fscale1p0_K": row["surface_temp_K"] - ref_temp if pd.notna(row["surface_temp_K"]) and pd.notna(ref_temp) else np.nan,
                })
    tables["vs_fscale1p0"] = pd.DataFrame(rows)

    return tables


# -----------------------------
# PLOTTING
# -----------------------------
def save_overall_comparison_plot(df, out_dir):
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Panel 1: by star, grouped by fscale, colored by run
    ax = axes[0]
    x = np.arange(len(STAR_ORDER))
    width = 0.22
    for i, scenario in enumerate(["1", "2", "3"]):
        vals = []
        for star in STAR_ORDER:
            sub = df[(df["star"] == star) & (df["scenario"] == scenario)]
            vals.append(np.nanmedian(sub["surface_temp_K"]) if not sub.empty else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=f"FSCALE={SCENARIO_TO_FSCALE[scenario]:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(STAR_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Surface temperature (K)")
    ax.set_title("Median surface temperature by star")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: by run
    ax = axes[1]
    x = np.arange(len(RUN_ORDER))
    for i, scenario in enumerate(["1", "2", "3"]):
        vals = []
        for run in RUN_ORDER:
            sub = df[(df["run"] == run) & (df["scenario"] == scenario)]
            vals.append(np.nanmedian(sub["surface_temp_K"]) if not sub.empty else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=f"FSCALE={SCENARIO_TO_FSCALE[scenario]:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([RUN_LABELS.get(r, r) for r in RUN_ORDER], rotation=20, ha="right")
    ax.set_title("Median surface temperature by run")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Surface temperature comparison across all runs and planets", y=1.02)
    save_figure(fig, os.path.join(out_dir, "surface_temp_overview.png"))
    fig.savefig(os.path.join(out_dir, "surface_temp_overview.pdf"), bbox_inches="tight")
    print(f"Saved figure: {os.path.join(out_dir, 'surface_temp_overview.pdf')}")
    plt.close(fig)


def save_star_panels(df, out_dir):
    for star in STAR_ORDER:
        sub = df[df["star"] == star].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        for run in RUN_ORDER:
            run_sub = sub[sub["run"] == run].sort_values("fscale")
            if run_sub.empty:
                continue
            x = pd.to_numeric(run_sub["fscale"], errors="coerce")
            y = pd.to_numeric(run_sub["surface_temp_K"], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            if np.any(mask):
                ax.plot(x[mask], y[mask], marker="o", label=RUN_LABELS.get(run, run))

        ax.set_xlabel("FSCALE")
        ax.set_ylabel("Surface temperature (K)")
        ax.set_title(star)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        png = os.path.join(out_dir, f"surface_temp_{sanitize_for_filename(star)}.png")
        pdf = os.path.join(out_dir, f"surface_temp_{sanitize_for_filename(star)}.pdf")
        save_figure(fig, png)
        fig.savefig(pdf, bbox_inches="tight")
        print(f"Saved figure: {pdf}")
        plt.close(fig)


def save_heatmap_table(df, out_dir):
    # heatmap-style pivot: rows stars, columns run|fscale
    pivot = df.pivot_table(index="star", columns=["run", "fscale"], values="surface_temp_K", aggfunc="first")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{r}\n{f:.2f}" for r, f in pivot.columns], rotation=45, ha="right")
    ax.set_title("Surface temperature heatmap (K)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            label = f"{val:.1f}" if np.isfinite(val) else ""
            ax.text(j, i, label, ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Surface temperature (K)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "surface_temp_heatmap.png"), dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "surface_temp_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {os.path.join(out_dir, 'surface_temp_heatmap.png')}")
    print(f"Saved figure: {os.path.join(out_dir, 'surface_temp_heatmap.pdf')}")


# -----------------------------
# TEXT SUMMARY
# -----------------------------
def build_summary(df):
    lines = []
    lines.append("Surface temperature comparison summary")
    lines.append("===================================")
    lines.append("")

    if df.empty:
        lines.append("No surface temperatures were extracted.")
        return "\n".join(lines)

    hottest = df.loc[df["surface_temp_K"].idxmax()] if df["surface_temp_K"].notna().any() else None
    coldest = df.loc[df["surface_temp_K"].idxmin()] if df["surface_temp_K"].notna().any() else None

    if hottest is not None:
        lines.append(
            f"Hottest case: {hottest['star']} | {hottest['run_label']} | FSCALE={hottest['fscale']:.2f} "
            f"-> {hottest['surface_temp_K']:.2f} K"
        )
    if coldest is not None:
        lines.append(
            f"Coldest case: {coldest['star']} | {coldest['run_label']} | FSCALE={coldest['fscale']:.2f} "
            f"-> {coldest['surface_temp_K']:.2f} K"
        )

    lines.append("")
    for star in STAR_ORDER:
        sub = df[df["star"] == star].sort_values(["run", "fscale"])
        if sub.empty:
            continue
        lines.append(star)
        lines.append("-" * len(star))
        for _, row in sub.iterrows():
            lines.append(
                f"{row['run_label']}, FSCALE={row['fscale']:.2f}: "
                f"{row['surface_temp_K']:.2f} K ({row['surface_temp_C']:.2f} C)"
            )
        lines.append("")

    return "\n".join(lines)


# -----------------------------
# MAIN
# -----------------------------
def main(base_dir=BASE_DIR, out_dir=OUT_DIR):
    ensure_dir(out_dir)
    df = collect_surface_temperatures(base_dir)
    if df.empty:
        print("No surface temperature data found.")
        return

    csv_path = os.path.join(out_dir, "surface_temperatures_all_cases.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved table: {csv_path}")

    comparison_tables = build_comparison_tables(df)
    for name, table in comparison_tables.items():
        out_path = os.path.join(out_dir, f"surface_temperature_{name}.csv")
        table.to_csv(out_path, index=False)
        print(f"Saved table: {out_path}")

    save_overall_comparison_plot(df, out_dir)
    save_star_panels(df, out_dir)
    save_heatmap_table(df, out_dir)

    summary_path = os.path.join(out_dir, "surface_temperature_summary.txt")
    with open(summary_path, "w") as f:
        f.write(build_summary(df))
    print(f"Saved text: {summary_path}")

    print("Done.")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
