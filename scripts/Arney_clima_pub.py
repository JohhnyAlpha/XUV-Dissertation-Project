# arney_clima_temperature_profiles_clean.py

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.expanduser("~/atmos/OutputStorage")
OUT_DIR = os.path.expanduser("~/atmos/BA/Plots/arney_clima_profiles_clean")

STARS = {
    "18": "Epsilon_Eri",
    "24": "HD_40307",
    "25": "HD_85512",
    "26": "HD_97658",
}

SAVE_DPI = 300
SHOW_PLOTS = False

AGE_LABELS = {
    "Run_1": "Young Star",
    "Run_2": "Intermediate Age",
    "Run_3": "Evolved Star",
}

RUN_STYLES = {
    "Run_1": "-",
    "Run_2": "--",
    "Run_3": ":",
}


# -----------------------------
# HELPERS
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize(value):
    return str(value).replace(" ", "_").replace("/", "_").replace("\\", "_")


def numeric_sort_key(value):
    try:
        return float(value)
    except Exception:
        return str(value)


def star_name_from_id(star_id):
    return STARS.get(str(star_id), str(star_id))


def discover_run_dirs(base_dir):
    runs = []

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)

        if os.path.isdir(full) and re.fullmatch(r"Run_\d+", name):
            runs.append(name)

    return sorted(runs, key=lambda x: int(x.split("_")[1]))


def pick_column(df, candidates=None, contains_candidates=None):
    candidates = candidates or []
    contains_candidates = contains_candidates or []

    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for col in df.columns:
        lc = col.lower()
        for cand in contains_candidates:
            if cand.lower() in lc:
                return col

    return None


def read_table(path):
    try:
        return pd.read_csv(
            path,
            sep=r"\s+",
            engine="python",
            comment="#",
        )
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return None


# -----------------------------
# FILE DISCOVERY
# -----------------------------
def find_clima_files(scale_dir):
    clima_dir = os.path.join(scale_dir, "CLIMA_OUTPUT")
    files = defaultdict(lambda: None)

    if not os.path.isdir(clima_dir):
        return {}

    pattern = re.compile(r"^(Run_\d+)_([0-9]+(?:\.[0-9]+)?)_(\d+)_(.+)$")

    for fname in os.listdir(clima_dir):
        match = pattern.match(fname)

        if not match:
            continue

        _, _, scenario, suffix = match.groups()

        if suffix in ("clima_last.tab", "clima_last.out"):
            files[scenario] = os.path.join(clima_dir, fname)

    return dict(files)


# -----------------------------
# PARSING
# -----------------------------
def parse_clima_profile(path):
    df = read_table(path)

    if df is None or df.empty:
        return None

    alt_col = pick_column(
        df,
        candidates=["ALT", "Z", "Alt", "alt", "z"],
        contains_candidates=["altitude"],
    )

    temp_col = pick_column(
        df,
        candidates=["T", "TEMP", "Temp", "Temperature", "temperature"],
        contains_candidates=["temp"],
    )

    pressure_col = pick_column(
        df,
        candidates=["P", "PRESS", "Pressure", "pressure"],
        contains_candidates=["press"],
    )

    if alt_col is None or temp_col is None:
        print(f"Missing altitude or temperature column in: {path}")
        print(f"Columns found: {list(df.columns)}")
        return None

    alt = pd.to_numeric(df[alt_col], errors="coerce").to_numpy(dtype=float)
    temp = pd.to_numeric(df[temp_col], errors="coerce").to_numpy(dtype=float)

    pressure = None
    if pressure_col is not None:
        pressure = pd.to_numeric(df[pressure_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(alt) & np.isfinite(temp)

    alt = alt[mask]
    temp = temp[mask]

    if pressure is not None:
        pressure = pressure[mask]

    if len(alt) < 2:
        return None

    order = np.argsort(alt)

    return {
        "alt": alt[order],
        "temp": temp[order],
        "pressure": pressure[order] if pressure is not None else None,
        "path": path,
        "columns": {
            "alt": alt_col,
            "temp": temp_col,
            "pressure": pressure_col,
        },
    }


# -----------------------------
# DATASET BUILDING
# -----------------------------
def build_clima_dataset(base_dir):
    rows = []

    for run in discover_run_dirs(base_dir):
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

                scenario_files = find_clima_files(scale_dir)

                for scenario, clima_path in sorted(
                    scenario_files.items(),
                    key=lambda x: int(x[0]),
                ):
                    profile = parse_clima_profile(clima_path)

                    if profile is None:
                        continue

                    rows.append(
                        {
                            "run": run,
                            "star": star,
                            "scale": scale,
                            "scenario": scenario,
                            "profile": profile,
                        }
                    )

                    print(
                        f"Loaded {run} | {star} | "
                        f"scale={scale} | scenario={scenario}"
                    )

    return rows


# -----------------------------
# STYLE / DIAGNOSTICS
# -----------------------------
def apply_arney_style(ax):
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Altitude (km)")


def get_global_axis_limits(rows):
    temps = []
    alts = []

    for row in rows:
        p = row["profile"]
        temps.extend(p["temp"])
        alts.extend(p["alt"])

    temps = np.asarray(temps, dtype=float)
    alts = np.asarray(alts, dtype=float)

    temps = temps[np.isfinite(temps)]
    alts = alts[np.isfinite(alts)]

    if len(temps) == 0 or len(alts) == 0:
        return 0, 1, 0, 1

    return (
        np.nanmin(temps),
        np.nanmax(temps),
        np.nanmin(alts),
        np.nanmax(alts),
    )


def add_profile_diagnostics(ax, temp, alt):
    temp = np.asarray(temp, dtype=float)
    alt = np.asarray(alt, dtype=float)

    mask = np.isfinite(temp) & np.isfinite(alt)
    temp = temp[mask]
    alt = alt[mask]

    if len(temp) < 5:
        return

    order = np.argsort(alt)
    temp = temp[order]
    alt = alt[order]

    try:
        dtdz = np.gradient(temp, alt)
    except Exception:
        return

    if not np.any(np.isfinite(dtdz)):
        return

    inversion_idx = np.nanargmax(dtdz)

    if np.isfinite(dtdz[inversion_idx]) and dtdz[inversion_idx] > 0:
        ax.axhline(
            alt[inversion_idx],
            linestyle=":",
            linewidth=0.8,
            alpha=0.25,
        )


# -----------------------------
# CLEAN PUBLICATION PLOTS
# -----------------------------
def plot_star_profiles_clean_grid(rows, star, out_dir):
    star_rows = [r for r in rows if r["star"] == star]

    if not star_rows:
        return

    tmin, tmax, zmin, zmax = get_global_axis_limits(star_rows)

    tpad = 0.05 * (tmax - tmin) if tmax > tmin else 1.0
    zpad = 0.03 * (zmax - zmin) if zmax > zmin else 1.0

    scales = sorted({r["scale"] for r in star_rows}, key=lambda x: float(x))
    scenarios = sorted({r["scenario"] for r in star_rows}, key=lambda x: int(x))
    runs = sorted({r["run"] for r in star_rows}, key=lambda x: int(x.split("_")[1]))

    fig, axes = plt.subplots(
        len(scenarios),
        len(scales),
        figsize=(12, 10),
        sharex=True,
        sharey=True,
    )

    axes = np.asarray(axes)

    for i, scenario in enumerate(scenarios):
        for j, scale in enumerate(scales):
            ax = axes[i, j]

            panel_rows = [
                r
                for r in star_rows
                if r["scenario"] == scenario and r["scale"] == scale
            ]

            for row in sorted(panel_rows, key=lambda r: int(r["run"].split("_")[1])):
                profile = row["profile"]

                ax.plot(
                    profile["temp"],
                    profile["alt"],
                    linestyle=RUN_STYLES.get(row["run"], "-"),
                    linewidth=2.0,
                    alpha=0.9,
                    label=AGE_LABELS.get(row["run"], row["run"]),
                )

                add_profile_diagnostics(
                    ax,
                    profile["temp"],
                    profile["alt"],
                )

            apply_arney_style(ax)

            ax.set_xlim(tmin - tpad, tmax + tpad)
            ax.set_ylim(zmin - zpad, zmax + zpad)

            if i == 0:
                ax.set_title(f"Flux scale = {scale}")

            if j == 0:
                ax.set_ylabel(f"Scenario {scenario}\nAltitude (km)")
            else:
                ax.set_ylabel("")

            if i == len(scenarios) - 1:
                ax.set_xlabel("Temperature (K)")
            else:
                ax.set_xlabel("")

            ax.text(
                0.03,
                0.95,
                f"S{scenario}, f={scale}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10,
        title="Atmospheric Age",
    )

    fig.suptitle(
        f"{star}: Temperature–Altitude Climate Profiles",
        y=0.98,
        fontsize=14,
    )

    ensure_dir(out_dir)

    out_png = os.path.join(
        out_dir,
        f"arney_clima_temperature_profiles_clean_grid_{sanitize(star)}.png",
    )

    out_pdf = os.path.join(
        out_dir,
        f"arney_clima_temperature_profiles_clean_grid_{sanitize(star)}.pdf",
    )

    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=SAVE_DPI, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


def plot_all_stars_overview(rows, out_dir):
    stars = sorted(set(r["star"] for r in rows))

    if not stars:
        return

    tmin, tmax, zmin, zmax = get_global_axis_limits(rows)

    tpad = 0.05 * (tmax - tmin) if tmax > tmin else 1.0
    zpad = 0.03 * (zmax - zmin) if zmax > zmin else 1.0

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for ax, star in zip(axes, stars):
        star_rows = [r for r in rows if r["star"] == star]

        for row in star_rows:
            profile = row["profile"]

            linestyle = RUN_STYLES.get(row["run"], "-")

            ax.plot(
                profile["temp"],
                profile["alt"],
                linestyle=linestyle,
                linewidth=1.2,
                alpha=0.35,
                label=AGE_LABELS.get(row["run"], row["run"]),
            )

        apply_arney_style(ax)
        ax.set_xlim(tmin - tpad, tmax + tpad)
        ax.set_ylim(zmin - zpad, zmax + zpad)
        ax.set_title(star)

    for ax in axes[len(stars):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()

    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10,
        title="Atmospheric Age",
    )

    fig.suptitle(
        "All Stars: Temperature–Altitude Overview",
        y=0.98,
        fontsize=14,
    )

    ensure_dir(out_dir)

    out_png = os.path.join(
        out_dir,
        "arney_clima_temperature_profiles_all_stars_overview.png",
    )

    out_pdf = os.path.join(
        out_dir,
        "arney_clima_temperature_profiles_all_stars_overview.pdf",
    )

    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=SAVE_DPI, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


# -----------------------------
# MANIFEST
# -----------------------------
def write_manifest(rows, out_dir):
    ensure_dir(out_dir)

    manifest = []

    for row in rows:
        profile = row["profile"]

        manifest.append(
            {
                "run": row["run"],
                "star": row["star"],
                "scale": row["scale"],
                "scenario": row["scenario"],
                "source_file": profile["path"],
                "altitude_column": profile["columns"]["alt"],
                "temperature_column": profile["columns"]["temp"],
                "pressure_column": profile["columns"]["pressure"],
                "n_points": len(profile["alt"]),
                "min_alt": np.nanmin(profile["alt"]),
                "max_alt": np.nanmax(profile["alt"]),
                "min_temp": np.nanmin(profile["temp"]),
                "max_temp": np.nanmax(profile["temp"]),
            }
        )

    df = pd.DataFrame(manifest)

    out_csv = os.path.join(
        out_dir,
        "arney_clima_temperature_profiles_manifest.csv",
    )

    df.to_csv(out_csv, index=False)

    print(f"Saved {out_csv}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    ensure_dir(OUT_DIR)

    rows = build_clima_dataset(BASE_DIR)

    print(f"\nLoaded {len(rows)} clima profiles")

    write_manifest(rows, OUT_DIR)

    for star in sorted(set(r["star"] for r in rows)):
        plot_star_profiles_clean_grid(rows, star, OUT_DIR)

    plot_all_stars_overview(rows, OUT_DIR)

    print("\nDone.")
    print(f"Outputs written to: {OUT_DIR}")
