import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_OUTPUT_DIR = os.path.expanduser("~/atmos/OutputStorage")
BASE_PLOTS_DIR = os.path.expanduser("~/atmos/BA/Plots")
OUT_DIR = os.path.join(BASE_PLOTS_DIR, "final_paper_figures")

SPECIES_FILE = os.path.join(BASE_PLOTS_DIR, "species_analysis_pub", "tables", "species_metrics.csv")
AEROSOL_FILE = os.path.join(BASE_PLOTS_DIR, "aerosol_analysis_pub", "tables", "aerosol_metrics.csv")

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

STAR_ORDER = ["Epsilon_Eri", "HD40307", "HD85512", "HD97658"]
STAR_COLORS = {
    "Epsilon_Eri": "#1f77b4",
    "HD40307": "#ff7f0e",
    "HD85512": "#2ca02c",
    "HD97658": "#d62728",
}

SCENARIO_TO_FSCALE = {
    1: 0.75,
    2: 1.00,
    3: 1.50,
    "1": 0.75,
    "2": 1.00,
    "3": 1.50,
}
SCENARIO_MARKERS = {
    1: "o",
    2: "s",
    3: "^",
    "1": "o",
    "2": "s",
    "3": "^",
}
RUN_LINESTYLES = {
    "Run_1": "-",
    "Run_2": "--",
    "Run_3": ":",
}

# Keep haze proxy focused on aerosol loading
EXCLUDE_AEROSOL_SPECIES = {"RFRAC", "CONVER"}

DPI = 300


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def numeric_sort_key(value):
    try:
        return float(value)
    except Exception:
        return str(value)


def read_table(file_path, skiprows=None):
    try:
        return pd.read_csv(
            file_path,
            sep=r"\s+",
            engine="python",
            skiprows=skiprows,
            comment="#",
        )
    except Exception:
        return None


def pick_altitude_column(df):
    for candidate in ["ALT", "Z", "Alt", "alt", "z"]:
        if candidate in df.columns:
            return candidate
    return None


def pick_temperature_column(df):
    for candidate in [
        "T", "TEMP", "Temp", "temp",
        "TEMPERATURE", "Temperature", "temperature",
        "TK", "t",
    ]:
        if candidate in df.columns:
            return candidate

    for col in df.columns:
        cl = str(col).lower()
        if "temp" in cl or cl == "t":
            return col
    return None


def safe_ratio(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")

    out = pd.Series(np.nan, index=a.index, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (b > 0)
    out.loc[mask] = a.loc[mask] / b.loc[mask]
    return out


def positive_finite(series):
    s = pd.to_numeric(series, errors="coerce")
    return s[np.isfinite(s) & (s > 0)]


def significance_from_log_change(x):
    if not np.isfinite(x):
        return "unknown"
    ax = abs(x)
    if ax >= 1.0:
        return "major"
    if ax >= 0.5:
        return "moderate"
    if ax >= 0.2:
        return "minor"
    return "small"


def discover_run_dirs(base_dir):
    if not os.path.isdir(base_dir):
        return []

    out = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and re.fullmatch(r"Run_\d+", name):
            out.append(name)

    return sorted(out, key=lambda x: int(x.split("_")[1]))


def build_clima_index(scale_dir):
    clima_dir = os.path.join(scale_dir, "CLIMA_OUTPUT")
    found = {}
    pattern = re.compile(r"^(Run_\d+)_([0-9]+(?:\.[0-9]+)?)_(\d+)_(.+)$")

    if not os.path.isdir(clima_dir):
        return found

    for fname in os.listdir(clima_dir):
        m = pattern.match(fname)
        if not m:
            continue
        _, _, scenario, suffix = m.groups()
        if suffix in ("clima_last.tab", "clima_last.out"):
            found[str(scenario)] = os.path.join(clima_dir, fname)

    return found


def extract_surface_temperature(file_path):
    df = read_table(file_path)
    if df is None or df.empty:
        return np.nan, None

    alt_col = pick_altitude_column(df)
    temp_col = pick_temperature_column(df)

    if alt_col is None or temp_col is None:
        return np.nan, temp_col

    alt = pd.to_numeric(df[alt_col], errors="coerce")
    temp = pd.to_numeric(df[temp_col], errors="coerce")
    mask = np.isfinite(alt) & np.isfinite(temp)

    if not np.any(mask):
        return np.nan, temp_col

    sub = pd.DataFrame({"alt": alt[mask], "temp": temp[mask]}).sort_values("alt")
    return float(sub.iloc[0]["temp"]), temp_col


def load_surface_temperatures(base_dir):
    rows = []

    for run in discover_run_dirs(base_dir):
        run_dir = os.path.join(base_dir, run)

        for star_id in sorted(os.listdir(run_dir), key=numeric_sort_key):
            star_dir = os.path.join(run_dir, star_id)
            if not os.path.isdir(star_dir):
                continue

            star_name = STAR_MAP.get(str(star_id), str(star_id))

            for scale in sorted(os.listdir(star_dir), key=numeric_sort_key):
                scale_dir = os.path.join(star_dir, scale)
                if not os.path.isdir(scale_dir):
                    continue

                clima_files = build_clima_index(scale_dir)

                for scenario, file_path in sorted(clima_files.items(), key=lambda kv: int(kv[0])):
                    surface_temp, temp_col = extract_surface_temperature(file_path)
                    rows.append(
                        {
                            "run": run,
                            "run_label": RUN_LABELS.get(run, run),
                            "star": star_name,
                            "scenario": int(scenario),
                            "fscale": SCENARIO_TO_FSCALE.get(int(scenario), np.nan),
                            "scale_dir": str(scale),
                            "surface_temp_k": surface_temp,
                            "temperature_column": temp_col,
                            "clima_file": file_path,
                        }
                    )

    return pd.DataFrame(rows)


def load_species_wide(path):
    df = pd.read_csv(path)

    needed = ["run", "run_label", "star", "scenario", "fscale", "scale_dir", "species", "column_proxy"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"species_metrics.csv missing columns: {missing}")

    case_cols = ["run", "run_label", "star", "scenario", "fscale", "scale_dir"]

    wide = (
        df[case_cols + ["species", "column_proxy"]]
        .pivot_table(
            index=case_cols,
            columns="species",
            values="column_proxy",
            aggfunc="first",
        )
        .reset_index()
    )

    rename_map = {}
    for sp in ["H2O", "CH4", "CO", "CO2", "O2", "O3", "H2"]:
        if sp in wide.columns:
            rename_map[sp] = f"{sp}_column"

    return wide.rename(columns=rename_map)


def load_aerosol_case_sums(path):
    df = pd.read_csv(path)

    needed = ["run", "run_label", "star", "scenario", "fscale", "scale_dir", "species", "column_proxy"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"aerosol_metrics.csv missing columns: {missing}")

    df = df.copy()
    df["species"] = df["species"].astype(str)
    use_df = df[~df["species"].isin(EXCLUDE_AEROSOL_SPECIES)].copy()

    case_cols = ["run", "run_label", "star", "scenario", "fscale", "scale_dir"]

    case_sum = (
        use_df.groupby(case_cols, as_index=False)["column_proxy"]
        .sum()
        .rename(columns={"column_proxy": "haze_proxy"})
    )

    aersol_only = use_df[use_df["species"] == "AERSOL"].copy()
    aersol_sum = (
        aersol_only.groupby(case_cols, as_index=False)["column_proxy"]
        .sum()
        .rename(columns={"column_proxy": "aersol_proxy"})
    )

    return pd.merge(case_sum, aersol_sum, on=case_cols, how="outer")


def normalize_case_keys(df):
    df = df.copy()

    if "run" in df.columns:
        df["run"] = df["run"].astype(str)
    if "run_label" in df.columns:
        df["run_label"] = df["run_label"].astype(str)
    if "star" in df.columns:
        df["star"] = df["star"].astype(str)
    if "scenario" in df.columns:
        df["scenario"] = pd.to_numeric(df["scenario"], errors="coerce").astype("Int64")
    if "fscale" in df.columns:
        df["fscale"] = pd.to_numeric(df["fscale"], errors="coerce")
    if "scale_dir" in df.columns:
        df["scale_dir"] = df["scale_dir"].astype(str)

    return df


def build_case_table():
    species = normalize_case_keys(load_species_wide(SPECIES_FILE))
    aerosol = normalize_case_keys(load_aerosol_case_sums(AEROSOL_FILE))
    temps = normalize_case_keys(load_surface_temperatures(BASE_OUTPUT_DIR))

    # Do not merge on scale_dir to avoid float/string mismatches and duplicate identifiers
    merge_cols = ["run", "run_label", "star", "scenario", "fscale"]

    df = pd.merge(species, aerosol, on=merge_cols, how="outer", suffixes=("", "_aer"))

    temp_cols = merge_cols + ["scale_dir", "surface_temp_k", "temperature_column", "clima_file"]
    temp_cols = [c for c in temp_cols if c in temps.columns]

    df = pd.merge(
        df,
        temps[temp_cols],
        on=merge_cols,
        how="outer",
        suffixes=("", "_temp"),
    )

    for col in ["H2_column", "CH4_column", "CO2_column", "O2_column"]:
        if col not in df.columns:
            df[col] = np.nan

    df["reduced_proxy"] = df["H2_column"].fillna(0) + df["CH4_column"].fillna(0)
    df["oxidized_proxy"] = df["CO2_column"].fillna(0) + df["O2_column"].fillna(0)
    df["redox_proxy"] = safe_ratio(df["reduced_proxy"], df["oxidized_proxy"])

    return df


def build_sensitivity_table(case_df):
    rows = []

    for star in STAR_ORDER:
        for scenario in [1, 2, 3]:
            sub = case_df[(case_df["star"] == star) & (case_df["scenario"] == scenario)].copy()
            ref = sub[sub["run"] == "Run_1"]

            if sub.empty or ref.empty:
                continue

            ref = ref.iloc[0]

            for _, row in sub.iterrows():
                rows.append(
                    {
                        "comparison_type": "run_at_fixed_fscale",
                        "star": star,
                        "scenario": scenario,
                        "fscale": row["fscale"],
                        "run": row["run"],
                        "reference": f"Run_1|{star}|s{scenario}",
                        "case_id": f"{row['run']}|{star}|s{scenario}",
                        "delta_temp_k": (
                            row["surface_temp_k"] - ref["surface_temp_k"]
                            if np.isfinite(row["surface_temp_k"]) and np.isfinite(ref["surface_temp_k"])
                            else np.nan
                        ),
                        "delta_log10_haze": (
                            np.log10(row["aersol_proxy"] / ref["aersol_proxy"])
                            if np.isfinite(row.get("aersol_proxy", np.nan))
                            and np.isfinite(ref.get("aersol_proxy", np.nan))
                            and row.get("aersol_proxy", np.nan) > 0
                            and ref.get("aersol_proxy", np.nan) > 0
                            else np.nan
                        ),
                        "delta_log10_ch4": (
                            np.log10(row["CH4_column"] / ref["CH4_column"])
                            if np.isfinite(row.get("CH4_column", np.nan))
                            and np.isfinite(ref.get("CH4_column", np.nan))
                            and row.get("CH4_column", np.nan) > 0
                            and ref.get("CH4_column", np.nan) > 0
                            else np.nan
                        ),
                    }
                )

    for star in STAR_ORDER:
        for run in RUN_LABELS:
            sub = case_df[(case_df["star"] == star) & (case_df["run"] == run)].copy()
            ref = sub[sub["scenario"] == 2]

            if sub.empty or ref.empty:
                continue

            ref = ref.iloc[0]

            for _, row in sub.iterrows():
                rows.append(
                    {
                        "comparison_type": "fscale_at_fixed_run",
                        "star": star,
                        "scenario": row["scenario"],
                        "fscale": row["fscale"],
                        "run": run,
                        "reference": f"{run}|{star}|s2",
                        "case_id": f"{run}|{star}|s{row['scenario']}",
                        "delta_temp_k": (
                            row["surface_temp_k"] - ref["surface_temp_k"]
                            if np.isfinite(row["surface_temp_k"]) and np.isfinite(ref["surface_temp_k"])
                            else np.nan
                        ),
                        "delta_log10_haze": (
                            np.log10(row["aersol_proxy"] / ref["aersol_proxy"])
                            if np.isfinite(row.get("aersol_proxy", np.nan))
                            and np.isfinite(ref.get("aersol_proxy", np.nan))
                            and row.get("aersol_proxy", np.nan) > 0
                            and ref.get("aersol_proxy", np.nan) > 0
                            else np.nan
                        ),
                        "delta_log10_ch4": (
                            np.log10(row["CH4_column"] / ref["CH4_column"])
                            if np.isfinite(row.get("CH4_column", np.nan))
                            and np.isfinite(ref.get("CH4_column", np.nan))
                            and row.get("CH4_column", np.nan) > 0
                            and ref.get("CH4_column", np.nan) > 0
                            else np.nan
                        ),
                    }
                )

    out = pd.DataFrame(rows)

    if not out.empty:
        out["temp_change_significance"] = out["delta_temp_k"].apply(
            lambda x: (
                "unknown"
                if not np.isfinite(x)
                else "major" if abs(x) >= 10
                else "moderate" if abs(x) >= 3
                else "minor" if abs(x) >= 1
                else "small"
            )
        )
        out["haze_change_significance"] = out["delta_log10_haze"].apply(significance_from_log_change)
        out["ch4_change_significance"] = out["delta_log10_ch4"].apply(significance_from_log_change)

    return out


def scatter_points(ax, df, xcol, ycol, xlabel, ylabel, title, use_logx=False, use_logy=False, color_by_temp=False):
    plotted = False
    scatter_obj = None

    for star in STAR_ORDER:
        sub_star = df[df["star"] == star].copy()
        if sub_star.empty:
            continue

        for scenario in [1, 2, 3]:
            sub = sub_star[sub_star["scenario"] == scenario].copy()
            if sub.empty:
                continue

            x = pd.to_numeric(sub[xcol], errors="coerce")
            y = pd.to_numeric(sub[ycol], errors="coerce")

            mask = np.isfinite(x) & np.isfinite(y)
            if use_logx:
                mask &= x > 0
            if use_logy:
                mask &= y > 0

            if not np.any(mask):
                continue

            if color_by_temp:
                c = pd.to_numeric(sub.loc[mask, "surface_temp_k"], errors="coerce")
                scatter_obj = ax.scatter(
                    x[mask],
                    y[mask],
                    c=c,
                    cmap="viridis",
                    marker=SCENARIO_MARKERS[scenario],
                    s=80,
                    edgecolors="black",
                    linewidths=0.4,
                    alpha=0.95,
                )
            else:
                ax.scatter(
                    x[mask],
                    y[mask],
                    color=STAR_COLORS.get(star),
                    marker=SCENARIO_MARKERS[scenario],
                    s=80,
                    edgecolors="black",
                    linewidths=0.4,
                    alpha=0.95,
                    label=f"{star}, FSCALE={SCENARIO_TO_FSCALE[scenario]:.2f}",
                )
            plotted = True

    if use_logx:
        ax.set_xscale("log")
    if use_logy:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    return plotted, scatter_obj


def save_main_figure(df, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    scatter_points(
        axes[0],
        df,
        "aersol_proxy",
        "surface_temp_k",
        "AERSOL haze proxy",
        "Surface temperature (K)",
        "(a) Climate–haze coupling",
        use_logx=True,
        use_logy=False,
    )

    scatter_points(
        axes[1],
        df,
        "CH4_column",
        "aersol_proxy",
        r"CH$_4$ column proxy",
        "AERSOL haze proxy",
        "(b) Methane–haze relationship",
        use_logx=True,
        use_logy=True,
    )

    _, sc = scatter_points(
        axes[2],
        df,
        "reduced_proxy",
        "aersol_proxy",
        r"Reduced-gas proxy (H$_2$ + CH$_4$)",
        "AERSOL haze proxy",
        "(c) Reducing chemistry vs haze",
        use_logx=True,
        use_logy=True,
        color_by_temp=True,
    )

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label("Surface temperature (K)")

    ax = axes[3]
    plotted_any = False

    for star in STAR_ORDER:
        for run in RUN_LABELS:
            sub = df[(df["star"] == star) & (df["run"] == run)].copy().sort_values("fscale")
            if sub.empty:
                continue

            x = pd.to_numeric(sub["fscale"], errors="coerce")
            y = pd.to_numeric(sub["surface_temp_k"], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)

            if not np.any(mask):
                continue

            ax.plot(
                x[mask],
                y[mask],
                color=STAR_COLORS.get(star),
                linestyle=RUN_LINESTYLES.get(run, "-"),
                marker="o",
                linewidth=1.8,
                label=f"{star}, {RUN_LABELS[run]}",
            )
            plotted_any = True

    ax.set_xlabel("FSCALE")
    ax.set_ylabel("Surface temperature (K)")
    ax.set_title("(d) Climate response to FSCALE")
    ax.grid(True, alpha=0.3)

    if plotted_any:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="best")

    fig.suptitle("Final synthesis figures: Archean-haze Earth around K-dwarfs", y=0.98, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out_dir, "final_paper_figure.png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "final_paper_figure.pdf"), bbox_inches="tight")
    plt.close(fig)


def save_star_temperature_figure(df, out_dir):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plotted_any = False

    for star in STAR_ORDER:
        sub = df[df["star"] == star].copy()
        if sub.empty:
            continue

        grouped = sub.groupby("fscale", as_index=False)["surface_temp_k"].median().sort_values("fscale")
        x = pd.to_numeric(grouped["fscale"], errors="coerce")
        y = pd.to_numeric(grouped["surface_temp_k"], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)

        if not np.any(mask):
            continue

        ax.plot(
            x[mask],
            y[mask],
            marker="o",
            linewidth=2.0,
            color=STAR_COLORS.get(star),
            label=star,
        )
        plotted_any = True

    ax.set_xlabel("FSCALE")
    ax.set_ylabel("Median surface temperature (K)")
    ax.set_title("Star-by-star surface temperature response")
    ax.grid(True, alpha=0.3)

    if plotted_any:
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "star_temperature_response.png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "star_temperature_response.pdf"), bbox_inches="tight")
    plt.close(fig)


def build_summary(case_df, sens_df):
    lines = []
    lines.append("Final synthesis summary")
    lines.append("======================")
    lines.append("")
    lines.append("Diagnostics included:")
    lines.append("- Surface temperature from the lowest-altitude clima_last row")
    lines.append("- Haze proxy from aerosol AERSOL column proxy")
    lines.append("- Methane column proxy from species_metrics.csv")
    lines.append("- Reduced-gas proxy = H2 + CH4")
    lines.append("")

    for star in STAR_ORDER:
        sub = case_df[case_df["star"] == star].copy()
        if sub.empty:
            continue

        lines.append(star)
        lines.append("-" * len(star))

        valid_t = positive_finite(sub["surface_temp_k"])
        if len(valid_t) > 0:
            lines.append(
                f"Surface temperature spans {np.nanmin(valid_t):.2f}–{np.nanmax(valid_t):.2f} K across runs and FSCALE."
            )

        valid_h = positive_finite(sub["aersol_proxy"])
        if len(valid_h) > 0:
            lines.append(f"AERSOL haze proxy spans {np.nanmin(valid_h):.3e}–{np.nanmax(valid_h):.3e}.")

        pair = sub[["surface_temp_k", "aersol_proxy"]].apply(pd.to_numeric, errors="coerce")
        mask = np.isfinite(pair["surface_temp_k"]) & np.isfinite(pair["aersol_proxy"]) & (pair["aersol_proxy"] > 0)
        if np.sum(mask) >= 3:
            corr = np.corrcoef(
                pair.loc[mask, "surface_temp_k"],
                np.log10(pair.loc[mask, "aersol_proxy"])
            )[0, 1]
            if np.isfinite(corr):
                direction = "decreases with increasing haze" if corr < 0 else "increases with increasing haze"
                lines.append(
                    f"Within this star set, surface temperature broadly {direction} "
                    f"(r = {corr:.2f} against log10 haze proxy)."
                )

        ss = sens_df[sens_df["star"] == star].copy()
        if not ss.empty:
            valid = ss["delta_temp_k"].abs().fillna(-np.inf)
            if np.isfinite(valid).any():
                top = ss.iloc[int(valid.argmax())]
                lines.append(
                    f"Largest temperature shift: {top['case_id']} relative to {top['reference']} "
                    f"with ΔT = {top['delta_temp_k']:.2f} K ({top['temp_change_significance']})."
                )

        lines.append("")

    return "\n".join(lines)


def main():
    ensure_dir(OUT_DIR)

    if not os.path.exists(SPECIES_FILE):
        raise FileNotFoundError(f"Missing species metrics file: {SPECIES_FILE}")
    if not os.path.exists(AEROSOL_FILE):
        raise FileNotFoundError(f"Missing aerosol metrics file: {AEROSOL_FILE}")

    case_df = build_case_table().sort_values(["star", "run", "scenario"]).reset_index(drop=True)
    sens_df = build_sensitivity_table(case_df)

    case_csv = os.path.join(OUT_DIR, "final_synthesis_cases.csv")
    sens_csv = os.path.join(OUT_DIR, "final_synthesis_sensitivities.csv")

    case_df.to_csv(case_csv, index=False)
    sens_df.to_csv(sens_csv, index=False)

    save_main_figure(case_df, OUT_DIR)
    save_star_temperature_figure(case_df, OUT_DIR)

    summary_path = os.path.join(OUT_DIR, "final_synthesis_summary.txt")
    with open(summary_path, "w") as f:
        f.write(build_summary(case_df, sens_df))

    print("Done.")
    print(f"Saved: {case_csv}")
    print(f"Saved: {sens_csv}")
    print(f"Saved: {os.path.join(OUT_DIR, 'final_paper_figure.png')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'final_paper_figure.pdf')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'star_temperature_response.png')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'star_temperature_response.pdf')}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
