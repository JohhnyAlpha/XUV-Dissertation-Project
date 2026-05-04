import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.expanduser('~/atmos/BA/Plots')
SPECIES_FILE = os.path.join(BASE, 'species_analysis_pub', 'tables', 'species_metrics.csv')
AEROSOL_FILE = os.path.join(BASE, 'aerosol_analysis_pub', 'tables', 'aerosol_metrics.csv')
OUT_DIR = os.path.join(BASE, 'haze_interpretation_refined')

RUN_ORDER = ['Run_1', 'Run_2', 'Run_3']
RUN_LABELS = {
    'Run_1': '2.7 Gy star',
    'Run_2': '0.5 Gy star',
    'Run_3': '9.9 Gy star',
}
SCENARIO_TO_FSCALE = {1: 0.75, 2: 1.00, 3: 1.50, '1': 0.75, '2': 1.00, '3': 1.50}
STAR_ORDER = ['Epsilon_Eri', 'HD40307', 'HD85512', 'HD97658']
STAR_COLORS = {
    'Epsilon_Eri': 'tab:blue',
    'HD40307': 'tab:orange',
    'HD85512': 'tab:green',
    'HD97658': 'tab:red',
}
FSCALE_MARKERS = {1: 'o', 2: 's', 3: '^', '1': 'o', '2': 's', '3': '^'}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize_for_filename(value):
    value = str(value)
    for old, new in [(' ', '_'), ('/', '_'), ('\\', '_'), ('=', ''), ('.', 'p')]:
        value = value.replace(old, new)
    return value


def safe_ratio(a, b):
    a = pd.to_numeric(a, errors='coerce')
    b = pd.to_numeric(b, errors='coerce')
    out = pd.Series(np.nan, index=a.index, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (b > 0)
    out.loc[mask] = a.loc[mask] / b.loc[mask]
    return out


def log10_change(v, ref):
    if pd.isna(v) or pd.isna(ref) or v <= 0 or ref <= 0:
        return np.nan
    return np.log10(v / ref)


def significance_from_log_change(x):
    if not np.isfinite(x):
        return 'unknown'
    ax = abs(x)
    if ax >= 2.0:
        return 'extreme'
    if ax >= 1.0:
        return 'major'
    if ax >= 0.5:
        return 'moderate'
    if ax >= 0.2:
        return 'minor'
    return 'small'


def classify_haze(x):
    if not np.isfinite(x):
        return 'unknown'
    if x >= 1e12:
        return 'very_thick_haze'
    if x >= 1e10:
        return 'thick_haze'
    if x >= 1e8:
        return 'weak_haze'
    return 'clear'


def classify_reducing(x):
    if not np.isfinite(x):
        return 'unknown'
    if x >= 10:
        return 'strongly_reducing'
    if x >= 1:
        return 'reducing'
    if x >= 0.1:
        return 'weakly_reducing'
    return 'oxidized_or_low_reduced_gases'


def positive_finite(series):
    s = pd.to_numeric(series, errors='coerce')
    return s[np.isfinite(s) & (s > 0)]


def nanmedian_safe(series):
    s = pd.to_numeric(series, errors='coerce')
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return np.nan
    return float(np.nanmedian(s))


def load_species_long(path):
    df = pd.read_csv(path)
    required = ['run', 'run_label', 'star', 'scenario', 'fscale', 'scale_dir', 'species', 'column_proxy']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'species_metrics.csv missing columns: {missing}')
    return df.copy()


def load_aerosol_long(path):
    df = pd.read_csv(path)
    required = ['run', 'run_label', 'star', 'scenario', 'fscale', 'scale_dir', 'species', 'column_proxy']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'aerosol_metrics.csv missing columns: {missing}')
    return df.copy()


def build_case_table(species_df, aerosol_df):
    case_cols = ['run', 'run_label', 'star', 'scenario', 'fscale', 'scale_dir']

    sp_wide = (
        species_df[case_cols + ['species', 'column_proxy']]
        .pivot_table(index=case_cols, columns='species', values='column_proxy', aggfunc='first')
        .reset_index()
    )

    aerosol_use = aerosol_df[aerosol_df['species'].astype(str) == 'AERSOL'].copy()
    aer_case = (
        aerosol_use.groupby(case_cols, as_index=False)['column_proxy']
        .sum()
        .rename(columns={'column_proxy': 'haze_proxy'})
    )

    df = pd.merge(sp_wide, aer_case, on=case_cols, how='outer')

    for col in ['CH4', 'CO', 'H2', 'O2', 'O3', 'H2O', 'CO2']:
        if col not in df.columns:
            df[col] = np.nan

    df['reducing_proxy'] = df[['H2', 'CH4', 'CO']].fillna(0).sum(axis=1)
    df['oxidizing_proxy'] = df[['O2', 'O3']].fillna(0).sum(axis=1)
    df['reduced_to_oxidized'] = safe_ratio(df['reducing_proxy'], df['oxidizing_proxy'])
    df['CH4_to_H2O'] = safe_ratio(df['CH4'], df['H2O'])
    df['CH4_to_O2'] = safe_ratio(df['CH4'], df['O2'])
    df['CO_to_CH4'] = safe_ratio(df['CO'], df['CH4'])

    if df['CO2'].notna().any():
        df['CH4_to_CO2'] = safe_ratio(df['CH4'], df['CO2'])
    else:
        df['CH4_to_CO2'] = np.nan

    df['haze_regime'] = df['haze_proxy'].apply(classify_haze)
    df['reducing_regime'] = df['reduced_to_oxidized'].apply(classify_reducing)
    return df


def build_comparisons(case_df):
    rows = []
    metrics = ['haze_proxy', 'reducing_proxy', 'oxidizing_proxy', 'reduced_to_oxidized', 'CH4', 'CO', 'H2', 'O2', 'O3', 'H2O', 'CH4_to_H2O', 'CH4_to_O2', 'CO_to_CH4']
    if case_df['CH4_to_CO2'].notna().any():
        metrics.append('CH4_to_CO2')

    for star in sorted(case_df['star'].dropna().unique()):
        for scenario in sorted(case_df['scenario'].dropna().unique(), key=lambda x: float(x)):
            sub = case_df[(case_df['star'] == star) & (case_df['scenario'] == scenario)].copy()
            if sub.empty:
                continue
            ref = sub[sub['run'] == 'Run_1']
            if ref.empty:
                continue
            ref = ref.iloc[0]
            for _, row in sub.iterrows():
                rec = {
                    'comparison_type': 'run_at_fixed_fscale',
                    'star': star,
                    'scenario': row['scenario'],
                    'fscale': row['fscale'],
                    'run': row['run'],
                    'reference': 'Run_1',
                }
                for m in metrics:
                    rec[f'log10_{m}_change'] = log10_change(row.get(m, np.nan), ref.get(m, np.nan))
                rows.append(rec)

    for star in sorted(case_df['star'].dropna().unique()):
        for run in sorted(case_df['run'].dropna().unique()):
            sub = case_df[(case_df['star'] == star) & (case_df['run'] == run)].copy()
            if sub.empty:
                continue
            ref = sub[sub['scenario'].astype(str) == '2']
            if ref.empty:
                continue
            ref = ref.iloc[0]
            for _, row in sub.iterrows():
                rec = {
                    'comparison_type': 'fscale_at_fixed_run',
                    'star': star,
                    'scenario': row['scenario'],
                    'fscale': row['fscale'],
                    'run': row['run'],
                    'reference': 'scenario_2',
                }
                for m in metrics:
                    rec[f'log10_{m}_change'] = log10_change(row.get(m, np.nan), ref.get(m, np.nan))
                rows.append(rec)

    comp = pd.DataFrame(rows)
    if comp.empty:
        return comp
    for m in metrics:
        col = f'log10_{m}_change'
        if col in comp.columns:
            comp[f'{m}_significance'] = comp[col].apply(significance_from_log_change)
    return comp


def save_ch4_vs_haze(case_df, out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    plotted = False

    for star in STAR_ORDER:
        sub_star = case_df[case_df['star'] == star]
        if sub_star.empty:
            continue
        for scenario in sorted(sub_star['scenario'].dropna().unique(), key=lambda x: float(x)):
            sub = sub_star[sub_star['scenario'] == scenario].copy()
            x = pd.to_numeric(sub['CH4'], errors='coerce')
            y = pd.to_numeric(sub['haze_proxy'], errors='coerce')
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if not np.any(mask):
                continue
            ax.scatter(
                x[mask], y[mask],
                label=f"{star}, FSCALE={SCENARIO_TO_FSCALE.get(scenario, scenario):.2f}",
                color=STAR_COLORS.get(star),
                marker=FSCALE_MARKERS.get(scenario, 'o'),
                s=70, alpha=0.9, edgecolors='black', linewidths=0.4,
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        print('No positive finite data available for CH4 vs haze diagram.')
        return

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('CH4 column proxy')
    ax.set_ylabel('Haze proxy (AERSOL column proxy)')
    ax.set_title('CH4 versus haze proxy')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    fig.savefig(os.path.join(out_dir, 'ch4_vs_haze.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'ch4_vs_haze.pdf'), bbox_inches='tight')
    plt.close(fig)


def save_reduced_ratio_vs_haze(case_df, out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    plotted = False

    for star in STAR_ORDER:
        sub_star = case_df[case_df['star'] == star]
        if sub_star.empty:
            continue
        for scenario in sorted(sub_star['scenario'].dropna().unique(), key=lambda x: float(x)):
            sub = sub_star[sub_star['scenario'] == scenario].copy()
            x = pd.to_numeric(sub['reduced_to_oxidized'], errors='coerce')
            y = pd.to_numeric(sub['haze_proxy'], errors='coerce')
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if not np.any(mask):
                continue
            ax.scatter(
                x[mask], y[mask],
                label=f"{star}, FSCALE={SCENARIO_TO_FSCALE.get(scenario, scenario):.2f}",
                color=STAR_COLORS.get(star),
                marker=FSCALE_MARKERS.get(scenario, 'o'),
                s=70, alpha=0.9, edgecolors='black', linewidths=0.4,
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        print('No positive finite data available for reduced ratio vs haze diagram.')
        return

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('(H2 + CH4 + CO) / (O2 + O3) column proxy ratio')
    ax.set_ylabel('Haze proxy (AERSOL column proxy)')
    ax.set_title('Reduced-to-oxidized proxy versus haze')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    fig.savefig(os.path.join(out_dir, 'reduced_ratio_vs_haze.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'reduced_ratio_vs_haze.pdf'), bbox_inches='tight')
    plt.close(fig)


def save_star_delta_table(case_df, comp_df, out_dir):
    records = []
    for star in sorted(case_df['star'].dropna().unique()):
        sub = comp_df[comp_df['star'] == star] if not comp_df.empty else pd.DataFrame()
        if sub.empty:
            continue
        for metric in ['haze_proxy', 'CH4', 'CO', 'H2', 'O2', 'O3', 'H2O', 'reduced_to_oxidized']:
            col = f'log10_{metric}_change'
            if col not in sub.columns:
                continue
            s = pd.to_numeric(sub[col], errors='coerce')
            s_abs = s.abs()
            if not np.isfinite(s_abs).any():
                continue
            idx = s_abs.idxmax()
            row = sub.loc[idx]
            records.append({
                'star': star,
                'metric': metric,
                'comparison_type': row['comparison_type'],
                'run': row['run'],
                'scenario': row['scenario'],
                'fscale': row['fscale'],
                'log10_change': row[col],
                'significance': significance_from_log_change(row[col]),
            })
    out = pd.DataFrame(records)
    out.to_csv(os.path.join(out_dir, 'star_largest_deltas.csv'), index=False)
    return out


def save_star_comparison(case_df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    metrics = [
        ('CH4', 'CH4 column proxy'),
        ('haze_proxy', 'Haze proxy'),
        ('reduced_to_oxidized', 'Reduced/Oxidized proxy'),
    ]

    for ax, (metric, label) in zip(axes, metrics):
        x = np.arange(len(STAR_ORDER))
        width = 0.22
        added_legend = False
        for i, scenario in enumerate([1, 2, 3]):
            vals = []
            for star in STAR_ORDER:
                sub = case_df[(case_df['star'] == star) & (case_df['scenario'] == scenario)]
                vals.append(nanmedian_safe(sub[metric]) if not sub.empty else np.nan)
            vals_arr = np.asarray(vals, dtype=float)
            ax.bar(x + (i - 1) * width, vals_arr, width=width, label=f"FSCALE={SCENARIO_TO_FSCALE[scenario]:.2f}", alpha=0.9)
            if np.any(np.isfinite(vals_arr)):
                added_legend = True
        ax.set_xticks(x)
        ax.set_xticklabels(STAR_ORDER, rotation=20, ha='right')
        ax.set_title(label)
        ax.grid(True, axis='y', alpha=0.3)
        vals_all = positive_finite(case_df[metric])
        if len(vals_all) > 0:
            ax.set_yscale('log')
        if added_legend:
            ax.legend(fontsize=8)

    fig.suptitle('Median star-by-star comparison', y=1.02)
    fig.savefig(os.path.join(out_dir, 'star_comparison.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'star_comparison.pdf'), bbox_inches='tight')
    plt.close(fig)


def save_fscale_sensitivity(case_df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    metrics = [
        ('CH4', 'CH4 column proxy'),
        ('haze_proxy', 'Haze proxy'),
        ('reduced_to_oxidized', 'Reduced/Oxidized proxy'),
    ]

    for ax, (metric, label) in zip(axes, metrics):
        plotted = False
        for star in STAR_ORDER:
            sub = case_df[case_df['star'] == star].copy()
            if sub.empty:
                continue
            grouped = sub.groupby('fscale', as_index=False)[metric].median().sort_values('fscale')
            x = pd.to_numeric(grouped['fscale'], errors='coerce')
            y = pd.to_numeric(grouped[metric], errors='coerce')
            mask = np.isfinite(x) & np.isfinite(y)
            if not np.any(mask):
                continue
            ax.plot(x[mask], y[mask], marker='o', label=star)
            plotted = True
        ax.set_xlabel('FSCALE')
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        vals_all = positive_finite(case_df[metric])
        if len(vals_all) > 0 and plotted:
            ax.set_yscale('log')
        if plotted:
            ax.legend(fontsize=8)

    axes[0].set_ylabel('Median value')
    fig.suptitle('FSCALE sensitivity across stars', y=1.02)
    fig.savefig(os.path.join(out_dir, 'fscale_sensitivity.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fscale_sensitivity.pdf'), bbox_inches='tight')
    plt.close(fig)


def build_summary_text(case_df, comp_df, delta_df):
    lines = []
    lines.append('Refined Archean-haze interpretation')
    lines.append('==================================')
    lines.append('')
    lines.append('Diagnostics used:')
    lines.append('- haze proxy: AERSOL column proxy only')
    lines.append('- reducing proxy: H2 + CH4 + CO')
    lines.append('- oxidizing proxy: O2 + O3')
    lines.append('- reduced/oxidized proxy: (H2 + CH4 + CO) / (O2 + O3)')
    lines.append('- CH4/CO2 omitted when CO2 is unavailable in the species table')
    lines.append('')

    for star in STAR_ORDER:
        sub = case_df[case_df['star'] == star].copy()
        if sub.empty:
            continue
        lines.append(star)
        lines.append('-' * len(star))
        sub = sub.sort_values(['run', 'scenario'])
        for _, row in sub.iterrows():
            lines.append(
                f"{row['run_label']}, FSCALE={row['fscale']:.2f}: "
                f"CH4={row['CH4']:.3e}, CO={row['CO']:.3e}, H2={row['H2']:.3e}, "
                f"O2={row['O2']:.3e}, O3={row['O3']:.3e}, "
                f"haze={row['haze_proxy']:.3e} ({row['haze_regime']}), "
                f"reduced/oxidized={row['reduced_to_oxidized']:.3e} ({row['reducing_regime']})"
            )

        sub_delta = delta_df[delta_df['star'] == star] if not delta_df.empty else pd.DataFrame()
        if not sub_delta.empty:
            lines.append('Largest shifts:')
            for _, drow in sub_delta.sort_values(['metric']).iterrows():
                lines.append(
                    f"- {drow['metric']}: {drow['comparison_type']} at {drow['run']}, "
                    f"scenario {drow['scenario']} (FSCALE={drow['fscale']:.2f}) gives "
                    f"Δlog10={drow['log10_change']:.2f} ({drow['significance']})."
                )

        star_comp = comp_df[comp_df['star'] == star] if not comp_df.empty else pd.DataFrame()
        if not star_comp.empty:
            haze_col = 'log10_haze_proxy_change'
            if haze_col in star_comp.columns:
                s = pd.to_numeric(star_comp[haze_col], errors='coerce').abs()
                if np.isfinite(s).any():
                    row = star_comp.loc[s.idxmax()]
                    sign = 'increase' if row[haze_col] > 0 else 'decrease'
                    lines.append(
                        f"Strongest haze response is a {sign} in {row['comparison_type']} for "
                        f"{row['run']} at scenario {row['scenario']} (FSCALE={row['fscale']:.2f}), "
                        f"relative to {row['reference']} with Δlog10 haze={row[haze_col]:.2f}."
                    )
        lines.append('')
    return '\n'.join(lines)


def main():
    ensure_dir(OUT_DIR)
    species_df = load_species_long(SPECIES_FILE)
    aerosol_df = load_aerosol_long(AEROSOL_FILE)
    case_df = build_case_table(species_df, aerosol_df)
    comp_df = build_comparisons(case_df)
    delta_df = save_star_delta_table(case_df, comp_df, OUT_DIR)

    case_df = case_df.sort_values(['star', 'run', 'scenario']).reset_index(drop=True)
    case_df.to_csv(os.path.join(OUT_DIR, 'haze_interpretation_cases.csv'), index=False)
    comp_df.to_csv(os.path.join(OUT_DIR, 'haze_interpretation_comparisons.csv'), index=False)

    save_ch4_vs_haze(case_df, OUT_DIR)
    save_reduced_ratio_vs_haze(case_df, OUT_DIR)
    save_star_comparison(case_df, OUT_DIR)
    save_fscale_sensitivity(case_df, OUT_DIR)

    with open(os.path.join(OUT_DIR, 'haze_interpretation_summary.txt'), 'w') as f:
        f.write(build_summary_text(case_df, comp_df, delta_df))

    print('Done.')
    for name in [
        'haze_interpretation_cases.csv',
        'haze_interpretation_comparisons.csv',
        'star_largest_deltas.csv',
        'ch4_vs_haze.png',
        'ch4_vs_haze.pdf',
        'reduced_ratio_vs_haze.png',
        'reduced_ratio_vs_haze.pdf',
        'star_comparison.png',
        'star_comparison.pdf',
        'fscale_sensitivity.png',
        'fscale_sensitivity.pdf',
        'haze_interpretation_summary.txt',
    ]:
        print(f'Saved: {os.path.join(OUT_DIR, name)}')


if __name__ == '__main__':
    main()
