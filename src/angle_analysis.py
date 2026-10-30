"""Error by approach angle: distance-only vs. PI-only, from the per-point
leave-one-electrophile-out predictions in results/per_point_predictions.csv
(NH3 probe, 39 electrophiles x 5 seeds x 2 conditions). Run from the repo root:

    python src/angle_analysis.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon

_ROOT = os.path.join(os.path.dirname(__file__), '..')
_RESULTS = os.path.join(_ROOT, 'results')
_FIGURES = os.path.join(_ROOT, 'figures')


def _summarize(per_system: pd.DataFrame, label) -> dict:
    n, p = per_system['none'].mean(), per_system['pi_only'].mean()
    return {'angle': label, 'mae_distance': round(n, 2), 'mae_pi_only': round(p, 2),
            'improvement_pct': round(100 * (n - p) / n, 1),
            'electrophiles_pi_wins': int((per_system['pi_only'] < per_system['none']).sum()),
            'wilcoxon_p': float(wilcoxon(per_system['none'], per_system['pi_only']).pvalue)}


def main():
    df = pd.read_csv(os.path.join(_RESULTS, 'per_point_predictions.csv'))
    df['abs_err'] = (df['pred'] - df['y']).abs()
    # MAE per (held-out electrophile, angle, condition), then compared across electrophiles
    per_system = df.groupby(['system', 'theta', 'mode'])['abs_err'].mean().unstack()

    rows = [_summarize(s, f'{t:g}') for t, s in per_system.groupby('theta')]
    thetas = per_system.index.get_level_values('theta')
    rows.append(_summarize(per_system[thetas <= 15].groupby('system').mean(), '<=15'))
    rows.append(_summarize(per_system[thetas >= 45].groupby('system').mean(), '>=45'))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(_RESULTS, 'angle_breakdown.csv'), index=False)
    print(out.to_string(index=False))

    single = out[~out['angle'].str.contains('=')]
    x = range(len(single))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.35
    ax.bar([i - w / 2 for i in x], single['mae_distance'], w, label='distance only', color='#94a3b8')
    ax.bar([i + w / 2 for i in x], single['mae_pi_only'], w, label='PI only', color='#2563eb')
    for i, pct in zip(x, single['improvement_pct']):
        ax.text(i + w / 2, single['mae_pi_only'].iloc[i] + 0.1, f'+{pct:.0f}%',
                ha='center', fontsize=9, color='#2563eb')
    ax.set_xticks(list(x))
    ax.set_xticklabels([f'{a}°' for a in single['angle']])
    ax.set_xlabel('Approach angle (0° = ideal)')
    ax.set_ylabel('Test MAE (kcal/mol)')
    ax.set_title('PI-only gain grows with distance from the ideal angle')
    ax.legend(frameon=False, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(_FIGURES, 'mae_by_angle.png'), dpi=160)


if __name__ == '__main__':
    main()
