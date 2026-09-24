"""Generates the figures used in the README from the results/ CSVs. Run
from the repo root:

    python src/make_figures.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

_ROOT = os.path.join(os.path.dirname(__file__), '..')
_RESULTS = os.path.join(_ROOT, 'results')
_FIGURES = os.path.join(_ROOT, 'figures')

FAMILY_LABELS = {
    'halogen_bond': 'Halogen', 'chalcogen_bond': 'Chalcogen',
    'pnictogen_bond': 'Pnictogen', 'tetrel_bond': 'Tetrel', 'hydrogen_bond': 'Hydrogen',
}
FAMILY_ORDER = ['Halogen', 'Chalcogen', 'Pnictogen', 'Tetrel', 'Hydrogen']
BASELINE_COLOR = '#94a3b8'
PI_COLOR = '#2563eb'


def fig_mae_by_family():
    """Primary comparison: distance-only vs. PI-only (no raw distance at
    all) - the more striking result, since PI alone can't see angle."""
    df = pd.read_csv(os.path.join(_RESULTS, 'report_pi_ablation.csv'))
    by_family = df.groupby('family')[['mae_none', 'mae_pi_only']].mean()
    by_family.index = [FAMILY_LABELS.get(f, f) for f in by_family.index]
    by_family = by_family.reindex(FAMILY_ORDER)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(by_family))
    width = 0.35
    ax.bar([i - width / 2 for i in x], by_family['mae_none'], width,
           label='distance only', color=BASELINE_COLOR)
    ax.bar([i + width / 2 for i in x], by_family['mae_pi_only'], width,
           label='PI only (no distance)', color=PI_COLOR)
    ax.set_xticks(list(x))
    ax.set_xticklabels(by_family.index)
    ax.set_ylabel('Test MAE (kcal/mol)')
    ax.set_title('Leave-one-electrophile-out test MAE, by interaction family')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(_FIGURES, 'mae_by_family.png'), dpi=160)
    plt.close(fig)


def fig_probe_robustness():
    """Does the PI-only effect replicate across different nucleophile
    probes, or is it a fluke of the NH3 battery specifically?"""
    df = pd.read_csv(os.path.join(_RESULTS, 'probe_robustness.csv'))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = range(len(df))
    ax.bar(list(x), df['win_rate_pi_only'], width=0.5, color=PI_COLOR)
    ax.axhline(0.5, color='#333', linewidth=0.8, linestyle='--')
    ax.set_xticks(list(x))
    ax.set_xticklabels(df['probe'])
    ax.set_ylabel("PI-only win rate vs. distance-only baseline")
    ax.set_ylim(0, 1)
    ax.set_title('PI-only beats distance-only across every probe tested')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(_FIGURES, 'probe_robustness.png'), dpi=160)
    plt.close(fig)


def main():
    os.makedirs(_FIGURES, exist_ok=True)
    fig_mae_by_family()
    fig_probe_robustness()
    print(f"Saved figures to {_FIGURES}")


if __name__ == '__main__':
    main()
