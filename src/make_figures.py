"""Generates the two summary figures used in the README from
results/report_39systems.csv. Run from the repo root:

    python src/make_figures.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

_ROOT = os.path.join(os.path.dirname(__file__), '..')
_RESULTS = os.path.join(_ROOT, 'results', 'report_39systems.csv')
_FIGURES = os.path.join(_ROOT, 'figures')

FAMILY_LABELS = {
    'halogen_bond': 'Halogen', 'chalcogen_bond': 'Chalcogen',
    'pnictogen_bond': 'Pnictogen', 'tetrel_bond': 'Tetrel', 'hydrogen_bond': 'Hydrogen',
}
BASELINE_COLOR = '#94a3b8'
PI_COLOR = '#2563eb'


def main():
    os.makedirs(_FIGURES, exist_ok=True)
    df = pd.read_csv(_RESULTS)
    full = df[df['train_fraction'] == 1.0]

    # --- Figure 1: MAE by family, baseline vs. PI-augmented ---
    by_family = full.groupby('family')[['mae_none', 'mae_full']].mean()
    by_family.index = [FAMILY_LABELS.get(f, f) for f in by_family.index]
    by_family = by_family.reindex(['Halogen', 'Chalcogen', 'Pnictogen', 'Tetrel', 'Hydrogen'])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(by_family))
    width = 0.35
    ax.bar([i - width / 2 for i in x], by_family['mae_none'], width,
           label='distance only', color=BASELINE_COLOR)
    ax.bar([i + width / 2 for i in x], by_family['mae_full'], width,
           label='distance + PI', color=PI_COLOR)
    ax.set_xticks(list(x))
    ax.set_xticklabels(by_family.index)
    ax.set_ylabel('Test MAE (kcal/mol)')
    ax.set_title('Leave-one-electrophile-out test MAE, by interaction family')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(_FIGURES, 'mae_by_family.png'), dpi=160)
    plt.close(fig)

    # --- Figure 2: MAE vs. dataset size (data efficiency) ---
    by_frac = df.groupby('train_fraction')[['mae_none', 'mae_full']].mean()

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(by_frac.index, by_frac['mae_none'], marker='o', color=BASELINE_COLOR,
            label='distance only')
    ax.plot(by_frac.index, by_frac['mae_full'], marker='o', color=PI_COLOR,
            label='distance + PI')
    ax.set_xlabel('Fraction of training electrophiles used')
    ax.set_ylabel('Mean test MAE (kcal/mol)')
    ax.set_title('Data efficiency: 39-electrophile battery')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(_FIGURES, 'mae_vs_data_size.png'), dpi=160)
    plt.close(fig)

    print(f"Saved figures to {_FIGURES}")


if __name__ == '__main__':
    main()
