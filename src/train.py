"""Core experiment: does adding the Penetration Index (computed for every
atom pair) as an edge feature improve a small message-passing net's
prediction of non-covalent interaction energy, over the same architecture
given only interatomic distance?

Two edge-feature conditions, architecture otherwise identical (see
model.py's examples_to_tensors for exact definitions):
  - none: distance only (baseline).
  - full: distance + PI for every atom pair.

Evaluation is leave-one-system-out (LOSO): train on N-1 electrophiles, test
on the one held out - a genuine generalization test to an unseen molecule,
not just an unseen geometry of a system already seen during training. Run
at several training-set fractions (data efficiency), 5 seeds each with
validation-based early stopping (see training.py's train_one_model
docstring for why that matters here).

Usage:
    python train.py [--epochs 300] [--n-seeds 5] [--out ../results/report.csv]
"""

import argparse
import os
import random

import numpy as np
import pandas as pd

from dataset import build_examples
from training import train_one_model

PI_MODES = ['none', 'full']


def loso_folds(examples):
    systems = sorted({ex.system for ex in examples})
    for held_out in systems:
        train = [ex for ex in examples if ex.system != held_out]
        test = [ex for ex in examples if ex.system == held_out]
        yield held_out, train, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--out', default=os.path.join('..', 'results', 'report.csv'))
    parser.add_argument('--fractions', nargs='+', type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--modes', nargs='+', default=PI_MODES, choices=['none', 'full'])
    args = parser.parse_args()
    modes = args.modes
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    examples = build_examples()
    print(f"Loaded {len(examples)} examples across "
          f"{len(set(ex.system for ex in examples))} systems.\n")

    rows = []
    for held_out, train, test in loso_folds(examples):
        family = next(ex.family for ex in examples if ex.system == held_out)
        split_rng = random.Random(12345)  # fixed subset selection, independent of model seed

        for frac in args.fractions:
            n_train = max(8, int(len(train) * frac))
            train_sub = split_rng.sample(train, n_train) if frac < 1.0 else train

            maes = {mode: [] for mode in modes}
            for seed in range(args.n_seeds):
                for mode in modes:
                    maes[mode].append(train_one_model(
                        train_sub, test, pi_mode=mode, epochs=args.epochs, seed=seed))

            row = {'held_out_system': held_out, 'family': family,
                   'train_fraction': frac, 'n_train': n_train}
            for mode in modes:
                row[f'mae_{mode}'] = round(float(np.mean(maes[mode])), 3)
                row[f'mae_{mode}_std'] = round(float(np.std(maes[mode])), 3)
            rows.append(row)

            print(f"[{held_out:10s} frac={frac:.2f} n={n_train:3d}] " +
                  "  ".join(f"{mode}={row[f'mae_{mode}']:6.2f}+/-{row[f'mae_{mode}_std']:4.2f}"
                            for mode in modes))

    report = pd.DataFrame(rows)
    report.to_csv(args.out, index=False)
    print(f"\nSaved detailed results to '{args.out}'.")

    n_folds = report['held_out_system'].nunique()
    print(f"\n=== Mean test MAE by training fraction (across all {n_folds} LOSO folds) ===")
    print(report.groupby('train_fraction')[[f'mae_{m}' for m in modes]].mean().round(2))

    print("\n=== Mean test MAE by family (at full training data) ===")
    full = report[report['train_fraction'] == max(args.fractions)]
    print(full.groupby('family')[[f'mae_{m}' for m in modes]].mean().round(2))

    if 'none' in modes:
        print("\n=== Win rate vs. baseline ('none') by family, full training data ===")
        for mode in modes:
            if mode == 'none':
                continue
            wins = (full[f'mae_{mode}'] < full['mae_none']).mean()
            print(f"  {mode}: {wins:.2f}")


if __name__ == '__main__':
    main()
