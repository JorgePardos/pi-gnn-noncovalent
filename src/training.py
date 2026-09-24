"""Minimal, dependency-light training core (torch + numpy + model.py only),
split out of train.py so it can run on remote compute machines without the
rest of the project. train.py re-exports these names unchanged."""

import random

import numpy as np
import torch
import torch.nn as nn

from model import MessagePassingNet, examples_to_tensors


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_model(train_examples, test_examples, pi_mode: str, epochs: int,
                     lr: float = 3e-3, seed: int = 0, val_frac: float = 0.15,
                     patience: int = 30) -> float:
    """Trains a fresh MessagePassingNet on train_examples, with a held-out
    validation split used ONLY for early stopping (keeps the checkpoint
    with the lowest validation loss seen), returns test MAE (kcal/mol) on
    test_examples.

    Early stopping + seed averaging (see main()) were added after the
    first pass through this experiment came back wildly noisy
    (single-seed, fixed-epoch-count results swinging by hundreds of
    percent between adjacent training-set fractions, including for the
    baseline model, where more training data sometimes made test MAE
    WORSE - not a real chemistry effect, just training variance)."""
    set_seed(seed)
    rng = random.Random(seed)
    shuffled = train_examples[:]
    rng.shuffle(shuffled)
    n_val = max(4, int(len(shuffled) * val_frac))
    val_examples, fit_examples = shuffled[:n_val], shuffled[n_val:]

    z_tr, e_tr, m_tr, am_tr, y_tr = examples_to_tensors(fit_examples, pi_mode)
    z_val, e_val, m_val, am_val, y_val = examples_to_tensors(val_examples, pi_mode)
    z_te, e_te, m_te, am_te, y_te = examples_to_tensors(test_examples, pi_mode)

    y_mean, y_std = y_tr.mean(), y_tr.std().clamp(min=1e-3)
    y_tr_norm = (y_tr - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    model = MessagePassingNet(n_edge_features=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()

    best_val, best_state, epochs_since_best = float('inf'), None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(z_tr, e_tr, m_tr, am_tr)
        loss = loss_fn(pred, y_tr_norm)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(z_val, e_val, m_val, am_val)
            val_loss = loss_fn(val_pred, y_val_norm).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_te_norm = model(z_te, e_te, m_te, am_te)
        pred_te = pred_te_norm * y_std + y_mean
        mae = (pred_te - y_te).abs().mean().item()
    return mae
