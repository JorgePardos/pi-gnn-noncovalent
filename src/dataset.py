"""Loads the pre-built graph dataset (`data/graph_dataset.pt`): 1170 DFT
single-point energies (39 electrophiles x 5 distances x 6 angles, all
probed with a fixed NH3 nucleophile), each already carrying the full N x N
pairwise distance and Penetration Index matrices computed with
`penindex_mini` at dataset-build time.

Original source: geometries generated deterministically from a distance x
angle grid around each electrophile, energies from Gaussian16 CCSD(T)/DFT
single points, PI computed for every atom pair via the Penetration Index
formula (Echeverria & Alvarez, Chem. Sci. 2023). The raw QM logs and
generation code live in the full research repo; this file only re-hydrates
the already-computed result so the model can be trained standalone.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'graph_dataset.pt')


@dataclass
class GraphExample:
    system: str
    family: str
    r: float
    theta: float
    z: np.ndarray            # (N,) atomic numbers
    dist: np.ndarray         # (N, N) pairwise distances, Angstrom
    pi: np.ndarray           # (N, N) pairwise Penetration Indices
    energy_kcal_mol: float
    anchor_idx: int           # index of the electrophilic atom (grid origin)
    probe_n_idx: int          # index of the probe's headline contact atom (N)
    probe: str = 'NH3'


def build_examples(path: str = None) -> list:
    if path is None:
        path = _DEFAULT_PATH
    rows = torch.load(path, weights_only=False)
    return [
        GraphExample(
            system=row['system'], family=row['family'], r=row['r'], theta=row['theta'],
            z=np.array(row['z']), dist=np.array(row['dist']), pi=np.array(row['pi']),
            energy_kcal_mol=row['energy_kcal_mol'], anchor_idx=row['anchor_idx'],
            probe_n_idx=row['probe_n_idx'], probe=row['probe'],
        )
        for row in rows
    ]


if __name__ == '__main__':
    examples = build_examples()
    print(f"Loaded {len(examples)} graph examples.")
    systems_seen = sorted({e.system for e in examples})
    print(f"Systems ({len(systems_seen)}): {systems_seen}")
    families = sorted({e.family for e in examples})
    print(f"Families: {families}")
