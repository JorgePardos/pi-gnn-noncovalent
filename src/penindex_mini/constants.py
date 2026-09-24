"""Covalent (r) and van der Waals (v) radii in Angstroms, from Alvarez et al.
(Cordero et al., Dalton Trans. 2008, 2832 for r; Alvarez, Dalton Trans. 2013,
8617 for v) - the same empirical sets used by the Penetration Index papers
this project builds on.

Trimmed down from the full `PenIndex` package (a separate, general-purpose
structural-scan tool this project depends on for PI calculation) to just the
elements used here: only the RADII table, needed by compute_pi_for_elements.
"""

from typing import Dict

RADII: Dict[str, Dict[str, float]] = {
    'H':  {'r': 0.31, 'v': 1.20}, 'HE': {'r': 0.28, 'v': 1.43},
    'LI': {'r': 1.28, 'v': 2.12}, 'BE': {'r': 0.96, 'v': 1.98},
    'B':  {'r': 0.84, 'v': 1.91}, 'C':  {'r': 0.73, 'v': 1.77},
    'N':  {'r': 0.71, 'v': 1.66}, 'O':  {'r': 0.66, 'v': 1.50},
    'F':  {'r': 0.57, 'v': 1.46}, 'NE': {'r': 0.58, 'v': 1.58},
    'NA': {'r': 1.66, 'v': 2.50}, 'MG': {'r': 1.41, 'v': 2.51},
    'AL': {'r': 1.21, 'v': 2.25}, 'SI': {'r': 1.11, 'v': 2.19},
    'P':  {'r': 1.07, 'v': 1.90}, 'S':  {'r': 1.05, 'v': 1.89},
    'CL': {'r': 1.02, 'v': 1.82}, 'AR': {'r': 1.06, 'v': 1.94},
    'GE': {'r': 1.20, 'v': 2.29}, 'AS': {'r': 1.19, 'v': 1.88},
    'SE': {'r': 1.20, 'v': 1.82}, 'BR': {'r': 1.20, 'v': 1.86},
    'SN': {'r': 1.39, 'v': 2.42}, 'SB': {'r': 1.39, 'v': 2.47},
    'TE': {'r': 1.38, 'v': 1.99}, 'I':  {'r': 1.39, 'v': 2.04},
}
