"""The Penetration Index (PI) formula itself - trimmed from the full
`PenIndex` package to the two functions this project actually calls. The
graphs in `data/graph_dataset.pt` already have PI baked in for every atom
pair, so this module isn't on the training critical path; it's here so the
core physics is inspectable and runnable standalone, e.g.:

    >>> from penindex_mini import compute_pi_for_elements
    >>> compute_pi_for_elements('N', 'Cl', 3.2)   # NH3...Cl- contact
"""

from .constants import RADII


def compute_pi(distance: float, cov_radius_a: float, vdw_radius_a: float,
                cov_radius_b: float, vdw_radius_b: float) -> float:
    """Penetration Index for a single atom pair A-B (eqn 1 in Echeverria &
    Alvarez, Chem. Sci. 2023, 14, 11647-11688):

        PI = 100 * (v_A + v_B - d_AB) / (v_A + v_B - r_A - r_B)

    where r is the covalent radius, v the van der Waals radius, and d_AB the
    interatomic distance. PI ~ 0% marks a van der Waals contact, ~20-70% a
    hydrogen bond or other secondary interaction, >= ~90% a covalent bond.
    """
    return (100 * (vdw_radius_a + vdw_radius_b - distance)
            / (vdw_radius_a + vdw_radius_b - cov_radius_a - cov_radius_b))


def compute_pi_for_elements(element_1: str, element_2: str, distance: float) -> float:
    """Convenience wrapper around compute_pi() that looks up both atoms in
    the built-in RADII table by element symbol (case-insensitive)."""
    r1 = RADII[element_1.upper()]
    r2 = RADII[element_2.upper()]
    return compute_pi(distance, r1['r'], r1['v'], r2['r'], r2['v'])
