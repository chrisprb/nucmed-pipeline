"""Radioactive decay chains (Bateman equations).

General solver: for a linear chain with decay constants lambda_i and
branching fractions b_i (fraction of nuclide i-1 decays that produce i),

    dN/dt = M N,   M[i, i] = -lambda_i,   M[i, i-1] = b_i * lambda_{i-1}

so N(t) = expm(M t) N(0). The closed-form two-member solution is provided
separately and used as the test oracle.

Nuclide data below are working values to be checked against a nuclear data
library (e.g. NNDC / ENSDF) before use in any write-up; they are tagged [verify].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

LN2 = np.log(2.0)


@dataclass(frozen=True)
class Nuclide:
    name: str
    half_life_h: float

    @property
    def lam(self) -> float:  # per hour
        return LN2 / self.half_life_h


MO99 = Nuclide("Mo-99", 65.94)  # [verify]
TC99M = Nuclide("Tc-99m", 6.007)  # [verify]
LU177 = Nuclide("Lu-177", 6.647 * 24)  # [verify]
F18 = Nuclide("F-18", 109.77 / 60)  # [verify]

# Fraction of Mo-99 decays that feed Tc-99m (the rest go to Tc-99 ground state).
MO99_TO_TC99M_BRANCH = 0.876  # [verify]


def chain_matrix(chain: list[Nuclide], branches: list[float] | None = None) -> np.ndarray:
    n = len(chain)
    b = [1.0] * (n - 1) if branches is None else branches
    M = np.zeros((n, n))
    for i, nuc in enumerate(chain):
        M[i, i] = -nuc.lam
        if i > 0:
            M[i, i - 1] = b[i - 1] * chain[i - 1].lam
    return M


def populations(chain: list[Nuclide], n0: np.ndarray, t_h: np.ndarray,
                branches: list[float] | None = None) -> np.ndarray:
    """Number of atoms of each nuclide at times t_h (hours). Shape (len(t), len(chain))."""
    M = chain_matrix(chain, branches)
    n0 = np.asarray(n0, float)
    return np.array([expm(M * t) @ n0 for t in np.atleast_1d(t_h)])


def activities(chain: list[Nuclide], n0: np.ndarray, t_h: np.ndarray,
               branches: list[float] | None = None) -> np.ndarray:
    lam = np.array([c.lam for c in chain])
    return populations(chain, n0, t_h, branches) * lam


def parent_daughter_analytic(lp: float, ld: float, n_p0: float, t: np.ndarray,
                             branch: float = 1.0, n_d0: float = 0.0) -> np.ndarray:
    """Closed-form daughter population for a two-member chain (lp != ld)."""
    t = np.asarray(t, float)
    return (branch * lp / (ld - lp) * n_p0 * (np.exp(-lp * t) - np.exp(-ld * t))
            + n_d0 * np.exp(-ld * t))


def time_of_max_daughter_activity(lp: float, ld: float) -> float:
    """t_max = ln(ld/lp) / (ld - lp) for a daughter starting at zero."""
    return float(np.log(ld / lp) / (ld - lp))


def generator_elution(t_elutions_h: list[float], a_mo0: float, t_end_h: float,
                      dt_h: float = 0.1, efficiency: float = 1.0) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Simulate a Mo-99/Tc-99m generator eluted at the given times.

    Returns (t, Tc-99m activity in the column, eluted activities). Activities
    are in the same unit as a_mo0. Each elution removes ``efficiency`` of the
    Tc-99m present.
    """
    chain = [MO99, TC99M]
    n = np.array([a_mo0 / MO99.lam, 0.0])
    M = chain_matrix(chain, [MO99_TO_TC99M_BRANCH])
    step = expm(M * dt_h)
    ts = np.arange(0.0, t_end_h + 1e-9, dt_h)
    pending = sorted(t_elutions_h)
    tc, eluted = [], []
    for t in ts:
        while pending and t >= pending[0] - 1e-9:
            eluted.append(efficiency * n[1] * TC99M.lam)
            n[1] *= 1 - efficiency
            pending.pop(0)
        tc.append(n[1] * TC99M.lam)
        n = step @ n
    return ts, np.array(tc), eluted
