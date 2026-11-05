"""Statistical reconstruction: MLEM and OSEM.

MLEM (Shepp & Vardi 1982 [verify]) update for Poisson data y ~ Poisson(A x + r):

    x <- x / (A^T 1) * A^T ( y / (A x + r) )

OSEM (Hudson & Larkin 1994 [verify]) applies the same update over ordered
subsets of projection angles; with one subset it is exactly MLEM.
``r`` is an optional known additive term (e.g. a scatter estimate), handled
in the denominator as in standard practice.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .projector import SystemMatrix

EPS = 1e-12


def subset_angles(n_angles: int, n_subsets: int) -> list[np.ndarray]:
    """Interleaved subsets (0, S, 2S, ...), (1, S+1, ...), ... so each subset
    spans the full arc."""
    if n_angles % n_subsets:
        raise ValueError(f"n_subsets={n_subsets} must divide n_angles={n_angles}")
    return [np.arange(k, n_angles, n_subsets) for k in range(n_subsets)]


def osem(
    sm: SystemMatrix,
    sino: np.ndarray,
    n_iter: int = 4,
    n_subsets: int = 8,
    x0: np.ndarray | None = None,
    additive: np.ndarray | None = None,
    callback: Callable[[int, np.ndarray], None] | None = None,
) -> np.ndarray:
    """OSEM. ``callback(iteration, image)`` is called after every full iteration."""
    y = sino.ravel().astype(np.float64)
    r = np.zeros_like(y) if additive is None else additive.ravel().astype(np.float64)
    x = np.ones(sm.grid.n * sm.grid.n) if x0 is None else x0.ravel().astype(np.float64).copy()

    subsets = []
    for ang in subset_angles(sm.acq.n_angles, n_subsets):
        rows = sm.angle_rows(ang)
        As = sm.A[rows]
        sens = np.asarray(As.sum(axis=0)).ravel()
        subsets.append((rows, As, sens))

    # Pixels no projection sees stay zero instead of dividing by zero.
    seen = np.asarray(sm.A.sum(axis=0)).ravel() > 0
    x[~seen] = 0.0

    for it in range(1, n_iter + 1):
        for rows, As, sens in subsets:
            ybar = As @ x + r[rows]
            ratio = np.where(ybar > EPS, y[rows] / np.maximum(ybar, EPS), 0.0)
            x = np.where(sens > 0, x * (As.T @ ratio) / np.maximum(sens, EPS), 0.0)
        if callback is not None:
            callback(it, x.reshape(sm.grid.n, sm.grid.n))
    return x.reshape(sm.grid.n, sm.grid.n)


def mlem(sm: SystemMatrix, sino: np.ndarray, n_iter: int = 50, **kw) -> np.ndarray:
    return osem(sm, sino, n_iter=n_iter, n_subsets=1, **kw)


def poisson_loglik(sm: SystemMatrix, sino: np.ndarray, x: np.ndarray,
                   additive: np.ndarray | None = None) -> float:
    """Poisson log-likelihood up to a constant: sum y log(ybar) - ybar."""
    ybar = sm.A @ x.ravel()
    if additive is not None:
        ybar = ybar + additive.ravel()
    y = sino.ravel()
    return float(np.sum(y * np.log(np.maximum(ybar, EPS)) - ybar))
