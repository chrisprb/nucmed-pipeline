"""Count statistics."""

from __future__ import annotations

import numpy as np


def scale_to_counts(sino: np.ndarray, total_counts: float) -> tuple[np.ndarray, float]:
    """Scale a noise-free sinogram to an expected total count. Returns (scaled, factor)."""
    f = total_counts / sino.sum()
    return sino * f, f


def poisson(sino: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.poisson(np.clip(sino, 0, None)).astype(np.float64)
