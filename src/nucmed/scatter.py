"""Scatter estimation from energy windows.

Triple-energy-window (TEW) estimate of the scatter counts in the photopeak
window, per projection bin: the scatter spectrum under the photopeak is
approximated by a trapezoid whose heights are the count densities (counts per
keV) in two narrow windows abutting the photopeak:

    S = (C_L / W_L + C_U / W_U) * W_P / 2

Setting C_U = 0 gives the lower-window-only (triangle) variant. The estimate
enters OSEM as the known additive term r in y ~ Poisson(A x + r).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def tew(c_lower: np.ndarray, c_upper: np.ndarray | None,
        w_lower: float, w_upper: float, w_peak: float) -> np.ndarray:
    up = 0.0 if c_upper is None else np.asarray(c_upper) / w_upper
    return (np.asarray(c_lower) / w_lower + up) * w_peak / 2.0


def windows_widths(windows: dict[str, tuple[float, float]]) -> tuple[float, float, float]:
    """(W_L, W_U, W_P) in keV from a {'lower', 'upper', 'photopeak'} window dict."""
    w = {k: hi - lo for k, (lo, hi) in windows.items()}
    return w["lower"], w["upper"], w["photopeak"]


def smooth_sinogram(sino: np.ndarray, sigma_bins: float, sigma_angles: float) -> np.ndarray:
    """2D Gaussian smoothing of a (n_angles, n_bins) sinogram; periodic in angle
    (full 360-degree arc), zero beyond the detector edges. Total counts are
    preserved up to what leaks off the detector edges."""
    return gaussian_filter(sino, sigma=(sigma_angles, sigma_bins), mode=("wrap", "constant"))
