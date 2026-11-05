"""Stage 3: TEW estimator and OSEM with a known additive term."""

import numpy as np
import pytest

from nucmed.geometry import Acquisition, ImageGrid
from nucmed.metrics import core_mask, roi_mean
from nucmed.phantom import uniform_disk
from nucmed.projector import build_system_matrix
from nucmed.recon import osem
from nucmed.scatter import smooth_sinogram, tew, windows_widths
from nucmed.mc import tc99m_windows


def test_tew_exact_for_linear_spectrum():
    """If the scatter spectrum is linear in energy across the windows, the
    trapezoid estimate equals the true counts in the photopeak window."""
    a, b = 50.0, -0.3  # counts per keV = a + b E
    win = tc99m_windows()
    integral = lambda lo, hi: a * (hi - lo) + 0.5 * b * (hi**2 - lo**2)
    C = {k: integral(*v) for k, v in win.items()}
    wl, wu, wp = windows_widths(win)
    assert tew(C["lower"], C["upper"], wl, wu, wp) == pytest.approx(C["photopeak"], rel=1e-12)


def test_tew_lower_only_is_triangle():
    assert tew(np.array([6.0]), None, 3.0, 3.0, 28.0)[0] == pytest.approx(28.0)


def test_smoothing_preserves_counts_and_wraps_in_angle():
    s = np.zeros((36, 40)); s[0, 20] = 1.0
    out = smooth_sinogram(s, 2.0, 2.0)
    assert out.sum() == pytest.approx(1.0, rel=1e-6)
    assert out[-1, 20] == pytest.approx(out[1, 20], rel=1e-9)  # wraps across angle 0


def test_osem_with_exact_additive_term_removes_bias():
    """Noise-free data contaminated by a known smooth additive term: with r
    modelled, the uniform disk is recovered as well as without contamination;
    ignoring r biases it upward."""
    grid = ImageGrid(n=48, pixel_cm=0.5)
    ph = uniform_disk(grid, radius_cm=9.0)
    sm = build_system_matrix(grid, Acquisition(n_angles=48), mu=ph.mu, support=ph.mu > 0)
    prim = sm.forward(ph.activity)
    r = smooth_sinogram(0.3 * prim, 6.0, 4.0)  # broad, scatter-like
    inner = core_mask(ph.rois["disk"])
    clean = roi_mean(osem(sm, prim, n_iter=20, n_subsets=8), inner)
    with_r = roi_mean(osem(sm, prim + r, n_iter=20, n_subsets=8, additive=r), inner)
    without = roi_mean(osem(sm, prim + r, n_iter=20, n_subsets=8), inner)
    assert with_r == pytest.approx(clean, rel=0.01)
    assert without > 1.15 * clean
