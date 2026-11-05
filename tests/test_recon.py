"""MLEM/OSEM checked against known algorithmic properties."""

import numpy as np
import pytest

from nucmed.geometry import Acquisition, ImageGrid
from nucmed.noise import poisson, scale_to_counts
from nucmed.phantom import uniform_disk
from nucmed.projector import build_system_matrix
from nucmed.recon import mlem, osem, poisson_loglik, subset_angles
from nucmed.metrics import core_mask, roi_mean

GRID = ImageGrid(n=48, pixel_cm=0.5)


@pytest.fixture(scope="module")
def setup():
    ph = uniform_disk(GRID, radius_cm=9.0)
    acq = Acquisition(n_angles=48)
    sm = build_system_matrix(GRID, acq, mu=ph.mu, support=ph.mu > 0)
    clean, _ = scale_to_counts(sm.forward(ph.activity), 2e5)
    noisy = poisson(clean, np.random.default_rng(1))
    return ph, sm, clean, noisy


def test_subsets_partition_angles():
    s = subset_angles(48, 8)
    assert sorted(np.concatenate(s).tolist()) == list(range(48))
    with pytest.raises(ValueError):
        subset_angles(48, 7)


def test_mlem_preserves_counts_and_increases_likelihood(setup):
    """Each MLEM iterate satisfies sum(A x) = sum(y), and the Poisson
    log-likelihood never decreases (EM monotonicity)."""
    _, sm, _, noisy = setup
    lls, totals = [], []

    def cb(_, x):
        lls.append(poisson_loglik(sm, noisy, x))
        totals.append(sm.forward(x).sum())

    mlem(sm, noisy, n_iter=15, callback=cb)
    np.testing.assert_allclose(totals, noisy.sum(), rtol=1e-6)
    assert np.all(np.diff(lls) >= -1e-6 * abs(lls[0]))


def test_osem_one_subset_equals_mlem(setup):
    _, sm, _, noisy = setup
    np.testing.assert_allclose(osem(sm, noisy, n_iter=5, n_subsets=1), mlem(sm, noisy, n_iter=5))


def test_attenuation_correction_recovers_uniform_activity(setup):
    """Noise-free data: with attenuation modelled the interior is flat and
    correct; without it the centre is strongly depressed (the classic
    cupping artefact), which shows the correction is doing the work."""
    ph, sm, clean, _ = setup
    scale = clean.sum() / sm.forward(ph.activity).sum()
    x = osem(sm, clean, n_iter=30, n_subsets=8) / scale

    inner = core_mask(ph.rois["disk"]) * (ph.mu > 0)
    rr = np.hypot(*GRID.centres())
    centre, rim = (rr < 2.0) & (inner > 0), (rr > 6.0) & (rr < 7.5) & (inner > 0)
    assert roi_mean(x, inner) == pytest.approx(1.0, rel=0.03)
    assert x[centre].mean() / x[rim].mean() == pytest.approx(1.0, abs=0.05)

    sm_nac = build_system_matrix(GRID, sm.acq, mu=None, support=ph.mu > 0)
    x_nac = osem(sm_nac, clean, n_iter=30, n_subsets=8)
    assert x_nac[centre].mean() / x_nac[rim].mean() < 0.7
