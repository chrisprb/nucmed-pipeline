"""Forward model checked against analytic results."""

import numpy as np
import pytest

from nucmed.geometry import Acquisition, Collimator, ImageGrid
from nucmed.phantom import disk_mask, point_source
from nucmed.projector import attenuation_factors, build_system_matrix

IDEAL = Collimator(0.0, 0.0)
GRID = ImageGrid(n=81, pixel_cm=0.4)  # odd n: a pixel sits exactly on the axis


def test_adjoint_identity():
    """<A x, y> == <x, A^T y> for random x, y (matched projector pair)."""
    rng = np.random.default_rng(0)
    mu = 0.15 * disk_mask(GRID, 0, 0, 12)
    sm = build_system_matrix(GRID, Acquisition(n_angles=36), mu=mu)
    x = rng.random((GRID.n, GRID.n))
    y = rng.random(sm.sino_shape)
    assert np.isclose(np.vdot(sm.forward(x), y), np.vdot(x, sm.back(y)), rtol=1e-5)


@pytest.mark.parametrize("collimator", [IDEAL, Collimator()])
def test_count_conservation_without_attenuation(collimator):
    """With no attenuation every projection holds all counts of an object
    that stays inside the detector."""
    img = disk_mask(GRID, 2.0, -1.0, 8.0)
    sm = build_system_matrix(GRID, Acquisition(n_angles=24, collimator=collimator))
    per_angle = sm.forward(img).sum(axis=1)
    np.testing.assert_allclose(per_angle, img.sum(), rtol=1e-4)


@pytest.mark.parametrize("theta_deg", [0, 30, 90, 135, 250])
def test_point_source_centroid_and_width(theta_deg):
    """Point source projects to s = x cos(t) + y sin(t) with variance
    sigma(d)^2 + ds^2/12 (Gaussian blur integrated over a bin)."""
    x0, y0 = 4.0, -6.0
    p = point_source(GRID, x0, y0)
    acq = Acquisition(n_angles=360, collimator=Collimator(0.3, 0.06), radius_cm=25.0)
    sm = build_system_matrix(GRID, acq)
    a = theta_deg  # 1 degree per angle
    th = np.deg2rad(theta_deg)
    prof = sm.forward(p.activity)[a]
    n_bins, ds = acq.bins(GRID)
    s_bins = (np.arange(n_bins) - (n_bins - 1) / 2) * ds
    mean = np.sum(prof * s_bins) / prof.sum()
    var = np.sum(prof * (s_bins - mean) ** 2) / prof.sum()

    assert mean == pytest.approx(x0 * np.cos(th) + y0 * np.sin(th), abs=0.02)
    t = -x0 * np.sin(th) + y0 * np.cos(th)
    sigma = acq.collimator.sigma_cm(np.array(25.0 - t))
    assert var == pytest.approx(sigma**2 + ds**2 / 12, rel=0.03)


def _chord_to_circle_edge(px, py, ux, uy, R):
    pu = px * ux + py * uy
    return -pu + np.sqrt(pu**2 - (px**2 + py**2 - R**2))


@pytest.mark.parametrize("point", [(0.0, 0.0), (4.0, 3.2), (-6.0, -2.0)])
@pytest.mark.parametrize("theta_deg", [0, 45, 100, 200, 315])
def test_attenuation_matches_chord_length(point, theta_deg):
    """Inside a uniform disk, att = exp(-mu * distance to the edge along u)."""
    mu0, R = 0.154, 11.0
    mu = mu0 * disk_mask(GRID, 0, 0, R, ss=8)
    th = np.deg2rad(theta_deg)
    att = attenuation_factors(mu, GRID, th)
    row, col = GRID.to_index(np.array(point[0]), np.array(point[1]))
    got = att[int(round(float(row))), int(round(float(col)))]
    expected = np.exp(-mu0 * _chord_to_circle_edge(*point, -np.sin(th), np.cos(th), R))
    assert got == pytest.approx(expected, rel=0.01)


def test_point_at_centre_of_attenuating_disk():
    """Total counts of a centred point source fall by exp(-mu R) at every angle."""
    mu0, R = 0.154, 10.0
    mu = mu0 * disk_mask(GRID, 0, 0, R, ss=8)
    p = point_source(GRID, 0.0, 0.0, mu)
    sm = build_system_matrix(GRID, Acquisition(n_angles=12, collimator=IDEAL), mu=mu)
    np.testing.assert_allclose(sm.forward(p.activity).sum(axis=1), np.exp(-mu0 * R), rtol=0.01)
