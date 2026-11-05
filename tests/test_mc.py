"""Stage 2: cross-sections and Monte Carlo transport against analytic and
NIST XCOM references (targets fixed in DESIGN.md before implementation)."""

import numpy as np
import pytest
from scipy import integrate, stats
from scipy.special import erf

from nucmed.geometry import Acquisition, Collimator, ImageGrid
from nucmed.mc import MCConfig, isotropic, rotate, simulate, to_sinogram, woodcock_flight
from nucmed.phantom import MU_WATER_140KEV, disk_mask
from nucmed.projector import build_system_matrix
from nucmed.xsections import (E_TC99M_KEV, ELECTRONS_PER_G_WATER, MEC2_KEV, _table,
                              compton_energy, kn_pdf_omega, kn_sigma_total,
                              mass_coefficients, mu_water, sample_kn_cos,
                              sample_thomson_cos, thomson_pdf_omega)

E0 = E_TC99M_KEV


# --- cross-section data -----------------------------------------------------

def test_xcom_partials_sum_to_reported_total():
    """The stored XCOM partials add up to XCOM's own total (rounding only)."""
    t = _table()
    np.testing.assert_allclose(t[:, 1:4].sum(axis=1), t[:, 4], rtol=2e-3)


def test_mu_water_140kev_matches_xcom_and_phantom_constant():
    assert mu_water(E0) == pytest.approx(0.1537, rel=1e-3)
    assert MU_WATER_140KEV == pytest.approx(float(mu_water(E0)), rel=2e-3)


def test_kn_total_vs_xcom_incoherent():
    """(a) Free-electron KN x electron density is within 3 % of XCOM
    incoherent, and above it (binding reduces the cross-section)."""
    kn = kn_sigma_total(E0) * ELECTRONS_PER_G_WATER
    inc = mass_coefficients(E0)["incoherent"]
    assert kn > inc
    assert kn / inc - 1 < 0.03


@pytest.mark.parametrize("e", [30.0, E0, 200.0, 511.0])
def test_kn_pdf_normalised(e):
    val, _ = integrate.quad(lambda c: 2 * np.pi * kn_pdf_omega(e, c), -1, 1)
    assert val == pytest.approx(1.0, rel=1e-8)


def test_kn_thomson_limit():
    """As E -> 0 Klein-Nishina tends to Thomson; the closed-form total and the
    low-energy series agree where they meet."""
    c = np.linspace(-1, 1, 11)
    np.testing.assert_allclose(kn_pdf_omega(1e-3, c), thomson_pdf_omega(c), rtol=1e-5)
    e_switch = 1e-3 * MEC2_KEV
    assert kn_sigma_total(e_switch * (1 - 1e-9)) == pytest.approx(kn_sigma_total(e_switch * (1 + 1e-9)), rel=1e-6)


# --- sampling ---------------------------------------------------------------

def _chi2_p(samples, pdf_cos, n_bins=50):
    edges = np.linspace(-1, 1, n_bins + 1)
    obs, _ = np.histogram(samples, edges)
    exp = np.array([integrate.quad(pdf_cos, a, b)[0] for a, b in zip(edges[:-1], edges[1:])])
    exp *= len(samples) / exp.sum()
    return stats.chisquare(obs, exp).pvalue


def test_kn_sampling_chi2():
    """Sampled Compton angles follow Klein-Nishina (chi2, p > 0.01, 1e6 samples)."""
    rng = np.random.default_rng(1)
    s = sample_kn_cos(np.full(1_000_000, E0), rng)
    assert _chi2_p(s, lambda c: 2 * np.pi * kn_pdf_omega(E0, c)) > 0.01


def test_thomson_sampling_chi2():
    rng = np.random.default_rng(2)
    s = sample_thomson_cos(1_000_000, rng)
    assert _chi2_p(s, lambda c: 2 * np.pi * thomson_pdf_omega(c)) > 0.01


def test_rotation_gives_requested_angle_and_unit_vectors():
    rng = np.random.default_rng(3)
    d = isotropic(100_000, rng)
    d[:10] = [0, 0, 1]; d[10:20] = [0, 0, -1]  # poles
    c = rng.uniform(-1, 1, len(d))
    out = rotate(d, c, rng.uniform(0, 2 * np.pi, len(d)))
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1, atol=1e-12)
    np.testing.assert_allclose((d * out).sum(axis=1), c, atol=1e-9)


def test_isotropic_emission_is_isotropic():
    d = isotropic(1_000_000, np.random.default_rng(4))
    assert np.abs(d.mean(axis=0)).max() < 3e-3
    np.testing.assert_allclose((d**2).mean(axis=0), 1 / 3, atol=3e-3)


def test_compton_energy_formula():
    """Single scatter: E' = E/(1 + k(1 - cos)); forward unchanged, backscatter
    at the Compton edge E/(1 + 2k) (90.7 keV for Tc-99m)."""
    k = E0 / MEC2_KEV
    assert compton_energy(E0, 1.0) == E0
    assert compton_energy(E0, -1.0) == pytest.approx(E0 / (1 + 2 * k), rel=1e-12)
    assert compton_energy(E0, -1.0) == pytest.approx(90.7, abs=0.05)
    c = np.linspace(-1, 1, 7)
    np.testing.assert_allclose(1 / compton_energy(E0, c) - 1 / E0, (1 - c) / MEC2_KEV, rtol=1e-12)


# --- transport --------------------------------------------------------------

def _pencil_transmission(rho_of_x, L, n=1_000_000, seed=5):
    rng = np.random.default_rng(seed)
    pos = np.zeros((n, 3)); pos[:, 0] = 1e-9
    d = np.tile([1.0, 0, 0], (n, 1))
    mu = np.full(n, float(mu_water(E0)))
    _, esc = woodcock_flight(pos, d, mu, lambda p: rho_of_x(p[:, 0]), 1.0,
                             lambda p: (p[:, 0] >= 0) & (p[:, 0] <= L), rng)
    return esc.mean()  # escaped without a real collision = unscattered transmission


def test_narrow_beam_mu_matches_xcom():
    """Narrow-beam mu from simulated transmission through 10 cm of water is
    within 2 % of XCOM."""
    L = 10.0
    T = _pencil_transmission(lambda x: np.ones_like(x), L)
    assert -np.log(T) / L == pytest.approx(float(mu_water(E0)), rel=0.02)


def test_woodcock_two_density_slab():
    """(b) 5 cm water + 5 cm lung (0.3): T = exp(-mu (5 + 1.5)) within 2 %."""
    T = _pencil_transmission(lambda x: np.where(x < 5.0, 1.0, 0.3), 10.0)
    assert T == pytest.approx(np.exp(-float(mu_water(E0)) * 6.5), rel=0.02)


def test_mc_primary_matches_analytic_projector():
    """(c) Primary-only MC (2 mm phantom) vs the Stage 1 projector (4 mm),
    same collimator and attenuation: totals within 2 %, NRMSE < 5 %."""
    coarse, fine = ImageGrid(80, 0.4), ImageGrid(160, 0.2)
    acq = Acquisition(n_angles=24, n_bins=80, bin_cm=0.4, radius_cm=20.0, collimator=Collimator())
    r = 10.0
    act_f = disk_mask(fine, 0, 0, r)
    res = simulate(act_f, act_f, fine, acq,
                   MCConfig(n_photons=200_000, scatter=False, seed=6))
    mc = to_sinogram(res, fine)["photopeak"]["primary"]
    # Photopeak window is E0 +/- 1 FWHM, so its acceptance is erf(2 sqrt(ln 2)).
    acc = float(erf(2 * np.sqrt(np.log(2))))
    mc = mc / acc / coarse.pixel_cm**2  # per-pixel activity units, as the projector

    act_c = disk_mask(coarse, 0, 0, r)
    sm = build_system_matrix(coarse, acq, mu=MU_WATER_140KEV * act_c)
    ref = sm.forward(act_c)
    assert mc.sum() == pytest.approx(ref.sum(), rel=0.02)
    assert np.sqrt(np.mean((mc - ref) ** 2)) / ref.mean() < 0.05


def test_scatter_adds_counts_and_respects_compton_edge():
    """With scatter on, scatter is a sizeable part of the photopeak; scattered
    photons never gain energy, and none scores below the Compton edge of the
    lowest energy still tracked (cut E_c: E_c / (1 + 2 E_c / m c^2))."""
    g = ImageGrid(80, 0.4)
    acq = Acquisition(n_angles=12, radius_cm=20.0)
    act = disk_mask(g, 0, 0, 10.0)
    res = simulate(act, act, g, acq, MCConfig(n_photons=20_000, seed=7))
    s = to_sinogram(res, g)["photopeak"]
    assert s["scatter"].sum() > 0.05 * s["primary"].sum()
    lo, hi = res.spectrum_edges[:-1], res.spectrum_edges[1:]
    cut = res.cfg.e_cut_kev
    assert res.spectrum[1][hi <= cut / (1 + 2 * cut / MEC2_KEV)].sum() == 0
    assert res.spectrum[1][lo > E0].sum() == 0
    assert res.spectrum[0][(hi <= E0) | (lo > E0)].sum() == 0  # primaries all at E0
