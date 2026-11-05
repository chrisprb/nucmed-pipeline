"""Stage 4: Lu-177 decay data, absorbed fractions, TAC integration
(targets T4.1-T4.6 fixed in DESIGN.md before implementation)."""

import csv
from pathlib import Path

import numpy as np
import pytest

from nucmed import dosimetry as dz
from nucmed.photon3d import absorbed_fraction_sphere, point_kernel, uniform_in_sphere
from nucmed.mc import isotropic
from nucmed.radionuclide import lu177

DATA = Path(dz.__file__).with_name("data")


# --- T4.1 decay data --------------------------------------------------------

def test_lu177_half_life_and_beta_spectrum():
    nuc = lu177()
    assert nuc.half_life_s / 86400 == pytest.approx(6.6443, abs=1e-4)
    e, p = nuc.beta_spectrum
    b = nuc.beta_branches
    mean_spec = np.trapezoid(e * p, e) / np.trapezoid(p, e)
    mean_branches = (b[:, 0] * b[:, 1]).sum() / b[:, 1].sum()
    assert mean_spec == pytest.approx(mean_branches, rel=0.01)
    assert b[:, 1].sum() == pytest.approx(1.0, abs=0.01)  # 100 % beta-minus
    assert np.trapezoid(p, e) == pytest.approx(b[:, 1].sum(), rel=0.01)  # spectrum per decay
    assert e.max() <= b[:, 2].max() + 1  # spectrum ends at the endpoint (496.8 keV)


def test_lu177_principal_photons():
    ph = lu177().photon_lines
    for e0, y0 in ((112.9498, 0.0623), (208.3662, 0.1041)):
        k = np.argmin(abs(ph[:, 0] - e0))
        assert ph[k, 0] == pytest.approx(e0) and ph[k, 1] == pytest.approx(y0)


# --- T4.2 photon absorbed fractions vs Stabin & Konijnenberg 2000 --------------

def _photon_rows():
    rows = [r for r in dz.stabin2000() if r["table"].startswith("photon") and 20 <= r["energy_MeV"] * 1000 <= 208.4]
    # Keep the test affordable: every energy, three masses spanning each table.
    out = []
    for tab in ("photon_small", "photon_large"):
        for e in sorted({r["energy_MeV"] for r in rows if r["table"] == tab}):
            sub = sorted((r for r in rows if r["table"] == tab and r["energy_MeV"] == e), key=lambda r: r["mass_g"])
            out += [sub[0], sub[len(sub) // 2], sub[-1]]
    return out


@pytest.mark.parametrize("row", _photon_rows(), ids=lambda r: f"{r['table']}-{r['energy_MeV']}MeV-{r['mass_g']}g")
def test_photon_af_regression_guard(row):
    """REGRESSION GUARD, not the pre-registered T4.2 criterion. T4.2 is
    evaluated on every table row by scripts/run_dosimetry_demo.py and reported
    there. Here: sphere embedded in water (the reference geometry), within
    20 % of the recommended value, to catch code breakage."""
    af, se = absorbed_fraction_sphere(row["energy_MeV"] * 1000, row["radius_cm"], n=200_000, seed=11)
    assert se / af < 0.02
    assert abs(af / row["recommended"] - 1) < 0.20, (af, row["recommended"])


# --- T4.3 small-sphere limit --------------------------------------------------

def _mu_en(e_kev):
    rows = list(csv.DictReader(l for l in open(DATA / "nist_muen_water.csv") if not l.startswith("#")))
    e = np.array([float(r["energy_MeV"]) for r in rows]) * 1000
    m = np.array([float(r["mu_en_rho_cm2_g"]) for r in rows])
    return float(np.exp(np.interp(np.log(e_kev), np.log(e), np.log(m))))


@pytest.mark.parametrize("e_kev", [100.0, 208.4])
def test_photon_af_small_sphere_limit(e_kev):
    """Isolated sphere, mu R << 1: AF -> mu_en * mean escape distance (3R/4)."""
    R = 0.02  # mu R ~ 0.003
    af, se = absorbed_fraction_sphere(e_kev, R, n=2_000_000, seed=12, surround="vacuum")
    assert af == pytest.approx(_mu_en(e_kev) * 0.75 * R, rel=0.05)


# --- T4.4 electron sanity -------------------------------------------------------

def test_escape_distance_distribution_by_sampling():
    """3L/4R - L^3/16R^3 against sampled distances to the surface."""
    rng = np.random.default_rng(13)
    R = 1.0
    p = uniform_in_sphere(400_000, R, rng)
    d = isotropic(len(p), rng)
    b = (p * d).sum(1)
    s = -b + np.sqrt(b * b - (p * p).sum(1) + R**2)
    for L in (0.1, 0.5, 1.0, 1.7):
        assert (s <= L).mean() == pytest.approx(dz.escape_distance_cdf(L, R), abs=3e-3)


def test_csda_straight_line_is_lower_bound_on_reference():
    for r in dz.stabin2000():
        if r["table"] == "electron" and r["energy_MeV"] <= 1.0:
            lb = dz.electron_af_csda_straight(r["energy_MeV"] * 1000, r["radius_cm"])
            assert lb <= r["recommended"] + 0.005, (r, lb)


def test_csda_range_matches_estar_node():
    assert dz.csda_range_cm(500.0) == pytest.approx(0.1766, rel=1e-6)
    assert dz.energy_with_range(dz.csda_range_cm(123.0)) == pytest.approx(123.0, rel=1e-9)


def test_electron_af_interpolation_hits_table_and_bounds():
    rows = [r for r in dz.stabin2000() if r["table"] == "electron"]
    for r in rows[::7]:
        assert dz.electron_af(r["energy_MeV"] * 1000, r["mass_g"]) == pytest.approx(r["recommended"])
    e = np.array([10.0, 50.0])
    lo, mid, hi = (dz.electron_af(e, 0.01, k) for k in ("constant", "linear", "one"))
    assert np.all(lo <= mid) and np.all(mid <= hi)


# --- T4.5 point kernel ----------------------------------------------------------

def test_point_kernel_energy_conservation_and_sphere_consistency():
    edges = np.concatenate([[0], np.geomspace(0.01, 100.0, 300)])
    k = point_kernel(208.4, edges, n=300_000, seed=14)
    assert k.sum() >= 0.99
    R = 2.88  # 100 g sphere
    rc = 0.5 * (edges[1:] + edges[:-1])
    # Probability that a deposit at distance r from a uniform source point is inside.
    p_in = 1 - dz.escape_distance_cdf(rc, R)
    af_kernel = float(np.sum(k * p_in))
    af_direct, _ = absorbed_fraction_sphere(208.4, R, n=300_000, seed=15, surround="water")
    assert af_kernel == pytest.approx(af_direct, rel=0.03)


# --- T4.6 TAC -----------------------------------------------------------------------

LAM_PHYS = np.log(2) / (6.6443 * 24)  # per hour


def test_mono_exp_fit_and_tia_exact():
    t = np.array([4.0, 24, 96, 168])
    a = dz.mono_exp(t, 5.0, 0.012)
    a0, lam = dz.fit_mono(t, a)
    assert dz.tia_mono(a0, lam) == pytest.approx(5.0 / 0.012, rel=1e-6)


def test_uptake_washout_fit_exact():
    t = np.array([1.0, 4, 24, 48, 96, 168])
    p = (10.0, 0.01, 0.4)
    a0, l1, l2 = dz.fit_uptake_washout(t, dz.uptake_washout(t, *p), LAM_PHYS)
    assert dz.tia_uptake_washout(a0, l1, l2) == pytest.approx(dz.tia_uptake_washout(*p), rel=1e-6)


def test_trapezoid_converges_with_dense_sampling():
    p = (10.0, 0.01, 0.4)
    t = np.linspace(0.01, 2000, 200_001)
    tia = dz.tia_trapezoid(t, dz.uptake_washout(t, *p), lam_tail=0.01)
    assert tia == pytest.approx(dz.tia_uptake_washout(*p), rel=1e-4)


def test_s_value_units_large_sphere_limit():
    """Units: for a 1 kg sphere, S_electrons * m is the electron energy
    absorbed per decay, which lies just below the energy emitted."""
    nuc = lu177()
    s = dz.s_value_sphere(nuc, 1000.0, photon_af=0.0)
    e_el = sum(v for k, v in nuc.energy_per_decay_kev().items() if k != "photons")
    absorbed = s["electrons"] * 1.0 / 1.602176634e-16  # keV per decay (m = 1 kg)
    assert 0.98 * e_el <= absorbed <= e_el  # reference AFs at 1 kg are 0.988-0.999
