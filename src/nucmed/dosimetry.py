"""MIRD-style internal dosimetry: time-activity curves, absorbed fractions and
sphere S-values.

    D = Ã * S,   S(m) = sum_i Delta_i * phi_i(m) / m

* Electron absorbed fractions for unit-density spheres come from Stabin &
  Konijnenberg, J Nucl Med 2000;41:149-160 (recommended values; data in
  data/stabin2000_sphere_af.csv), used as reference input data. Tabulated
  energies are 0.1-2 MeV and masses 0.01-1000 g. Below 0.1 MeV the fraction
  is interpolated linearly to 1 at E = 0 (``low_energy="linear"``); the
  bounds "constant" (phi(0.1 MeV)) and "one" are available to bound that
  assumption.
* A straight-line CSDA model (NIST ESTAR ranges) gives a lower bound on the
  electron absorbed fraction, used as a sanity check.
* Photon absorbed fractions come from our own Monte Carlo (``photon3d``),
  validated against the same paper's photon tables.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from .radionuclide import KEV_TO_J, Radionuclide

_DATA = Path(__file__).with_name("data")


# --- geometry ---------------------------------------------------------------

def sphere_radius_cm(mass_g: float, density: float = 1.0) -> float:
    return (3 * mass_g / (4 * np.pi * density)) ** (1 / 3)


def escape_distance_cdf(L, R):
    """P(distance to the surface <= L) for a uniform point in a sphere of
    radius R and an isotropic direction: 3L/(4R) - L^3/(16R^3), L <= 2R."""
    L = np.minimum(np.asarray(L, float), 2 * R)
    return 3 * L / (4 * R) - L**3 / (16 * R**3)


def escape_distance_pdf(s, R):
    s = np.asarray(s, float)
    return np.where(s <= 2 * R, 3 / (4 * R) - 3 * s**2 / (16 * R**3), 0.0)


# --- reference tables -------------------------------------------------------

def _csv(name: str) -> list[dict[str, str]]:
    with open(_DATA / name) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


@lru_cache(maxsize=1)
def stabin2000() -> list[dict[str, float | str]]:
    return [{k: (v if k == "table" else float(v)) for k, v in r.items()} for r in _csv("stabin2000_sphere_af.csv")]


@lru_cache(maxsize=1)
def _electron_grid():
    rows = [r for r in stabin2000() if r["table"] == "electron"]
    energies = np.array(sorted({r["energy_MeV"] for r in rows}))
    masses = np.array(sorted({r["mass_g"] for r in rows}))
    grid = np.full((len(energies), len(masses)), np.nan)
    for r in rows:
        grid[np.searchsorted(energies, r["energy_MeV"]), np.searchsorted(masses, r["mass_g"])] = r["recommended"]
    assert not np.isnan(grid).any(), "incomplete electron table"
    return energies * 1000.0, masses, grid  # keV


def electron_af(e_kev, mass_g: float, low_energy: str = "linear") -> np.ndarray:
    """Electron absorbed fraction for a unit-density sphere (Stabin & Konijnenberg 2000),
    linear in energy and in log(mass) between table nodes."""
    energies, masses, grid = _electron_grid()
    if not masses[0] <= mass_g <= masses[-1]:
        raise ValueError(f"mass {mass_g} g outside the electron table ({masses[0]}-{masses[-1]} g)")
    lm = np.log(masses)
    at_m = np.array([np.interp(np.log(mass_g), lm, grid[i]) for i in range(len(energies))])
    e = np.asarray(e_kev, float)
    if e.max(initial=0) > energies[-1]:
        raise ValueError("energy above the electron table")
    phi = np.interp(e, energies, at_m)
    lowmask = e < energies[0]
    if low_energy == "linear":
        phi = np.where(lowmask, 1 + (at_m[0] - 1) * e / energies[0], phi)
    elif low_energy == "constant":
        phi = np.where(lowmask, at_m[0], phi)
    elif low_energy == "one":
        phi = np.where(lowmask, 1.0, phi)
    else:
        raise ValueError(low_energy)
    return phi


@lru_cache(maxsize=1)
def _estar():
    rows = _csv("estar_water.csv")
    e = np.array([float(r["energy_MeV"]) for r in rows]) * 1000
    r = np.array([float(r["csda_range_g_cm2"]) for r in rows])
    return e, r


def csda_range_cm(e_kev) -> np.ndarray:
    """CSDA range in water (cm), log-log interpolation of ESTAR; below the first
    node (10 keV) extrapolated with the local power law."""
    e, r = _estar()
    le, lr = np.log(e), np.log(r)
    x = np.log(np.maximum(np.asarray(e_kev, float), 1e-3))
    slope = (lr[1] - lr[0]) / (le[1] - le[0])
    return np.exp(np.where(x < le[0], lr[0] + slope * (x - le[0]), np.interp(x, le, lr)))


def energy_with_range(x_cm) -> np.ndarray:
    """Inverse of csda_range_cm."""
    e, r = _estar()
    le, lr = np.log(e), np.log(r)
    slope = (lr[1] - lr[0]) / (le[1] - le[0])
    lx = np.log(np.maximum(np.asarray(x_cm, float), 1e-30))
    return np.exp(np.where(lx < lr[0], le[0] + (lx - lr[0]) / slope, np.interp(lx, lr, le)))


def electron_af_csda_straight(e_kev: float, radius_cm: float, n: int = 4000) -> float:
    """Absorbed fraction if electrons travelled in straight lines losing energy
    continuously (ESTAR CSDA). Real tracks are tortuous and penetrate less, so
    this is a lower bound on the true absorbed fraction."""
    r0 = float(csda_range_cm(e_kev))
    smax = min(r0, 2 * radius_cm)
    s = (np.arange(n) + 0.5) * smax / n
    residual = energy_with_range(r0 - s)  # energy left when crossing the surface
    escaped = np.sum(escape_distance_pdf(s, radius_cm) * residual) * smax / n
    return float(1 - escaped / e_kev)


# --- time-activity curves ---------------------------------------------------

def mono_exp(t, a0, lam):
    return a0 * np.exp(-lam * t)


def uptake_washout(t, a0, lam_out, lam_in):
    """A0 (exp(-lam_out t) - exp(-lam_in t)), lam_in > lam_out (both effective)."""
    return a0 * (np.exp(-lam_out * t) - np.exp(-lam_in * t))


def tia_mono(a0, lam):
    """Time-integrated activity of a mono-exponential, 0 to infinity."""
    return a0 / lam


def tia_uptake_washout(a0, lam_out, lam_in):
    return a0 * (1 / lam_out - 1 / lam_in)


def fit_mono(t, a):
    """Least-squares fit in log space (equal relative weights)."""
    slope, intercept = np.polyfit(t, np.log(a), 1)
    return float(np.exp(intercept)), float(-slope)


def fit_uptake_washout(t, a, lam_phys):
    """Fit A0 (e^-l1 t - e^-l2 t) with l1 >= lam_phys (no faster-than-physical
    clearance is forbidden; slower than physical is not), relative weights."""
    t, a = np.asarray(t, float), np.asarray(a, float)
    a0_guess = a.max() * 1.5
    p, _ = curve_fit(uptake_washout, t, a, p0=(a0_guess, 2 * lam_phys, 20 * lam_phys),
                     sigma=a, bounds=([0, lam_phys, lam_phys], [np.inf, np.inf, np.inf]), maxfev=20000)
    return tuple(float(x) for x in p)


def tia_trapezoid(t, a, lam_tail: float, start: str = "zero") -> float:
    """Trapezoid between samples; from 0 to the first sample a straight line
    from A(0) = 0 (``start="zero"``) or a constant (``"constant"``); after the
    last sample a mono-exponential tail with rate ``lam_tail``."""
    t, a = np.asarray(t, float), np.asarray(a, float)
    head = 0.5 * a[0] * t[0] if start == "zero" else a[0] * t[0]
    return float(head + np.trapezoid(a, t) + a[-1] / lam_tail)


# --- S-values ---------------------------------------------------------------

def electron_energy_absorbed_kev(nuc: Radionuclide, mass_g: float, low_energy: str = "linear") -> float:
    """Mean electron energy absorbed per decay in a sphere (beta continuum + lines)."""
    e, n = nuc.beta_spectrum_normalised()
    beta = np.trapezoid(e * n * electron_af(e, mass_g, low_energy), e)
    lines = nuc.electron_lines
    return float(beta + np.sum(lines[:, 0] * lines[:, 1] * electron_af(lines[:, 0], mass_g, low_energy)))


def s_value_sphere(nuc: Radionuclide, mass_g: float, photon_af: float,
                   low_energy: str = "linear") -> dict[str, float]:
    """Self-dose S-value (Gy per Bq s) of a unit-density sphere, split by
    electrons and photons. ``photon_af``: energy-weighted photon absorbed
    fraction for this sphere (from photon3d)."""
    m_kg = mass_g / 1000.0
    e_el = electron_energy_absorbed_kev(nuc, mass_g, low_energy)
    e_ph = nuc.energy_per_decay_kev()["photons"] * photon_af
    return {"electrons": e_el * KEV_TO_J / m_kg, "photons": e_ph * KEV_TO_J / m_kg,
            "total": (e_el + e_ph) * KEV_TO_J / m_kg}
