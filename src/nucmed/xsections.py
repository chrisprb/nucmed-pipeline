"""Photon interaction data for water, and Klein-Nishina / Thomson kinematics.

Sources
-------
* Mass coefficients: NIST XCOM, water (H2O), queried 2026-11-05; the table and
  query are in ``data/xcom_water.csv``. Interpolated log-log, valid 20-208 keV.
* r_e = 2.8179403205e-13 cm and m_e c^2 = 510.99895069 keV: CODATA 2022 via
  https://physics.nist.gov/cuu/Constants (checked 2026-11-05).
* Electron density of water uses the XCOM weight fractions (H 0.111898,
  O 0.888102) and IUPAC standard atomic weights H 1.008, O 15.999.

Energies are in keV, lengths in cm, mass coefficients in cm^2/g. Materials
are water scaled by relative density, so mu(E, r) = rho(r) * mu_water(E).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

R_E_CM = 2.8179403205e-13
MEC2_KEV = 510.99895069
N_A = 6.02214076e23
ELECTRONS_PER_G_WATER = N_A * (0.111898 * 1 / 1.008 + 0.888102 * 8 / 15.999)
E_TC99M_KEV = 140.5  # principal gamma of Tc-99m

_DATA = Path(__file__).with_name("data") / "xcom_water.csv"


@lru_cache(maxsize=1)
def _table() -> np.ndarray:
    return np.loadtxt(_DATA, delimiter=",", comments="#", skiprows=6)


def mass_coefficients(e_kev) -> dict[str, np.ndarray]:
    """Water mass coefficients (cm^2/g) at energy ``e_kev``: keys
    'coherent', 'incoherent', 'photoelectric', 'total'."""
    t = _table()
    e_mev = np.asarray(e_kev, dtype=float) / 1000.0
    if np.any(e_mev < t[0, 0] * (1 - 1e-9)) or np.any(e_mev > t[-1, 0] * (1 + 1e-9)):
        raise ValueError(f"energy outside tabulated range {t[0,0]*1e3:.0f}-{t[-1,0]*1e3:.0f} keV")
    le = np.log(t[:, 0])
    out = {}
    for name, col in (("coherent", 1), ("incoherent", 2), ("photoelectric", 3)):
        out[name] = np.exp(np.interp(np.log(e_mev), le, np.log(t[:, col])))
    out["total"] = out["coherent"] + out["incoherent"] + out["photoelectric"]
    return out


def mu_water(e_kev) -> np.ndarray:
    """Total linear attenuation coefficient of water (rho = 1 g/cm^3), cm^-1."""
    return mass_coefficients(e_kev)["total"]


class WaterTable:
    """Fast lookup on a fine log-spaced grid, for the transport inner loop."""

    def __init__(self, e_min: float = 20.0, e_max: float = 208.0, n: int = 2000):
        self.le = np.linspace(np.log(e_min), np.log(e_max), n)
        c = mass_coefficients(np.exp(self.le))
        self.log_tot = np.log(c["total"])
        self.f_inc = c["incoherent"] / c["total"]
        self.f_coh = c["coherent"] / c["total"]

    def lookup(self, e_kev: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(mu_total cm^-1, P(incoherent), P(coherent)) for water."""
        l = np.log(e_kev)
        return (np.exp(np.interp(l, self.le, self.log_tot)),
                np.interp(l, self.le, self.f_inc), np.interp(l, self.le, self.f_coh))


# --- Compton kinematics -----------------------------------------------------

def compton_energy(e_kev, cos_theta):
    """Scattered photon energy E' = E / (1 + k (1 - cos theta)), k = E / m_e c^2."""
    k = np.asarray(e_kev) / MEC2_KEV
    return np.asarray(e_kev) / (1.0 + k * (1.0 - np.asarray(cos_theta)))


def kn_dsigma_domega(e_kev, cos_theta):
    """Klein-Nishina differential cross-section per electron, cm^2/sr:
    (r_e^2 / 2) P^2 (P + 1/P - sin^2 theta), P = E'/E."""
    p = compton_energy(e_kev, cos_theta) / np.asarray(e_kev)
    return 0.5 * R_E_CM**2 * p**2 * (p + 1.0 / p - (1.0 - np.asarray(cos_theta) ** 2))


def kn_sigma_total(e_kev):
    """Klein-Nishina total cross-section per electron, cm^2. Closed form; for
    k < 1e-3, where it cancels catastrophically, the series
    sigma_T (1 - 2k + 26 k^2 / 5) with sigma_T = 8 pi r_e^2 / 3."""
    k = np.atleast_1d(np.asarray(e_kev, dtype=float) / MEC2_KEV)
    out = np.empty_like(k)
    small = k < 1e-3
    ks = k[small]
    out[small] = 8 * np.pi / 3 * R_E_CM**2 * (1 - 2 * ks + 26 / 5 * ks**2)
    kb = k[~small]
    a = 1.0 + 2.0 * kb
    out[~small] = 2.0 * np.pi * R_E_CM**2 * (
        (1 + kb) / kb**2 * (2 * (1 + kb) / a - np.log(a) / kb)
        + np.log(a) / (2 * kb)
        - (1 + 3 * kb) / a**2
    )
    return out.reshape(np.shape(e_kev)) if np.ndim(e_kev) else float(out[0])


def kn_pdf_omega(e_kev, cos_theta):
    """Klein-Nishina angular density per steradian (integrates to 1 over 4 pi)."""
    return kn_dsigma_domega(e_kev, cos_theta) / kn_sigma_total(e_kev)


def thomson_pdf_omega(cos_theta):
    """Thomson angular density per steradian: 3 (1 + cos^2) / (16 pi)."""
    return 3.0 * (1.0 + np.asarray(cos_theta) ** 2) / (16.0 * np.pi)


def sample_kn_cos(e_kev: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample cos(theta) from Klein-Nishina by rejection against a uniform
    envelope. The unnormalised density P^2 (P + 1/P - sin^2) has its maximum, 2,
    at cos = 1 for every energy."""
    e_kev = np.asarray(e_kev, dtype=float)
    out = np.empty_like(e_kev)
    todo = np.arange(e_kev.size)
    while todo.size:
        mu = rng.uniform(-1.0, 1.0, todo.size)
        p = 1.0 / (1.0 + e_kev[todo] / MEC2_KEV * (1.0 - mu))
        f = p**2 * (p + 1.0 / p - (1.0 - mu**2))
        ok = rng.random(todo.size) * 2.0 <= f
        out[todo[ok]] = mu[ok]
        todo = todo[~ok]
    return out


def sample_thomson_cos(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample cos(theta) from (1 + cos^2) by rejection (envelope 2)."""
    out = np.empty(n)
    todo = np.arange(n)
    while todo.size:
        mu = rng.uniform(-1.0, 1.0, todo.size)
        ok = rng.random(todo.size) * 2.0 <= 1.0 + mu**2
        out[todo[ok]] = mu[ok]
        todo = todo[~ok]
    return out
