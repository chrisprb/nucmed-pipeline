"""Lu-177 decay data (ENSDF via the IAEA LiveChart API; see data/lu177_*.csv).

Energies in keV, yields per decay (not per 100 decays).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_DATA = Path(__file__).with_name("data")
KEV_TO_J = 1.602176634e-16  # exact (SI 2019): 1 eV = 1.602176634e-19 J


def _rows(name: str) -> list[dict[str, str]]:
    with open(_DATA / name) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


@dataclass(frozen=True)
class Radionuclide:
    name: str
    half_life_s: float
    beta_branches: np.ndarray  # (n, 3): mean energy keV, yield, max energy keV
    beta_spectrum: tuple[np.ndarray, np.ndarray]  # energy keV, dN/dE per keV (as tabulated)
    electron_lines: np.ndarray  # (n, 2): energy keV, yield (conversion + Auger electrons)
    photon_lines: np.ndarray  # (n, 2): energy keV, yield (gamma + X-ray)

    @property
    def decay_constant_s(self) -> float:
        return np.log(2) / self.half_life_s

    def beta_spectrum_normalised(self) -> tuple[np.ndarray, np.ndarray]:
        """Spectrum scaled to integrate to the total beta yield per decay."""
        e, p = self.beta_spectrum
        return e, p * self.beta_branches[:, 1].sum() / np.trapezoid(p, e)

    def energy_per_decay_kev(self) -> dict[str, float]:
        """Mean energy emitted per decay, by radiation type (Delta / k)."""
        b = self.beta_branches
        return {
            "beta": float((b[:, 0] * b[:, 1]).sum()),
            "electron_lines": float((self.electron_lines[:, 0] * self.electron_lines[:, 1]).sum()),
            "photons": float((self.photon_lines[:, 0] * self.photon_lines[:, 1]).sum()),
        }


@lru_cache(maxsize=1)
def lu177() -> Radionuclide:
    rows = _rows("lu177_emissions.csv")
    hl = float((_DATA / "lu177_half_life.txt").read_text().strip().splitlines()[-1])

    def pick(*kinds):
        out = [(float(r["energy_keV"]), float(r["intensity_pct"]) / 100.0)
               for r in rows if r["kind"] in kinds and r["intensity_pct"]]
        return np.array(out)

    beta = np.array([(float(r["energy_keV"]), float(r["intensity_pct"]) / 100.0, float(r["max_energy_keV"]))
                     for r in rows if r["kind"] == "beta_minus_mean"])
    spec = np.array([(float(r["energy_keV"]), float(r["dN_dE"])) for r in _rows("lu177_beta_spectrum.csv")])
    return Radionuclide("Lu-177", hl, beta, (spec[:, 0], spec[:, 1]),
                        pick("ce", "auger"), pick("gamma", "xray"))
