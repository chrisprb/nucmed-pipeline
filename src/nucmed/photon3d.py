"""3D photon transport in homogeneous water for absorbed fractions and point kernels.

Analog Monte Carlo with the same interaction physics as ``mc`` (XCOM water
cross-sections, Klein-Nishina Compton angles, Thomson Rayleigh angles), but
scoring *energy deposited* instead of detector counts. Kerma approximation:
the electron set in motion deposits its energy at the interaction point
(photoelectric: all of E; Compton: E - E'). Photons below ``e_min_kev`` deposit
their energy locally. No fluorescence (water). Density 1 g/cm^3.
"""

from __future__ import annotations

import numpy as np

from .mc import isotropic, rotate
from .xsections import WaterTable, compton_energy, sample_kn_cos, sample_thomson_cos

_TABLES: dict[str, WaterTable] = {}


def table(material: str = "water") -> WaterTable:
    if material not in _TABLES:
        _TABLES[material] = WaterTable(e_min=1.0, e_max=1000.0, n=4000, material=material)
    return _TABLES[material]


def _sample_lines(lines: np.ndarray, n: int, rng) -> np.ndarray:
    lines = np.atleast_2d(lines)
    p = lines[:, 1] / lines[:, 1].sum()
    return lines[rng.choice(len(lines), size=n, p=p), 0]


def transport(pos: np.ndarray, e: np.ndarray, rng: np.random.Generator,
              exit_distance, score, e_min_kev: float = 1.0, material: str = "water") -> None:
    """Track photons from ``pos`` with energies ``e`` until absorbed or escaped.

    ``exit_distance(pos, d)``: distance to the region boundary (np.inf for an
    infinite medium). ``score(pos, energy_deposited, idx)`` is called at every
    interaction, with ``idx`` the index of each photon in the input arrays.
    """
    tab = table(material)
    d = isotropic(len(e), rng)
    idx = np.arange(len(e))
    while len(e):
        mu, f_inc, f_coh = tab.lookup(e)
        step = -np.log(rng.random(len(e))) / mu
        stay = step < exit_distance(pos, d)
        pos, d, e, step, idx = pos[stay], d[stay], e[stay], step[stay], idx[stay]
        f_inc, f_coh = f_inc[stay], f_coh[stay]
        pos = pos + step[:, None] * d
        u = rng.random(len(e))
        photo = u >= f_inc + f_coh
        comp = u < f_inc
        ray = ~photo & ~comp
        cos_t = np.ones(len(e))
        cos_t[comp] = sample_kn_cos(e[comp], rng)
        cos_t[ray] = sample_thomson_cos(int(ray.sum()), rng)
        e_new = np.where(comp, compton_energy(e, cos_t), e)
        low = ~photo & (e_new < e_min_kev)  # below cut-off: deposit everything left
        score(pos, np.where(photo | low, e, e - e_new), idx)
        alive = ~photo & ~low
        pos, e, idx = pos[alive], e_new[alive], idx[alive]
        d = rotate(d[alive], cos_t[alive], rng.uniform(0, 2 * np.pi, int(alive.sum())))


def sphere_exit_distance(radius: float):
    def f(p, d):
        b = (p * d).sum(axis=1)
        c = (p * p).sum(axis=1) - radius**2
        return -b + np.sqrt(np.maximum(b * b - c, 0.0))
    return f


def uniform_in_sphere(n: int, radius: float, rng) -> np.ndarray:
    r = radius * rng.random(n) ** (1 / 3)
    return isotropic(n, rng) * r[:, None]


def absorbed_fraction_sphere(lines, radius_cm: float, n: int = 200_000, seed: int = 0,
                             chunk: int = 100_000, surround: str = "water",
                             kill_margin_cm: float = 40.0, material: str = "water") -> tuple[float, float]:
    """Fraction of emitted photon energy absorbed in a water sphere with a
    uniform source. ``lines``: energy (keV) or array of (energy, yield).

    ``surround="water"``: the sphere sits in an unbounded water medium (the
    geometry of Stabin & Konijnenberg 2000), so photons that leave can scatter
    back in; only deposits inside the sphere count. Photons are dropped once
    they are ``kill_margin_cm`` beyond the surface (return probability is
    negligible). ``surround="vacuum"``: an isolated sphere.
    Returns (absorbed fraction, standard error).
    """
    rng = np.random.default_rng(seed)
    lines = np.array([[float(lines), 1.0]]) if np.isscalar(lines) else np.asarray(lines, float)
    if surround == "water":
        ex = sphere_exit_distance(radius_cm + kill_margin_cm)
    elif surround == "vacuum":
        ex = sphere_exit_distance(radius_cm)
    else:
        raise ValueError(surround)
    r2 = radius_cm**2
    deps, e0s = [], []
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        e0 = _sample_lines(lines, m, rng)
        dep = np.zeros(m)

        def score(p, de, idx, dep=dep):
            inside = (p * p).sum(axis=1) <= r2
            np.add.at(dep, idx[inside], de[inside])

        transport(uniform_in_sphere(m, radius_cm, rng), e0.copy(), rng, ex, score, material=material)
        deps.append(dep); e0s.append(e0)
    dep, e0 = np.concatenate(deps), np.concatenate(e0s)
    af = dep.sum() / e0.sum()
    se = np.sqrt(np.var(dep - af * e0, ddof=1) * len(e0)) / e0.sum()  # ratio estimator
    return float(af), float(se)


def point_kernel(lines, r_edges_cm: np.ndarray, n: int = 500_000, seed: int = 0,
                 chunk: int = 100_000) -> np.ndarray:
    """Fraction of emitted photon energy deposited in each spherical shell
    [r_edges[i], r_edges[i+1]) around a point source in infinite water."""
    rng = np.random.default_rng(seed)
    lines = np.array([[float(lines), 1.0]]) if np.isscalar(lines) else np.asarray(lines, float)
    hist = np.zeros(len(r_edges_cm) - 1)
    e_total = 0.0

    def score(p, de, idx):
        r = np.linalg.norm(p, axis=1)
        h, _ = np.histogram(r, r_edges_cm, weights=de)
        hist[:] += h

    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        e0 = _sample_lines(lines, m, rng)
        e_total += e0.sum()
        transport(np.zeros((m, 3)), e0, rng, lambda p, d: np.full(len(p), np.inf), score)
    return hist / e_total
