"""Monte Carlo photon transport for 2D SPECT projections, with scatter.

Model (see DESIGN.md, Stage 2):

* Medium: a 2D density map (g/cm^3, water-equivalent) extruded infinitely
  along z. Photons move in 3D; only their (x, y) position is used to look up
  density. With activity uniform in z, tracking photons from one plane and
  scoring them at any z gives the counts in one axial detector slice.
* Interactions (water, NIST XCOM): photoelectric (implicit capture: weight is
  multiplied by the scattering probability), Compton (free-electron
  Klein-Nishina angles, Compton energy), Rayleigh (Thomson angles, no form
  factor). Free paths by Woodcock (delta) tracking.
* Detection by forced detection: at emission and at every real collision,
  the probability of leaving along each detector's collimator axis u_a and
  reaching it unattenuated is added to the tally. Scores are multiplied by
  4 pi, so a primary photon of weight w scores w * exp(-tau), the same
  normalisation as the analytic projector.
* The tally is binned by (primary/scattered, energy window, angle, detector
  position s on a fine grid, source-to-collimator depth). ``to_sinogram``
  then applies the depth-dependent Gaussian collimator response, so the
  collimator model is shared with Stage 1 but attenuation, scatter and
  sampling are not.
* Energy resolution: Gaussian, FWHM = fwhm_frac * E0 * sqrt(E / E0); window
  acceptance is integrated analytically for each scored energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import erf

from .geometry import Acquisition, ImageGrid
from .projector import line_integrals
from .xsections import (E_TC99M_KEV, WaterTable, compton_energy, kn_pdf_omega,
                        sample_kn_cos, sample_thomson_cos, thomson_pdf_omega)

FOUR_PI = 4.0 * np.pi


# --- kinematics -------------------------------------------------------------

def isotropic(n: int, rng: np.random.Generator) -> np.ndarray:
    cz = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    sz = np.sqrt(1 - cz**2)
    return np.stack([sz * np.cos(phi), sz * np.sin(phi), cz], axis=1)


def rotate(d: np.ndarray, cos_t: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """New unit directions at polar angle acos(cos_t) and azimuth phi about d."""
    sin_t = np.sqrt(np.clip(1 - cos_t**2, 0, None))
    ux, uy, uz = d.T
    out = np.empty_like(d)
    near_pole = 1 - uz**2 < 1e-12  # within 1e-6 rad of the z axis
    cp, sp = np.cos(phi), np.sin(phi)
    # General case.
    s = np.sqrt(np.clip(1 - uz**2, 1e-30, None))
    out[:, 0] = ux * cos_t + sin_t * (ux * uz * cp - uy * sp) / s
    out[:, 1] = uy * cos_t + sin_t * (uy * uz * cp + ux * sp) / s
    out[:, 2] = uz * cos_t - sin_t * cp * s
    # Direction (anti)parallel to z.
    sg = np.sign(uz[near_pole])
    out[near_pole, 0] = sin_t[near_pole] * cp[near_pole]
    out[near_pole, 1] = sin_t[near_pole] * sp[near_pole]
    out[near_pole, 2] = sg * cos_t[near_pole]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def woodcock_flight(pos, d, mu_water_e, rho_fn, rho_max, inside_fn, rng):
    """Move photons to their next real collision with Woodcock tracking.

    ``mu_water_e``: water attenuation (cm^-1) at each photon's energy;
    ``rho_fn(pos) -> density``; ``inside_fn(pos) -> bool`` is a convex
    bounding region outside which density is zero. Returns (pos, escaped).
    """
    pos = pos.copy()
    escaped = np.zeros(len(pos), bool)
    todo = np.arange(len(pos))
    mu_max = mu_water_e * rho_max
    while todo.size:
        step = -np.log(rng.random(todo.size)) / mu_max[todo]
        pos[todo] += step[:, None] * d[todo]
        out = ~inside_fn(pos[todo])
        escaped[todo[out]] = True
        todo = todo[~out]
        real = rng.random(todo.size) * rho_max < rho_fn(pos[todo])
        todo = todo[~real]
    return pos, escaped


# --- configuration ----------------------------------------------------------

def tc99m_windows() -> dict[str, tuple[float, float]]:
    """20 % photopeak window plus TEW side windows (3 keV each, abutting)."""
    e0 = E_TC99M_KEV
    lo, hi = 0.9 * e0, 1.1 * e0
    return {"photopeak": (lo, hi), "lower": (lo - 3.0, lo), "upper": (hi, hi + 3.0)}


@dataclass
class MCConfig:
    n_photons: int = 1_000_000
    seed: int = 20261105
    e0_kev: float = E_TC99M_KEV
    fwhm_frac: float = 0.10
    windows: dict[str, tuple[float, float]] = field(default_factory=tc99m_windows)
    e_cut_kev: float = 90.0  # photons below this cannot reach any window
    s_fine_cm: float = 0.1
    depth_bin_cm: float = 0.5
    los_step_cm: float = 0.1  # step of the line integrals to the detector
    chunk: int = 20_000
    roulette_w: float = 0.05  # relative weight below which Russian roulette applies
    scatter: bool = True  # False: primaries only (for validation)
    spectrum_bin_kev: float = 0.5


@dataclass
class MCResult:
    tally: np.ndarray  # (2, n_win, n_angles, n_sfine, n_depth); [0]=primary, [1]=scatter
    spectrum: np.ndarray  # (2, n_ebins): true-energy weight summed over all detectors
    spectrum_edges: np.ndarray
    windows: dict[str, tuple[float, float]]
    s_edges0: float
    depth_edges0: float
    cfg: MCConfig
    acq: Acquisition
    n_collisions: int


# --- main simulation --------------------------------------------------------

def simulate(activity: np.ndarray, density: np.ndarray, grid: ImageGrid,
             acq: Acquisition, cfg: MCConfig = MCConfig()) -> MCResult:
    """Simulate SPECT projections of ``activity`` (concentration per pixel) in
    a water-equivalent ``density`` map on ``grid``. The total photon weight is
    the integral of activity over area, sum(activity) * pixel_cm^2."""
    rng = np.random.default_rng(cfg.seed)
    table = WaterTable()
    angles = acq.angles()
    ca, sa = np.cos(angles), np.sin(angles)
    ux, uy = -sa, ca  # collimator axes
    win_names = list(cfg.windows)
    w_lo = np.array([cfg.windows[k][0] for k in win_names])
    w_hi = np.array([cfg.windows[k][1] for k in win_names])
    sig_e0 = cfg.fwhm_frac * cfg.e0_kev / (2 * np.sqrt(2 * np.log(2)))

    # Line integrals of density from every pixel centre to each detector.
    D = np.stack([line_integrals(density, grid, th, step_cm=cfg.los_step_cm, order=0)
                  for th in angles]).astype(np.float32)

    r_obj = np.sqrt(2) * grid.fov_cm / 2
    yy, xx = grid.centres()[1], grid.centres()[0]
    r_occ = np.sqrt(xx**2 + yy**2)[density > 0].max() + grid.pixel_cm  # tight bound
    r_obj = min(r_obj, r_occ)
    rho_max = float(density.max())

    n_bins, bin_cm = acq.bins(grid)
    s_half = n_bins * bin_cm / 2
    n_sf = int(round(2 * s_half / cfg.s_fine_cm))
    d_min = acq.radius_cm - r_obj
    n_d = int(np.ceil(2 * r_obj / cfg.depth_bin_cm)) + 1
    n_win, n_ang = len(win_names), len(angles)
    block = n_ang * n_sf * n_d
    tally = np.zeros((2, n_win, block))
    e_edges = np.arange(0, cfg.e0_kev + 20 + cfg.spectrum_bin_kev, cfg.spectrum_bin_kev)
    spectrum = np.zeros((2, len(e_edges) - 1))
    ang_idx = np.arange(n_ang)

    def rho_fn(p):
        r, c = grid.to_index(p[:, 0], p[:, 1])
        r = np.rint(r).astype(int); c = np.rint(c).astype(int)
        ok = (r >= 0) & (r < grid.n) & (c >= 0) & (c < grid.n)
        out = np.zeros(len(p))
        out[ok] = density[r[ok], c[ok]]
        return out

    def inside_fn(p):
        return p[:, 0] ** 2 + p[:, 1] ** 2 < r_obj**2

    def score(order, x, y, e_out, wgt):
        """Add contributions; all arrays are (M, n_ang) except x, y (M,)."""
        s = x[:, None] * ca + y[:, None] * sa
        t = -x[:, None] * sa + y[:, None] * ca
        sf = np.floor((s + s_half) / cfg.s_fine_cm).astype(np.int64)
        db = np.floor((acq.radius_cm - t - d_min) / cfg.depth_bin_cm).astype(np.int64)
        ok = (sf >= 0) & (sf < n_sf) & (db >= 0) & (db < n_d) & (wgt > 0)
        sig = sig_e0 * np.sqrt(e_out / cfg.e0_kev)
        inv = 1 / (np.sqrt(2) * sig)
        spectrum[order] += np.bincount(np.digitize(e_out[ok], e_edges) - 1, wgt[ok],
                                       minlength=len(e_edges))[: len(e_edges) - 1]
        a = np.broadcast_to(ang_idx, s.shape)
        base = ((a[ok] * n_sf + sf[ok]) * n_d + db[ok])
        for k in range(n_win):
            acc = 0.5 * (erf((w_hi[k] - e_out[ok]) * inv[ok]) - erf((w_lo[k] - e_out[ok]) * inv[ok]))
            tally[order, k] += np.bincount(base, wgt[ok] * acc, minlength=block)

    def tau_to_detectors(x, y):
        r, c = grid.to_index(x, y)
        M = len(x)
        coords = [np.broadcast_to(ang_idx, (M, n_ang)).ravel().astype(float),
                  np.repeat(r, n_ang), np.repeat(c, n_ang)]
        return map_coordinates(D, coords, order=1, mode="nearest").reshape(M, n_ang)

    # Emission sampling: pixel by activity, then uniform within the pixel.
    p = activity.ravel().clip(min=0)
    total_weight = p.sum() * grid.pixel_cm**2
    p = p / p.sum()
    xc, yc = (g.ravel() for g in grid.centres())
    w0 = total_weight / cfg.n_photons
    n_coll = 0

    for start in range(0, cfg.n_photons, cfg.chunk):
        m = min(cfg.chunk, cfg.n_photons - start)
        pix = rng.choice(p.size, size=m, p=p)
        x = xc[pix] + (rng.random(m) - 0.5) * grid.pixel_cm
        y = yc[pix] + (rng.random(m) - 0.5) * grid.pixel_cm
        e = np.full(m, cfg.e0_kev)
        w = np.full(m, w0)

        mu0 = table.lookup(e[:1])[0][0]
        score(0, x, y, np.full((m, n_ang), cfg.e0_kev),
              w[:, None] * np.exp(-mu0 * tau_to_detectors(x, y)))
        if not cfg.scatter:
            continue

        pos = np.stack([x, y, np.zeros(m)], axis=1)
        d = isotropic(m, rng)
        while len(pos):
            mu_e, f_inc, f_coh = table.lookup(e)
            pos, esc = woodcock_flight(pos, d, mu_e, rho_fn, rho_max, inside_fn, rng)
            keep = ~esc
            pos, d, e, w = pos[keep], d[keep], e[keep], w[keep]
            mu_e, f_inc, f_coh = mu_e[keep], f_inc[keep], f_coh[keep]
            if not len(pos):
                break
            n_coll += len(pos)

            # Forced detection from this collision site.
            tau = tau_to_detectors(pos[:, 0], pos[:, 1])
            cos_a = d[:, 0:1] * ux + d[:, 1:2] * uy  # (M, n_ang)
            e_c = compton_energy(e[:, None], cos_a)
            mu_c = table.lookup(e_c.ravel())[0].reshape(e_c.shape)
            w_c = w[:, None] * f_inc[:, None] * FOUR_PI * kn_pdf_omega(e[:, None], cos_a) * np.exp(-mu_c * tau)
            w_r = w[:, None] * f_coh[:, None] * FOUR_PI * thomson_pdf_omega(cos_a) * np.exp(-mu_e[:, None] * tau)
            score(1, pos[:, 0], pos[:, 1], e_c, w_c)
            score(1, pos[:, 0], pos[:, 1], np.broadcast_to(e[:, None], cos_a.shape), w_r)

            # Implicit capture, then choose Compton or Rayleigh and scatter.
            p_scat = f_inc + f_coh
            w = w * p_scat
            is_c = rng.random(len(e)) * p_scat < f_inc
            cos_t = np.empty(len(e))
            cos_t[is_c] = sample_kn_cos(e[is_c], rng)
            cos_t[~is_c] = sample_thomson_cos(int((~is_c).sum()), rng)
            e = np.where(is_c, compton_energy(e, cos_t), e)
            d = rotate(d, cos_t, rng.uniform(0, 2 * np.pi, len(e)))

            # Energy cut and Russian roulette.
            alive = e >= cfg.e_cut_kev
            low = alive & (w < cfg.roulette_w * w0)
            survive = rng.random(len(w)) < 0.5
            w = np.where(low & survive, 2 * w, w)
            alive &= ~(low & ~survive)
            pos, d, e, w = pos[alive], d[alive], e[alive], w[alive]

    return MCResult(tally.reshape(2, n_win, n_ang, n_sf, n_d), spectrum, e_edges,
                    dict(cfg.windows), -s_half, d_min, cfg, acq, n_coll)


def to_sinogram(res: MCResult, grid: ImageGrid) -> dict[str, dict[str, np.ndarray]]:
    """Apply the depth-dependent collimator blur and rebin onto detector bins.
    Returns {window: {"primary", "scatter", "total"}} sinograms (n_angles, n_bins)."""
    acq, cfg = res.acq, res.cfg
    n_bins, bin_cm = acq.bins(grid)
    _, _, n_ang, n_sf, n_d = res.tally.shape
    s_c = res.s_edges0 + (np.arange(n_sf) + 0.5) * cfg.s_fine_cm
    lo = res.s_edges0 + np.arange(n_bins) * bin_cm
    kernels = np.empty((n_d, n_sf, n_bins))
    for j in range(n_d):
        dc = res.depth_edges0 + (j + 0.5) * cfg.depth_bin_cm
        sig = float(acq.collimator.sigma_cm(np.asarray(dc)))
        if sig < 1e-9:
            K = np.zeros((n_sf, n_bins))
            b = np.clip(np.floor((s_c - res.s_edges0) / bin_cm).astype(int), 0, n_bins - 1)
            K[np.arange(n_sf), b] = 1.0
        else:
            inv = 1 / (np.sqrt(2) * sig)
            K = 0.5 * (erf((lo[None, :] + bin_cm - s_c[:, None]) * inv)
                       - erf((lo[None, :] - s_c[:, None]) * inv))
        kernels[j] = K
    out = {}
    for k, name in enumerate(res.windows):
        parts = {}
        for o, label in ((0, "primary"), (1, "scatter")):
            parts[label] = np.einsum("asd,dsb->ab", res.tally[o, k], kernels)
        parts["total"] = parts["primary"] + parts["scatter"]
        out[name] = parts
    return out
