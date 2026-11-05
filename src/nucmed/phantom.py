"""Digital phantoms: activity map, attenuation map and ROI masks.

All shapes are rendered with sub-pixel supersampling so that partial-volume
effects at edges are represented.

The IEC-like phantom is a *2D slice through the sphere centres* of a phantom
modelled on the NEMA NU 2 / IEC 61675-1 body phantom. Dimensions below are
approximations for testing reconstruction methods, not a certified replica;
values marked [verify] must be checked against the standard before any
quantitative claim about the real phantom is made.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import ImageGrid

# Linear attenuation coefficients at 140 keV (Tc-99m), cm^-1.
# Water ~0.15 cm^-1 is the commonly used narrow-beam value [verify against NIST XCOM].
MU_WATER_140KEV = 0.154  # [verify]
MU_LUNG_140KEV = 0.3 * MU_WATER_140KEV  # lung-equivalent insert, rough [verify]


def _supersampled(grid: ImageGrid, inside, ss: int = 4) -> np.ndarray:
    """Fraction of each pixel covered by the region ``inside(x, y) -> bool``."""
    fine = ImageGrid(grid.n * ss, grid.pixel_cm / ss)
    x, y = fine.centres()
    m = inside(x, y).astype(float)
    return m.reshape(grid.n, ss, grid.n, ss).mean(axis=(1, 3))


def disk_mask(grid: ImageGrid, cx: float, cy: float, r: float, ss: int = 4) -> np.ndarray:
    return _supersampled(grid, lambda x, y: (x - cx) ** 2 + (y - cy) ** 2 <= r**2, ss)


def ellipse_mask(grid: ImageGrid, cx: float, cy: float, a: float, b: float, ss: int = 4) -> np.ndarray:
    return _supersampled(grid, lambda x, y: ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1.0, ss)


@dataclass
class Phantom:
    activity: np.ndarray  # arbitrary units per pixel (relative concentration)
    mu: np.ndarray  # cm^-1
    grid: ImageGrid
    rois: dict[str, np.ndarray] = field(default_factory=dict)  # fractional masks
    truth: dict[str, float] = field(default_factory=dict)  # true concentrations


def uniform_disk(grid: ImageGrid, radius_cm: float = 10.0, activity: float = 1.0,
                 mu: float = MU_WATER_140KEV) -> Phantom:
    m = disk_mask(grid, 0.0, 0.0, radius_cm)
    return Phantom(activity * m, mu * m, grid, rois={"disk": m}, truth={"disk": activity})


def point_source(grid: ImageGrid, x_cm: float, y_cm: float, mu_map: np.ndarray | None = None) -> Phantom:
    """Unit activity in the single pixel nearest (x_cm, y_cm)."""
    row, col = grid.to_index(np.asarray(x_cm), np.asarray(y_cm))
    a = np.zeros((grid.n, grid.n))
    a[int(round(float(row))), int(round(float(col)))] = 1.0
    mu = np.zeros_like(a) if mu_map is None else mu_map
    return Phantom(a, mu, grid)


@dataclass(frozen=True)
class IECConfig:
    # Elliptical approximation of the body cross-section, cm (real phantom is
    # a half-circle on a rounded rectangle, roughly 30 x 23 cm) [verify].
    body_a_cm: float = 15.0
    body_b_cm: float = 11.5
    # Sphere inner diameters, cm (NEMA NU 2 set) [verify].
    sphere_d_cm: tuple[float, ...] = (1.0, 1.3, 1.7, 2.2, 2.8, 3.7)
    ring_radius_cm: float = 5.72  # sphere centres on this circle [verify]
    lung_d_cm: float = 5.0  # central lung insert diameter [verify]
    sphere_to_background: float = 4.0
    background: float = 1.0
    mu_body: float = MU_WATER_140KEV
    mu_lung: float = MU_LUNG_140KEV


def iec_like(grid: ImageGrid, cfg: IECConfig = IECConfig()) -> Phantom:
    body = ellipse_mask(grid, 0, 0, cfg.body_a_cm, cfg.body_b_cm)
    lung = disk_mask(grid, 0, 0, cfg.lung_d_cm / 2)
    hot = cfg.background * cfg.sphere_to_background

    activity = cfg.background * (body - lung)
    mu = cfg.mu_body * (body - lung) + cfg.mu_lung * lung
    rois: dict[str, np.ndarray] = {"lung": lung}
    truth: dict[str, float] = {"background": cfg.background, "lung": 0.0}

    # Spheres placed every 60 degrees, smallest first, counter-clockwise from +x.
    for k, d in enumerate(cfg.sphere_d_cm):
        ang = np.deg2rad(60.0 * k)
        cx, cy = cfg.ring_radius_cm * np.cos(ang), cfg.ring_radius_cm * np.sin(ang)
        m = disk_mask(grid, cx, cy, d / 2)
        activity = activity * (1 - m) + hot * m
        name = f"sphere_{int(round(d * 10))}mm"
        rois[name] = m
        truth[name] = hot

    rois["background"] = background_rois(grid, cfg)
    return Phantom(activity, mu, grid, rois=rois, truth=truth)


def background_rois(grid: ImageGrid, cfg: IECConfig, d_cm: float = 3.0) -> np.ndarray:
    """Union of 12 circular background ROIs, NEMA-style (simplified placement):
    6 between the spheres (r = 9.3 cm) and 6 radially outside them (r = 10 cm),
    kept at least one pixel clear of every sphere (checked by a test)."""
    out = np.zeros((grid.n, grid.n))
    for k in range(6):
        for r, ang in ((9.3, 60.0 * k + 30.0), (10.0, 60.0 * k)):
            a = np.deg2rad(ang)
            out = np.maximum(out, disk_mask(grid, r * np.cos(a), r * np.sin(a), d_cm / 2))
    return out
