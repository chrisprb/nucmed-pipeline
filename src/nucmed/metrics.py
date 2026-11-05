"""Image quality metrics, NEMA NU 2-style (simplified to 2D).

* Contrast recovery coefficient for a hot sphere j:
      CRC_j = (C_j / C_B - 1) / (a_H / a_B - 1)
  where C_j is the mean in the sphere ROI, C_B the mean background, and
  a_H / a_B the true sphere-to-background ratio.
* Background variability: SD / mean across the background ROIs, here
  approximated by the pixel-level coefficient of variation inside the
  background ROI union.

These follow the structure of NEMA NU 2 [verify] but are not a compliant
implementation (2D only, one slice, simplified ROI placement).
"""

from __future__ import annotations

import numpy as np

from .phantom import Phantom


def roi_mean(image: np.ndarray, mask: np.ndarray) -> float:
    """Mask-weighted mean (masks may be fractional)."""
    return float(np.sum(image * mask) / np.sum(mask))


def core_mask(mask: np.ndarray, threshold: float = 0.999) -> np.ndarray:
    """Pixels essentially fully inside a region (drops partial-volume edge pixels)."""
    return (mask >= threshold).astype(float)


def crc(image: np.ndarray, phantom: Phantom) -> dict[str, float]:
    bkg = roi_mean(image, phantom.rois["background"])
    ratio_true = None
    out = {}
    for name, m in phantom.rois.items():
        if not name.startswith("sphere_"):
            continue
        ratio_true = phantom.truth[name] / phantom.truth["background"]
        out[name] = (roi_mean(image, m) / bkg - 1.0) / (ratio_true - 1.0)
    return out


def background_variability(image: np.ndarray, phantom: Phantom) -> float:
    m = core_mask(phantom.rois["background"]) > 0
    v = image[m]
    return float(v.std(ddof=1) / v.mean())


def nrmse(image: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> float:
    m = np.ones_like(truth, bool) if mask is None else mask > 0
    return float(np.sqrt(np.mean((image[m] - truth[m]) ** 2)) / np.mean(truth[m]))
