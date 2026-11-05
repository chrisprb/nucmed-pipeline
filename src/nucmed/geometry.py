"""Image and acquisition geometry.

Conventions (2D, one transaxial slice):

* The image is N x N pixels of size ``pixel_cm``, centred on the rotation axis.
  ``image[i, j]`` has centre x = (j - (N-1)/2) * dx, y = ((N-1)/2 - i) * dx,
  so row 0 is the top of the slice (anterior) and x increases to the right.
* At gantry angle theta the detector coordinate of a point is
      s = x cos(theta) + y sin(theta)
  and photons accepted by a parallel-hole collimator travel along
      u = (-sin(theta), cos(theta))
  towards the detector, which sits at  t = radius_cm  where
      t = -x sin(theta) + y cos(theta).
  The source-to-collimator distance is therefore  d = radius_cm - t.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ImageGrid:
    # Default: 80 x 80 at 4 mm (32 cm FOV), close to clinical SPECT sampling.
    n: int = 80
    pixel_cm: float = 0.4

    @property
    def fov_cm(self) -> float:
        return self.n * self.pixel_cm

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (x, y) pixel-centre coordinates in cm, each shaped (n, n)."""
        c = (np.arange(self.n) - (self.n - 1) / 2) * self.pixel_cm
        x = np.broadcast_to(c[None, :], (self.n, self.n))
        y = np.broadcast_to(-c[:, None], (self.n, self.n))
        return x, y

    def to_index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Continuous (row, col) indices for physical coordinates (for interpolation)."""
        col = x / self.pixel_cm + (self.n - 1) / 2
        row = (self.n - 1) / 2 - y / self.pixel_cm
        return row, col


@dataclass(frozen=True)
class Collimator:
    """Gaussian geometric resolution model: FWHM(d) = fwhm0_cm + slope * d.

    Default values are representative of a low-energy high-resolution (LEHR)
    collimator at 140 keV (about 7-8 mm FWHM at 10 cm) and are a modelling
    choice, not a specification of a particular product. Set both to 0 for an
    ideal (line-integral) collimator.
    """

    fwhm0_cm: float = 0.25
    slope: float = 0.05

    def sigma_cm(self, d_cm: np.ndarray) -> np.ndarray:
        fwhm = self.fwhm0_cm + self.slope * np.clip(d_cm, 0.0, None)
        return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass(frozen=True)
class Acquisition:
    n_angles: int = 120
    arc_deg: float = 360.0
    n_bins: int | None = None  # defaults to grid.n
    bin_cm: float | None = None  # defaults to grid.pixel_cm
    radius_cm: float = 25.0
    collimator: Collimator = field(default_factory=Collimator)

    def angles(self) -> np.ndarray:
        return np.deg2rad(np.arange(self.n_angles) * self.arc_deg / self.n_angles)

    def bins(self, grid: ImageGrid) -> tuple[int, float]:
        n_bins = self.n_bins if self.n_bins is not None else grid.n
        bin_cm = self.bin_cm if self.bin_cm is not None else grid.pixel_cm
        return n_bins, bin_cm
