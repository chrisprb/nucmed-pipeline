"""2D parallel-hole SPECT system matrix.

The matrix A maps an activity image (N*N,) to a sinogram (n_angles*n_bins,):

    A[(a, b), p] = att(p, a) * blur(b | s_p(a), sigma(d_p(a)))

* att(p, a)  = exp(-integral of mu from pixel p to the detector along the
               acceptance direction u(a)); midpoint rule, step = pixel size,
               bilinear interpolation of the mu map.
* blur(...)  = a Gaussian of width sigma(d) (depth-dependent collimator
               response) integrated over detector bin b. With an ideal
               collimator (sigma = 0) each pixel is linearly interpolated onto
               its two nearest bins.

The matrix is pixel-driven and stored explicitly (scipy CSR), so the
back-projector is its exact transpose: A and A^T are a matched pair, which
MLEM/OSEM rely on. Not modelled (yet): scatter, septal penetration, finite
pixel width in the blur, detector intrinsic response beyond the Gaussian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.ndimage import map_coordinates
from scipy.special import erf

from .geometry import Acquisition, ImageGrid


def attenuation_factors(mu: np.ndarray, grid: ImageGrid, theta: float) -> np.ndarray:
    """exp(-path integral of mu) from each pixel centre to the detector at angle theta."""
    x, y = grid.centres()
    ux, uy = -np.sin(theta), np.cos(theta)
    h = grid.pixel_cm
    n_steps = int(np.ceil(grid.fov_cm * np.sqrt(2) / h)) + 1
    k = (np.arange(n_steps) + 0.5) * h  # midpoints of each step
    xs = x[..., None] + ux * k
    ys = y[..., None] + uy * k
    row, col = grid.to_index(xs, ys)
    vals = map_coordinates(mu, [row.ravel(), col.ravel()], order=1, mode="constant", cval=0.0)
    line_integral = vals.reshape(xs.shape).sum(axis=-1) * h
    return np.exp(-line_integral)


@dataclass
class SystemMatrix:
    A: sparse.csr_matrix
    grid: ImageGrid
    acq: Acquisition
    n_bins: int

    @property
    def sino_shape(self) -> tuple[int, int]:
        return (self.acq.n_angles, self.n_bins)

    def forward(self, image: np.ndarray) -> np.ndarray:
        return (self.A @ image.ravel()).reshape(self.sino_shape)

    def back(self, sino: np.ndarray) -> np.ndarray:
        return (self.A.T @ sino.ravel()).reshape(self.grid.n, self.grid.n)

    def angle_rows(self, angle_idx: np.ndarray) -> np.ndarray:
        """Row indices of A belonging to the given projection angles."""
        return (np.asarray(angle_idx)[:, None] * self.n_bins + np.arange(self.n_bins)).ravel()


def build_system_matrix(
    grid: ImageGrid,
    acq: Acquisition,
    mu: np.ndarray | None = None,
    support: np.ndarray | None = None,
    truncate_sigmas: float = 4.0,
    dtype=np.float32,
) -> SystemMatrix:
    """Build A. ``support`` (bool N x N) limits columns to pixels that can hold
    activity, which cuts memory; pixels outside it get all-zero columns."""
    n_bins, ds = acq.bins(grid)
    x, y = grid.centres()
    pix = np.arange(grid.n * grid.n).reshape(grid.n, grid.n)
    keep = np.ones((grid.n, grid.n), bool) if support is None else support.astype(bool)
    xk, yk, pk = x[keep], y[keep], pix[keep]
    edges0 = -n_bins / 2 * ds  # left edge of bin 0

    rows, cols, vals = [], [], []
    for a, th in enumerate(acq.angles()):
        s = xk * np.cos(th) + yk * np.sin(th)
        t = -xk * np.sin(th) + yk * np.cos(th)
        att = np.ones_like(s) if mu is None else attenuation_factors(mu, grid, th)[keep]
        sigma = acq.collimator.sigma_cm(acq.radius_cm - t)

        if np.all(sigma < 1e-9):  # ideal collimator: linear interpolation onto bins
            f = (s - edges0) / ds - 0.5  # continuous bin index of the pixel centre
            b0 = np.floor(f).astype(int)
            frac = f - b0
            for b, w in ((b0, 1 - frac), (b0 + 1, frac)):
                ok = (b >= 0) & (b < n_bins) & (w > 0)
                rows.append(a * n_bins + b[ok]); cols.append(pk[ok]); vals.append((w * att)[ok])
            continue

        half = int(np.ceil(truncate_sigmas * sigma.max() / ds)) + 1
        centre_bin = np.floor((s - edges0) / ds).astype(int)
        inv = 1.0 / (np.sqrt(2.0) * sigma)
        for off in range(-half, half + 1):
            b = centre_bin + off
            lo = edges0 + b * ds
            w = 0.5 * (erf((lo + ds - s) * inv) - erf((lo - s) * inv))
            ok = (b >= 0) & (b < n_bins) & (w > 1e-6)
            rows.append(a * n_bins + b[ok]); cols.append(pk[ok]); vals.append((w * att)[ok])

    A = sparse.csr_matrix(
        (np.concatenate(vals).astype(dtype), (np.concatenate(rows), np.concatenate(cols))),
        shape=(acq.n_angles * n_bins, grid.n * grid.n),
    )
    return SystemMatrix(A, grid, acq, n_bins)
