import numpy as np
import pytest

from nucmed.geometry import ImageGrid
from nucmed.metrics import crc, nrmse
from nucmed.phantom import IECConfig, Phantom, disk_mask, ellipse_mask, iec_like


def test_disk_area_partial_volume():
    """Supersampled disk area matches pi r^2."""
    g = ImageGrid(n=64, pixel_cm=0.5)
    m = disk_mask(g, 1.3, -0.7, 5.0, ss=8)
    assert m.sum() * g.pixel_cm**2 == pytest.approx(np.pi * 25.0, rel=0.005)


def test_iec_rois_are_clean():
    """Background ROIs lie inside the body and touch neither spheres nor lung."""
    g = ImageGrid()
    cfg = IECConfig()
    ph = iec_like(g, cfg)
    body = ellipse_mask(g, 0, 0, cfg.body_a_cm, cfg.body_b_cm)
    bkg = ph.rois["background"] > 0
    assert np.all(body[bkg] > 0.99)
    for name, m in ph.rois.items():
        if name != "background":
            assert not np.any((m > 0) & bkg), name
    assert sum(k.startswith("sphere_") for k in ph.rois) == 6


def test_crc_is_one_for_perfect_image():
    g = ImageGrid(n=32, pixel_cm=1.0)
    s = (disk_mask(g, 5, 0, 3) > 0.5).astype(float)
    b = (disk_mask(g, -6, 0, 3) > 0.5).astype(float)
    img = 1.0 + 3.0 * s
    ph = Phantom(img, img * 0, g, rois={"sphere_60mm": s, "background": b},
                 truth={"sphere_60mm": 4.0, "background": 1.0})
    assert crc(img, ph)["sphere_60mm"] == pytest.approx(1.0)
    assert crc(1.0 + 1.5 * s, ph)["sphere_60mm"] == pytest.approx(0.5)
    assert nrmse(img, img) == 0.0
