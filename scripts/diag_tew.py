"""Exploratory diagnostics for Stage 3 (not pre-registered): noise-free data,
to separate (i) the magnitude and (ii) the spatial shape of the TEW estimate,
and (iii) convergence with an additive term. Writes results/tew/diagnostics.json.

    uv run python scripts/diag_tew.py
"""
import json
from pathlib import Path

import numpy as np

from nucmed.geometry import Acquisition, Collimator, ImageGrid
from nucmed.metrics import crc, roi_mean
from nucmed.phantom import iec_like
from nucmed.projector import build_system_matrix
from nucmed.recon import osem
from nucmed.scatter import smooth_sinogram, tew, windows_widths

ROOT = Path(__file__).resolve().parents[1]
meta = json.loads((ROOT / "results/mc_iec/metrics.json").read_text())
proj = np.load(ROOT / "results/mc_iec/projections.npz")
unit = float(proj["to_projector_units"])
wl, wu, wp = windows_widths({k: v["range_keV"] for k, v in meta["windows"].items()})
g = meta["meta"]["expected_primary_counts"] / (proj["photopeak_primary"] / unit).sum()
P = {k: proj[k] / unit * g for k in proj.files if k.endswith(("_total", "_primary", "_scatter"))}

grid = ImageGrid(80, 0.4)
a = meta["meta"]["acquisition"]
acq = Acquisition(n_angles=a["n_angles"], n_bins=80, bin_cm=0.4, radius_cm=a["radius_cm"],
                  collimator=Collimator(a["fwhm0_cm"], a["slope"]))
ph = iec_like(grid)
sm = build_system_matrix(grid, acq, mu=ph.mu, support=ph.mu > 0)
S = P["photopeak_scatter"]
y = P["photopeak_total"]

def metrics(x):
    x = x / g
    bkg = roi_mean(x, ph.rois["background"])
    c = crc(x, ph)
    return {"bias": round(bkg - 1, 4), "lung": round(roi_mean(x, ph.rois["lung"]) / bkg, 4),
            "crc_10": round(c["sphere_10mm"], 3), "crc_37": round(c["sphere_37mm"], 3)}

tew_l = tew(P["lower_total"], None, wl, wu, wp)
tew_full = tew(P["lower_total"], P["upper_total"], wl, wu, wp)
tew_scat_only = tew(P["lower_scatter"], P["upper_scatter"], wl, wu, wp)
out = {}
for it in (12, 48):
    out[f"scatter-free data, {it} it"] = metrics(osem(sm, P["photopeak_primary"], it, 8))
    out[f"oracle, {it} it"] = metrics(osem(sm, y, it, 8, additive=S))
cases = {
    "TEW (noise-free)": tew_full,
    "TEW rescaled to true total (shape only)": tew_full * S.sum() / tew_full.sum(),
    "TEW from scatter counts only (no primary spill)": tew_scat_only,
    "lower-only (noise-free)": tew_l,
    "true scatter, smoothed like TEW": smooth_sinogram(S, 2, 2),
}
for name, r in cases.items():
    m = metrics(osem(sm, y, 12, 8, additive=r))
    m["est_over_true_total"] = round(float(r.sum() / S.sum()), 4)
    # Where does the estimate put its excess? Sinogram bins that see the 37 mm sphere vs the rest.
    out[name] = m
# Spatial shape: estimate / truth in bins with high primary (sphere-crossing) vs low primary.
prim = P["photopeak_primary"]
hot = prim > np.percentile(prim[prim > 0], 90)
cold = (prim > 0) & (prim < np.percentile(prim[prim > 0], 50))
out["shape: TEW/true in top-10% primary bins vs bottom-50%"] = [
    round(float(tew_full[hot].sum() / S[hot].sum()), 3), round(float(tew_full[cold].sum() / S[cold].sum()), 3)]
out["shape: TEW-from-scatter-only/true, same bins"] = [
    round(float(tew_scat_only[hot].sum() / S[hot].sum()), 3), round(float(tew_scat_only[cold].sum() / S[cold].sum()), 3)]
(ROOT / "results/tew/diagnostics.json").write_text(json.dumps(out, indent=2))
for k, v in out.items():
    print(f"{k:52s} {v}")
