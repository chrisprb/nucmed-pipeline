"""Stage 2 demo: Monte Carlo projections of the IEC-like phantom (with scatter),
reconstructed with the Stage 1 OSEM + AC + RR model.

Three data sets are reconstructed with the *same* reconstructor and the same
calibration (counts per unit activity), so their differences isolate effects:

  A. analytic projector data  - Stage 1 setting, inverse crime
  B. MC, primaries only       - independent forward model, no scatter
  C. MC, primaries + scatter  - photopeak window, scatter not corrected

    uv run python scripts/run_mc_demo.py [--photons 1e6] [--counts 1e6] [--seed 20261105]

Writes results/mc_iec/{metrics.json, projections.npz, figure.png}.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

import nucmed
from nucmed.geometry import Acquisition, Collimator, ImageGrid
from nucmed.mc import MCConfig, simulate, to_sinogram
from nucmed.metrics import background_variability, crc, roi_mean
from nucmed.noise import poisson
from nucmed.phantom import MU_WATER_140KEV, iec_like
from nucmed.projector import build_system_matrix
from nucmed.recon import osem

OUT = Path(__file__).resolve().parents[1] / "results" / "mc_iec"
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}  # categorical slots 1-3
LABELS = {"A": "A. analytic data (inverse crime)",
          "B": "B. MC primaries only",
          "C": "C. MC with scatter, uncorrected"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photons", type=float, default=1e6)
    ap.add_argument("--counts", type=float, default=1e6, help="expected primary photopeak counts")
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--subsets", type=int, default=8)
    ap.add_argument("--reps", type=int, default=10, help="Poisson noise realisations per data set")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    coarse, fine = ImageGrid(80, 0.4), ImageGrid(160, 0.2)
    acq = Acquisition(n_angles=120, n_bins=80, bin_cm=0.4, radius_cm=20.0, collimator=Collimator())
    ph_f, ph_c = iec_like(fine), iec_like(coarse)

    cfg = MCConfig(n_photons=int(args.photons), seed=args.seed)
    res = simulate(ph_f.activity, ph_f.mu / MU_WATER_140KEV, fine, acq, cfg)
    sino = to_sinogram(res, fine)
    t_mc = time.time() - t0

    # Convert MC units (activity x cm^2, window-weighted) to the projector's
    # per-coarse-pixel units, using the photopeak acceptance for primaries.
    acc = float(erf(2 * np.sqrt(np.log(2))))
    unit = acc * coarse.pixel_cm**2
    pp = {k: v / unit for k, v in sino["photopeak"].items()}

    support = ph_c.mu > 0
    sm = build_system_matrix(coarse, acq, mu=ph_c.mu, support=support)
    analytic = sm.forward(ph_c.activity)
    g = args.counts / pp["primary"].sum()  # counts per unit activity, fixed for all data sets

    clean = {"A": analytic * g, "B": pp["primary"] * g, "C": pp["total"] * g}
    rng = np.random.default_rng(args.seed + 1)
    spheres = [k for k in ph_c.rois if k.startswith("sphere_")]

    def summary(vals):
        a = np.array(vals)
        return {"mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=1)), 4)} if len(a) > 1 \
            else {"mean": round(float(a[0]), 4), "sd": None}

    results, images = {}, {}
    for k, lam in clean.items():
        per = {"crc": {s: [] for s in spheres}, "bv": [], "bias": [], "lung": []}
        for rep in range(args.reps):
            x = osem(sm, poisson(lam, rng), n_iter=args.iters, n_subsets=args.subsets) / g
            if rep == 0:
                images[k] = x
            for s, v in crc(x, ph_c).items():
                per["crc"][s].append(v)
            per["bv"].append(background_variability(x, ph_c))
            bkg = roi_mean(x, ph_c.rois["background"])
            per["bias"].append(bkg / ph_c.truth["background"] - 1)
            per["lung"].append(roi_mean(x, ph_c.rois["lung"]) / bkg)
        results[k] = {
            "label": LABELS[k],
            "expected_counts": round(float(lam.sum())),
            "crc": {s: summary(v) for s, v in per["crc"].items()},
            "background_variability": summary(per["bv"]),
            "background_bias": summary(per["bias"]),
            "lung_residual_over_background": summary(per["lung"]),
        }
    # Pre-registered check (DESIGN.md, Stage 2): CRC(C) < CRC(A) for every sphere; bias(C) > 0.
    check = {"crc_C_below_A_all_spheres": all(results["C"]["crc"][s]["mean"] < results["A"]["crc"][s]["mean"]
                                              for s in spheres),
             "background_bias_C_positive": results["C"]["background_bias"]["mean"] > 0}

    windows = {}
    for name, w in sino.items():
        windows[name] = {"range_keV": list(res.windows[name]),
                         "scatter_fraction": round(float(w["scatter"].sum() / w["total"].sum()), 4),
                         "counts_relative_to_photopeak": round(float(w["total"].sum() / sino["photopeak"]["total"].sum()), 4)}
    consistency = float(pp["primary"].sum() / analytic.sum())

    np.savez_compressed(OUT / "projections.npz",
                        **{f"{w}_{p}": sino[w][p] for w in sino for p in sino[w]},
                        spectrum=res.spectrum, spectrum_edges=res.spectrum_edges,
                        to_projector_units=unit)
    meta = {
        "nucmed_version": nucmed.__version__, "python": platform.python_version(), "numpy": np.__version__,
        "seed": args.seed, "photons": cfg.n_photons, "expected_primary_counts": args.counts,
        "mc": {"fwhm_frac": cfg.fwhm_frac, "e_cut_keV": cfg.e_cut_kev, "fine_grid_cm": fine.pixel_cm,
               "collisions_per_photon": round(res.n_collisions / cfg.n_photons, 3)},
        "recon": {"grid_cm": coarse.pixel_cm, "model": "OSEM + AC + RR (Stage 1 system matrix)",
                  "iters": args.iters, "subsets": args.subsets, "noise_realisations": args.reps},
        "acquisition": {"n_angles": acq.n_angles, "radius_cm": acq.radius_cm,
                        "fwhm0_cm": acq.collimator.fwhm0_cm, "slope": acq.collimator.slope},
        "mc_primary_over_analytic_total": round(consistency, 4),
        "note": "2D slice of an axially infinite phantom; Rayleigh without form factor; "
                "metrics are 2D NEMA-style approximations, not NU 2 compliant.",
        "runtime_mc_s": round(t_mc, 1), "runtime_s": round(time.time() - t0, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(
        {"meta": meta, "windows": windows, "preregistered_check": check, "results": results}, indent=2))
    plot(res, sino, ph_c, images, results, spheres)
    print(json.dumps({"windows": windows, "consistency": consistency, "check": check}, indent=1))
    for k, r in results.items():
        print(k, " ".join(f"{s[7:]}={r['crc'][s]['mean']:.2f}±{r['crc'][s]['sd'] or 0:.2f}" for s in spheres),
              f"bias={r['background_bias']['mean']:+.3f} BV={r['background_variability']['mean']:.3f}",
              f"lung={r['lung_residual_over_background']['mean']:.3f}")
    print(f"wrote {OUT} in {meta['runtime_s']} s")


def plot(res, sino, ph, images, results, spheres) -> None:
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})
    fig = plt.figure(figsize=(14, 8), constrained_layout=True, facecolor="#fcfcfb")
    gs = fig.add_gridspec(2, 4)

    # Measured energy spectrum: true-energy tally blurred by the energy resolution.
    ax = fig.add_subplot(gs[0, :2])
    edges = res.spectrum_edges
    ec = 0.5 * (edges[:-1] + edges[1:])
    sig = res.cfg.fwhm_frac * res.cfg.e0_kev / 2.3548 * np.sqrt(np.maximum(ec, 1) / res.cfg.e0_kev)
    def blur(h):
        k = np.exp(-0.5 * ((ec[:, None] - ec[None, :]) / sig[None, :]) ** 2) / (sig[None, :] * np.sqrt(2 * np.pi))
        return k @ h
    prim, scat = blur(res.spectrum[0]), blur(res.spectrum[1])
    norm = (prim + scat).max()
    for name, (lo, hi) in res.windows.items():
        ax.axvspan(lo, hi, color="#f0efec" if name == "photopeak" else "#e6e5e1", zorder=0, lw=0)
        ax.text((lo + hi) / 2, 1.04, name if name == "photopeak" else name[0].upper(),
                ha="center", fontsize=8, color="#52514e")
    ax.plot(ec, (prim + scat) / norm, color="#0b0b0b", lw=2, label="total")
    ax.plot(ec, prim / norm, color=COLORS["A"], lw=2, label="primary")
    ax.plot(ec, scat / norm, color=COLORS["B"], lw=2, label="scattered")
    ax.set_xlim(60, 165); ax.set_ylim(0, 1.1)
    ax.set_xlabel("measured energy (keV)"); ax.set_ylabel("counts (normalised)")
    sf = sino["photopeak"]["scatter"].sum() / sino["photopeak"]["total"].sum()
    ax.set_title(f"Detected spectrum, all angles (scatter = {sf:.0%} of photopeak window)", loc="left")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(sino["photopeak"]["total"], cmap="gray_r", aspect="auto")
    ax.set_title("Photopeak sinogram", loc="left"); ax.set_xlabel("detector bin"); ax.set_ylabel("angle"); ax.grid(False)
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(sino["photopeak"]["scatter"], cmap="gray_r", aspect="auto")
    ax.set_title("…of which scatter", loc="left"); ax.set_xlabel("detector bin"); ax.grid(False)

    vmax = ph.activity.max() * 1.1
    for i, k in enumerate(images):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(images[k], cmap="gray_r", vmin=0, vmax=vmax)
        ax.set_title(LABELS[k], loc="left", fontsize=9); ax.axis("off")

    ax = fig.add_subplot(gs[1, 3])
    diam = [int(s.split("_")[1][:-2]) for s in spheres]
    for k, r in results.items():
        m = np.array([r["crc"][s]["mean"] for s in spheres])
        sd = np.array([r["crc"][s]["sd"] or 0 for s in spheres])
        ax.fill_between(diam, m - sd, m + sd, color=COLORS[k], alpha=0.15, lw=0)
        ax.plot(diam, m, "-o", color=COLORS[k], lw=2, ms=8, mec="#fcfcfb", mew=2, label=k)
    ax.set_xlabel("sphere diameter (mm)"); ax.set_ylabel("contrast recovery")
    ax.set_ylim(0, 1); ax.set_title("CRC, OSEM + AC + RR (mean ± SD)", loc="left"); ax.legend(frameon=False)
    fig.savefig(OUT / "figure.png", dpi=130, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
