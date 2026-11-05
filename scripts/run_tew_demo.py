"""Stage 3 demo: triple-energy-window (TEW) scatter correction on the Stage 2
Monte Carlo projections, reconstructed with OSEM + AC + RR.

Conditions (pre-registered in DESIGN.md, Stage 3 plan):
  0  no scatter correction
  1  TEW, smoothed (primary comparison)
  2  TEW, unsmoothed
  3  lower window only, smoothed
  4  oracle: noise-free true scatter as the additive term

    uv run python scripts/run_mc_demo.py      # first, writes results/mc_iec/
    uv run python scripts/run_tew_demo.py [--reps 10] [--seed 20261106]

Writes results/tew/{metrics.json, figure.png}.
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

import nucmed
from nucmed.geometry import Acquisition, Collimator, ImageGrid
from nucmed.metrics import background_variability, crc, roi_mean
from nucmed.noise import poisson
from nucmed.phantom import iec_like
from nucmed.projector import build_system_matrix
from nucmed.recon import osem
from nucmed.scatter import smooth_sinogram, tew, windows_widths

ROOT = Path(__file__).resolve().parents[1]
MC = ROOT / "results" / "mc_iec"
OUT = ROOT / "results" / "tew"
SIGMA_BINS, SIGMA_ANGLES = 2.0, 2.0  # fixed in DESIGN.md before running
LABELS = {0: "0. no scatter correction", 1: "1. TEW, smoothed", 2: "2. TEW, unsmoothed",
          3: "3. lower window only, smoothed", 4: "4. oracle (true scatter)"}
COLORS = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a", 3: "#eda100", 4: "#e87ba4"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--subsets", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    mc_meta = json.loads((MC / "metrics.json").read_text())
    proj = np.load(MC / "projections.npz")
    windows = {k: tuple(v["range_keV"]) for k, v in mc_meta["windows"].items()}
    wl, wu, wp = windows_widths(windows)
    unit = float(proj["to_projector_units"])

    coarse = ImageGrid(80, 0.4)
    a = mc_meta["meta"]["acquisition"]
    acq = Acquisition(n_angles=a["n_angles"], n_bins=80, bin_cm=0.4, radius_cm=a["radius_cm"],
                      collimator=Collimator(a["fwhm0_cm"], a["slope"]))
    ph = iec_like(coarse)
    sm = build_system_matrix(coarse, acq, mu=ph.mu, support=ph.mu > 0)

    # Same calibration as Stage 2: counts per unit activity.
    counts = mc_meta["meta"]["expected_primary_counts"]
    g = counts / (proj["photopeak_primary"] / unit).sum()
    lam = {w: proj[f"{w}_total"] / unit * g for w in ("photopeak", "lower", "upper")}
    true_scatter = proj["photopeak_scatter"] / unit * g

    spheres = [k for k in ph.rois if k.startswith("sphere_")]
    rng = np.random.default_rng(args.seed)
    per = {c: {"crc": {s: [] for s in spheres}, "bv": [], "bias": [], "lung": [],
               "s_ratio": [], "s_nrmse": []} for c in LABELS}
    images, est_example = {}, {}

    for rep in range(args.reps):
        y = {w: poisson(v, rng) for w, v in lam.items()}  # windows are independent
        raw = tew(y["lower"], y["upper"], wl, wu, wp)
        est = {0: None, 1: smooth_sinogram(raw, SIGMA_BINS, SIGMA_ANGLES), 2: raw,
               3: smooth_sinogram(tew(y["lower"], None, wl, wu, wp), SIGMA_BINS, SIGMA_ANGLES),
               4: true_scatter}
        for c, r in est.items():
            x = osem(sm, y["photopeak"], n_iter=args.iters, n_subsets=args.subsets, additive=r) / g
            if rep == 0:
                images[c] = x
                est_example[c] = r
            p = per[c]
            for s, v in crc(x, ph).items():
                p["crc"][s].append(v)
            bkg = roi_mean(x, ph.rois["background"])
            p["bv"].append(background_variability(x, ph))
            p["bias"].append(bkg / ph.truth["background"] - 1)
            p["lung"].append(roi_mean(x, ph.rois["lung"]) / bkg)
            if r is not None:
                p["s_ratio"].append(r.sum() / true_scatter.sum())
                p["s_nrmse"].append(np.sqrt(np.mean((r - true_scatter) ** 2)) / true_scatter.mean())

    def summ(v):
        v = np.asarray(v)
        return None if not len(v) else {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4)}

    results = {LABELS[c]: {"crc": {s: summ(v) for s, v in p["crc"].items()},
                           "background_bias": summ(p["bias"]), "background_variability": summ(p["bv"]),
                           "lung_residual_over_background": summ(p["lung"]),
                           "scatter_estimate_total_over_true": summ(p["s_ratio"]),
                           "scatter_estimate_nrmse": summ(p["s_nrmse"])} for c, p in per.items()}
    b = {c: np.mean(per[c]["bias"]) for c in LABELS}
    check = {"abs_bias_TEW_smoothed_below_no_correction": bool(abs(b[1]) < abs(b[0])),
             "prediction_TEW_bias_negative": bool(b[1] < 0),
             "prediction_lower_only_closer_to_zero_than_TEW": bool(abs(b[3]) < abs(b[1]))}

    meta = {"nucmed_version": nucmed.__version__, "python": platform.python_version(), "numpy": np.__version__,
            "seed": args.seed, "reps": args.reps, "input": "results/mc_iec/projections.npz",
            "input_mc_photons": mc_meta["meta"]["photons"], "expected_primary_counts": counts,
            "windows_keV": windows, "smoothing": {"sigma_bins": SIGMA_BINS, "sigma_angles": SIGMA_ANGLES},
            "recon": {"model": "OSEM + AC + RR", "iters": args.iters, "subsets": args.subsets},
            "runtime_s": round(time.time() - t0, 1)}
    (OUT / "metrics.json").write_text(json.dumps(
        {"meta": meta, "preregistered_check": check, "results": results}, indent=2))
    plot(ph, images, est_example, true_scatter, results, spheres)

    print(json.dumps(check, indent=1))
    for name, r in results.items():
        sr = r["scatter_estimate_total_over_true"]
        print(f"{name:34s} bias={r['background_bias']['mean']:+.3f}±{r['background_bias']['sd']:.3f} "
              f"BV={r['background_variability']['mean']:.3f} lung={r['lung_residual_over_background']['mean']:.3f} "
              f"S/S_true={'-' if sr is None else format(sr['mean'], '.3f')} | "
              + " ".join(f"{s[7:]}={r['crc'][s]['mean']:.2f}" for s in spheres))
    print(f"wrote {OUT} in {meta['runtime_s']} s")


def plot(ph, images, est, true_scatter, results, spheres) -> None:
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})
    fig = plt.figure(figsize=(15, 8), constrained_layout=True, facecolor="#fcfcfb")
    gs = fig.add_gridspec(2, 5)
    vmax = ph.activity.max() * 1.1
    for i, c in enumerate(LABELS):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(images[c], cmap="gray_r", vmin=0, vmax=vmax)
        ax.set_title(LABELS[c], loc="left", fontsize=9); ax.axis("off")

    ax = fig.add_subplot(gs[1, :2])
    a = 30
    ax.plot(true_scatter[a], color="#0b0b0b", lw=2, label="true scatter (noise-free MC)")
    for c in (2, 1, 3):
        ax.plot(est[c][a], color=COLORS[c], lw=2 if c != 2 else 1, alpha=1 if c != 2 else 0.6, label=LABELS[c][3:])
    ax.set_xlabel("detector bin"); ax.set_ylabel("counts per bin")
    ax.set_title(f"Scatter in the photopeak window, one projection (angle {a * 3}°)", loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    names = list(results)
    m = [100 * results[n]["background_bias"]["mean"] for n in names]
    sd = [100 * results[n]["background_bias"]["sd"] for n in names]
    ax.bar(range(len(names)), m, yerr=sd, color=[COLORS[c] for c in LABELS], width=0.6,
           edgecolor="#fcfcfb", linewidth=2, capsize=3, error_kw={"ecolor": "#52514e", "lw": 1})
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(range(len(names)), [str(c) for c in LABELS])
    for i, v in enumerate(m):
        ax.text(i, v + (2 if v >= 0 else -2), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_ylabel("background bias (%)"); ax.set_xlabel("condition"); ax.set_title("Background bias", loc="left")
    ax.grid(axis="x", visible=False)

    ax = fig.add_subplot(gs[1, 3:])
    diam = [int(s.split("_")[1][:-2]) for s in spheres]
    for c, n in zip(LABELS, names):
        mm = np.array([results[n]["crc"][s]["mean"] for s in spheres])
        ax.plot(diam, mm, "-o", color=COLORS[c], lw=2, ms=7, mec="#fcfcfb", mew=2, label=LABELS[c])
    ax.set_ylim(0, 1); ax.set_xlabel("sphere diameter (mm)"); ax.set_ylabel("contrast recovery")
    ax.set_title("CRC, OSEM + AC + RR (mean of realisations)", loc="left"); ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "figure.png", dpi=130, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
