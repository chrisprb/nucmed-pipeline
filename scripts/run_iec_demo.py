"""End-to-end demo: IEC-like phantom -> SPECT projections (attenuation +
collimator blur + Poisson noise) -> OSEM with and without corrections ->
contrast recovery vs iteration.

    uv run python scripts/run_iec_demo.py [--counts 1e6] [--seed 20261105]

Writes results/iec_demo/{metrics.json, figure.png}. Every number in the
README comes from this script.
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
from nucmed.metrics import background_variability, crc, nrmse
from nucmed.noise import poisson, scale_to_counts
from nucmed.phantom import iec_like
from nucmed.projector import build_system_matrix
from nucmed.recon import osem

OUT = Path(__file__).resolve().parents[1] / "results" / "iec_demo"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=float, default=1e6)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--subsets", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    grid = ImageGrid()
    ph = iec_like(grid)
    support = ph.mu > 0
    acq = Acquisition(n_angles=120, radius_cm=20.0, collimator=Collimator())

    # Truth model generates the data.
    sm_full = build_system_matrix(grid, acq, mu=ph.mu, support=support)
    clean, scale = scale_to_counts(sm_full.forward(ph.activity), args.counts)
    data = poisson(clean, np.random.default_rng(args.seed))

    # Reconstruction models: what the scanner software corrects for.
    models = {
        "OSEM (no corrections)": build_system_matrix(
            grid, Acquisition(acq.n_angles, radius_cm=acq.radius_cm, collimator=Collimator(0, 0)),
            mu=None, support=support),
        "OSEM + AC": build_system_matrix(
            grid, Acquisition(acq.n_angles, radius_cm=acq.radius_cm, collimator=Collimator(0, 0)),
            mu=ph.mu, support=support),
        "OSEM + AC + RR": sm_full,  # attenuation + resolution recovery
    }

    results, images = {}, {}
    for name, sm in models.items():
        curve = []

        def cb(it, x, curve=curve):
            curve.append({"iter": it, "crc": crc(x, ph),
                          "bv": background_variability(x, ph),
                          "nrmse_body": nrmse(x / scale, ph.activity, support)})

        x = osem(sm, data, n_iter=args.iters, n_subsets=args.subsets, callback=cb)
        results[name] = curve
        images[name] = x / scale

    meta = {
        "nucmed_version": nucmed.__version__, "python": platform.python_version(),
        "numpy": np.__version__, "seed": args.seed, "total_counts": args.counts,
        "grid": {"n": grid.n, "pixel_cm": grid.pixel_cm},
        "acquisition": {"n_angles": acq.n_angles, "radius_cm": acq.radius_cm,
                        "fwhm0_cm": acq.collimator.fwhm0_cm, "slope": acq.collimator.slope},
        "osem": {"iters": args.iters, "subsets": args.subsets},
        "note": "Data generated with the same projector used for AC+RR (an 'inverse crime'); "
                "no scatter. Metrics are 2D NEMA-style approximations, not NU 2 compliant.",
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    plot(ph, images, results)
    final = {k: {s: round(v, 3) for s, v in r[-1]["crc"].items()} | {"BV": round(r[-1]["bv"], 3)}
             for k, r in results.items()}
    print(json.dumps(final, indent=2))
    print(f"wrote {OUT} in {meta['runtime_s']} s")


def plot(ph, images, results) -> None:
    names = list(images)
    fig = plt.figure(figsize=(14, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, len(names) + 1)
    vmax = ph.activity.max() * 1.1
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(ph.activity, cmap="gray_r", vmin=0, vmax=vmax)
    ax.set_title("Truth (IEC-like, 4:1)"); ax.axis("off")
    for i, n in enumerate(names):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(images[n], cmap="gray_r", vmin=0, vmax=vmax); ax.set_title(n); ax.axis("off")

    spheres = [k for k in ph.rois if k.startswith("sphere_")]
    diam = [int(k.split("_")[1][:-2]) for k in spheres]
    ax = fig.add_subplot(gs[1, : (len(names) + 1) // 2 + 1])
    for n in names:
        last = results[n][-1]["crc"]
        ax.plot(diam, [last[s] for s in spheres], "o-", label=n)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("sphere diameter (mm)"); ax.set_ylabel("contrast recovery")
    ax.set_title("CRC at final iteration"); ax.legend()

    ax = fig.add_subplot(gs[1, (len(names) + 1) // 2 + 1:])
    for n in names:
        r = results[n]
        ax.plot([e["bv"] for e in r], [e["crc"]["sphere_22mm"] for e in r], "o-", ms=3, label=n)
    ax.set_xlabel("background variability (CoV)"); ax.set_ylabel("CRC, 22 mm sphere")
    ax.set_title("Contrast vs noise, by iteration"); ax.legend()
    fig.savefig(OUT / "figure.png", dpi=130)


if __name__ == "__main__":
    main()
