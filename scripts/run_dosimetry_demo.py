"""Stage 4: Lu-177 dosimetry validation and demo.

1. T4.1: decay data summary.
2. T4.2: photon absorbed fractions vs Stabin & Konijnenberg (2000), every
   table row with 20 keV <= E <= 208.4 keV. Three model variants:
     water, embedded  - primary (the reference geometry, our pipeline material)
     water, isolated  - the first implementation (wrong geometry; kept for the record)
     tissue, embedded - exploratory composition check (ICRU four-component tissue)
3. T4.4: straight-line CSDA lower bound vs the electron table.
4. Lu-177 S-values for unit-density spheres, with the low-energy electron bound.
5. Synthetic time-activity curves: integration methods and sampling schedules
   (descriptive; synthetic kinetics only).

    uv run python scripts/run_dosimetry_demo.py [--photons 200000] [--seed 20261107]

Writes results/dosimetry/{metrics.json, photon_af_validation.csv, figure.png}.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import nucmed
from nucmed import dosimetry as dz
from nucmed.photon3d import absorbed_fraction_sphere
from nucmed.radionuclide import lu177

OUT = Path(__file__).resolve().parents[1] / "results" / "dosimetry"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
VARIANTS = {"water, embedded": dict(surround="water", material="water"),
            "water, isolated": dict(surround="vacuum", material="water"),
            "tissue, embedded": dict(surround="water", material="tissue_icru4")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photons", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--tac-draws", type=int, default=2000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    nuc = lu177()
    out: dict = {}

    # 1. Decay data.
    e, p = nuc.beta_spectrum
    b = nuc.beta_branches
    out["decay"] = {"half_life_d": nuc.half_life_s / 86400,
                    "energy_per_decay_keV": {k: round(v, 2) for k, v in nuc.energy_per_decay_kev().items()},
                    "beta_mean_from_spectrum_keV": round(float(np.trapezoid(e * p, e) / np.trapezoid(p, e)), 2),
                    "beta_mean_from_branches_keV": round(float((b[:, 0] * b[:, 1]).sum() / b[:, 1].sum()), 2)}

    # 2. T4.2, every photon row in range.
    rows = [r for r in dz.stabin2000() if r["table"].startswith("photon") and 20 <= r["energy_MeV"] * 1000 <= 208.4]
    val = []
    for i, r in enumerate(rows):
        tol = max(0.05, abs(r["egs4"] / r["mcnp4b"] - 1))
        rec = {"table": r["table"], "energy_keV": r["energy_MeV"] * 1000, "mass_g": r["mass_g"],
               "recommended": r["recommended"], "egs4": r["egs4"], "mcnp4b": r["mcnp4b"], "tolerance": round(tol, 4)}
        for name, kw in VARIANTS.items():
            af, se = absorbed_fraction_sphere(r["energy_MeV"] * 1000, r["radius_cm"], n=args.photons,
                                              seed=args.seed + i, **kw)
            rec[f"{name}: af"] = round(af, 5)
            rec[f"{name}: se"] = round(se, 5)
            rec[f"{name}: ratio"] = round(af / r["recommended"], 4)
            rec[f"{name}: pass"] = bool(abs(af / r["recommended"] - 1) <= tol)
        val.append(rec)
    with open(OUT / "photon_af_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(val[0]))
        w.writeheader(); w.writerows(val)
    summ = {}
    for name in VARIANTS:
        ratios = np.array([v[f"{name}: ratio"] for v in val])
        passes = [v[f"{name}: pass"] for v in val]
        by_e = {}
        for v in val:
            by_e.setdefault(v["energy_keV"], []).append(v[f"{name}: pass"])
        summ[name] = {"rows": len(val), "pass": int(sum(passes)),
                      "ratio_median": round(float(np.median(ratios)), 3),
                      "ratio_min": round(float(ratios.min()), 3), "ratio_max": round(float(ratios.max()), 3),
                      "pass_by_energy_keV": {f"{k:g}": f"{sum(x)}/{len(x)}" for k, x in sorted(by_e.items())}}
    out["T4.2_photon_af"] = summ

    # 3. T4.4 CSDA lower bound.
    el = [r for r in dz.stabin2000() if r["table"] == "electron" and r["energy_MeV"] <= 1.0]
    lb_ok = [dz.electron_af_csda_straight(r["energy_MeV"] * 1000, r["radius_cm"]) <= r["recommended"] + 0.005 for r in el]
    out["T4.4_csda_lower_bound"] = {"rows": len(el), "bound_holds": int(sum(lb_ok))}

    # 4. Lu-177 sphere S-values.
    masses = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    svals = []
    for j, m in enumerate(masses):
        R = dz.sphere_radius_cm(m)
        af_ph, se_ph = absorbed_fraction_sphere(nuc.photon_lines, R, n=4 * args.photons, seed=args.seed + 1000 + j)
        s = {k: dz.s_value_sphere(nuc, m, af_ph, k) for k in ("linear", "constant", "one")}
        e_el = sum(v for k, v in nuc.energy_per_decay_kev().items() if k != "photons")
        svals.append({"mass_g": m, "radius_cm": round(R, 4), "photon_af": round(af_ph, 4), "photon_af_se": round(se_ph, 5),
                      "electron_af_effective": round(dz.electron_energy_absorbed_kev(nuc, m) / e_el, 4),
                      "S_total_Gy_per_Bq_s": s["linear"]["total"], "S_electrons": s["linear"]["electrons"],
                      "S_photons": s["linear"]["photons"],
                      "S_total_bounds_low_energy_electrons": [s["constant"]["total"], s["one"]["total"]],
                      "S_total_mGy_per_MBq_s": s["linear"]["total"] * 1e9,
                      "photon_share": s["linear"]["photons"] / s["linear"]["total"]})
    out["lu177_sphere_s_values"] = svals

    # 5. Synthetic TAC example (descriptive).
    out["tac"], tac_plot = tac_study(nuc, args.seed, args.tac_draws, svals)

    out["meta"] = {"nucmed_version": nucmed.__version__, "python": platform.python_version(),
                   "numpy": np.__version__, "seed": args.seed, "photons_per_row": args.photons,
                   "runtime_s": round(time.time() - t0, 1)}
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2))
    plot(val, svals, tac_plot)
    print(json.dumps({k: out[k] for k in ("decay", "T4.2_photon_af", "T4.4_csda_lower_bound")}, indent=1))
    for s in svals:
        lo, hi = s["S_total_bounds_low_energy_electrons"]
        print(f"{s['mass_g']:8g} g  S = {s['S_total_mGy_per_MBq_s']:.4g} mGy/(MBq s) "
              f"[{lo*1e9:.4g}, {hi*1e9:.4g}]  phi_e = {s['electron_af_effective']:.3f}  "
              f"phi_ph = {s['photon_af']:.3f}  photon share = {s['photon_share']:.1%}")
    for k, v in out["tac"]["methods"].items():
        print(f"{k:45s} bias {v['bias_pct']:+6.1f} %  SD {v['sd_pct']:5.1f} %")
    print(f"wrote {OUT} in {out['meta']['runtime_s']} s")


def tac_study(nuc, seed, draws, svals):
    """Synthetic tumour: uptake half-time 2 h, biological washout half-life
    120 h (modelling choices). Samples with 5 % multiplicative noise."""
    lam_p = np.log(2) / (nuc.half_life_s / 3600)  # per hour
    lam_out = lam_p + np.log(2) / 120.0
    lam_in = lam_p + np.log(2) / 2.0
    a0 = 1.0  # MBq scale
    true = dz.tia_uptake_washout(a0, lam_out, lam_in)  # MBq h
    schedules = {"4 points (4, 24, 96, 168 h)": np.array([4.0, 24, 96, 168]),
                 "3 points (24, 96, 168 h)": np.array([24.0, 96, 168])}
    rng = np.random.default_rng(seed)
    res = {}
    example = None
    for sname, t in schedules.items():
        methods = {"trapezoid + physical-decay tail": [], "trapezoid + fitted tail": [],
                   "mono-exponential fit": []}
        if len(t) >= 4:
            methods["uptake-washout fit"] = []
        for k in range(draws):
            a = dz.uptake_washout(t, a0, lam_out, lam_in) * np.exp(rng.normal(0, 0.05, len(t)))
            _, lam_tail = dz.fit_mono(t[-2:], a[-2:])
            lam_tail = max(lam_tail, 1e-6)
            methods["trapezoid + physical-decay tail"].append(dz.tia_trapezoid(t, a, lam_p))
            methods["trapezoid + fitted tail"].append(dz.tia_trapezoid(t, a, lam_tail))
            m0, ml = dz.fit_mono(t[t >= 24], a[t >= 24])
            methods["mono-exponential fit"].append(dz.tia_mono(m0, ml))
            if "uptake-washout fit" in methods:
                try:
                    methods["uptake-washout fit"].append(dz.tia_uptake_washout(*dz.fit_uptake_washout(t, a, lam_p)))
                except RuntimeError:
                    methods["uptake-washout fit"].append(np.nan)
            if example is None and len(t) >= 4:
                example = (t, a)
        for mname, v in methods.items():
            v = np.array(v)
            err = 100 * (v / true - 1)
            res[f"{sname}: {mname}"] = {"bias_pct": round(float(np.nanmean(err)), 2),
                                        "sd_pct": round(float(np.nanstd(err, ddof=1)), 2),
                                        "failed_fits": int(np.isnan(v).sum())}
    s10 = next(s for s in svals if s["mass_g"] == 10.0)["S_total_Gy_per_Bq_s"]
    dose_per_mbq = true * 3600 * 1e6 * s10  # Gy per MBq injected-equivalent A0 scale
    info = {"truth": {"lam_phys_per_h": lam_p, "lam_out_per_h": lam_out, "lam_in_per_h": lam_in,
                      "tia_MBq_h": true, "dose_10g_sphere_Gy_per_MBq_A0": dose_per_mbq},
            "noise": "5 % multiplicative (log-normal)", "draws": draws, "methods": res}
    return info, {"example": example, "lam": (a0, lam_out, lam_in), "methods": res}


def plot(val, svals, tac) -> None:
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True, facecolor="#fcfcfb")

    ax = axs[0, 0]
    for k, name in enumerate(VARIANTS):
        x = np.array([v["energy_keV"] for v in val]) * (1 + 0.025 * (k - 1))
        y = np.array([v[f"{name}: ratio"] for v in val])
        ax.scatter(x, y, s=22, color=C[k], edgecolor="#fcfcfb", linewidth=1, label=name, zorder=3)
    ax.axhspan(0.95, 1.05, color="#f0efec", zorder=0, lw=0)
    ax.axhline(1, color="#52514e", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("photon energy (keV)"); ax.set_ylabel("our AF / reference AF")
    ax.set_title("Photon absorbed fractions vs Stabin & Konijnenberg 2000\n(band: ±5 %; each point a sphere mass)", loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = axs[0, 1]
    m = np.array([s["mass_g"] for s in svals])
    for key, label, c in (("S_total_Gy_per_Bq_s", "total", "#0b0b0b"), ("S_electrons", "electrons", C[0]),
                          ("S_photons", "photons", C[1])):
        ax.plot(m, np.array([s[key] for s in svals]) * 1e9, "-o", color=c, lw=2, ms=7, mec="#fcfcfb", mew=2, label=label)
    lo = np.array([s["S_total_bounds_low_energy_electrons"][0] for s in svals]) * 1e9
    hi = np.array([s["S_total_bounds_low_energy_electrons"][1] for s in svals]) * 1e9
    ax.fill_between(m, lo, hi, color="#0b0b0b", alpha=0.12, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sphere mass (g)"); ax.set_ylabel("S (mGy per MBq·s)")
    ax.set_title("Lu-177 self-dose S-values, unit-density spheres", loc="left"); ax.legend(frameon=False)

    ax = axs[1, 0]
    t, a = tac["example"]
    a0, l1, l2 = tac["lam"]
    tt = np.linspace(0, 400, 800)
    ax.plot(tt, dz.uptake_washout(tt, a0, l1, l2), color="#0b0b0b", lw=2, label="true (synthetic)")
    ax.plot(t, a, "o", color=C[1], ms=8, mec="#fcfcfb", mew=2, label="samples (5 % noise)")
    ax.set_xlabel("time after administration (h)"); ax.set_ylabel("activity (relative)")
    ax.set_title("Synthetic tumour time–activity curve", loc="left"); ax.legend(frameon=False)

    ax = axs[1, 1]
    names = list(tac["methods"])
    bias = [tac["methods"][n]["bias_pct"] for n in names]
    sd = [tac["methods"][n]["sd_pct"] for n in names]
    y = np.arange(len(names))
    ax.barh(y, bias, xerr=sd, color=[C[0] if n.startswith("4") else C[2] for n in names], height=0.6,
            edgecolor="#fcfcfb", linewidth=2, error_kw={"ecolor": "#52514e", "lw": 1}, capsize=3)
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_yticks(y, [n.replace(": ", "\n") for n in names], fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("error in time-integrated activity (%), mean ± SD")
    ax.set_title("Integration method and schedule", loc="left"); ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "figure.png", dpi=130, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
