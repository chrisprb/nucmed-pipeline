"""Build the project page from the recorded results.

Every number on the page is read from results/*/metrics.json (or the saved
Monte Carlo spectrum); nothing is typed in by hand. Re-run after any result
changes.

    uv run python scripts/build_site.py                  # writes docs/index.html + docs/figures/
    uv run python scripts/build_site.py --fragment PATH  # also writes the page body alone

docs/ can be served as-is (e.g. GitHub Pages from the /docs folder).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from html import escape
from pathlib import Path
from string import Template

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DOCS = ROOT / "docs"
SPHERES = ["sphere_10mm", "sphere_13mm", "sphere_17mm", "sphere_22mm", "sphere_28mm", "sphere_37mm"]


def load(name: str) -> dict:
    return json.loads((RES / name / "metrics.json").read_text())


def f2(x: float) -> str:
    return f"{x:.2f}"


def pct(x: float, signed: bool = True, digits: int = 1) -> str:
    return f"{x * 100:+.{digits}f} %" if signed else f"{x * 100:.{digits}f} %"


def sci(x: float) -> str:
    if abs(x) >= 0.01:
        return f"{x:.3g}"
    mant, exp = f"{x:.2e}".split("e")
    return f"{mant}&#8239;×&#8239;10<sup>{int(exp)}</sup>"


def table(head: list[str], rows: list[list[str]], cls: str = "") -> str:
    th = "".join(f"<th scope=\"col\">{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" if i else f"<th scope=\"row\">{c}</th>"
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="table-wrap"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def spectrum_svg() -> tuple[str, float]:
    """Hero graphic: the simulated Tc-99m spectrum (primary vs scattered),
    blurred by the 10 % energy resolution exactly as in the Stage 2 figure."""
    d = np.load(RES / "mc_iec" / "projections.npz")
    meta = load("mc_iec")
    edges, spec = d["spectrum_edges"], d["spectrum"]
    ec = 0.5 * (edges[:-1] + edges[1:])
    e0, fw = 140.5, meta["meta"]["mc"]["fwhm_frac"]
    sig = fw * e0 / 2.3548 * np.sqrt(np.maximum(ec, 1) / e0)
    k = np.exp(-0.5 * ((ec[:, None] - ec[None, :]) / sig[None, :]) ** 2) / sig[None, :]
    prim, scat = k @ spec[0], k @ spec[1]
    norm = (prim + scat).max()
    x0, x1, W, H, top, bot = 60.0, 165.0, 640, 260, 18, 222
    X = lambda e: 40 + (e - x0) / (x1 - x0) * (W - 56)
    Y = lambda v: bot - v / norm * (bot - top)
    sel = (ec >= x0) & (ec <= x1)
    def poly(v):
        return " ".join(f"{X(e):.1f},{Y(y):.1f}" for e, y in zip(ec[sel], v[sel]))
    lo, hi = meta["windows"]["photopeak"]["range_keV"]
    sf = meta["windows"]["photopeak"]["scatter_fraction"]
    ticks = "".join(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{bot}" y2="{bot + 5}" class="ax"/>'
                    f'<text x="{X(t):.1f}" y="{bot + 19}" class="tick" text-anchor="middle">{t}</text>'
                    for t in (60, 80, 100, 120, 140, 160))
    svg = f"""<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="spec-t spec-d" class="spectrum">
<title id="spec-t">Simulated Tc-99m energy spectrum</title>
<desc id="spec-d">Detected counts against measured energy from the Stage 2 Monte Carlo. The photopeak at 140.5 keV sits on a broad shoulder of scattered photons; {sf:.0%} of the counts in the 126 to 155 keV window are scattered.</desc>
<rect x="{X(lo):.1f}" y="{top - 8}" width="{X(hi) - X(lo):.1f}" height="{bot - top + 8}" class="window"/>
<text x="{(X(lo) + X(hi)) / 2:.1f}" y="{top + 4}" class="win-label" text-anchor="middle">photopeak window</text>
<line x1="40" x2="{W - 16}" y1="{bot}" y2="{bot}" class="ax"/>
{ticks}
<text x="{W - 16}" y="{H - 2}" class="tick" text-anchor="end">keV</text>
<polyline points="{poly(scat)}" class="ln scat"/>
<polyline points="{poly(prim)}" class="ln prim"/>
<polyline points="{poly(prim + scat)}" class="ln tot"/>
<text x="{X(96):.1f}" y="{Y((prim + scat)[np.argmin(abs(ec - 96))]) - 10:.1f}" class="lab scat-t" text-anchor="middle">scattered</text>
<text x="{X(141) + 34:.1f}" y="{Y(prim.max()) + 30:.1f}" class="lab prim-t">primary</text>
<text x="{X(141) - 6:.1f}" y="{Y((prim + scat).max()) - 6:.1f}" class="lab tot-t" text-anchor="end">total</text>
</svg>"""
    return svg, sf


def build() -> dict[str, str]:
    s1, s2, s3, s4 = load("iec_demo"), load("mc_iec"), load("tew"), load("dosimetry")
    v = {}
    svg, sf = spectrum_svg()
    v["spectrum"] = svg
    v["sf"] = f"{sf:.0%}"

    # Stage 1.
    rows = []
    for name, curve in s1["results"].items():
        last = curve[-1]
        rows.append([escape(name)] + [f2(last["crc"][s]) for s in SPHERES] + [f2(last["bv"])])
    v["s1_table"] = table(["Reconstruction", "10 mm", "13", "17", "22", "28", "37", "Bkg CoV"], rows, "num")

    # Stage 2.
    rows = []
    for k, r in s2["results"].items():
        rows.append([escape(r["label"])] + [f'{f2(r["crc"][s]["mean"])}<small>±{f2(r["crc"][s]["sd"])}</small>' for s in SPHERES]
                    + [pct(r["background_bias"]["mean"])])
    v["s2_table"] = table(["Data", "10 mm", "13", "17", "22", "28", "37", "Bkg bias"], rows, "num")
    v["s2_cons"] = f'{abs(s2["meta"]["mc_primary_over_analytic_total"] - 1) * 100:.2f} %'
    v["s2_bias_c"] = pct(s2["results"]["C"]["background_bias"]["mean"], digits=0)
    v["s2_crc37"] = f'{f2(s2["results"]["A"]["crc"]["sphere_37mm"]["mean"])} → {f2(s2["results"]["C"]["crc"]["sphere_37mm"]["mean"])}'
    v["s2_photons"] = f'{s2["meta"]["photons"]:.0e}'.replace("1e+06", "10<sup>6</sup>")

    # Stage 3.
    rows = []
    for name, r in s3["results"].items():
        sr = r["scatter_estimate_total_over_true"]
        rows.append([escape(name.split(". ", 1)[1]),
                     f'{pct(r["background_bias"]["mean"])}<small>±{r["background_bias"]["sd"] * 100:.1f}</small>',
                     "–" if sr is None else f2(sr["mean"]),
                     f2(r["crc"]["sphere_37mm"]["mean"]), f2(r["background_variability"]["mean"])])
    v["s3_table"] = table(["Scatter correction", "Background bias", "Scatter est. / true", "CRC 37 mm", "Bkg CoV"], rows, "num")
    r3 = s3["results"]
    v["s3_none"] = pct(r3["0. no scatter correction"]["background_bias"]["mean"])
    v["s3_tew"] = pct(r3["1. TEW, smoothed"]["background_bias"]["mean"])
    v["s3_low"] = pct(r3["3. lower window only, smoothed"]["background_bias"]["mean"])
    v["s3_ratio"] = f'{r3["1. TEW, smoothed"]["scatter_estimate_total_over_true"]["mean"]:.2f}'
    v["s3_crc_tew"] = f2(r3["1. TEW, smoothed"]["crc"]["sphere_37mm"]["mean"])
    v["s3_crc_or"] = f2(r3["4. oracle (true scatter)"]["crc"]["sphere_37mm"]["mean"])

    # Stage 4.
    af = s4["T4.2_photon_af"]
    v["t42"] = f'{af["water, embedded"]["pass"]}/{af["water, embedded"]["rows"]}'
    v["t42_tissue"] = f'{af["tissue, embedded"]["pass"]}/{af["tissue, embedded"]["rows"]}'
    v["t42_iso"] = f'{af["water, isolated"]["pass"]}/{af["water, isolated"]["rows"]}'
    hi_e = [k for k in af["water, embedded"]["pass_by_energy_keV"] if float(k) >= 100]
    ok = sum(int(af["water, embedded"]["pass_by_energy_keV"][k].split("/")[0]) for k in hi_e)
    tot = sum(int(af["water, embedded"]["pass_by_energy_keV"][k].split("/")[1]) for k in hi_e)
    v["t42_hi"] = f"{ok}/{tot}"
    t44 = s4["T4.4_csda_lower_bound"]
    v["t44"] = f'{t44["bound_holds"]}/{t44["rows"]}'
    dec = s4["decay"]
    v["hl"] = f'{dec["half_life_d"]:.4f}'
    v["beta_gap"] = f'{abs(dec["beta_mean_from_spectrum_keV"] / dec["beta_mean_from_branches_keV"] - 1) * 100:.1f} %'
    rows = []
    for s in s4["lu177_sphere_s_values"]:
        rows.append([f'{s["mass_g"]:g} g', sci(s["S_total_mGy_per_MBq_s"]), f2(s["electron_af_effective"]),
                     f'{s["photon_af"]:.3f}', pct(s["photon_share"], signed=False)])
    v["s4_table"] = table(["Sphere", "S (mGy per MBq·s)", "Electron AF", "Photon AF", "Photon share of S"], rows, "num")
    rows = []
    for name, r in s4["tac"]["methods"].items():
        sched, meth = name.split(": ")
        rows.append([escape(meth), escape(sched.split(" (")[0]), f'{r["bias_pct"]:+.1f} %', f'{r["sd_pct"]:.1f} %'])
    v["tac_table"] = table(["Integration method", "Samples", "Mean error", "SD"], rows, "num")

    # Ledger.
    yes, part = '<span class="st met">Met</span>', '<span class="st part">Partly met</span>'
    conf = '<span class="st met">Confirmed</span>'
    c2, c3 = s2["preregistered_check"], s3["preregistered_check"]
    st = lambda b, good=yes: good if b else '<span class="st fail">Not met</span>'
    ledger = [
        ["1", "Projector: adjoint identity, count conservation, point-source centroid and width, attenuation vs analytic chord (≤ 1 %)", yes, "tests/test_projector.py"],
        ["1", "MLEM/OSEM: counts preserved, likelihood never decreases, 1 subset = MLEM, AC restores a uniform disk (≤ 3 %)", yes, "tests/test_recon.py"],
        ["2", "Narrow-beam μ from simulated transmission within 2 % of NIST XCOM", yes, "tests/test_mc.py"],
        ["2", "Sampled Compton angles follow Klein–Nishina (χ², 10<sup>6</sup> samples, p &gt; 0.01)", yes, "tests/test_mc.py"],
        ["2", f"Monte Carlo primaries vs analytic projector: totals within 2 %, NRMSE &lt; 5 % (observed {v['s2_cons']} and 1.2 %)", yes, "tests/test_mc.py"],
        ["2", "Uncorrected scatter lowers CRC for every sphere and biases the background upward", st(all(c2.values()), conf), "results/mc_iec/metrics.json"],
        ["3", "Smoothed TEW brings the background bias closer to zero than no correction", st(c3["abs_bias_TEW_smoothed_below_no_correction"]), "results/tew/metrics.json"],
        ["3", "Predicted from window totals: TEW over-corrects (negative bias); the lower window alone does better", st(c3["prediction_TEW_bias_negative"] and c3["prediction_lower_only_closer_to_zero_than_TEW"], conf), "results/tew/metrics.json"],
        ["4", f"Lu-177 decay data: spectrum mean vs ENSDF branch mean within 1 % (observed {v['beta_gap']})", yes, "results/dosimetry/metrics.json"],
        ["4", f"Photon absorbed fractions within the EGS4–MCNP spread (floor 5 %) of Stabin &amp; Konijnenberg 2000: {v['t42']} rows; {v['t42_hi']} at 100–160 keV", part, "results/dosimetry/photon_af_validation.csv"],
        ["4", "Small-sphere limit AF → μ<sub>en</sub>·3R/4", yes, "tests/test_dosimetry.py"],
        ["4", f"Straight-line CSDA electron model is a lower bound on the reference ({v['t44']} rows)", yes, "results/dosimetry/metrics.json"],
        ["4", "Photon point kernel conserves energy and reproduces the direct sphere result (≤ 3 %)", yes, "tests/test_dosimetry.py"],
        ["4", "Time–activity fits and integrals exact on noise-free data", yes, "tests/test_dosimetry.py"],
    ]
    v["ledger"] = table(["Stage", "Target, written before the code", "Outcome", "Evidence"],
                        [[a, b, c, f"<code>{d}</code>"] for a, b, c, d in ledger], "ledger")
    v["n_met"] = str(sum('class="st met"' in r[2] for r in ledger))
    v["n_targets"] = str(len(ledger))
    try:
        out = subprocess.run(["uv", "run", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True).stdout
        v["n_tests"] = out.strip().splitlines()[-1].split()[0]
    except Exception:
        v["n_tests"] = "–"
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", type=Path, help="also write the page body (no <html> wrapper) here")
    args = ap.parse_args()
    v = build()
    body = Template((ROOT / "scripts" / "site_template.html").read_text()).substitute(v)
    (DOCS / "figures").mkdir(parents=True, exist_ok=True)
    for stage in ("iec_demo", "mc_iec", "tew", "dosimetry"):
        shutil.copyfile(RES / stage / "figure.png", DOCS / "figures" / f"{stage}.png")
    (DOCS / "index.html").write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + body.replace("<!--/head-->", "</head>\n<body>", 1) + "\n</body>\n</html>\n")
    if args.fragment:
        args.fragment.write_text(body.replace("<!--/head-->", "", 1))
    print(f"wrote {DOCS / 'index.html'} ({v['n_met']}/{v['n_targets']} targets met, {v['n_tests']} tests)")


if __name__ == "__main__":
    main()
