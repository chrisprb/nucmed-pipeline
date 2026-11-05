"""Download the external reference data used by nucmed and write it, with
provenance headers, to src/nucmed/data/.

    uv run python scripts/fetch_reference_data.py [--date YYYY-MM-DD]

Sources (all public, no login):
  * NIST XCOM: water photon cross sections, 1 keV - 1 MeV.
  * IAEA LiveChart API (ENSDF evaluation): Lu-177 ground-state decay
    radiations and beta spectrum.
  * NIST ESTAR: electron CSDA range and radiative yield in liquid water.
  * NIST X-ray mass attenuation / energy-absorption coefficients, water.
  * Stabin MG, Konijnenberg MW. Re-evaluation of absorbed fractions for
    photons and electrons in spheres of various sizes. J Nucl Med 2000;41:149-160.
    Free full text; Tables 1-3 are parsed from the PDF and checked for internal
    consistency (EGS4/MCNP ratio column, recommended = mean of the two codes).

Downloads go to a temporary directory; only the parsed tables are kept.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import tempfile
from datetime import date
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "src" / "nucmed" / "data"
UA = "nucmed-research-client"


def curl(url: str, out: Path, post: list[str] | None = None) -> Path:
    cmd = ["curl", "-sS", "-m", "60", "-L", "-A", UA, "-o", str(out)]
    for p in post or []:
        cmd += ["--data-urlencode", p]
    subprocess.run(cmd + [url], check=True)
    return out


def strip_tags(html: str) -> str:
    return re.sub(r"<[^>]*>", " ", html)


XCOM_ENERGIES_KEV = sorted(set([1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17.5]
                               + list(range(20, 201, 5)) + [112.9, 140.5, 208.4]
                               + [250, 300, 350, 400, 450, 500, 600, 700, 800, 1000]))


def _xcom_query(tmp: Path, out_name: str, title: str, query: str, endpoint: str, fields: list[str], today: str) -> None:
    e = "\n".join(f"{x / 1000:.5f}" for x in XCOM_ENERGIES_KEV)
    f = curl(f"https://physics.nist.gov/cgi-bin/Xcom/{endpoint}", tmp / f"{out_name}.html",
             fields + [f"Energies={e}", "NumAdd=1", "WindowXmin=0.001", "WindowXmax=100000", "Graph6=on"])
    row = re.compile(r"^\s*(\d\.\d{3}E[-+]\d{2})((?:\s+\d\.\d{3}E[-+]\d{2}){7})\s*$")
    rows = []
    text = strip_tags(f.read_text())
    for line in text.splitlines():
        m = row.match(line)
        if m:
            v = [m.group(1)] + m.group(2).split()
            rows.append((v[0], v[1], v[2], v[3], v[6]))
    assert len(rows) >= 50, len(rows)
    constituents = " ".join(re.findall(r"Z=\d+\s*:\s*[\d.]+", text))
    with open(DATA / out_name, "w") as out:
        out.write(f"# NIST XCOM photon cross sections for {title}, mass coefficients in cm^2/g.\n"
                  "# Source: NIST XCOM, https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html\n"
                  f"# Query: POST https://physics.nist.gov/cgi-bin/Xcom/{endpoint}  {query}, user energy list, retrieved {today}.\n"
                  f"# Constituents reported by XCOM (Z : fraction by weight): {constituents}\n"
                  "# Columns: energy_MeV, coherent, incoherent, photoelectric, total_with_coherent\n"
                  "energy_MeV,coherent,incoherent,photoelectric,total_with_coherent\n")
        for r in rows:
            out.write(",".join(r) + "\n")


def xcom(tmp: Path, today: str) -> None:
    _xcom_query(tmp, "xcom_water.csv", "water (H2O)", "Formula=H2O", "xcom3_2",
                ["Formula=H2O", "Name=water"], today)


def xcom_tissue(tmp: Path, today: str) -> None:
    """ICRU four-component soft tissue (H 10.1174, C 11.1, N 2.6, O 76.1826 % by
    weight; composition as listed for NIST ESTAR/XCOM 'tissue, soft (ICRU
    four-component)' [verify against ICRU 44]). Used only in an exploratory
    composition check, not in the main pipeline."""
    f = "H 0.101174\nC 0.111000\nN 0.026000\nO 0.761826"
    _xcom_query(tmp, "xcom_tissue_icru4.csv", "ICRU four-component soft tissue", "Formulae=H/C/N/O by weight",
                "xcom3_3", [f"Formulae={f}", "Name=tissue"], today)


def iaea(tmp: Path, today: str) -> None:
    base = "https://nds.iaea.org/relnsd/v1/data?nuclides=177lu&fields="
    def table(fields, rad=None):
        f = curl(base + fields + (f"&rad_types={rad}" if rad else ""), tmp / f"{fields}_{rad}.csv")
        rows = list(csv.DictReader(io.StringIO(f.read_text())))
        return [r for r in rows if r.get("p_energy", "0") in ("0", "0.0")]  # ground state only

    gs = table("ground_states")[0]
    lines = []
    for r in table("decay_rads", "bm"):
        lines.append(("beta_minus_mean", r["mean_energy"], r["intensity_beta"], r["unc_ib"], r["max_energy"]))
    for rad, label in (("g", "gamma"), ("x", "xray")):
        for r in table("decay_rads", rad):
            lines.append((label, r["energy"], r["intensity"], r["unc_i"], ""))
    for r in table("decay_rads", "e"):
        lines.append(("ce" if r["type"] == "CE" else "auger", r["energy"], r["intensity"], r["unc_i"], ""))
    meta = (f"# Lu-177 ground state (7/2+), half-life {gs['half_life']} {gs['unit_hl']} "
            f"(+/- {gs['unc_hl']} in last digits), {gs['half_life_sec']} s.\n")
    with open(DATA / "lu177_emissions.csv", "w") as out:
        out.write("# Lu-177 decay radiations, ground-state decay only (metastable Lu-177m excluded).\n"
                  "# Source: IAEA LiveChart of Nuclides API (ENSDF evaluation, F.G. Kondev, cut-off 30-Aug-2019),\n"
                  f"# https://nds.iaea.org/relnsd/v1/data?fields=decay_rads&nuclides=177lu, retrieved {today}.\n"
                  + meta +
                  "# Energies in keV; intensity per 100 decays. For beta_minus_mean, energy is the branch mean energy.\n"
                  "kind,energy_keV,intensity_pct,unc_intensity,max_energy_keV\n")
        for l in lines:
            out.write(",".join(str(x) for x in l) + "\n")
    with open(DATA / "lu177_half_life.txt", "w") as out:
        out.write(meta + f"{gs['half_life_sec']}\n")

    rows = table("bin_beta", "bm")
    with open(DATA / "lu177_beta_spectrum.csv", "w") as out:
        out.write("# Lu-177 beta-minus spectrum (all branches), dN/dE per keV.\n"
                  f"# Source: IAEA LiveChart API fields=bin_beta&nuclides=177lu&rad_types=bm, retrieved {today}.\n"
                  "energy_keV,dN_dE\n")
        for r in rows:
            out.write(f"{r['bin_en']},{r['dn_de']}\n")


def estar(tmp: Path, today: str) -> None:
    f = tmp / "estar.html"
    subprocess.run(["curl", "-sS", "-m", "60", "-o", str(f), "-X", "POST",
                    "https://physics.nist.gov/cgi-bin/Star/e_table-t.pl", "-d", "matno=276&ShowDefault=on"], check=True)
    num = r"\d\.\d{3}E[-+]\d{2}"
    rows = re.findall(rf"({num})\s+({num})\s+({num})\s+({num})\s+({num})\s+({num})\s+({num})", strip_tags(f.read_text()))
    assert len(rows) > 50, len(rows)
    with open(DATA / "estar_water.csv", "w") as out:
        out.write("# NIST ESTAR, electrons in liquid water (density 1.0 g/cm3, I = 75 eV).\n"
                  f"# Source: https://physics.nist.gov/cgi-bin/Star/e_table-t.pl (matno=276), retrieved {today}.\n"
                  "energy_MeV,collision_stopping_MeVcm2_g,radiative_stopping_MeVcm2_g,total_stopping_MeVcm2_g,"
                  "csda_range_g_cm2,radiation_yield,density_effect\n")
        for r in rows:
            out.write(",".join(r) + "\n")


def nist_muen(tmp: Path, today: str) -> None:
    f = curl("https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html", tmp / "muen.html")
    num = r"\d\.\d{5}E[-+]\d{2}|\d\.\d{3}E[-+]\d{2}"
    toks = re.findall(num, strip_tags(f.read_text()))
    rows = [toks[i:i + 3] for i in range(0, len(toks) - len(toks) % 3, 3)]
    assert all(len(r[0].split("E")[0]) == 7 for r in rows), "unexpected table layout"
    with open(DATA / "nist_muen_water.csv", "w") as out:
        out.write("# NIST X-Ray Mass Attenuation Coefficients (Hubbell & Seltzer), water, liquid.\n"
                  f"# Source: https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html, retrieved {today}.\n"
                  "energy_MeV,mu_rho_cm2_g,mu_en_rho_cm2_g\n")
        for r in rows:
            out.write(",".join(r) + "\n")


def stabin2000(tmp: Path, today: str) -> None:
    from pypdf import PdfReader

    pdf = curl("https://jnm.snmjournals.org/content/jnumed/41/1/149.full.pdf", tmp / "sk2000.pdf")
    text = "\n".join(p.extract_text() for p in PdfReader(str(pdf)).pages)
    section, energy, out = "electron", None, []
    num = r"\d*\.?\d+"
    row = re.compile(rf"^\s*(I|{num})\s+({num})\s+({num})\s+({num})\s+({num})\s+({num})\s+({num})\s+({num})\s+({num})\s*$")
    for line in text.splitlines():
        if "SmallerSpheresfor Photons" in line or "Smaller Spheres for Photons" in line:
            section = "photon_small"
        if line.strip() == "TABLE3":
            section = "photon_large"
        m = re.search(r"Energy:\s*([\d.]+)\s*MeV", line)
        if m:
            energy = float(m.group(1))
        m = row.match(line)
        if m and energy is not None:
            v = [1.0 if m.group(1) == "I" else float(m.group(1))] + [float(m.group(i)) for i in range(2, 10)]
            mass, radius, egs, mcnp, old, _, _, egs_mcnp, rec = v
            # Internal consistency: the paper's EGS4/MCNP column and recommended = mean.
            if abs(egs / mcnp - egs_mcnp) > 0.015 or abs((egs + mcnp) / 2 - rec) > 0.0015:
                continue
            out.append((section, energy, mass, radius, egs, mcnp, old, rec))
    with open(DATA / "stabin2000_sphere_af.csv", "w") as f:
        f.write("# Absorbed fractions for self-irradiation of unit-density spheres, uniform source.\n"
                "# Stabin MG, Konijnenberg MW. Re-evaluation of absorbed fractions for photons and electrons in\n"
                "# spheres of various sizes. J Nucl Med 2000;41(1):149-160. https://jnm.snmjournals.org/content/41/1/149\n"
                f"# Parsed from the free full-text PDF on {today} by scripts/fetch_reference_data.py. Only rows whose\n"
                "# EGS4/MCNP ratio column and recommended value (= mean of the two codes) are internally consistent are kept.\n"
                "# 'reference_old' is Siegel & Stabin (electrons), MIRD Pamphlet 8 (photon_small) or MIRD 3 (photon_large).\n"
                "table,energy_MeV,mass_g,radius_cm,egs4,mcnp4b,reference_old,recommended\n")
        for r in out:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"stabin2000: kept {len(out)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for fn in (xcom, xcom_tissue, iaea, estar, nist_muen, stabin2000):
            fn(tmp, args.date)
            print("ok", fn.__name__)


if __name__ == "__main__":
    main()
