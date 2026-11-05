# DESIGN: nucmed-pipeline

**Type:** implementation project (portfolio and teaching), not a hypothesis-testing study. "Pre-registration" here means that the validation target for each stage is written down *before* that stage is implemented, so it can't be quietly loosened afterwards.

**Started:** 2026-11-05.

## Goal

Build an end-to-end nuclear medicine pipeline (phantom → SPECT forward model → statistical reconstruction → quantification → dosimetry) in which every stage is checked against an analytic result or a published reference value.

## Principles

1. Every reported number is produced by a script in `scripts/`, which writes to `results/` along with its seed and environment metadata.
2. Every physics module has at least one test against an analytic or published value, not only "it runs".
3. Physical constants carry `[verify]` until checked against a primary source (NIST XCOM, NNDC/ENSDF, ICRP, MIRD pamphlets). The source is then cited in a code comment and the tag removed.
4. Known limitations (inverse crime, 2D only, no scatter) are stated next to the results, not hidden.
5. Synthetic phantoms only.

## Stages and validation targets

### Stage 1: forward model and OSEM (done 2026-11-05)
- Projector: adjoint identity; count conservation; point-source centroid and width; attenuation factor matches the analytic chord length (≤1 %).
- MLEM/OSEM: count preservation; monotone log-likelihood; 1 subset ≡ MLEM; AC restores uniform disk to within 3 %.
- Demo: CRC rises with sphere size, and is ordered no-corr < AC < AC+RR at matched iterations. Observed in the first run (see changelog).

### Stage 2: Monte Carlo photon transport (done 2026-11-05)
- Photons at 140 keV in water, with photoelectric, Compton (Klein–Nishina sampling) and Rayleigh interactions.
- **Targets, fixed in advance:**
  - The narrow-beam attenuation coefficient estimated from the simulation matches NIST XCOM within 2 %.
  - The sampled Compton angular distribution matches the analytic Klein–Nishina cross-section (χ² test, p > 0.01, at 10⁶ samples).
  - The energy of single-scattered photons matches the Compton formula exactly.
- Use: generate projections with scatter from a *different* model than the reconstructor, to remove the inverse crime from Stage 1.

### Stage 3: scatter correction (next)
- Triple-energy-window (TEW) estimate fed into OSEM as the additive term.
- **Target:** on Monte Carlo data, TEW+AC brings uniform-region bias closer to zero than AC alone, measured against the known activity in the background ROI. Report the remaining bias whatever it is.

### Stage 4: Lu-177 dosimetry
- Fit time–activity curves (mono/bi-exponential), compute cumulated activity, and get organ dose with the MIRD formalism.
- Voxel-level dose by convolving with a dose point kernel.
- **Targets:** S-values and dose point kernels reproduce tabulated reference values within the tolerance those references state. The references must be chosen and verified before implementation and recorded here.

### Stretch
- GPU projector (JAX or MLX) with a CPU/GPU benchmark table.
- A 3D extension.
- A comparison against an established toolkit (STIR, CASToR or GATE) [verify names and licences before use].

## References (all [verify] until checked)
- Shepp LA, Vardi Y. Maximum likelihood reconstruction for emission tomography. IEEE Trans Med Imaging, 1982. [verify]
- Hudson HM, Larkin RS. Accelerated image reconstruction using ordered subsets of projection data. IEEE Trans Med Imaging, 1994. [verify]
- NEMA NU 2: Performance measurements of positron emission tomographs (IEC body phantom, CRC and background variability definitions). [verify edition]
- Berger MJ, Hubbell JH, Seltzer SM, et al. XCOM: Photon Cross Sections Database, NIST Standard Reference Database 8. https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html (queried 2026-11-05; URL verified, author list [verify] against the database's citation page).
- CODATA 2022 recommended values (r_e, m_e c²), https://physics.nist.gov/cuu/Constants (checked 2026-11-05).

## Changelog
- **2026-11-05:** Project created. Stage 1 implemented: 36 tests pass. First demo run (`scripts/run_iec_demo.py`, seed 20261105, 10⁶ counts, 12 it × 8 subsets): CRC for the 37 mm sphere is 0.44 (no corrections), 0.80 (AC), 0.88 (AC+RR); for the 10 mm sphere 0.08 / 0.38 / 0.48. During development the phantom test found background ROIs sharing edge pixels with the 37 mm sphere; the ROIs were repositioned before the first recorded run.
- **2026-11-05 (Stage 2 plan, written before any Stage 2 code):**
  - *Cross-sections.* Water photoelectric, incoherent and coherent mass coefficients are taken from NIST XCOM (queried 2026-11-05, stored with provenance in `src/nucmed/data/xcom_water.csv`) and interpolated log–log. Other materials are water scaled by relative density (lung insert = 0.3). At 140.5 keV XCOM gives μ/ρ = 0.1537 cm²/g, which confirms the Stage 1 working value of 0.154. The constant is updated and its `[verify]` tag removed.
  - *Interactions.* Compton directions are sampled from free-electron Klein–Nishina by rejection; the total incoherent probability comes from XCOM, which includes binding. Rayleigh directions are sampled from the Thomson distribution with **no atomic form factor**. This makes Rayleigh angles too wide; it is a known limitation, and Rayleigh is about 1.8 % of interactions at 140 keV. Photoelectric absorption is handled by implicit capture, i.e. survival weighting. No fluorescence (water: negligible).
  - *Geometry.* 3D photon directions in a phantom extruded infinitely along z, with activity uniform in z. By translation symmetry, tracking photons emitted from one plane and scoring them at any z equals the counts in one detector slice. Woodcock (delta) tracking is used in the heterogeneous density map.
  - *Detection.* Forced detection is used: at emission and at every real collision, the probability of reaching each detector angle along the collimator axis is scored. Collimator blur is the same Gaussian depth model as Stage 1, applied after binning by depth. Energy resolution is 10 % FWHM at 140.5 keV, scaling as √E (a modelling choice). The windows are a 20 % photopeak window (126.45–154.55 keV) plus TEW side windows, which are fixed in Stage 3.
  - *Differences from the Stage 1 reconstructor* (the point is to break the inverse crime): scatter is present; the phantom is rendered on a 2 mm grid while reconstruction uses 4 mm; emission points are continuous within pixels; line integrals use 1 mm steps on the fine map.
  - *Additional tests, fixed now:*
    - (a) Klein–Nishina total cross-section × electron density lies within 3 % of XCOM incoherent at 140.5 keV and above it, since binding reduces it.
    - (b) Woodcock tracking through a two-density slab reproduces exp(−Σ μᵢLᵢ) within 2 %.
    - (c) A primary-only Monte Carlo sinogram agrees with the Stage 1 analytic projector (same collimator, attenuation): total counts within 2 % and NRMSE < 5 % on the uniform disk.
  - *Expectation for the demo, stated before running.* On Monte Carlo data (photopeak window, scatter not corrected), OSEM+AC+RR will give a lower CRC than the Stage 1 inverse-crime value for every sphere, and a positive background bias, because scatter adds counts. The values are reported whatever they are.
- **2026-11-05 (before the main Stage 2 run):** A smoke run with 2·10⁴ photons showed that one Poisson realisation makes small-sphere CRC too noisy to judge the ordering. The demo therefore reconstructs **10 noise realisations** per data set (A analytic, B MC primaries, C MC with scatter) and judges the pre-registered expectation on the means. Main run: 10⁶ photons, 10⁶ expected primary counts, seed 20261105. The smoke-run numbers are discarded and not reported.
- **2026-11-05 (Stage 2 results):** 53 tests pass. All pre-registered Stage 2 targets are met: narrow-beam μ within 2 % of XCOM; KN χ² p > 0.01 at 10⁶ samples; Compton energy exact. Additional checks (a)–(c) also pass; for (c), the MC primary total is within 0.06 % of the analytic projector, with NRMSE 1.2 %. Two bugs were found by tests during development and fixed before the main run. (1) The near-pole branch of the direction rotation fired for |u_z| > 0.99999, treating directions up to about 4.5 mrad off the z-axis as on it. (2) The closed-form KN total cancels catastrophically at k ≲ 10⁻⁴; a series is now used for k < 10⁻³, which does not affect energies ≥ 20 keV. One of my own test assertions was also wrong: forced detection can legitimately score below the tracking cut-off. It was corrected. Main run (`scripts/run_mc_demo.py`, 10⁶ photons, 70 s): photopeak scatter fraction 26.3 %. With 10 realisations, CRC for A (analytic) and B (MC primaries) agree within noise. C (with scatter) is lower for every sphere (37 mm: 0.75 ± 0.02 vs 0.87 ± 0.02) and its background bias is +41 %. **The pre-registered expectation is confirmed.** *Exploratory, not pre-registered:* the lung-insert residual rises from 0.25 to 0.32 of background with scatter.
- **2026-11-05 (Stage 1 numbers revised):** Updating μ_water from 0.154 to 0.1537 changed the Stage 1 single-realisation CRCs by up to 0.05 (e.g. 37 mm AC+RR 0.88 → 0.85; 10 mm 0.48 → 0.45). This is a different Poisson draw from the same seed, not a physics effect. It shows that Stage 1's table carries roughly ±0.05 of noise. The README now shows the re-run values and says this; Stage 2 onwards reports means over realisations.
