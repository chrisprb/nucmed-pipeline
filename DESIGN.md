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

### Stage 3: scatter correction (done 2026-11-05)
- Triple-energy-window (TEW) estimate fed into OSEM as the additive term.
- **Target:** on Monte Carlo data, TEW+AC brings uniform-region bias closer to zero than AC alone, measured against the known activity in the background ROI. Report the remaining bias whatever it is.

### Stage 4: Lu-177 dosimetry (done 2026-11-05; sphere level; see changelog)
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
- **2026-11-05 (Stage 3 plan, written before any Stage 3 code):**
  - *Estimator.* TEW per projection bin is Ŝ = (C_L/W_L + C_U/W_U) · W_P / 2, with W_L = W_U = 3 keV and W_P = 28.1 keV (windows as simulated in Stage 2). Ŝ is used as the known additive term r in OSEM + AC + RR (Stage 1 system matrix), with y ~ Poisson(Ax + r).
  - *Smoothing (fixed now).* The side windows hold only ~6 % of photopeak counts, so raw Ŝ is very noisy. The primary method smooths Ŝ with a 2D Gaussian, σ = 2 detector bins (0.8 cm) and σ = 2 angular steps (6°), periodic in angle.
  - *Data.* The Stage 2 Monte Carlo sinograms (`results/mc_iec/projections.npz`) at the Stage 2 calibration (10⁶ expected primary photopeak counts). There are independent Poisson draws per window and 10 realisations, with a seed scheme separate from Stage 2.
  - *Conditions.*
    - (0) No scatter correction: the same as Stage 2 condition C, recomputed here.
    - (1) TEW, smoothed. **This is the primary comparison.**
    - (2) TEW, unsmoothed.
    - (3) Lower window only (C_U = 0; the dual-window trapezoid with zero at the upper edge).
    - (4) Oracle: the noise-free true Monte Carlo scatter as r. This is a reference for the best a perfect additive estimate can do.
  - *Target (as originally fixed).* |background bias| of (1) < that of (0).
  - *Quantitative prediction, from the noise-free Stage 2 window totals, before any reconstruction.*
    - TEW overestimates the total photopeak scatter by ~14 %, because primaries spill into the side windows (9 % of lower-window counts, 94 % of upper-window counts at 10 % FWHM). The lower-window-only estimate is ~4 % high.
    - So (1) is expected to over-correct, leaving a small *negative* background bias of a few percent, and (3) should be closer to zero.
    - CRC should rise from (0) towards the oracle (4).
  - *Additional tests.* The TEW trapezoid is exact for a spectrum that is linear across the windows. OSEM given the exact additive term on noise-free data recovers a uniform disk as well as OSEM on scatter-free data does.
- **2026-11-05 (Stage 3 results):** 57 tests pass. Main run: `scripts/run_tew_demo.py`, 10 realisations, seed 20261106, about 6 s. No smoke run preceded it.
  - *Clarification.* Condition (3), lower window only, used the same smoothing as (1). The plan didn't say so explicitly; this was decided before the run.
  - *Pre-registered target met.* Background bias is +41.1 ± 1.5 % with no correction and −4.5 ± 1.3 % with smoothed TEW.
  - *Quantitative prediction confirmed.* TEW total / true = 1.139, giving negative bias; lower-only gives 1.038 and bias −0.9 %. The oracle gives +0.6 %.
  - *Unpredicted, exploratory* (`scripts/diag_tew.py`, noise-free).
    - TEW's 37 mm CRC (0.95) exceeds the oracle's (0.85). About half the excess is reproduced noise-free and is explained by an estimate that is too large and too smooth: shape-only rescaling gives CRC 0.88; smoothing the true scatter also gives 0.88; TEW/true is 1.19 in low-count bins vs 1.10 in sphere-crossing bins.
    - The remaining excess appears only with noise and is not explained yet.
    - The oracle's lung residual (0.29) exceeds scatter-free data (0.25) at 12 iterations. The gap shrinks at 48 iterations (0.17 vs 0.15). This is consistent with slower OSEM convergence in cold regions when a positive additive term is present, but not proven.
  - *Lesson recorded.* CRC alone rewards over-correction; report bias against truth alongside it.
- **2026-11-05 (Stage 4 plan, written before any Stage 4 code):**
  - *References chosen and retrieved* by `scripts/fetch_reference_data.py`, which writes `src/nucmed/data/` with provenance headers:
    - Lu-177 decay radiations, beta spectrum and T½ = 6.6443 d: ENSDF evaluation (Kondev, cut-off 2019-08-30) via the IAEA LiveChart API.
    - Electron CSDA ranges in water: NIST ESTAR.
    - Absorbed fractions for unit-density spheres: **Stabin MG, Konijnenberg MW. J Nucl Med 2000;41:149–160** (EGS4 and MCNP4B, monoenergetic electrons and photons). Title, authors, journal and year were verified from the journal page's citation metadata. Only rows passing an internal consistency check are kept: the EGS4/MCNP ratio column, and recommended = mean of the two codes (278 rows).
    - XCOM is extended to 1 keV–1 MeV; the previous 40 rows are unchanged.
  - *Deviation from the original Stage 4 wording.* The planned voxel S-value / dose point kernel reference was the Lanconelli et al. 2012 database (Phys Med Biol 57:517). Its data files require a registered login, and the paper only plots Lu-177 values. So there is no freely checkable Lu-177 voxel kernel, and I am **not** writing an electron transport code to compare against nothing. Instead:
    - (i) Electron absorbed fractions are taken from Stabin & Konijnenberg as reference input data, like XCOM, and are not "validated".
    - (ii) Voxel dosimetry uses local deposition for electrons plus a photon point kernel from our own Monte Carlo.
    - (iii) A Lanconelli comparison can be added later as Stage 4b if the user registers.
  - *Targets, fixed now:*
    - **T4.1 Decay data.** The mean of the IAEA beta spectrum equals the intensity-weighted ENSDF branch mean within 1 %. The spectrum integrates to the total β⁻ intensity within 1 %.
    - **T4.2 Photon absorbed fractions.** Our 3D photon Monte Carlo (water, XCOM, analog, kerma approximation: electron energy deposited at the interaction point) reproduces Stabin & Konijnenberg's recommended photon AFs. The criterion is |ours/recommended − 1| ≤ max(5 %, |EGS4/MCNP − 1|), i.e. the references' own inter-code spread with a 5 % floor. It is checked at every tabulated energy in 20–208 keV and every mass in the tables.
    - **T4.3 Small-sphere limit.** For μR ≪ 1, the photon AF → μ_en·(3R/4), with μ_en from NIST (X-ray mass energy-absorption table, water), within 5 %.
    - **T4.4 Electron sanity.** A straight-line CSDA model (ESTAR ranges) gives AFs ≤ the reference values at every tabulated energy ≤ 1 MeV and mass: it is a lower bound, because real paths are tortuous. Its escape probability for a monoenergetic, constant-stopping-power case matches the analytic chord result 3L/4R − L³/16R³.
    - **T4.5 Point kernel.** The photon point-kernel convolution over a uniform sphere reproduces the directly simulated sphere AF within 3 %. The kernel's energy integral in a large medium accounts for the emitted photon energy (≥ 99 % within 1 m).
    - **T4.6 TAC fitting.** Noise-free mono- and bi-exponential data return Ã to 10⁻⁶. Analytic Ã = A₀/λ_eff.
  - *Reported, with no external pass/fail possible.* Lu-177 S-values for unit-density spheres (0.01 g–6 kg), split into electrons and photons. Below 0.1 MeV, where the reference has no electron values, the electron AF is interpolated linearly to 1 at E = 0, and the effect of that assumption is reported as a bound. Also reported: a synthetic time–activity example comparing integration methods and sampling schedules. All kinetics are synthetic, no patient data, and this part is descriptive.
- **2026-11-05 (Stage 4 results):** 106 tests pass. Validation run: `scripts/run_dosimetry_demo.py`, seed 20261107, 2·10⁵ photons per table row, 86 s.
  - **T4.1 met.** Spectrum mean 132.63 keV vs branch mean 133.64 keV (0.76 %); integral 1.00006 per decay.
  - **T4.3 met.** **T4.4 met** (95/95 rows). **T4.5 met.** **T4.6 met.**
  - **T4.2 partially met: 59/112 rows** with water, sphere embedded in water. All 37 rows at 100–160 keV pass; most rows at 30–60 keV fail, with ours 5–16 % high.
  - *What happened during development.*
    - (1) The first photon model was an isolated sphere. The guard test failed, ours 5–19 % low at ≥ 60 keV. On re-reading, the reference's methods state spheres *"in an infinite unit density medium"*. That was my reading error, so the geometry was changed to match the reference. The tolerance was not changed.
    - (2) With the correct geometry, ours is high at low energy. The reference states that it used the tissue compositions of MIRD 3/8.
  - *Exploratory composition check, not pre-registered:* XCOM ICRU four-component soft tissue, embedded geometry. 104/112 rows pass, all rows ≥ 60 keV pass, median ratio 1.03. This supports composition as the main cause at low energy. It is not the exact MIRD composition, so the residual at 30–40 keV is left unexplained.
  - *Test-suite changes made after seeing failures, all disclosed.*
    - The unit test for photon AFs is now a 20 % regression guard. The pre-registered criterion is evaluated on all rows by the demo script instead.
    - T4.3 explicitly uses the isolated sphere, which is the geometry where the analytic limit holds.
    - T4.5 compares against the embedded sphere.
    - A units test I had written expected ≥ 99 % electron absorption in 1 kg; the reference itself gives 0.988–0.999, so it now checks 98–100 %.
  - *Lu-177 S-values reported* (0.01 g: 2.01; 10 g: 2.34·10⁻³; 1 kg: 2.45·10⁻⁵ mGy per MBq·s). The low-energy electron assumption moves S by < 0.5 %. The photon share is ≤ 4.6 %, so the T4.2 water-vs-tissue issue affects S by < 0.3 %. No external Lu-177 S-value check was possible (see the Stage 4 plan entry).
  - *TAC example (descriptive, synthetic):* trapezoid + physical tail +26 % / +18 % (4 / 3 points); trapezoid + fitted tail +0.9 % / −6.3 %; mono-exponential fit +3.1 %; uptake–washout fit 0.0 % (SD 2.8 %, 4 points).
