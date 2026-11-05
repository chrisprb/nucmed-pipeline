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

### Stage 2: Monte Carlo photon transport (next)
- Photons at 140 keV in water, with photoelectric, Compton (Klein–Nishina sampling) and Rayleigh interactions.
- **Targets, fixed in advance:**
  - The narrow-beam attenuation coefficient estimated from the simulation matches NIST XCOM within 2 %.
  - The sampled Compton angular distribution matches the analytic Klein–Nishina cross-section (χ² test, p > 0.01, at 10⁶ samples).
  - The energy of single-scattered photons matches the Compton formula exactly.
- Use: generate projections with scatter from a *different* model than the reconstructor, to remove the inverse crime from Stage 1.

### Stage 3: scatter correction
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
- NIST XCOM photon cross-section database. [verify URL]

## Changelog
- **2026-11-05:** Project created. Stage 1 implemented: 36 tests pass. First demo run (`scripts/run_iec_demo.py`, seed 20261105, 10⁶ counts, 12 it × 8 subsets): CRC for the 37 mm sphere is 0.44 (no corrections), 0.80 (AC), 0.88 (AC+RR); for the 10 mm sphere 0.08 / 0.38 / 0.48. During development the phantom test found background ROIs sharing edge pixels with the 37 mm sphere; the ROIs were repositioned before the first recorded run.
