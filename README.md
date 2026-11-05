# nucmed-pipeline

A small nuclear medicine imaging pipeline written from scratch in Python. It goes from emitted photon to reconstructed image, and next to absorbed dose. **Every stage is tested against an analytic or published reference.**

```
phantom ──► SPECT forward model ──► OSEM reconstruction ──► quantification ──► dosimetry
 (IEC-like)  analytic projector, or   MLEM / OSEM with        contrast recovery,   (planned:
             Monte Carlo photon       attenuation and         background           Lu-177 MIRD +
             transport with scatter   resolution recovery     variability, bias    voxel kernels)
```

![IEC demo](results/iec_demo/figure.png)

*Top: a 2D slice through an IEC-like phantom (4:1 sphere-to-background), reconstructed with OSEM (12 iterations × 8 subsets, 10⁶ counts): with no corrections, with attenuation correction (AC), and with AC plus resolution recovery (RR). All images share one intensity scale, so the uncorrected image is faint: without AC, SPECT underestimates absolute activity. Bottom: contrast recovery per sphere, and the contrast–noise trade-off as iterations increase.*

## Result (from `scripts/run_iec_demo.py`, seed 20261105)

Contrast recovery coefficient (CRC) at the final iteration:

| Sphere (mm) | 10 | 13 | 17 | 22 | 28 | 37 | Background CoV |
|---|---|---|---|---|---|---|---|
| OSEM, no corrections | 0.14 | 0.23 | 0.27 | 0.23 | 0.37 | 0.42 | 0.39 |
| OSEM + AC | 0.43 | 0.48 | 0.56 | 0.56 | 0.71 | 0.76 | 0.32 |
| OSEM + AC + RR | 0.45 | 0.61 | 0.70 | 0.71 | 0.83 | 0.85 | 0.16 |

Full per-iteration values and run metadata are in [`results/iec_demo/metrics.json`](results/iec_demo/metrics.json).

**Caveats, stated up front.**
- This table is **one Poisson noise realisation**, so treat each value as ±0.05 or so. When the water μ was corrected from 0.154 to the XCOM value 0.1537 (Stage 2), the same seed produced a different noise draw and values moved by up to 0.05. Stage 2 below uses 10 realisations and reports mean ± SD.
- The AC+RR result is an *inverse crime*: the data were simulated with the same system model used to reconstruct them, so it is an upper bound.
- There is no scatter yet.
- Metrics are 2D NEMA-style approximations, not NU 2 compliant.
- Without corrections, the background CoV includes the cupping artefact (non-uniformity) as well as noise.

## Stage 2: Monte Carlo with scatter (from `scripts/run_mc_demo.py`, seed 20261105)

![MC demo](results/mc_iec/figure.png)

*A photon-transport Monte Carlo (Compton, Rayleigh and photoelectric absorption in water, with cross-sections from NIST XCOM; forced detection; 10 % energy resolution) generates projections of the IEC-like phantom on a 2 mm grid. The Stage 1 reconstructor (4 mm, OSEM + AC + RR, 12 it × 8 subsets) then reconstructs three data sets at the same calibration and 10⁶ expected primary counts, with 10 noise realisations each. The spectrum below ~100 keV is truncated by the 90 keV tracking cut-off.*

In the 20 % photopeak window (126–154 keV), **26 % of detected counts are scattered photons**. Mean ± SD over 10 realisations:

| Data | CRC 10 mm | 13 | 17 | 22 | 28 | 37 | Background bias | Lung / background (truth 0) |
|---|---|---|---|---|---|---|---|---|
| A. analytic projector (inverse crime) | 0.35 ± 0.09 | 0.53 ± 0.05 | 0.70 ± 0.05 | 0.76 ± 0.02 | 0.83 ± 0.02 | 0.87 ± 0.02 | +0.2 % | 0.25 |
| B. MC, primaries only | 0.39 ± 0.05 | 0.60 ± 0.05 | 0.74 ± 0.05 | 0.77 ± 0.03 | 0.83 ± 0.03 | 0.86 ± 0.02 | +0.2 % | 0.25 |
| C. MC with scatter, uncorrected | 0.32 ± 0.06 | 0.48 ± 0.06 | 0.60 ± 0.05 | 0.65 ± 0.04 | 0.69 ± 0.02 | 0.75 ± 0.02 | **+41 %** | 0.32 |

What this shows:
- **Removing the inverse crime costs almost nothing here (A vs B).** An independent forward model, with a finer grid, continuous emission points and its own ray tracing, gives the same CRC within noise. This is because the reconstructor's attenuation and collimator models are correct; the only mismatch is discretisation.
- **Scatter is what hurts (B vs C).** Uncorrected scatter lowers CRC by about 0.1 for the larger spheres, adds 41 % to the background activity estimate, and fills the cold lung insert. Matching the pre-registered expectation, CRC(C) < CRC(A) for every sphere and the bias is positive. Stage 3 corrects it.

Caveats: 2D slice of an axially infinite phantom (this overstates scatter from out-of-slice activity relative to a finite phantom); Rayleigh angles use Thomson without a form factor; the collimator response is the same Gaussian model as the reconstructor (no septal penetration); metrics are 2D NEMA-style.

## Stage 3: TEW scatter correction (from `scripts/run_tew_demo.py`, seed 20261106)

![TEW demo](results/tew/figure.png)

*Triple-energy-window (TEW) scatter estimates are computed from two 3 keV windows beside the 126–154 keV photopeak and used as the additive term in OSEM + AC + RR. The data are the Stage 2 Monte Carlo projections (10⁶ expected primary counts, 10 realisations, independent Poisson noise per window).*

| Condition | Background bias | Scatter estimate / true | CRC 10 mm | 22 mm | 37 mm | Background CoV |
|---|---|---|---|---|---|---|
| 0. no scatter correction | +41.1 ± 1.5 % | – | 0.35 | 0.63 | 0.76 | 0.14 |
| **1. TEW, smoothed** (pre-specified primary) | **−4.5 ± 1.3 %** | 1.14 | 0.39 | 0.81 | 0.95 | 0.17 |
| 2. TEW, unsmoothed | −3.6 ± 1.5 % | 1.14 | 0.33 | 0.74 | 0.91 | 0.25 |
| 3. lower window only, smoothed | −0.9 ± 1.3 % | 1.04 | 0.39 | 0.79 | 0.94 | 0.16 |
| 4. oracle (true scatter) | +0.6 ± 1.4 % | 1.00 | 0.37 | 0.73 | 0.85 | 0.16 |

What this shows:
- **TEW removes most of the scatter bias** (+41 % → −4.5 %), meeting the pre-registered target. It over-corrects, by the amount predicted before the run from the window totals: primary photons spill into the side windows at 10 % energy resolution, so TEW overestimates the scatter by 14 %. Dropping the upper window, which is 94 % primaries here, brings the bias to −0.9 %.
- **Without smoothing, TEW is noisy.** Background CoV rises from 0.17 to 0.25, because the side windows hold only about 6 % of the photopeak counts.
- **Be suspicious of a CRC above the oracle's.** TEW gives a 37 mm CRC of 0.95, *higher* than perfect scatter knowledge (0.85) or scatter-free data (0.86). That is not better imaging. *Exploratory noise-free diagnostics* (`scripts/diag_tew.py` → `results/tew/diagnostics.json`) attribute it to the estimate being too large and too smooth:
  - Rescaling TEW to the true total removes the bias but still gives a CRC of 0.88.
  - Smoothing the *true* scatter the same way also lifts CRC from 0.85 to 0.88.
  - TEW over-estimates more in low-count, peripheral projection bins (+19 %) than in sphere-crossing ones (+10 %), so it subtracts relatively more from the background.
  - The rest of the gap (noise-free 0.905 vs noisy 0.95) appears only with noise. I haven't explained that part yet.
  - Contrast recovery read alone would reward over-correction. Bias against known truth catches it.

## What is validated (`uv run pytest`)

| Module | Check | Reference |
|---|---|---|
| `projector` | ⟨Ax, y⟩ = ⟨x, Aᵀy⟩ | matched projector/backprojector pair |
| `projector` | counts conserved per angle without attenuation | geometry |
| `projector` | point-source centroid s = x cos θ + y sin θ; variance = σ(d)² + Δs²/12 | analytic |
| `projector` | attenuation factor = exp(−μ · chord length) in a uniform disk | analytic chord length |
| `recon` | MLEM preserves total counts every iteration | EM property |
| `recon` | Poisson log-likelihood never decreases | EM monotonicity |
| `recon` | OSEM with 1 subset ≡ MLEM | definition |
| `recon` | AC restores a flat uniform disk; no AC → centre/rim < 0.7 | analytic truth |
| `phantom` | supersampled disk area = πr² | analytic |
| `phantom` | background ROIs inside the body and clear of spheres and lung | geometry |
| `decay` | matrix-exponential Bateman solution = closed-form two-member solution | Bateman equations |
| `decay` | Tc-99m activity peaks at t_max = ln(λd/λp)/(λd−λp) ≈ 23 h | analytic |
| `decay` | transient equilibrium A_d/A_p → b·λd/(λd−λp) | analytic |
| `xsections` | μ(water, 140.5 keV) = 0.1537 cm⁻¹; partials sum to the total | NIST XCOM |
| `xsections` | Klein–Nishina total × electron density within 3 % of XCOM incoherent, and above it | XCOM (binding) |
| `xsections` | KN angular pdf integrates to 1; → Thomson as E → 0 | analytic |
| `mc` | sampled Compton angles fit Klein–Nishina (χ², 10⁶ samples, p > 0.01); same for Thomson | analytic |
| `mc` | direction rotation gives the requested scattering angle; Compton energy formula and 90.7 keV edge | kinematics |
| `mc` | narrow-beam μ from simulated transmission through 10 cm of water within 2 % of XCOM | NIST XCOM |
| `mc` | Woodcock tracking through a water + lung slab gives exp(−ΣμᵢLᵢ) within 2 % | analytic |
| `scatter` | TEW trapezoid exact for a linear spectrum; smoothing preserves counts and wraps in angle | analytic |
| `recon` | OSEM with the exact additive term recovers a contaminated uniform disk (ignoring it: > 15 % high) | analytic truth |
| `mc` | primary-only MC sinogram = analytic projector: total within 2 %, NRMSE < 5 % (observed 0.06 %, 1.2 %) | independent model |

## Run it

```bash
uv sync
uv run pytest                          # 57 tests, ~15 s
uv run python scripts/run_iec_demo.py  # Stage 1, ~2 s on an Apple M5
uv run python scripts/run_mc_demo.py   # Stage 2, 10⁶ photons, ~70 s
uv run python scripts/run_tew_demo.py  # Stage 3, uses Stage 2 output, ~6 s
```

## Layout

```
src/nucmed/
  geometry.py    image grid, acquisition, collimator response model
  phantom.py     IEC-like, uniform disk, point source; ROI masks
  projector.py   sparse system matrix: attenuation + depth-dependent blur
  recon.py       MLEM / OSEM, Poisson log-likelihood
  metrics.py     CRC, background variability, NRMSE
  noise.py       count scaling, Poisson noise
  decay.py       Bateman chains, Mo-99/Tc-99m generator
  xsections.py   NIST XCOM water data, Klein–Nishina, Thomson
  mc.py          Monte Carlo transport: Woodcock tracking, forced detection, energy windows
  scatter.py     TEW scatter estimate, sinogram smoothing
  data/          XCOM table with query provenance
tests/           analytic and property tests
scripts/         reproducible runs that write to results/
```

## Roadmap

See [`DESIGN.md`](DESIGN.md). Done: Monte Carlo photon transport (Stage 2), TEW scatter correction (Stage 3). Next: Lu-177 dosimetry (time–activity curve fitting, MIRD S-values, voxel dose-point-kernel convolution) → optional GPU (JAX/MLX).

## Data and scope

Synthetic phantoms only; no patient data. This is an educational and research implementation, not a medical device. Physical constants marked `[verify]` in the source are working values that have not yet been checked against their primary source.

MIT licence.
