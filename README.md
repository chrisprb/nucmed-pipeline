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
- **Scatter is what hurts (B vs C).** Uncorrected scatter lowers CRC by about 0.1 for the larger spheres, adds 41 % to the background activity estimate, and fills the cold lung insert. Matching the pre-registered expectation, CRC(C) < CRC(A) for every sphere and the bias is positive. Fixing this is Stage 3 (TEW scatter correction); the lower and upper windows are already simulated and saved in `results/mc_iec/projections.npz`.

Caveats: 2D slice of an axially infinite phantom (this overstates scatter from out-of-slice activity relative to a finite phantom); Rayleigh angles use Thomson without a form factor; the collimator response is the same Gaussian model as the reconstructor (no septal penetration); metrics are 2D NEMA-style.

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
| `mc` | primary-only MC sinogram = analytic projector: total within 2 %, NRMSE < 5 % (observed 0.06 %, 1.2 %) | independent model |

## Run it

```bash
uv sync
uv run pytest                          # 53 tests, ~15 s
uv run python scripts/run_iec_demo.py  # Stage 1, ~2 s on an Apple M5
uv run python scripts/run_mc_demo.py   # Stage 2, 10⁶ photons, ~70 s
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
  data/          XCOM table with query provenance
tests/           analytic and property tests
scripts/         reproducible runs that write to results/
```

## Roadmap

See [`DESIGN.md`](DESIGN.md). Done: Monte Carlo photon transport (Stage 2). Next, in order: triple-energy-window (TEW) scatter correction → Lu-177 dosimetry (time–activity curve fitting, MIRD S-values, voxel dose-point-kernel convolution) → optional GPU (JAX/MLX).

## Data and scope

Synthetic phantoms only; no patient data. This is an educational and research implementation, not a medical device. Physical constants marked `[verify]` in the source are working values that have not yet been checked against their primary source.

MIT licence.
