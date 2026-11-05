# nucmed-pipeline

A small nuclear medicine imaging pipeline written from scratch in Python. It goes from emitted photon to reconstructed image, and next to absorbed dose. **Every stage is tested against an analytic or published reference.**

```
phantom ──► SPECT forward model ──► OSEM reconstruction ──► quantification ──► dosimetry
 (IEC-like)  attenuation + depth-     MLEM / OSEM with        contrast recovery,   (planned:
             dependent collimator     attenuation and         background           Lu-177 MIRD +
             blur + Poisson noise     resolution recovery     variability          voxel kernels)
```

![IEC demo](results/iec_demo/figure.png)

*Top: a 2D slice through an IEC-like phantom (4:1 sphere-to-background), reconstructed with OSEM (12 iterations × 8 subsets, 10⁶ counts): with no corrections, with attenuation correction (AC), and with AC plus resolution recovery (RR). All images share one intensity scale, so the uncorrected image is faint: without AC, SPECT underestimates absolute activity. Bottom: contrast recovery per sphere, and the contrast–noise trade-off as iterations increase.*

## Result (from `scripts/run_iec_demo.py`, seed 20261105)

Contrast recovery coefficient (CRC) at the final iteration:

| Sphere (mm) | 10 | 13 | 17 | 22 | 28 | 37 | Background CoV |
|---|---|---|---|---|---|---|---|
| OSEM, no corrections | 0.08 | 0.17 | 0.25 | 0.27 | 0.38 | 0.44 | 0.42 |
| OSEM + AC | 0.38 | 0.43 | 0.52 | 0.63 | 0.72 | 0.80 | 0.32 |
| OSEM + AC + RR | 0.48 | 0.54 | 0.63 | 0.78 | 0.82 | 0.88 | 0.18 |

Full per-iteration values and run metadata are in [`results/iec_demo/metrics.json`](results/iec_demo/metrics.json).

**Caveats, stated up front.**
- The AC+RR result is an *inverse crime*: the data were simulated with the same system model used to reconstruct them, so it is an upper bound.
- There is no scatter yet.
- Metrics are 2D NEMA-style approximations, not NU 2 compliant.
- Without corrections, the background CoV includes the cupping artefact (non-uniformity) as well as noise.

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

## Run it

```bash
uv sync
uv run pytest                          # 36 tests, ~10 s
uv run python scripts/run_iec_demo.py  # ~2 s on an Apple M5
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
tests/           analytic and property tests
scripts/         reproducible runs that write to results/
```

## Roadmap

See [`DESIGN.md`](DESIGN.md). In order: Monte Carlo photon transport, used to break the inverse crime and to add scatter → triple-energy-window (TEW) scatter correction → Lu-177 dosimetry (time–activity curve fitting, MIRD S-values, voxel dose-point-kernel convolution) → optional GPU (JAX/MLX).

## Data and scope

Synthetic phantoms only; no patient data. This is an educational and research implementation, not a medical device. Physical constants marked `[verify]` in the source are working values that have not yet been checked against their primary source.

MIT licence.
