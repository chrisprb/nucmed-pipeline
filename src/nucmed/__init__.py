"""nucmed: a small, validated nuclear medicine imaging pipeline.

Stages (each one is tested against a known reference):

    phantom    -> digital activity and attenuation maps (IEC-like body, disk, point)
    projector  -> 2D parallel-hole SPECT system matrix with attenuation and
                  depth-dependent collimator blur
    recon      -> MLEM / OSEM
    metrics    -> contrast recovery, background variability, RMSE
    decay      -> Bateman decay chains (e.g. Mo-99 -> Tc-99m generator)

Planned: Monte Carlo photon transport, scatter correction (TEW), Lu-177 dosimetry.
"""

__version__ = "0.1.0"
