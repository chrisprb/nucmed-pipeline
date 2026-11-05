"""nucmed: a small, validated nuclear medicine imaging pipeline.

Stages (each one is tested against a known reference):

    phantom    -> digital activity and attenuation maps (IEC-like body, disk, point)
    projector  -> 2D parallel-hole SPECT system matrix with attenuation and
                  depth-dependent collimator blur
    recon      -> MLEM / OSEM
    metrics    -> contrast recovery, background variability, RMSE
    decay      -> Bateman decay chains (e.g. Mo-99 -> Tc-99m generator)
    xsections  -> NIST XCOM water cross-sections, Klein-Nishina / Thomson
    mc         -> Monte Carlo photon transport (Compton, Rayleigh, photoelectric)
                  with forced detection and energy windows
    scatter    -> triple-energy-window scatter estimate
    radionuclide, photon3d, dosimetry
               -> Lu-177 decay data, photon absorbed fractions and point
                  kernels, TAC integration, MIRD sphere S-values
"""

__version__ = "0.1.0"
