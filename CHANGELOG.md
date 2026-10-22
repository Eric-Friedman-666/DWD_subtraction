# Change log

## 2026-10-22

- Synchronize the complete active OzSTAR scientific Python source, including simulation/SNR entrypoints, global and single-bank workflows, sampler/waveform helpers, reference cache, status writer and worker supervisor. Preserve the public repository's flat source layout.
- Publish the current main-guarded controller initialization and native three-channel residual handling; preserve the source's current validation/error-handling behavior without introducing publication-only scientific fixes.
- Synchronize matching deployed test modules. Retain other portable/historical tests without claiming they all pass for this snapshot.
- Update the portable launcher to the deployed seven-day resource limit and CUDA/cache environment defaults. Keep four A100 GPUs as its default and document a submission-only one-GPU override.
- Record file-level source provenance and fresh static/launcher-only checks. Retain the previous check report separately, preserve the unchanged prebuilt CUDA helper and third-party notices, and exclude all data, environments and run products.
- Do not modify OzSTAR deployment source, queued/running jobs or their outputs; no scientific or GPU validation was executed for this publication.

Scientific controller SHA-256: d03fe8a35c40e249c083b57b55a3835492a74bffd5b7640efd3a4ca0fed01ad7.

## 2026-10-20

- Publish the frozen frequency-dependent signed-log-magnitude chirp policy and matching per-bank padding, preserving ownership edges, the nonzero gap and the 4e-12 Hz/s global ceiling.
- Update the global controller to fixed saved SNR_r strictly >7, F=16, 500+1500 cycles and maximum 100 sources.
- Include winner-only observed-data nonnegative amplitude refits before combined native subtraction.
- Include physical unweighted v4 reference caching, per-round refreshes, nonempty-bank skips and atomic finalized-round/ID status.
- Include rolling recycled multi-GPU workers, staggered per-device initialization, the 600-second no-progress watchdog and up to 3 total fresh attempts.
- Add a reusable fresh-run Slurm template with no historical paths or consumed deployment guards, plus packaging-only temporary-controller tests.
- Add current focused regressions and preserve the prebuilt helper/third-party notices. Exclude machine-specific archived replay fixtures, private inputs and generated scientific outputs.
- Keep historical PDF/TeX walkthroughs but label their scope; preserve failed/incomplete scientific qualification limits.

Scientific controller SHA-256: e125f9555c95255ab214cdb993bac548ce908e4d57bbcfbba1d64d27d9dbd8b1.
Chirp-policy source SHA-256: 34afa82058570b71b7ac68706709a28c8d18a2115c2a401da505941490d759af.
Unchanged native-helper SHA-256: bcca137c714ca5dcbce18cd9a73d2302e595748f90ed03dd93cdf7184d3d15ca.
