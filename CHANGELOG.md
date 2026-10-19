# Change log

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
