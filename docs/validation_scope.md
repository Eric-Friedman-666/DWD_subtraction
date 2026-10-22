# Validation scope and limitations

The 22 October 2026 publication synchronizes the active OzSTAR source and portable launch packaging. It does not change deployed source, jobs, environments, guards or outputs. No scientific controller, sampler or GPU/native test is executed for publication. The current source is not the frozen 20 October release.

## What is checked

- All scientific Python controller/helper files are synchronized byte-for-byte from the active deployment's `source/` tree, mapped to the public repository root. File hashes and the layout mapping are in [source_snapshot.json](source_snapshot.json).
- Python source is parsed without imports or execution. Shell syntax and launcher-only temporary-controller fixtures are checked, including a single allocated device and fresh run identity. No scientific tests are run.
- The reusable Slurm template is syntax checked and its preparation/exec logic is exercised only against a fake temporary controller. Those packaging fixtures neither submit Slurm work nor import scientific/GPU runtimes.
- No live scientific HDF5, pickle, reference cache or sampler chain is used for publication checks.
- Data, credentials, local paths and machine-specific deployment receipts are not published.

See [publication_checks.json](publication_checks.json) for the checks actually performed on this update. The previous 124-pass/11-skip report is preserved separately in [publication_checks_20261020.json](publication_checks_20261020.json); it is not a test result for the new source. Retained scientific and historical tests have not been requalified and may require adaptation to the current source/schema.

This synchronization preserves current behavior, including removal of several defensive input, cache-shape, amplitude-refit solver/KKT/objective and worker-argument checks compared with the previous publication. It does not silently restore those checks or treat their removal as a validated improvement. The supervisor now uses per-attempt status filenames and exits on unrecoverable errors. No fresh numerical accuracy, retry exhaustion, worker cleanup or failure-path qualification is claimed.

## Preserved scientific limitations

The frequency-dependent branch was used for explicitly accepted exploratory work, not promoted as passing the original high-frequency native-accuracy gate. Earlier bounded tests measured 94.06231905127603% native power removal for the fastest 42.97 mHz source on the 900-second grid, below the original strictly-greater-than-95% requirement. The 3,600-second comparison measured 93.64582767170032%; the earlier 300-second result of 94.06140412376406% also remains below that gate. These historical values are not recomputed for publication.

Prior support checks establish that templates fall inside declared support, not that they were recovered. Overlap-matched unique IDs, aggregate foreground-power reduction, completed sampler files, scheduler success and a terminated empty round do not establish accurate amplitudes, posterior convergence, blind completeness or per-source native subtraction quality.

No validated paired-GPU speedup, 16-workers-per-GPU memory-capacity qualification, or new full-catalogue GPU/native accuracy claim is made here. The prebuilt Linux CUDA helper is unchanged and is not reproducibly rebuilt because its kernel source is unavailable.

The final amplitude refit uses observed data, nonnegative scales and fixed winner shapes without extra Gaussian shrinkage. It cannot undo earlier selection or shape biases. The sigma=0.1 prior is on scaled linear coefficients, not catalogue-centred physical amplitude.

## Fresh-run safety

The public Slurm template uses a unique output directory per submission, a new source snapshot and input symlinks. It does not inherit residuals, reference caches or partial chains, and it never submits replacement jobs. It is separate from historical one-use deployment wrappers. Do not delete a consumed guard or edit a historical receipt to make an old deployment look like a new run.

A submitted job retains its stored batch script, but this launcher copies mutable source only when it actually starts. A later change to that deployment source could therefore change a queued job's source snapshot. This GitHub publication makes no such deployment changes. New submissions, cancellations and deployments are separate user actions.
