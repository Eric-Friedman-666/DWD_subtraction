# Validation scope and limitations

This publication updates the source snapshot and portable launch packaging only. It does not change any deployed server/OzSTAR source, job, environment, guard or output. Publishing or running CPU fixture tests is not a new scientific test pass.

## What is checked

- The scientific controller and helper source are synchronized from the frozen frequency-dependent 900-second multi-GPU snapshot.
- Signed-log prior support, per-bank chirp bounds/padding, fixed SNR filtering, matching/winner refit, atomic status and worker supervision have focused CPU/mock/temporary-fixture regressions.
- The reusable Slurm template is syntax checked and its preparation/exec logic is exercised only against a fake temporary controller. Those packaging fixtures neither submit Slurm work nor import scientific/GPU runtimes.
- No live scientific HDF5, pickle, reference cache or sampler chain is used for publication checks.
- Data, credentials, local paths and machine-specific deployment receipts are not published.

See [publication_checks.json](publication_checks.json) for the actual checks performed on this update: 124 passed and 11 skipped across 17 selected CPU/mock modules on macOS, plus Python AST parsing and shell syntax checks. This is not a full-suite result. Six monitor fixtures require Linux /proc, three require installed Eryn, and two require optional historical-reference sources. These skips are not passes. Some temporary-fixture teardowns emitted macOS multiprocessing resource-tracker duplicate-cleanup warnings after unittest success; no clean Linux worker-lifecycle claim follows. The monitor implementation and scientific controller were not changed to hide those warnings or platform limitations.

## Preserved scientific limitations

The frequency-dependent branch was used for explicitly accepted exploratory work, not promoted as passing the original high-frequency native-accuracy gate. Earlier bounded tests measured 94.06231905127603% native power removal for the fastest 42.97 mHz source on the 900-second grid, below the original strictly-greater-than-95% requirement. The 3,600-second comparison measured 93.64582767170032%; the earlier 300-second result of 94.06140412376406% also remains below that gate. These historical values are not recomputed for publication.

Prior support checks establish that templates fall inside declared support, not that they were recovered. Overlap-matched unique IDs, aggregate foreground-power reduction, completed sampler files, scheduler success and a terminated empty round do not establish accurate amplitudes, posterior convergence, blind completeness or per-source native subtraction quality.

No validated paired-GPU speedup, 16-workers-per-GPU memory-capacity qualification, or new full-catalogue GPU/native accuracy claim is made here. The prebuilt Linux CUDA helper is unchanged and is not reproducibly rebuilt because its kernel source is unavailable.

The final amplitude refit uses observed data, nonnegative scales and fixed winner shapes without extra Gaussian shrinkage. It cannot undo earlier selection or shape biases. The sigma=0.1 prior is on scaled linear coefficients, not catalogue-centred physical amplitude.

## Fresh-run safety

The public Slurm template uses a unique output directory per submission, a new source snapshot and input symlinks. It does not inherit residuals, reference caches or partial chains, and it never submits replacement jobs. It is separate from historical one-use deployment wrappers. Do not delete a consumed guard or edit a historical receipt to make an old deployment look like a new run.

A submitted job retains its stored batch script. Deploying or pulling this GitHub update does not change a pending/running job; any new submission or cancellation is a separate user action.
