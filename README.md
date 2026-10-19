# DWD_subtraction

Catalogue-informed LISA double-white-dwarf (DWD) simulation, SNR calculation and iterative native-TDI subtraction, with F-statistic screening and Eryn RJMCMC.

The October 2026 update publishes the frequency-dependent, multi-GPU pipeline snapshot: signed chirp bounds with matching per-bank padding, a physical-waveform reference cache, recycled workers and mandatory winner-only amplitude refitting. This is exploratory research code, not a blind-search or scientific-accuracy certification. See [validation scope](docs/validation_scope.md).

## Scientific workflow

1. `eric_simulation_DWD.py` generates the original native XYZ observation and companion orbit file.
2. `eric_snr.py` saves the catalogue and its absolute/relative SNRs.
3. `eric_subtraction_all.py` repeats global rounds. Each round freezes one residual/covariance snapshot, fits the remaining catalogue-informed banks, matches recovered components, jointly refits only unique winners using the observed data, and performs one combined native TD/TDI subtraction after every required bank succeeds and every worker exits.
4. The round is finalized only after its output save closes. The run stops when no new identified components remain; full completion follows the final output close.

The saved catalogue relative SNR remains fixed and is selected strictly `SNR_r > 7`. It is never recomputed from the residual. Earlier-finalized IDs are excluded from later rounds. Unique overlap-matched IDs are not a convergence, amplitude-accuracy, physical-detection or blind-completeness claim.

### Published defaults

| Setting | Value |
|---|---|
| Ownership banks | 0.1-50 mHz, 10 microHz wide; 4,990 selected banks |
| Chirp prior | Signed log-magnitude, both signs; nonzero gap retained |
| Global chirp ceiling | Magnitude 4e-12 Hz/s |
| Local bounds/padding | Frequency-dependent per-bank envelope and matching support |
| Detector-response grid | 900 s (15 minutes) |
| F-statistic threshold | 16 |
| RJMCMC | 500 burn-in + 1,500 production cycles; maximum 100 sources |
| Linear-amplitude prior | Zero-mean Gaussian, sigma 0.1 in the scaled four-coefficient basis |
| Final amplitude refit | Winner-only observed-data joint nonnegative fit, no extra Gaussian shrinkage |
| Worker supervision | Rolling single-use processes; 600 s no-forward-progress watchdog; up to 3 total attempts |

The Gaussian is not a 10% Gaussian around each catalogue/template amplitude. Failed winner refits abort before subtraction; there is no unscaled fallback. The reference cache stores physical unweighted XYZ templates, while covariance, weighting, screening and fits are refreshed each round.

## Inputs and environment

Obtain the LDC Sangria catalogue separately and follow its access/use conditions. In an existing compatible Linux/CUDA LISA environment, the sequential preparation commands are:

```bash
python3 eric_simulation_DWD.py
python3 eric_snr.py
```

Scripts use paths/configuration near their top. Preparation can replace its configured output; preserve existing observations before rerunning it. Default observation duration is 360 days at 0.1 Hz.

Subtraction needs these externally prepared files:

```text
data/dwd_observation_full_catalogue.h5
data/dwd_observation_full_catalogue.orbits.h5
data/ldc_dwd_data.h5
```

`requirements.txt` records the deployed Linux/CUDA package versions, not a newly qualified clean installation. A compatible stream-based `lisainstrument.Instrument.export_numpy_full()` installation is required.

The existing `script/td_full_horner.so` is a Linux x86-64 CUDA/JAX-FFI binary, not a macOS library. Its kernel build source is not included; this update does not rebuild it. Its SHA-256 and limitations are recorded in `script/td_full_horner.provenance.json`; third-party notices remain in `script/LICENSES/`.

## Run on OzSTAR with Slurm

Activate your compatible environment, change into this checkout, and submit the reusable template:

```bash
sbatch --account=YOUR_PROJECT eric_subtraction_all.sh
```

The template requests one node, four A100 GPUs, 64 CPUs, 256 GiB and 48 hours on `milan-gpu`. It configures 16 rolling worker slots per GPU (64 logical slots), not 64 continuously alive processes. Adjust resource requests to your allocation and memory budget before submission. This repository update is not a measured 64-way capacity or speedup qualification.

Optional environment settings, supplied before submission:

```bash
export DWD_PYTHON=/path/to/compatible/environment/bin/python
export DWD_DATA_DIR=/path/to/prepared/inputs
export DWD_RUNS_ROOT=/path/to/new/run/directories
export DWD_BLOCK_WORKERS=16  # per allocated GPU
```

The script uses Slurm's `CUDA_VISIBLE_DEVICES`, creates a unique fresh run directory and source snapshot, links only the three prepared inputs, records `launch.json`, and executes the controller. No residual, worker result, chain/backend or reference cache is inherited. It never calls `sbatch` recursively. A new submission gets a new run; it does not resume an interrupted run.

This template does not reference an old machine-specific `formal.py` deployment or its consumed one-time launch guard. Do not reset old guards to reuse a previous release. Updating GitHub does not modify, repair, cancel or replace a script already submitted to Slurm.

For an already allocated single-GPU shell, choose a **new, empty** output directory, keep the three inputs under the checkout's `data/`, set `DWD_SUBTRACTION_OUTPUT` and `DWD_GPU_DEVICES`, and run `python3 eric_subtraction_all.py`. Do not run production on a login node. The controller's standalone default is three worker slots per GPU unless overridden.

## Outputs and safe monitoring

Each fresh Slurm run contains a source snapshot, `launch.json`, `subtraction.status.json`, per-bank settings, `subtraction.h5` and its own checkpoint/cache directory. The launch record is not completion: verify the atomic pipeline status and Slurm exit state.

Use scheduler metadata, small atomic JSON, bounded own logs and checkpoint **filenames** while a run is live. Do not open live HDF5/pickle/cache/backend payloads to reconstruct progress. Nonempty-bank completion is separate from input-empty skips, completed RJMCMC fits, winner refit, native subtraction, round finalization and full output closure. GPU allocation is not current utilization telemetry.

`script/monitor_dwd_run.py` is a local compute-host diagnostic for its matching single-GPU launch schema; do not run it on an OzSTAR login node as a multi-GPU monitor.

## Checks and documentation

Examples of CPU-only current-contract checks using generated temporary fixtures:

```bash
python3 -m unittest discover -s tests -p 'test_signed_prior_math.py' -v
python3 -m unittest discover -s tests -p 'test_frequency_dependent_chirp.py' -v
python3 -m unittest discover -s tests -p 'test_release_controller.py' -v
python3 -m unittest discover -s tests -p 'test_portable_slurm_launcher.py' -v
bash -n eric_subtraction_all.sh
```

Some retained tests cover historical output schemas or require optional isolated reference fixtures. Passing focused CPU/mock regressions does not mean a fresh native/GPU recovery benchmark passed. The compute-host-only qualification runner retains its host guard; GPU smoke scripts require separate explicitly authorized allocated resources and must not run against production outputs. Machine-specific archived native replays and hard-coded GPU smoke launchers are not redistributed.

The [publication checks](docs/publication_checks.json) record 124 passed and 11 explicitly skipped tests across 17 selected CPU/mock modules; they are not a full-suite or GPU qualification result.

The PDF/TeX walkthrough and `docs/subtraction_workflow.md` are retained as historical documentation, not a review of all current features. This README, [change log](CHANGELOG.md) and [validation scope](docs/validation_scope.md) describe the current publication.

External catalogues, observations, reference caches, results, sampler chains, plots, credentials, local environments and private deployment receipts are excluded from Git.
