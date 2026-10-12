# DWD_subtraction

LISA double-white-dwarf (DWD) simulation, SNR calculation, and iterative subtraction using F-statistic screening and Eryn RJMCMC.

## Workflow

Run from the repository directory in the existing Linux/CUDA `lisa_pipeline` environment:

```bash
conda activate lisa_pipeline
python3 eric_simulation_DWD.py
python3 eric_snr.py
python3 eric_subtraction_all.py
```

These commands are sequential, not concurrent. Settings and file paths are defined at the top of each script. The default simulation spans 360 days at 0.1 Hz. Running a script can replace its configured output file; preserve any result you need before rerunning it.

1. `eric_simulation_DWD.py`: simulate continuous ACC+OMS noise, sum the complete DWD catalogue on the GPU, combine instrumental measurements, form second-generation XYZ TDI, and save the time-domain observation and companion orbit file.
2. `eric_snr.py`: calculate absolute and relative SNRs and save the eleven-column DWD table.
3. `eric_subtraction_all.py`: repeat global rounds over the selected frequency blocks. Estimate effective covariance from the residual; screen the recorded templates with the F-statistic; perform joint RJMCMC fits; combine accepted sources; subtract their time-domain XYZ signal; and stop when no new sources are accepted.

Every block in one global round uses the same residual snapshot. The residual changes only after the block sweep. The candidate bank is catalogue-informed and filtered by saved relative SNR; this is not a blind search or a guarantee that every recoverable source is found.

`eric_subtraction_1block.py` and `oneblock_experiment.py` retain the earlier single-block workflow for reference and regression coverage. Use `eric_subtraction_all.py` for the current global workflow.

## Inputs and outputs

Place the separately obtained LDC Sangria catalogue at:

```text
data/LDC2_sangria_training_v2.h5
```

The pipeline creates:

```text
data/dwd_observation_full_catalogue.h5
data/dwd_observation_full_catalogue.orbits.h5
data/ldc_dwd_data.h5
data/dwd_subtraction_all.h5
```

The catalogue, observations, sampler checkpoints, generated plots, temporary workspaces, credentials, and local environment files are not included in this repository.

## Dependencies and GPU helper

Production execution requires a CUDA-capable NVIDIA GPU and the native LISA packages. `requirements.txt` records package versions from the working server environment. It is not a fully locked or newly qualified clean-environment installation.

The LISA installation must provide the stream-based `lisainstrument.Instrument.export_numpy_full()` API used by the scripts. Preserve the compatible native LISA installation when reproducing the working server environment.

The original `script/td_full_horner.so` is included because the SNR waveform helper loads it through JAX FFI. It is a prebuilt Linux x86-64 CUDA helper for the working environment, not a portable macOS binary. Its kernel build source is not part of this snapshot. Third-party redistribution notices are preserved under `script/LICENSES/`.

## Checks and documentation

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

The upload snapshot passed 43 lightweight regression tests. These include mocked workflow branches and numerical/HDF5 checks; they do not constitute a new full-catalogue GPU simulation or an end-to-end recovery benchmark. GPU smoke scripts are supplied separately and require the LISA/CUDA environment and their documented input files.

The detailed walkthrough is in [the code-review PDF](docs/DWD%20injection%20and%20subtraction%20workflow.pdf), with its matching TeX source in `docs/`. The older single-block notes are retained as `docs/subtraction_workflow.md`.
