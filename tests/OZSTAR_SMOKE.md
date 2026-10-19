# One-node, four-GPU execution qualification

Run only on the allocated OzSTAR compute node, with the same pinned Python/JAX
environment as formal production. Never run these scientific checks on macOS or
an OzSTAR login node. The parent Slurm wrapper owns account, modules and paths.
It must request exactly one node, four A100 GPUs, and 64 worker CPU cores.

No proxy/SSH setting changes are involved. These fixtures use only new synthetic
files. No production HDF5, pickle, cache, residual or Eryn backend is read.

Set `DWD_GPU_DEVICES` to the four CUDA devices visible within this allocation;
set `DWD_BLOCK_WORKERS=16`, `DWD_WORKER_GPU_MEM_FRACTION=.045`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. **Do not set the controller's global
`XLA_PYTHON_CLIENT_MEM_FRACTION=.045`**: that small pool is assigned only in each
spawned worker's inherited environment, and restored in the parent before it
performs native combined subtraction after the all-reaped round barrier.

From this `source/` directory, with `$PYTHON` denoting the pinned environment:

```sh
"$PYTHON" -B -m unittest discover -s tests -p test_gpu_assignment.py -v
"$PYTHON" -B tests/run_release_regressions.py /NEW_ABSOLUTE_SMOKE_DIR/focused_regressions.json
"$PYTHON" -B tests/check_native_helper.py /NEW_ABSOLUTE_SMOKE_DIR/native_helper.json
"$PYTHON" -B tests/smoke_gpu_assignment.py /NEW_ABSOLUTE_SMOKE_DIR/capacity
"$PYTHON" -B tests/smoke_ozstar_pipeline.py /NEW_ABSOLUTE_SMOKE_DIR/native
```

Use fresh destination directories. The capacity test creates and synchronizes
64 separate real GPU contexts, each after a full-year/900-second real padded
128-template basis. It verifies four physical UUIDs with sixteen live CUDA PIDs
each and at least 1 GiB free per device at simultaneous residency. This is
**not** proof of 64 concurrent max100 RJMCMC fits, arbitrary crowded banks or
full posterior convergence. The separate native controller test uses real
500+1500 Eryn cycles, max100, F16, sigma0.1, fixed saved SNR>7, shared-snapshot
immutability, winner-only refit, strict same-bank identification, round exclusion
and the combined native TD/TDI barrier. It injects four sources over 30 days;
the configured limit is 4x16, but only four nonempty fixture banks exist.

The native gate remains **strictly >95%** foreground-power removal for every
injection. A test can pass operational integration and still fail this quality
gate; its JSON and nonzero exit retain that failure. Previous 300-second native
quality failure is not overwritten by these new results. A successful FD or
capacity check alone is insufficient for a scientific native-accuracy claim.

The focused CPU runner reports actual passed/skipped/failed counts in atomic
JSON, with per-module logs and test IDs. The two unavailable historical-output
comparisons are explicit skips, never passes. All other skips fail qualification.
The runner adds analytic signed/positive prior checks, chirp/grid and current
configuration checks, winner refitting/association/identification, input-empty
and cache behavior, filtered-worker contracts, current-controller and monitor
accounting, round timing/shared-memory barriers and installed-Eryn birth/death
mechanics. It is a focused suite, not every legacy repository test. No archived
or live scientific output is opened by this runner.

## Native helper provenance and rebuild limitation

The exact existing `script/td_full_horner.so` is transported unchanged, hash
`bcca137c714ca5dcbce18cd9a73d2302e595748f90ed03dd93cdf7184d3d15ca`.
It contains sm80 PTX version8.4, is Linux x86-64, and statically links cudart.
The server file and local transport artifact have identical hashes. Its CUDA
source `eric_td_full_horner.cu` was **not found** in the scoped DWD source search.
Therefore no OzSTAR rebuild has been performed or claimed. Dynamic linking and
the exported FFI symbol are checked on the allocated node; only actual kernel
execution by the independent native smoke can establish this binary's tested
runtime compatibility with the pinned JAX0.6.2/CUDA environment.

If the original source is subsequently recovered, a reproducible sm80 rebuild
must use the pinned JAX FFI headers and an OzSTAR-compatible CUDA compiler, with
ordinary IEEE arithmetic (no `--use_fast_math`), for example:

```sh
JAX_FFI_INCLUDE=$("$PYTHON" -c 'import jax.ffi; print(jax.ffi.include_dir())')
nvcc -std=c++17 -O3 --shared -Xcompiler -fPIC \
  --generate-code arch=compute_80,code='[sm_80,compute_80]' \
  -I "$JAX_FFI_INCLUDE" eric_td_full_horner.cu -o td_full_horner.so
```

That is a **conditional recipe, not an executed build**. Record the source,
compiler, headers, flags, output hashes and fresh native smoke result before
replacing the recorded binary. Never rebuild against guessed/reconstructed
source or call a symbol-only admission check a scientific validation.
