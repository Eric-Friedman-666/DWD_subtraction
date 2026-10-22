#!/bin/bash
#SBATCH --job-name=dwd-subtraction
#SBATCH --partition=milan-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --no-requeue
#SBATCH --output=dwd-subtraction-%j.log

# Submit from the checkout: sbatch --account=YOUR_PROJECT eric_subtraction_all.sh
# This reusable template does not reference historical deployment guards.
set -euo pipefail
umask 027
: "${SLURM_JOB_ID:?Submit this script with sbatch; do not run it on a login node.}"
source_dir="${DWD_SOURCE_DIR:-${SLURM_SUBMIT_DIR:?}}"
data_dir="${DWD_DATA_DIR:-${source_dir}/data}"
runs_dir="${DWD_RUNS_ROOT:-${source_dir}/output/runs}"
python_executable="${DWD_PYTHON:-python3}"
export TMPDIR="${source_dir}/output/tmp" XDG_CACHE_HOME="${source_dir}/output/cache"
export CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda" MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset XLA_PYTHON_CLIENT_MEM_FRACTION
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DWD_BLOCK_WORKERS="${DWD_BLOCK_WORKERS:-16}"
export DWD_WORKER_PROGRESS_TIMEOUT="${DWD_WORKER_PROGRESS_TIMEOUT:-600}"
export DWD_WORKER_MAX_ATTEMPTS="${DWD_WORKER_MAX_ATTEMPTS:-3}"
export DWD_WORKER_GPU_MEM_FRACTION="${DWD_WORKER_GPU_MEM_FRACTION:-.045}"

exec srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" --cpu-bind=threads \
    "$python_executable" -u -B - "$source_dir" "$data_dir" "$runs_dir" <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile

source, inputs, destination = (Path(p).expanduser().resolve() for p in sys.argv[1:4])
job = os.environ.get("SLURM_JOB_ID", "")
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if not job.isdecimal():
    raise SystemExit("A valid SLURM_JOB_ID is required.")
if not visible or visible in ("-1", "NoDevFiles"):
    raise SystemExit("CUDA_VISIBLE_DEVICES must be supplied by the GPU allocation.")
if not (source / "eric_subtraction_all.py").is_file() or not (source / "script").is_dir():
    raise SystemExit("DWD_SOURCE_DIR must be a pipeline checkout.")
required = ("dwd_observation_full_catalogue.h5",
            "dwd_observation_full_catalogue.orbits.h5", "ldc_dwd_data.h5")
missing = [name for name in required if not (inputs / name).is_file()]
if missing:
    raise SystemExit("Missing input files: " + ", ".join(missing))
destination.mkdir(parents=True, exist_ok=True)
run = Path(tempfile.mkdtemp(prefix=f"subtraction_{job}_", dir=destination))
snapshot = run / "source_snapshot"
snapshot.mkdir()
files = list(source.glob("*.py"))
allowed = {".py", ".so", ".json", ".md", ".txt", ".cu", ".h", ".hpp"}
for path in (source / "script").rglob("*"):
    relative = path.relative_to(source)
    if path.is_file() and not any(p.startswith(".") or p == "__pycache__"
                                  for p in relative.parts) and path.suffix in allowed:
        files.append(path)
hashes = {}
for path in sorted(files):
    relative = path.relative_to(source)
    target = snapshot / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
(snapshot / "data").mkdir()
input_metadata = {}
for name in required:
    original = (inputs / name).resolve(strict=True)
    (snapshot / "data" / name).symlink_to(original)
    stat = original.stat()
    input_metadata[name] = {"path": str(original), "bytes": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns}
output = run / "subtraction.h5"
os.environ["DWD_SUBTRACTION_OUTPUT"] = str(output)
os.environ["DWD_GPU_DEVICES"] = visible
os.chdir(snapshot)
stat_file = Path("/proc/self/stat")
ticks = stat_file.read_text().rpartition(")")[2].split()[19] if stat_file.exists() else None
record = {
    "schema_version": 1, "status": "launching",
    "slurm_job_id": job, "run": str(run), "cwd": str(snapshot),
    "host": socket.gethostname(), "pid": os.getpid(), "controller_start_ticks": ticks,
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "gpu_devices": visible.split(","),
    "gpu_assignment_scope": "Slurm CUDA_VISIBLE_DEVICES; not physical utilization telemetry",
    "block_workers_per_gpu": int(os.environ["DWD_BLOCK_WORKERS"]),
    "configured_workers": int(os.environ["DWD_BLOCK_WORKERS"]) * len(visible.split(",")),
    "response_step_seconds": 900,
    "output": str(output), "progress_snapshot": str(output.with_suffix(".status.json")),
    "cache_progress_snapshot": str(output.with_suffix(".checkpoints") / "reference_waveforms.status.json"),
    "source_hashes": hashes, "input_metadata": input_metadata,
    "fresh_start": True, "inherited_residual_results_backends_cache": False,
    "scientific_qualification_claimed": False,
}
with (run / "launch.json").open("x") as stream:
    json.dump(record, stream, indent=2, allow_nan=False)
    stream.flush()
    os.fsync(stream.fileno())
print("DWD_RUN_DIRECTORY=" + str(run), flush=True)
print("Launch metadata is not completion; inspect the atomic pipeline status and Slurm exit.", flush=True)
os.execv(sys.executable, [sys.executable, "-u", "-B", "eric_subtraction_all.py"])
PY
