"""Packaging-only fixtures: no Slurm submission, scientific import, HDF5 or GPU use."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "eric_subtraction_all.sh"
PROGRAM = SCRIPT.read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
REQUIRED = ("dwd_observation_full_catalogue.h5",
            "dwd_observation_full_catalogue.orbits.h5", "ldc_dwd_data.h5")


class PortableSlurmLauncherTests(unittest.TestCase):
    def fixture(self, base):
        source = base / "source with spaces"
        data = base / "external inputs"
        runs = base / "fresh runs"
        (source / "script" / "__pycache__").mkdir(parents=True)
        data.mkdir()
        (source / "eric_subtraction_all.py").write_text(
            "import json,os\nfrom pathlib import Path\n"
            "p=Path(os.environ['DWD_SUBTRACTION_OUTPUT']).with_suffix('.status.json')\n"
            "p.write_text(json.dumps({'phase':'complete','complete':True,'controller_pid':os.getpid()}))\n")
        (source / "script" / "helper.py").write_text("FIXTURE_ONLY = True\n")
        (source / "script" / "ignored.h5").write_bytes(b"must not copy")
        (source / "script" / "__pycache__" / "ignored.pyc").write_bytes(b"must not copy")
        for name in REQUIRED:
            (data / name).write_bytes(b"fixture input; no scientific reader opens this")
        return source, data, runs

    def launch(self, paths, **overrides):
        env = os.environ.copy()
        env.update(SLURM_JOB_ID="900123", CUDA_VISIBLE_DEVICES="0,1",
                   DWD_BLOCK_WORKERS="2", PYTHONDONTWRITEBYTECODE="1")
        env.update(overrides)
        return subprocess.run([sys.executable, "-B", "-c", PROGRAM, *map(str, paths)],
                              env=env, capture_output=True, text=True, timeout=15)

    def run_path(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return Path(next(line.split("=", 1)[1] for line in result.stdout.splitlines()
                         if line.startswith("DWD_RUN_DIRECTORY=")))

    def test_fresh_runs_never_reuse_outputs_or_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fixture(Path(folder))
            first = self.run_path(self.launch(paths))
            cache = first / "subtraction.checkpoints"
            cache.mkdir()
            sentinel = cache / "reference_waveforms.h5"
            sentinel.write_bytes(b"old fixture cache")
            second = self.run_path(self.launch(paths))
            self.assertNotEqual(first, second)
            self.assertFalse((second / "subtraction.checkpoints").exists())
            self.assertEqual(sentinel.read_bytes(), b"old fixture cache")

    def test_controller_identity_and_paths_match_exec_target(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fixture(Path(folder))
            run = self.run_path(self.launch(paths))
            launch = json.loads((run / "launch.json").read_text())
            status = json.loads((run / "subtraction.status.json").read_text())
            self.assertEqual(launch["slurm_job_id"], "900123")
            self.assertEqual(launch["pid"], status["controller_pid"])
            self.assertEqual(launch["cwd"], str(run / "source_snapshot"))
            self.assertEqual(launch["configured_workers"], 4)
            self.assertEqual(launch["gpu_devices"], ["0", "1"])
            self.assertFalse(launch["scientific_qualification_claimed"])
            self.assertFalse(launch["inherited_residual_results_backends_cache"])

    def test_inputs_are_links_and_only_source_files_are_copied(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fixture(Path(folder))
            run = self.run_path(self.launch(paths))
            snapshot = run / "source_snapshot"
            for name in REQUIRED:
                self.assertTrue((snapshot / "data" / name).is_symlink())
                self.assertEqual((snapshot / "data" / name).resolve(), (paths[1] / name).resolve())
            self.assertTrue((snapshot / "script" / "helper.py").exists())
            self.assertFalse((snapshot / "script" / "ignored.h5").exists())
            self.assertFalse((snapshot / "script" / "__pycache__").exists())

    def test_missing_gpu_allocation_fails_before_creating_run(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fixture(Path(folder))
            result = self.launch(paths, CUDA_VISIBLE_DEVICES="")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("GPU allocation", result.stderr)
            self.assertFalse(paths[2].exists())

    def test_missing_input_fails_before_creating_run(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fixture(Path(folder))
            (paths[1] / REQUIRED[-1]).unlink()
            result = self.launch(paths)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing input files", result.stderr)
            self.assertFalse(paths[2].exists())

    def test_no_slurm_job_and_no_old_guard_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fixture(Path(folder))
            result = self.launch(paths, SLURM_JOB_ID="")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(paths[2].exists())
        text = SCRIPT.read_text()
        self.assertNotIn("launch.once.json", text)
        self.assertNotIn("/fred/", text)
        self.assertNotIn("/home/", text)
        self.assertIn("--no-requeue", text)


if __name__ == "__main__":
    unittest.main()
