"""Bounded compute-host-only focused CPU qualification with explicit JSON counts.

All science files used by these tests are their own temporary fixtures. This is
not a live-output audit or native/GPU/convergence qualification. Each test module
runs in a fresh CPU-only interpreter; explicit skips never become passes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    'test_signed_prior_math', 'test_chirp_frequency_support', 'test_rjmcmc_configuration',
    'test_amplitude_only_refit', 'test_same_block_association', 'test_identification_gate',
    'test_empty_input_blocks', 'test_reference_waveform_cache', 'test_filtered_worker',
    'test_release_controller', 'test_run_monitor', 'test_round_reporting',
    'test_shared_worker_memory', 'test_rjmcmc_birthdeath',
)
ALLOWED_SKIP_IDS = {
    'test_empty_input_blocks.EmptyInputBlockTests.test_scientific_outputs_match_pre_skip_controller',
    'test_reference_waveform_cache.CachedControllerTests.test_full_scientific_output_matches_uncached_same_block_controller',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def compute_host_only():
    short_host = socket.gethostname().split('.')[0]
    if sys.platform != 'linux' or not (os.environ.get('SLURM_JOB_ID') or short_host in ('cpu1', 'gpu1')):
        raise SystemExit('Run on allocated OzSTAR compute or server cpu1/gpu1, never Mac/login')


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_ids = []

    def addSuccess(self, test):
        self.passed_ids.append(test.id())
        super().addSuccess(test)


def run_case(name, output):
    sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern=name+'.py')
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
    skipped = [dict(test=test.id(), reason=reason) for test, reason in result.skipped]
    unexpected_skips = [row for row in skipped if row['test'] not in ALLOWED_SKIP_IDS]
    fields = dict(name=name, status='passed' if result.wasSuccessful() and not unexpected_skips else 'failed',
        tests=result.testsRun, passed=len(result.passed_ids), skipped=len(skipped),
        passed_ids=result.passed_ids, skipped_tests=skipped, unexpected_skips=unexpected_skips,
        failures=[dict(test=test.id(), traceback=text[-12000:]) for test, text in result.failures],
        errors=[dict(test=test.id(), traceback=text[-12000:]) for test, text in result.errors],
        expected_failures=[test.id() for test, _ in result.expectedFailures],
        unexpected_successes=[test.id() for test in result.unexpectedSuccesses],
        seconds=time.monotonic()-started, completed_utc=now())
    if fields['expected_failures'] or fields['unexpected_successes'] or fields['tests'] == 0:
        fields['status'] = 'failed'
    save(output, fields)
    return 0 if fields['status'] == 'passed' else 1


def main():
    compute_host_only()
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--case-timeout', type=float, default=180.)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.case:
        return run_case(args.case, output)
    if output.exists():
        raise SystemExit('Use a fresh focused-regression JSON path')
    logs = output.with_suffix('.logs')
    logs.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='-1', DWD_GPU_DEVICES='-1', JAX_PLATFORMS='cpu',
        PYTHONDONTWRITEBYTECODE='1')
    # Deliberately unavailable archived comparisons are explicit skips; never
    # follow an inherited path into another run's scientific outputs.
    env.pop('DWD_EMPTY_SKIP_REFERENCE', None)
    env.pop('DWD_REFERENCE_CACHE_BASELINE', None)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    report = dict(status='running', started_utc=now(), results=[], passed=0, skipped=0,
        scope=__doc__, slurm_job_id=os.environ.get('SLURM_JOB_ID'), host=socket.gethostname(),
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(ROOT.rglob('*.py')) if '__pycache__' not in path.parts},
        explicitly_allowed_skip_ids=sorted(ALLOWED_SKIP_IDS))
    save(output, report)
    for name in CASES:
        case_output = logs/(name+'.json')
        log_path = logs/(name+'.log')
        command = [sys.executable, '-B', str(Path(__file__).resolve()), str(case_output), '--case', name]
        started = time.monotonic()
        with log_path.open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=args.case_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5.)
                code = 124
        if case_output.exists():
            row = json.loads(case_output.read_text())
        else:
            row = dict(name=name, status='failed', tests=0, passed=0, skipped=0,
                error='Timed out' if code == 124 else 'Child produced no committed test JSON')
        row.update(exit_code=code, log=str(log_path), command=command,
            wall_seconds=time.monotonic()-started)
        if code:
            row['status'] = 'failed'
        report['results'].append(row)
        report.update(passed=sum(item['passed'] for item in report['results']),
            skipped=sum(item['skipped'] for item in report['results']))
        save(output, report)
        print('FOCUSED_REGRESSION_CASE', json.dumps(row), flush=True)
    report.update(status='passed' if all(row['status'] == 'passed' for row in report['results']) else 'failed',
        completed_utc=now(), tests=sum(row['tests'] for row in report['results']))
    save(output, report)
    print('FOCUSED_REGRESSIONS_FINAL', json.dumps(dict(status=report['status'],
        tests=report['tests'], passed=report['passed'], skipped=report['skipped'],
        output=str(output))), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
