"""Allocated-node dynamic-link/FFI-symbol check of the recorded native helper.

This is admission evidence only. The real independent native-controller smoke
must still execute the kernel successfully before any compatibility claim.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    if platform.system() != 'Linux' or not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run on an allocated OzSTAR node, not a login node/Mac')
    root = Path(__file__).resolve().parents[1]
    helper = root/'script/td_full_horner.so'
    expected = 'bcca137c714ca5dcbce18cd9a73d2302e595748f90ed03dd93cdf7184d3d15ca'
    digest = hashlib.sha256(helper.read_bytes()).hexdigest()
    assert digest == expected, (digest, expected)
    assert platform.machine() == 'x86_64'
    library = ctypes.CDLL(str(helper))
    assert library.EricsFullHornerXYZ
    dependencies = subprocess.check_output(['ldd', str(helper)], text=True)
    assert 'not found' not in dependencies
    strings = subprocess.check_output(['strings', str(helper)], text=True)
    assert '.target sm_80' in strings, 'Recorded helper lacks its known sm80 PTX'
    metadata = dict(status='admission_passed_actual_kernel_not_yet_tested',
        path=str(helper), sha256=digest, bytes=helper.stat().st_size,
        exported_symbol='EricsFullHornerXYZ', sm80_ptx_present=True,
        dependencies=dependencies, source_available=False,
        compile_performed=False, actual_gpu_kernel_execution_test_required=True)
    if shutil.which('cuobjdump'):
        metadata['cuda_binary_objects'] = subprocess.check_output(
            ['cuobjdump', '--list-elf', str(helper)], text=True, stderr=subprocess.STDOUT)
    output = Path(sys.argv[1]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output)+'.partial')
    temporary.write_text(json.dumps(metadata, indent=2)+'\n')
    temporary.replace(output)
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
