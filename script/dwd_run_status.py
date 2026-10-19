"""Atomic, non-HDF5 progress snapshots; monitoring never opens live science files."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def write_run_status(path, **fields):
    """Publish one complete snapshot, without making reporting a science dependency."""
    path = Path(path)
    payload = dict(schema_version=1, controller_pid=os.getpid(),
        updated_utc=datetime.now(timezone.utc).isoformat())
    payload.update(fields)
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False) + '\n'
    temporary = path.with_name(f'{path.name}.{os.getpid()}.partial')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(encoded)
        temporary.replace(path)
    except OSError as error:
        print(f'Status snapshot unavailable: {path}: {error}', flush=True)
