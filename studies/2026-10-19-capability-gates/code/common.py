"""Resource checks shared by all jobs. The account permit is private."""
import json
import os
import resource
import sys
import time
from pathlib import Path


def check_stop(deadline):
    deadline = min(deadline, float(os.environ.get('MI_STAGE_DEADLINE', 'inf')))
    if time.monotonic() >= deadline:
        raise TimeoutError('Stage time limit reached')
    permit = json.loads(Path(os.environ['MI_USAGE_PERMIT']).read_text())
    if not permit.get('allowed') or time.time() >= permit['expires_at']:
        raise RuntimeError('Usage permit is absent, stopped, or expired')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == 'darwin' else peak * 1024
    if peak_bytes > 4 * 1024**3:
        raise MemoryError('Four GiB process memory limit reached')


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
