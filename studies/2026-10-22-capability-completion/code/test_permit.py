"""Dummy file-concurrency and fail-closed checks. No model is trained."""
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from permit import replace_permit

OLD = Path(__file__).resolve().parents[2] / '2026-10-19-capability-gates/code'
sys.path.insert(0, str(OLD))
from common import check_stop


def check():
    start = time.monotonic()
    errors, reads = [], [0]
    done = threading.Event()
    old_env = {k: os.environ.get(k) for k in ('MI_USAGE_PERMIT', 'MI_STAGE_DEADLINE')}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'dummy-permit.json'
        replace_permit(path, True)
        def reader():
            try:
                while not done.is_set() or reads[0] < 10000:
                    d = json.loads(path.read_text())
                    assert isinstance(d['allowed'], bool)
                    assert isinstance(d['expires_at'], (int, float))
                    reads[0] += 1
            except BaseException as exc:
                errors.append(repr(exc))
        worker = threading.Thread(target=reader)
        worker.start()
        for index in range(1000):
            replace_permit(path, bool(index % 2))
        done.set()
        worker.join(timeout=10)
        assert not worker.is_alive() and not errors, errors
        os.environ['MI_USAGE_PERMIT'] = str(path)
        os.environ['MI_STAGE_DEADLINE'] = str(float('inf'))
        controls = {}
        def rejects(label, action):
            try:
                action()
            except (RuntimeError, TimeoutError, FileNotFoundError, json.JSONDecodeError):
                controls[label] = True
            else:
                raise AssertionError(label + ' did not stop')
        replace_permit(path, True)
        check_stop(float('inf')); controls['valid_accepts'] = True
        replace_permit(path, False)
        rejects('denied_stops', lambda: check_stop(float('inf')))
        path.write_text(json.dumps({'allowed': True, 'expires_at': 0}))
        rejects('expired_stops', lambda: check_stop(float('inf')))
        path.write_text('')
        rejects('empty_file_stops', lambda: check_stop(float('inf')))
        path.unlink()
        rejects('missing_file_stops', lambda: check_stop(float('inf')))
        replace_permit(path, True)
        rejects('deadline_stops', lambda: check_stop(time.monotonic() - 1))
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return dict(passed=True, writes=1000, reads=reads[0], read_errors=errors,
                controls=controls, scientific_work=False, seconds=time.monotonic()-start,
                limit='Finite concurrency check; not proof for all filesystems or failures.')


if __name__ == '__main__':
    result = check()
    target = Path(__file__).resolve().parents[1] / 'permit-check.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
