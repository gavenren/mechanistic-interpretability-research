"""Fixed Experiment 5. Local CPU only; an external live meter issues permits."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib
from importlib.metadata import version
import json
import math
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
SOURCE = '8307227b856d6d6cb44155fdb99b1a47a85ef61e'
PRIOR_PREP = 453.1494834999976
PRIOR_SCIENCE = 1568.8212972499932
ROADMAP_CAP = 183600.0
PRELOCK_CHECK_CHARGE = 120.0
POSTRUN_AUDIT_RESERVE = 120.0
TOKEN_CAP = 0
STAGE_CAP = 7200.0
PREP_CAP = 3600.0
BRANCHES = ('bounds',)
LOCK_DOCS = ('analysis-plan.md', 'sources.md', 'README.md', 'BOUND_METHOD.md',
             'input-manifest.json', 'PLAN_REVIEW.md')


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def manifest(folder):
    return {p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts
            and not p.name.endswith('.tmp')}


def verify_bytes(commit, path):
    relative = path.relative_to(ROOT).as_posix()
    remote = subprocess.check_output(['git', 'show', commit + ':' + relative], cwd=ROOT)
    if remote != path.read_bytes(): raise RuntimeError('Public bytes differ: ' + relative)


def verify_lock(commit, preparation=False):
    for path in sorted((STUDY / 'code').glob('*.py')): verify_bytes(commit, path)
    for name in LOCK_DOCS: verify_bytes(commit, STUDY / name)
    inputs = json.loads((STUDY / 'input-manifest.json').read_text())
    if inputs['source_commit'] != SOURCE: raise RuntimeError('Wrong source commit')
    for relative, expected in inputs['files'].items():
        path = ROOT / relative
        verify_bytes(SOURCE, path)
        if digest(path) != expected: raise RuntimeError('Input hash differs: ' + relative)
    if preparation:
        folder = STUDY / 'preparation'
        for p in sorted(folder.rglob('*')):
            if p.is_file(): verify_bytes(commit, p)
        if manifest(folder) != json.loads((folder / 'manifest.json').read_text()):
            raise RuntimeError('Preparation manifest differs')
        receipt = json.loads((folder / 'status.json').read_text())
        if receipt['status'] != 'passed': raise RuntimeError('Preparation did not pass')
        verify_lock(receipt['plan_lock'])
        subprocess.run(['git', 'merge-base', '--is-ancestor', receipt['plan_lock'], commit],
                       cwd=ROOT, check=True, capture_output=True)
        return receipt


def peak_memory():
    n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return n if sys.platform == 'darwin' else n * 1024


class Guard:
    def __init__(self, deadline, tokens=0, memory_gib=16):
        self.deadline, self.tokens = deadline, tokens
        self.memory_bytes = memory_gib * 1024**3
        self.last_disk = -math.inf
        self.busy = False

    def __call__(self, tokens=0):
        if type(tokens) is not int or tokens < 0: raise RuntimeError('Invalid token charge')
        self.tokens += tokens
        if self.tokens > TOKEN_CAP: raise RuntimeError('Forwarded token cap reached')
        if time.monotonic() >= self.deadline: raise TimeoutError('Stage time cap reached')
        try:
            permit = json.loads(Path(os.environ['MI_USAGE_PERMIT']).read_text())
            expiry = permit['expires_at']; now = time.time()
            valid = (permit['allowed'] is True and type(expiry) in (int, float)
                     and math.isfinite(expiry) and now < expiry <= now + 120)
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            raise RuntimeError('Usage permit missing or invalid') from None
        if not valid: raise RuntimeError('Usage permit denied or expired')
        if peak_memory() > self.memory_bytes: raise RuntimeError('Peak memory cap reached')
        if time.monotonic() - self.last_disk >= 5:
            if shutil.disk_usage(ROOT).free < 20 * 1024**3:
                raise RuntimeError('Less than 20 GiB free disk')
            self.last_disk = time.monotonic()

    def alarm(self, *_):
        # Python executes this between native operations. Each model forward is
        # also charged and checked explicitly. A long native call can overrun.
        if self.busy: return
        self.busy = True
        try: self()
        finally: self.busy = False


def execute(stage, commit):
    started = time.monotonic()
    out = STUDY / ('preparation' if stage == 'prepare' else 'results')
    out.mkdir(exist_ok=False)
    available = PREP_CAP - PRIOR_PREP - PRELOCK_CHECK_CHARGE if stage == 'prepare' else STAGE_CAP - POSTRUN_AUDIT_RESERVE
    guard = Guard(started + available - 2, memory_gib=4 if stage == 'prepare' else 16)
    report = dict(status='running', stage=stage, plan_lock=commit, branches={},
                  prior_preparation_seconds=PRIOR_PREP, prelock_check_charge_seconds=PRELOCK_CHECK_CHARGE,
                  stage_cap_seconds=PREP_CAP if stage == 'prepare' else STAGE_CAP,
                  prior_science_seconds=PRIOR_SCIENCE, token_cap=TOKEN_CAP)
    prep = None
    def save():
        report.update(elapsed_seconds=time.monotonic() - started, forwarded_tokens=guard.tokens)
        atomic_json(out / 'status.json', report)
    try:
        guard()
        signal.signal(signal.SIGALRM, guard.alarm); signal.setitimer(signal.ITIMER_REAL, 1, 1)
        prep = verify_lock(commit, preparation=stage == 'experiment')
        prep_charge = PRIOR_PREP + PRELOCK_CHECK_CHARGE + (prep['charged_seconds'] if prep else 0)
        if prep_charge + PRIOR_SCIENCE + (PREP_CAP - PRIOR_PREP - PRELOCK_CHECK_CHARGE if stage == 'prepare' else STAGE_CAP) > ROADMAP_CAP:
            raise RuntimeError('Remaining roadmap cap cannot hold this fixed stage')
        if prep is not None:
            guard.tokens = prep['forwarded_tokens']; guard()
        import torch
        import numpy as np
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        report['runtime'] = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                 transformers=version('transformers'),
                                 threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())
        if (report['runtime']['torch'], report['runtime']['numpy'], report['runtime']['transformers']) != ('2.8.0', '2.3.5', '4.57.6'):
            raise RuntimeError('Runtime differs from fixed versions')
        if prep is not None and prep['runtime'] != report['runtime']:
            raise RuntimeError('Runtime differs from preparation')
        save()
        for name in BRANCHES:
            guard(); report['current_branch'] = name; save()
            module = importlib.import_module(name)
            begun = time.monotonic()
            value = (module.prepare if stage == 'prepare' else module.run)(out / name, guard)
            report['branches'][name] = dict(value, measured_seconds=time.monotonic() - begun)
            save()
            if value.get('passed') is not True: raise RuntimeError(name + ' checks failed')
        if stage == 'prepare':
            projected = sum(float(report['branches'][x]['projected_seconds']) for x in BRANCHES)
            # Add room for setup, source checks, saving, and the reserved audit.
            report['projected_experiment_seconds'] = projected + 120 + POSTRUN_AUDIT_RESERVE
            if not math.isfinite(projected) or projected < 0 or report['projected_experiment_seconds'] >= STAGE_CAP:
                raise RuntimeError('Conservative projection exceeds stage cap')
        verify_lock(commit, preparation=stage == 'experiment'); guard()
        report.update(status='passed' if stage == 'prepare' else 'complete', current_branch=None)
    except BaseException as exc:
        reason = str(exc).replace(str(ROOT), '[repository]')
        private = os.environ.get('MI_USAGE_PERMIT')
        if private: reason = reason.replace(private, '[private permit]')
        report.update(status='stopped', error_type=type(exc).__name__, reason=reason)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        save(); hashes = manifest(out)
        report.update(peak_memory_bytes=peak_memory(), measured_seconds=time.monotonic() - started,
                      closing_receipt_allowance_seconds=2.0)
        report['charged_seconds'] = report['measured_seconds'] + 2
        report['total_preparation_seconds'] = PRIOR_PREP + PRELOCK_CHECK_CHARGE + (report['charged_seconds'] if stage == 'prepare' else prep['charged_seconds'] if prep else 0)
        if stage == 'experiment':
            report['stage5_charge_with_audit_reserve_seconds'] = report['charged_seconds'] + POSTRUN_AUDIT_RESERVE
        if report['charged_seconds'] >= available:
            report.update(status='stopped', error_type='TimeoutError', reason='Stage cap reached during saving')
        closing = time.monotonic()
        atomic_json(out / 'status.json', report)
        hashes['status.json'] = digest(out / 'status.json')
        atomic_json(out / 'manifest.json', hashes)
        if time.monotonic() - closing > 2:
            report.update(status='stopped', reason='Closing receipt allowance exceeded')
            report['charged_seconds'] = time.monotonic() - started + 2
            report['total_preparation_seconds'] = PRIOR_PREP + PRELOCK_CHECK_CHARGE + (report['charged_seconds'] if stage == 'prepare' else prep['charged_seconds'] if prep else 0)
            if stage == 'experiment':
                report['stage5_charge_with_audit_reserve_seconds'] = report['charged_seconds'] + POSTRUN_AUDIT_RESERVE
            atomic_json(out / 'status.json', report)
            hashes['status.json'] = digest(out / 'status.json'); atomic_json(out / 'manifest.json', hashes)
    print(json.dumps(report, allow_nan=False), flush=True)
    return report['status'] in ('passed', 'complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'experiment')); parser.add_argument('--lock', required=True)
    args = parser.parse_args()
    (ROOT / '.local-state').mkdir(exist_ok=True)
    with (ROOT / '.local-state/research-compute.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(0 if execute(args.stage, args.lock) else 1)
