"""Fixed Experiment 2C. Local CPU only; an external live meter issues permits."""
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
SOURCE = '6f32f86922774d57639afbae6f753f26d3fdfa2a'
PRIOR_PREP = 328.7724097919963
PRIOR_SCIENCE = 1385.5104865419962
PRELOCK_CHECK_CHARGE = 120.0
POSTRUN_AUDIT_RESERVE = 120.0
TOKEN_CAP = 0
STAGE_CAP = 7200.0
PREP_CAP = 3600.0
BRANCHES = ('models', 'edits')
LOCK_DOCS = ('analysis-plan.md', 'sources.md', 'README.md', 'MODEL_METHOD.md',
             'EDIT_METHOD.md', 'input-manifest.json', 'PLAN_REVIEW.md')


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


def verify_training(commit):
    folder=STUDY/'training'
    for p in sorted(folder.rglob('*')):
        if p.is_file():verify_bytes(commit,p)
    receipt=json.loads((folder/'status.json').read_text())
    if receipt['status']!='complete' or receipt['branches']['models'].get('ability_gate_pass') is not True:
        raise RuntimeError('The full model panel did not pass ordinary task ability')
    if manifest(folder)!=json.loads((folder/'manifest.json').read_text()):
        raise RuntimeError('Training manifest differs')
    verify_lock(receipt['plan_lock'],preparation=True)
    subprocess.run(['git','merge-base','--is-ancestor',receipt['plan_lock'],commit],cwd=ROOT,check=True,capture_output=True)
    return receipt


def frozen_names(expected=None):
    """All edit-calibration files in the completed, pre-archive manifest."""
    if expected is None:
        expected=json.loads((STUDY/'calibration/manifest.json').read_text())
    names={name for name in expected if name.startswith('edits/')}
    if not {'edits/choices.json','edits/calibration-manifest.json'} <= names:
        raise RuntimeError('Calibration manifest lacks required edit records')
    return names


def verify_calibration(commit):
    import zipfile
    folder=STUDY/'calibration'
    for name in ('status.json','manifest.json','edits/choices.json','edits/calibration-manifest.json','frozen.zip'):
        verify_bytes(commit,folder/name)
    receipt=json.loads((folder/'status.json').read_text())
    if receipt['status']!='complete':raise RuntimeError('Edit calibration did not complete')
    expected=json.loads((folder/'manifest.json').read_text())
    # The receipt and manifest are closed before packaging. Only the root
    # manifest itself and the later archive are outside this fixed file list.
    local_names={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()
                 and p.relative_to(folder).as_posix() not in ('manifest.json','frozen.zip')}
    if local_names!=set(expected):
        raise RuntimeError('Calibration file membership differs from the public manifest')
    for name,h in expected.items():
        if digest(folder/name)!=h:raise RuntimeError('Calibration file differs: '+name)
    required_members=frozen_names(expected)
    with zipfile.ZipFile(folder/'frozen.zip') as archive:
        members=archive.namelist()
        if len(members)!=len(set(members)) or set(members)!=required_members:
            raise RuntimeError('Wrong or duplicate full-calibration archive members')
        for name in sorted(required_members):
            payload=archive.read(name)
            if hashlib.sha256(payload).hexdigest()!=expected[name] or payload!=(folder/name).read_bytes():
                raise RuntimeError('Public edit-calibration bytes differ: '+name)
    verify_lock(receipt['plan_lock'],preparation=True);verify_training(receipt['plan_lock'])
    subprocess.run(['git','merge-base','--is-ancestor',receipt['plan_lock'],commit],cwd=ROOT,check=True,capture_output=True)
    return receipt


def execute(stage,commit):
    started=time.monotonic()
    folder={'prepare':'preparation','train':'training','calibrate':'calibration','experiment':'results'}[stage]
    out=STUDY/folder;out.mkdir(exist_ok=False)
    available=PREP_CAP-PRIOR_PREP-PRELOCK_CHECK_CHARGE if stage=='prepare' else STAGE_CAP-POSTRUN_AUDIT_RESERVE
    guard=Guard(started+available-2,memory_gib=4 if stage=='prepare' else 16)
    prep,training,calibration=None,None,None
    report=dict(status='running',stage=stage,plan_lock=commit,branches={},
                prior_preparation_seconds=PRIOR_PREP,prelock_check_charge_seconds=PRELOCK_CHECK_CHARGE,
                prior_science_seconds=PRIOR_SCIENCE,stage_cap_seconds=PREP_CAP if stage=='prepare' else STAGE_CAP)
    def save():
        report.update(elapsed_seconds=time.monotonic()-started,forwarded_tokens=guard.tokens)
        atomic_json(out/'status.json',report)
    def previous_charges():
        return sum(x['charged_seconds'] for x in (training,calibration) if x is not None)
    def charges():
        report['total_preparation_seconds']=PRIOR_PREP+PRELOCK_CHECK_CHARGE+(report['charged_seconds'] if stage=='prepare' else prep['charged_seconds'] if prep else 0)
        if stage!='prepare':
            report['stage2c_charge_seconds']=report['charged_seconds']+previous_charges()
            report['stage2c_charge_with_audit_reserve_seconds']=report['stage2c_charge_seconds']+POSTRUN_AUDIT_RESERVE
    try:
        guard();signal.signal(signal.SIGALRM,guard.alarm);signal.setitimer(signal.ITIMER_REAL,1,1)
        prep=verify_lock(commit,preparation=stage!='prepare')
        if stage in ('calibrate','experiment'):training=verify_training(commit)
        if stage=='experiment':calibration=verify_calibration(commit)
        available-=previous_charges();guard.deadline=started+available-2;guard()
        import torch
        import numpy as np
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        report['runtime']=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
                              transformers=version('transformers'),threads=torch.get_num_threads(),
                              interop_threads=torch.get_num_interop_threads())
        if (report['runtime']['torch'],report['runtime']['numpy'],report['runtime']['transformers'])!=('2.8.0','2.3.5','4.57.6'):
            raise RuntimeError('Runtime differs from fixed versions')
        if prep is not None and prep['runtime']!=report['runtime']:raise RuntimeError('Runtime differs from preparation')
        names=BRANCHES if stage=='prepare' else ('models',) if stage=='train' else ('edits',)
        for name in names:
            guard();report['current_branch']=name;save();module=importlib.import_module(name);begun=time.monotonic()
            if stage=='prepare':value=module.prepare(out/name,guard)
            elif stage=='train':value=module.run(out/name,guard)
            elif stage=='calibrate':value=module.calibrate(out/name,guard,STUDY/'training/models')
            else:value=module.run(out/name,guard,STUDY/'training/models',STUDY/'calibration/edits')
            report['branches'][name]=dict(value,measured_seconds=time.monotonic()-begun);save()
            if value.get('passed') is not True:raise RuntimeError(name+' calculation checks failed')
        if stage=='prepare':
            projected=sum(float(report['branches'][x]['projected_seconds']) for x in BRANCHES)
            report['projected_combined_science_seconds']=projected+120+POSTRUN_AUDIT_RESERVE
            if not math.isfinite(projected) or projected<0 or report['projected_combined_science_seconds']>=STAGE_CAP:
                raise RuntimeError('Combined projection exceeds stage cap')
        if stage=='train':report['proceed_to_edit_calibration']=report['branches']['models']['ability_gate_pass']
        verify_lock(commit,preparation=stage!='prepare')
        if training is not None:verify_training(commit)
        if calibration is not None:verify_calibration(commit)
        guard();report.update(status='passed' if stage=='prepare' else 'complete',current_branch=None)
    except BaseException as exc:
        reason=str(exc).replace(str(ROOT),'[repository]');private=os.environ.get('MI_USAGE_PERMIT')
        if private:reason=reason.replace(private,'[private permit]')
        report.update(status='stopped',error_type=type(exc).__name__,reason=reason)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);save();hashes=manifest(out)
        report.update(peak_memory_bytes=peak_memory(),measured_seconds=time.monotonic()-started,closing_receipt_allowance_seconds=2.)
        report['charged_seconds']=report['measured_seconds']+2;charges()
        if report['charged_seconds']>=available:report.update(status='stopped',reason='Stage cap reached during saving')
        closing=time.monotonic();atomic_json(out/'status.json',report)
        hashes['status.json']=digest(out/'status.json');atomic_json(out/'manifest.json',hashes)
        if time.monotonic()-closing>2:
            report.update(status='stopped',reason='Closing receipt allowance exceeded')
            report['charged_seconds']=time.monotonic()-started+2;charges()
            atomic_json(out/'status.json',report);hashes['status.json']=digest(out/'status.json');atomic_json(out/'manifest.json',hashes)
    print(json.dumps(report,allow_nan=False),flush=True)
    return report['status'] in ('passed','complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','train','calibrate','experiment'))
    parser.add_argument('--lock',required=True);args=parser.parse_args()
    (ROOT/'.local-state').mkdir(exist_ok=True)
    with (ROOT/'.local-state/research-compute.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raise SystemExit(0 if execute(args.stage,args.lock) else 1)
