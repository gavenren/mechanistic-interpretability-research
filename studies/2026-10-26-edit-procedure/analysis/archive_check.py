"""Pack exact result files and check a fresh restore. No model execution."""
import time
START = time.monotonic()
import fcntl
import json
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import result_archive as archive
S = Path(__file__).resolve().parents[1]
R = S.parents[1]
sys.path.insert(0, str(S/'code'))
from run import Guard, atomic_json
with (R/'.local-state/research-compute.lock').open('a') as lease:
    fcntl.flock(lease, fcntl.LOCK_EX|fcntl.LOCK_NB)
    guard=Guard(START+58)
    signal.signal(signal.SIGALRM,guard.alarm);signal.setitimer(signal.ITIMER_REAL,1,1)
    guard()
    packed=archive.pack(S);guard()
    with tempfile.TemporaryDirectory(prefix='mi-2a-restore-') as tmp:
        target=Path(tmp)
        (target/'results').mkdir()
        for name in ('status.json','manifest.json'):
            shutil.copyfile(S/'results'/name,target/'results'/name)
        shutil.copytree(S/'artifacts',target/'artifacts')
        restored=archive.unpack(target);guard()
        repeated=archive.unpack(target)
        if repeated['files_added']!=0:raise RuntimeError('Identical restore changed files')
        first=json.loads((target/'artifacts/manifest.json').read_text())['archives'][0]['members'][0]['path']
        (target/first).write_bytes(b'deliberate conflict in temporary test folder')
        rejected=False
        try:archive.unpack(target)
        except FileExistsError:rejected=True
        if not rejected:raise RuntimeError('Conflicting restore was not rejected')
    guard()
    report=dict(status='passed',packed=packed,fresh_restore=restored,identical_restore_added=0,
                conflict_rejected=True,measured_seconds_before_receipt=time.monotonic()-START,
                charged_allowance_seconds=60,model_forwards=0)
    atomic_json(S/'analysis/archive-check.json',report)
    guard();signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps(report))
