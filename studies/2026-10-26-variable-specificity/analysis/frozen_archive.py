"""Publish the full calibration evidence before final tests. No model execution."""
import time
START=time.monotonic()
import fcntl,hashlib,json,signal,sys,zipfile
from pathlib import Path
S=Path(__file__).resolve().parents[1];R=S.parents[1]
sys.path.insert(0,str(S/'code'))
from run import Guard,atomic_json,frozen_names
with (R/'.local-state/research-compute.lock').open('a') as lease:
 fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 guard=Guard(START+18);guard()
 signal.signal(signal.SIGALRM,guard.alarm);signal.setitimer(signal.ITIMER_REAL,1,1)
 p=S/'calibration/frozen.zip';expected=json.loads((S/'calibration/manifest.json').read_text())
 names=frozen_names()
 with zipfile.ZipFile(p,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for name in sorted(names):guard();z.write(S/'calibration'/name,name)
 with zipfile.ZipFile(p) as z:
  assert len(z.namelist())==len(names) and set(z.namelist())==names
  for name in sorted(names):
   guard();b=z.read(name);assert hashlib.sha256(b).hexdigest()==expected[name]
   assert b==(S/'calibration'/name).read_bytes()
 report=dict(status='passed',files=len(names),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),measured_seconds=time.monotonic()-START,charged_allowance_seconds=20,model_forwards=0)
 atomic_json(S/'analysis/frozen-archive-check.json',report);guard();signal.setitimer(signal.ITIMER_REAL,0)
 print(json.dumps(report))
