"""Publish the exact development banks. No model execution."""
import time
START=time.monotonic()
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import sys
import zipfile
S=Path(__file__).resolve().parents[1];R=S.parents[1]
sys.path.insert(0,str(S/'code'))
from run import Guard,atomic_json
with (R/'.local-state/research-compute.lock').open('a') as lease:
 fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 guard=Guard(START+18);guard();signal.signal(signal.SIGALRM,guard.alarm);signal.setitimer(signal.ITIMER_REAL,1,1)
 root=S/'calibration';expected=json.loads((root/'manifest.json').read_text())
 if json.loads((root/'status.json').read_text())['status']!='complete':raise RuntimeError('Calibration is incomplete')
 names=[f'subsets/bank-seed-{seed}.npz' for seed in range(8)]
 target=root/'banks.zip'
 with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
  for name in names:
   guard();b=(root/name).read_bytes()
   if hashlib.sha256(b).hexdigest()!=expected[name]:raise RuntimeError('Bank hash differs')
   info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
   archive.writestr(info,b)
 with zipfile.ZipFile(target) as archive:
  if archive.namelist()!=names:raise RuntimeError('Archive names differ')
  for name in names:
   guard()
   if hashlib.sha256(archive.read(name)).hexdigest()!=expected[name]:raise RuntimeError('Archived bank differs')
 if target.stat().st_size>=24*1024**2:raise RuntimeError('Bank archive is too large')
 receipt=dict(status='passed',members=8,archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),archive_bytes=target.stat().st_size,measured_seconds_before_receipt=time.monotonic()-START,charged_allowance_seconds=20,model_forwards=0)
 atomic_json(S/'analysis/bank-archive-check.json',receipt);guard();signal.setitimer(signal.ITIMER_REAL,0)
 print(json.dumps(receipt))
