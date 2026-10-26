"""Exact split-archive transport for the unchanged Experiment 2C final runner."""
from pathlib import Path
import argparse,fcntl,hashlib,json,sys
S=Path(__file__).resolve().parents[1];R=S.parents[1]
sys.path.insert(0,str(S/'code'))
import run
ORIGINAL_VERIFY=run.verify_bytes

def verify_bytes(commit,path):
 path=Path(path)
 if path != S/'calibration/frozen.zip':return ORIGINAL_VERIFY(commit,path)
 for name in ('analysis/final_transport.py','TRANSPORT_PLAN.md','TRANSPORT_REVIEW.md','calibration/transport.json'):
  ORIGINAL_VERIFY(commit,S/name)
 record=json.loads((S/'calibration/transport.json').read_text())
 if record['source_sha256']!='64c6fe719663885b2db61f8a9c91e6a59abd5b16eef93c79bc8624e73b57aec7' or record['source_bytes']!=30963232:
  raise RuntimeError('Frozen calibration identity changed')
 if [x['path'] for x in record['parts']]!=['frozen.zip.part01','frozen.zip.part02']:
  raise RuntimeError('Unexpected transport parts')
 full=hashlib.sha256();total=0
 with path.open('rb') as source:
  for item in record['parts']:
   p=S/'calibration'/item['path'];ORIGINAL_VERIFY(commit,p)
   raw=p.read_bytes()
   if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
    raise RuntimeError('Transport part differs')
   if source.read(len(raw))!=raw:raise RuntimeError('Transport differs from frozen archive')
   full.update(raw);total+=len(raw)
  if source.read(1):raise RuntimeError('Frozen archive has trailing bytes')
 if total!=record['source_bytes'] or full.hexdigest()!=record['source_sha256']:
  raise RuntimeError('Transport archive identity differs')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--lock',required=True);args=parser.parse_args()
 run.verify_bytes=verify_bytes
 with (R/'.local-state/research-compute.lock').open('a') as lease:
  fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  raise SystemExit(0 if run.execute('experiment',args.lock) else 1)
