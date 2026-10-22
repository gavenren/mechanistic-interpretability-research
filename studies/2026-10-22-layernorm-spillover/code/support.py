import os,sys,time,json,csv,hashlib,platform,resource
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import numpy as np
STUDY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STUDY.parent/'2026-10-19-capability-gates/code'))
from common import check_stop
import argparse,subprocess,fcntl
parser=argparse.ArgumentParser();parser.add_argument('--lock',required=True);LOCK=parser.parse_args().lock
ROOT=STUDY.parents[1]
for source in [STUDY/'analysis-plan.md',STUDY/'sources.md',STUDY.parent/'2026-10-19-capability-gates/code/common.py',*sorted((STUDY/'code').glob('*.py'))]:
 if source.read_bytes()!=subprocess.check_output(['git','show',LOCK+':'+source.relative_to(ROOT).as_posix()],cwd=ROOT): raise RuntimeError('Public lock differs: '+str(source))
LEASE=(ROOT/'.local-state/research-compute.lock').open('a');fcntl.flock(LEASE,fcntl.LOCK_EX|fcntl.LOCK_NB)
START=time.monotonic()
DEADLINE=min(START+120,float(os.environ.get('MI_STAGE_DEADLINE','inf')))
def check(): check_stop(DEADLINE)
def dump(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def finish(out,metrics):
 metrics.update(public_lock=LOCK,seconds=time.monotonic()-START,python=platform.python_version(),numpy=np.__version__,peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,training_runs=0,plan_sha256=hashlib.sha256((STUDY/'analysis-plan.md').read_bytes()).hexdigest())
 dump(out/'metrics.json',metrics)
 dump(out/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'})
 print(json.dumps(metrics),flush=True)
