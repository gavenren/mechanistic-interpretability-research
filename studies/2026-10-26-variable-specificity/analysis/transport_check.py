"""Split and test exact file transport. No model execution."""
import time
START=time.monotonic()
from pathlib import Path
import fcntl,hashlib,importlib.util,json,signal,sys,tempfile,shutil
S=Path(__file__).resolve().parents[1];R=S.parents[1]
sys.path.insert(0,str(S/'code'))
from run import Guard,atomic_json
spec=importlib.util.spec_from_file_location('transport',S/'analysis/final_transport.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with (R/'.local-state/research-compute.lock').open('a') as lease:
 fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);guard=Guard(START+8);guard()
 signal.signal(signal.SIGALRM,guard.alarm);signal.setitimer(signal.ITIMER_REAL,1,1)
 raw=(S/'calibration/frozen.zip').read_bytes();parts=[]
 for i,start in enumerate(range(0,len(raw),20000000),1):
  guard();name=f'frozen.zip.part{i:02d}';b=raw[start:start+20000000]
  p=S/'transport'/name
  if p.exists():assert p.read_bytes()==b
  else:
   with p.open('xb') as f:f.write(b)
  parts.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
 record=dict(source_bytes=len(raw),source_sha256=hashlib.sha256(raw).hexdigest(),parts=parts)
 atomic_json(S/'transport/transport.json',record)
 with tempfile.TemporaryDirectory(prefix='mi-2c-transport-') as tmp:
  T=Path(tmp);(T/'calibration').mkdir();(T/'transport').mkdir()
  shutil.copyfile(S/'calibration/frozen.zip',T/'calibration/frozen.zip')
  for name in ('transport.json',*[x['path'] for x in parts]):shutil.copyfile(S/'transport'/name,T/'transport'/name)
  seen=[];m.S=T;m.ORIGINAL_VERIFY=lambda c,p:seen.append(p.relative_to(T).as_posix())
  m.verify_bytes('mock-public-lock',T/'calibration/frozen.zip');guard()
  assert len(seen)==6
  m.verify_bytes('mock-public-lock',T/'unchanged.txt');assert seen[-1]=='unchanged.txt'
  denied=[]
  def reject(label):
   try:m.verify_bytes('mock-public-lock',T/'calibration/frozen.zip')
   except (RuntimeError,FileNotFoundError):denied.append(label)
   else:raise AssertionError('Accepted '+label)
  p=T/'transport/frozen.zip.part02';original=p.read_bytes();p.write_bytes(b'X'+original[1:]);reject('changed-part');p.write_bytes(original)
  d=json.loads(json.dumps(record));d['parts'].reverse();atomic_json(T/'transport/transport.json',d);reject('reordered-parts');atomic_json(T/'transport/transport.json',record)
  p.unlink();reject('missing-part');p.write_bytes(original)
  with (T/'calibration/frozen.zip').open('ab') as f:f.write(b'X')
  reject('extra-archive-byte')
 m.S=S;m.ORIGINAL_VERIFY=lambda c,p:None
 import run
 run.verify_bytes=m.verify_bytes;run.verify_lock=lambda *a,**k:None;run.verify_training=lambda *a,**k:None
 run.subprocess.run=lambda *a,**k:None
 run.verify_calibration('mock-public-lock');guard()
 out=dict(status='passed',parts=2,source_bytes=len(raw),rejected=denied,unchanged_path_delegated=True,original_calibration_verification_passed=True,measured_seconds=time.monotonic()-START,charged_allowance_seconds=10,model_forwards=0)
 atomic_json(S/'analysis/transport-check.json',out);guard();signal.setitimer(signal.ITIMER_REAL,0);print(json.dumps(out))
