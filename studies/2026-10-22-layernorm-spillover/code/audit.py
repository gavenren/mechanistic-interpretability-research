from support import *
import math
import torch
from torch.nn import functional as F
torch.set_num_threads(2);torch.set_num_interop_threads(1)
check();out=STUDY/'results';rows=list(csv.DictReader((out/'cases.csv').open()));saved=np.load(out/'arrays.npz')
max_native=max_scalar=max_metric=0.
changed_count=0
for i,r in enumerate(rows):
 check();d=int(r['width']);x=saved[f'x_{d}'][int(r['sample'])].tolist();eps=float(r['epsilon']);delta=float(r['delta']);xp=x.copy();xp[0]+=delta
 def norm(v):
  m=math.fsum(v)/len(v);s=math.sqrt(math.fsum((a-m)**2 for a in v)/len(v)+eps);return [(a-m)/s for a in v]
 y=norm(x);yp=norm(xp);change=[b-a for a,b in zip(y,yp)]
 native=F.layer_norm(torch.tensor(xp,dtype=torch.float64),(d,),eps=eps).numpy()
 max_native=max(max_native,float(np.max(np.abs(native-saved[f'y_{i}']))));max_scalar=max(max_scalar,max(abs(a-b) for a,b in zip(yp,saved[f'y_{i}'])))
 fraction=math.fsum(v*v for v in change[1:])/math.fsum(v*v for v in change)
 max_metric=max(max_metric,abs(fraction-float(r['other_energy_fraction'])))
 changed=sum(abs(v)>1e-12 for v in change[1:])/(d-1)
 assert changed==float(r['other_changed_fraction'])
 changed_count+=changed==1
 mean=math.fsum(x)/d;z=[v-mean for v in x];scale=math.sqrt(math.fsum(v*v for v in z)/d+eps)
 linear=[(((1 if j==0 else 0)-1/d)/scale-z[j]*z[0]/(d*scale**3))*delta for j in range(d)]
 relative=math.sqrt(math.fsum((v-w)**2 for v,w in zip(change,linear)))/max(math.sqrt(math.fsum(v*v for v in change)),1e-12)
 max_metric=max(max_metric,abs(relative-float(r['linear_relative_error'])))
assert changed_count==json.loads((out/'metrics.json').read_text())['all_other_coordinates_changed_cases']
passed=max(max_native,max_scalar,max_metric)<=5e-12
result=dict(passed=passed,cases=len(rows),maximum_native_error=max_native,maximum_scalar_error=max_scalar,maximum_metric_error=max_metric,seconds=time.monotonic()-START,method='PyTorch native layer_norm plus scalar math.fsum; no training replay')
dump(out/'audit.json',result);dump(out/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'});print(json.dumps(result));assert passed
