from support import *
import math
check();out=STUDY/'results';saved=np.load(out/'draws.npz');rows=list(csv.DictReader((out/'cells.csv').open()));maxerror=0.;max_reference=0.
def phi(z):return (1+math.erf(z/math.sqrt(2)))/2
def inverse(p):
 lo,hi=-10.,10.
 for _ in range(100):
  mid=(lo+hi)/2
  if phi(mid)<p:lo=mid
  else:hi=mid
 return (lo+hi)/2
for r in rows:
 check();cell=int(r['cell']);K=int(r['candidates']);n=int(r['cases']);N=int(r['trials']);rho=float(r['correlation']);rng=np.random.default_rng(452000+cell);data=(math.sqrt(rho)*rng.normal(size=(N,1))+math.sqrt(1-rho)*rng.normal(size=(N,K)))/math.sqrt(n);validation=(math.sqrt(rho)*rng.normal(size=(N,1))+math.sqrt(1-rho)*rng.normal(size=(N,K)))/math.sqrt(n);index=saved[f'index_{cell}'];val=saved[f'validation_selected_{cell}'];cal=[]
 for i in range(N):
  j=max(range(K),key=lambda j:float(data[i,j]));assert j==int(index[i]);assert float(data[i,j])==float(saved[f'calibration_selected_{cell}'][i]);assert float(validation[i,j])==float(val[i]);cal.append(float(data[i,j]))
 assert np.isfinite(saved[f'calibration_selected_{cell}']).all() and np.isfinite(val).all() and np.isfinite(data).all()
 z=inverse(.975)
 for name,values,cutoff in [('naive',cal,inverse(.95)),('bonferroni',cal,inverse(1-.05/K)),('held_out',val,inverse(.95))]:
  p=sum(float(v)*math.sqrt(n)>cutoff for v in values)/N
  den=1+z*z/N;mid=(p+z*z/(2*N))/den;half=z*math.sqrt(p*(1-p)/N+z*z/(4*N*N))/den
  maxerror=max(maxerror,abs(p-float(r[name+'_rate'])),abs(mid-half-float(r[name+'_wilson_low'])),abs(mid+half-float(r[name+'_wilson_high'])))
 if r['naive_reference']:
  reference=1-phi(inverse(.95))**K;max_reference=max(max_reference,abs(reference-float(r['naive_reference'])))
result=dict(passed=max(maxerror,max_reference)<=1e-12,cells=len(rows),maximum_metric_error=maxerror,maximum_reference_error=max_reference,seconds=time.monotonic()-START,method='Scalar argmax/counts, erf CDF with bisection inverse, independent Wilson calculation; exact seed regeneration; no new trial seeds')
dump(out/'audit.json',result);dump(out/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'});print(json.dumps(result));assert result['passed']
