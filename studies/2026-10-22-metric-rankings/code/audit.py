from support import *
import math
check();out=STUDY/'results';rows=list(csv.DictReader((out/'comparisons.csv').open()));maxerror=0.;reversals=0;strict=0
sig=lambda v: math.exp(v)/(1+math.exp(v))
sign=lambda v: 0 if abs(v)<=1e-12 else (1 if v>0 else -1)
for r in rows:
 check();m=[float(r['m1']),float(r['m2'])];a=[float(r['a1']),float(r['a2'])];b=[float(r['b1']),float(r['b2'])]
 ld=math.fsum(x-y for x,y in zip(a,b))/2;pd=math.fsum(sig(v+x)-sig(v+y) for v,x,y in zip(m,a,b))/2
 maxerror=max(maxerror,abs(ld-float(r['logit_difference'])),abs(pd-float(r['probability_difference'])))
 assert sign(ld)==int(r['logit_sign']) and sign(pd)==int(r['probability_sign']);reversals+=sign(ld)*sign(pd)<0;strict+=sign(ld)!=0
metrics=json.loads((out/'metrics.json').read_text());assert reversals==metrics['strict_reversals'] and strict==metrics['strict_logit_comparisons']
result=dict(passed=maxerror<=1e-12,comparisons=len(rows),maximum_error=maxerror,reversals=reversals,seconds=time.monotonic()-START,method='Independent scalar exponentials and pairwise score differences')
dump(out/'audit.json',result);dump(out/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'});print(json.dumps(result));assert result['passed']
