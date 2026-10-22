from support import *
import itertools
check();out=STUDY/'results';out.mkdir(exist_ok=False)
sigmoid=lambda z: 1/(1+np.exp(-np.asarray(z,dtype=float)))
sgn=lambda v: 0 if abs(v)<=1e-12 else (1 if v>0 else -1)
candidates=list(itertools.product((0.,.25,.5,1.,2.),repeat=2));rows=[];error=0.;bound_error=0.;dominance_fail=0
for margins in itertools.product((-6.,-2.,0.,2.,6.),repeat=2):
 check();base=sigmoid(margins);gains=[]
 for a in candidates:
  shifted=np.asarray(margins)+a;z=np.stack([shifted,np.zeros(2)],axis=1);z-=z.max(axis=1,keepdims=True);soft=np.exp(z);soft/=soft.sum(axis=1,keepdims=True)
  prob=sigmoid(shifted)-base;error=max(error,float(np.max(np.abs(soft[:,0]-sigmoid(shifted)))));bound_error=max(bound_error,float(np.max(prob-np.asarray(a)/4)),float(np.max(-prob)))
  gains.append((float(np.mean(a)),float(np.mean(prob))))
 for ai,bi in itertools.combinations(range(25),2):
  a,b=candidates[ai],candidates[bi];lm=gains[ai][0]-gains[bi][0];pm=gains[ai][1]-gains[bi][1];ls,ps=sgn(lm),sgn(pm)
  dominance=1 if all(x>=y for x,y in zip(a,b)) else (-1 if all(x<=y for x,y in zip(a,b)) else 0)
  dominance_fail+=bool(dominance and ps*dominance<0)
  rows.append(dict(m1=margins[0],m2=margins[1],a1=a[0],a2=a[1],b1=b[0],b2=b[1],logit_difference=lm,probability_difference=pm,logit_sign=ls,probability_sign=ps,strict_reversal=int(ls*ps<0),dominance=dominance))
with (out/'comparisons.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
strict=sum(r['logit_sign']!=0 for r in rows);reversals=sum(r['strict_reversal'] for r in rows)
witness=dict(margins=[6,0],a=[2,0],b=[0,1],a_mean_logit=1.,b_mean_logit=.5,a_mean_probability=float(np.mean(sigmoid([8,0])-sigmoid([6,0]))),b_mean_probability=float(np.mean(sigmoid([6,1])-sigmoid([6,0]))))
passed=error<=1e-12 and bound_error<=1e-12 and dominance_fail==0
finish(out,dict(status='complete' if passed else 'failed_controls',controls_passed=passed,comparisons=len(rows),strict_logit_comparisons=strict,strict_reversals=reversals,reversal_fraction_of_strict_logit=reversals/strict,dominance_violations=dominance_fail,maximum_softmax_error=error,maximum_bound_violation=bound_error,witness=witness));assert passed
