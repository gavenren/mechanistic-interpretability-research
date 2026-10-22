from support import *
check()
out=STUDY/'results';out.mkdir(exist_ok=False)
rows=[];arrays={};errors=[];zeros=[];shift_errors=[]
for d in (4,8,16,64):
 xbank=np.random.default_rng(451000+d).normal(size=(32,d));arrays[f'x_{d}']=xbank
 for eps in (0.,1e-5):
  for sample,x in enumerate(xbank):
   check();z=x-x.mean();s=np.sqrt(np.mean(z*z)+eps);y=z/s
   J=(np.eye(d)-np.ones((d,d))/d)/s-np.outer(z,z)/(d*s**3)
   zeros.append(float(np.max(np.abs((x-x.mean())/s-y))))
   shifted=x+3.25;shift_errors.append(float(np.max(np.abs((shifted-shifted.mean())/np.sqrt(np.mean((shifted-shifted.mean())**2)+eps)-y))))
   for scale in (-4.,-1.,-.1,-.01,.01,.1,1.,4.):
    delta=scale*np.sqrt(np.mean(z*z));xp=x.copy();xp[0]+=delta;zp=xp-xp.mean();sp=np.sqrt(np.mean(zp*zp)+eps);yp=zp/sp
    change=yp-y;e0=np.eye(d)[0];formula=delta*e0/sp+(1/sp-1/s)*z-delta/(d*sp)
    error=float(np.max(np.abs(change-formula)));errors.append(error)
    rows.append(dict(width=d,epsilon=eps,sample=sample,scale=scale,delta=delta,other_changed_fraction=float(np.mean(np.abs(change[1:])>1e-12)),other_energy_fraction=float(np.sum(change[1:]**2)/np.sum(change**2)),linear_relative_error=float(np.linalg.norm(change-J[:,0]*delta)/max(np.linalg.norm(change),1e-12)),formula_error=error))
    arrays[f'y_{len(rows)-1}']=yp
with (out/'cases.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
np.savez_compressed(out/'arrays.npz',**arrays)
groups=[]
for d in (4,8,16,64):
 for scale in (-4.,-1.,-.1,-.01,.01,.1,1.,4.):
  a=[r for r in rows if r['width']==d and r['scale']==scale]
  groups.append(dict(width=d,scale=scale,cases=len(a),mean_other_energy=float(np.mean([r['other_energy_fraction'] for r in a])),mean_linear_error=float(np.mean([r['linear_relative_error'] for r in a]))))
passed=max(errors)<=5e-12 and max(zeros)==0 and max(shift_errors)<=5e-12
finish(out,dict(status='complete' if passed else 'failed_controls',cases=len(rows),controls_passed=passed,maximum_decomposition_error=max(errors),maximum_constant_shift_error=max(shift_errors),all_other_coordinates_changed_cases=sum(r['other_changed_fraction']==1 for r in rows),mean_other_energy_fraction=float(np.mean([r['other_energy_fraction'] for r in rows])),groups=groups))
assert passed
