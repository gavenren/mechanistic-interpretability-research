from support import *
from statistics import NormalDist
import math
check();out=STUDY/'results';out.mkdir(exist_ok=False);normal=NormalDist();threshold=normal.inv_cdf(.95);rows=[];arrays={};trials=5000
for K in (1,8,32,128):
 for n in (16,64,256):
  for rho in (0.,.5,.9):
   check();cell=len(rows);rng=np.random.default_rng(452000+cell)
   calibration=(np.sqrt(rho)*rng.normal(size=(trials,1))+np.sqrt(1-rho)*rng.normal(size=(trials,K)))/np.sqrt(n)
   validation=(np.sqrt(rho)*rng.normal(size=(trials,1))+np.sqrt(1-rho)*rng.normal(size=(trials,K)))/np.sqrt(n)
   index=calibration.argmax(1);cal=calibration[np.arange(trials),index];val=validation[np.arange(trials),index]
   # Save selected values. The exact seed regenerates the full candidate bank for audit.
   arrays[f'calibration_selected_{cell}']=cal;arrays[f'index_{cell}']=index;arrays[f'validation_selected_{cell}']=val
   rates={name:float(np.mean(values>cutoff)) for name,values,cutoff in (('naive',cal*np.sqrt(n),threshold),('bonferroni',cal*np.sqrt(n),normal.inv_cdf(1-.05/K)),('held_out',val*np.sqrt(n),threshold))}
   row=dict(cell=cell,candidates=K,cases=n,correlation=rho,trials=trials,naive_rate=rates['naive'],bonferroni_rate=rates['bonferroni'],held_out_rate=rates['held_out'],calibration_mean=float(cal.mean()),validation_mean=float(val.mean()),naive_reference=1-.95**K if rho==0 else None)
   for name,p in rates.items():
    z=normal.inv_cdf(.975);den=1+z*z/trials;mid=(p+z*z/(2*trials))/den;half=z*np.sqrt(p*(1-p)/trials+z*z/(4*trials**2))/den
    row[name+'_wilson_low']=float(mid-half);row[name+'_wilson_high']=float(mid+half)
   rows.append(row)
np.savez_compressed(out/'draws.npz',**arrays)
with (out/'cells.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
maxnull=max(abs(r['naive_rate']-r['naive_reference']) for r in rows if r['naive_reference'] is not None);maxheld=max(abs(r['held_out_rate']-.05) for r in rows);maxbonf=max(r['bonferroni_rate'] for r in rows)
passed=maxnull<=.03 and maxheld<=.03 and maxbonf<=.08
finish(out,dict(status='complete' if passed else 'failed_controls',controls_passed=passed,cells=len(rows),trials_per_cell=trials,total_trials=len(rows)*trials,maximum_independent_null_discrepancy=maxnull,maximum_held_out_discrepancy=maxheld,maximum_bonferroni_rate=maxbonf,cells_summary=rows));assert passed
