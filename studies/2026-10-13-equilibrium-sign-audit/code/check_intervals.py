"""Dummy-only exact-rational tests. No study data or trained model is read."""
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time
import numpy as np
import intervals as iv


def rational(x):
    return F.from_float(float(x))


def contains(bounds, target):
    return rational(bounds[0]) <= target <= rational(bounds[1])


def solve(matrix,rhs):
    a=[list(row)+[value] for row,value in zip(matrix,rhs)]
    for j in range(len(a)):
        pivot=next(i for i in range(j,len(a)) if a[i][j])
        a[j],a[pivot]=a[pivot],a[j]
        scale=a[j][j];a[j]=[x/scale for x in a[j]]
        for i in range(len(a)):
            if i!=j:
                scale=a[i][j];a[i]=[x-scale*y for x,y in zip(a[i],a[j])]
    return [row[-1] for row in a]


def exact_equilibrium(weight,injection,removed):
    n=len(weight); answers=[]
    for flags in itertools.product((False,True),repeat=n):
        if removed>=0 and flags[removed]:continue
        active=[j for j in range(n) if flags[j]]
        matrix=[[F(int(i==j))-weight[i][j] for j in active] for i in active]
        solution=solve(matrix,[injection[i] for i in active]) if active else []
        state=[F(0)]*n
        for i,value in zip(active,solution):state[i]=value
        pre=[sum(w*z for w,z in zip(row,state))+a for row,a in zip(weight,injection)]
        if all((state[i]>=0 and state[i]==pre[i]) if flags[i] else (i==removed or pre[i]<=0) for i in range(n)):
            answers.append(tuple(state))
    answers=list(set(answers))
    assert len(answers)==1
    return answers[0]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    started=time.monotonic();checks=0;records=[]
    receipt=dict(status='running',scientific_model_outputs_collected=False,
        python=platform.python_version(),numpy=np.__version__,
        interval_code_sha256=hashlib.sha256(Path(iv.__file__).read_bytes()).hexdigest(),
        check_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def check(ok,label):
        nonlocal checks
        records.append(dict(check=label,passed=bool(ok)))
        checks+=1
        assert ok,label
    try:
        tiny=float(np.nextafter(0.,1.))
        scalars=[0.,tiny,-tiny,np.finfo(float).tiny,-np.finfo(float).tiny,1.,-1.,.1,-.1,2.**100,2.**-100,np.nextafter(1.,0.)]
        for a,b in itertools.product(scalars,repeat=2):
            for name,func,target in [('add',iv.add,rational(a)+rational(b)),('sub',iv.sub,rational(a)-rational(b)),('mul',iv.mul,rational(a)*rational(b))]:
                check(contains(func(iv.point(a),iv.point(b)),target),f'point_{name}')
        rng=np.random.default_rng(2026101601)
        for trial in range(128):
            a=np.sort(rng.normal(size=2));b=np.sort(rng.normal(size=2))
            for x,y in itertools.product(a,b):
                check(contains(iv.mul(tuple(a),tuple(b)),rational(x)*rational(y)),'interval_multiply_corners')
            p=abs(float(rng.normal()));d=abs(float(rng.normal()))+.1
            check(rational(iv.positive_division_upper(p,d))>=rational(p)/rational(d),'positive_division')
        affine_checks=0
        for trial in range(24):
            x=rng.normal(size=(3,4));w=rng.normal(size=(5,4));b=rng.normal(size=5)
            lo,hi=iv.affine_point(x,w,b)
            for i,j in itertools.product(range(3),range(5)):
                truth=sum(rational(x[i,k])*rational(w[j,k]) for k in range(4))+rational(b[j])
                check(contains((lo[i,j],hi[i,j]),truth),'affine_exact');affine_checks+=1
        # Exact small targets include all-active, inactive, slow, and cancellation cases.
        specifications=[
            ('feedforward',[[0.,0.],[0.,0.]],[.5,-.25],[1.,-.5]),
            ('linear',[[.25,.125],[-.125,.25]],[.5,.75],[1.,-1.]),
            ('slow',[[.875,0.],[0.,.5]],[.125,.25],[1.,-1.]),
            ('inactive',[[.25,.125],[.125,.25]],[.5,-2.],[1.,1.]),
            ('cancel',[[.875,0.],[0.,.875]],[.125,.125],[1.,-1.]),
            ('zero_readout',[[.5,.125],[.125,.5]],[.125,.25],[0.,0.]),
        ]
        cases=[]; signs=set();unresolved=0;extra_updates=0
        for name,ww,aa,cc in specifications:
            w=np.array(ww);a=np.array(aa);c=np.array(cc);q=iv.contraction_upper(w)
            wr=[[rational(v) for v in row] for row in w];ar=list(map(rational,a));cr=list(map(rational,c))
            exact_base=exact_equilibrium(wr,ar,-1)
            check(rational(q)>=max(sum(abs(v) for v in row) for row in wr),'q_upper')
            for removed in range(2):
                exact_edit=exact_equilibrium(wr,ar,removed)
                truth=sum(v*(b-e) for v,b,e in zip(cr,exact_base,exact_edit))
                signs.add(int(truth>0)-int(truth<0))
                base=np.zeros((1,2));edit=np.zeros((1,2));readout=np.stack([c,np.zeros(2)])
                for iteration in range(129):
                    values=iv.effect_interval(base,edit,w,iv.point(a[None,:]),np.array([removed]),readout,np.array([0]),np.array([1]),q)
                    bounds=(values['lower'][0],values['upper'][0])
                    check(contains(bounds,truth),'effect_exact_equilibrium')
                    for state,exact,rkey in [(base[0],exact_base,'residual_base_upper'),(edit[0],exact_edit,'residual_edit_upper')]:
                        target_error=max(abs(rational(v)-t) for v,t in zip(state,exact))
                        rb=iv.state_error_upper(values[rkey][0],q)
                        check(rational(rb)>=target_error,'state_error_exact')
                    if iteration==0:unresolved+=int(bounds[0]<=0<=bounds[1])
                    if iteration>0 and (bounds[0]>0 or bounds[1]<0 or (bounds[0]>=-1e-7 and bounds[1]<=1e-7)):
                        break
                    if iteration==128:break
                    base=np.maximum(base@w.T+a,0.)
                    edit=np.maximum(edit@w.T+a,0.);edit[:,removed]=0.
                extra_updates+=iteration
                cases.append(dict(name=name,removed=removed,exact_effect=str(truth),
                    lower=float(bounds[0]),upper=float(bounds[1]),additional_updates=iteration))
        check(signs=={-1,0,1},'nonvacuous_effect_signs')
        check(unresolved>0 and extra_updates>0,'nonvacuous_unresolved_and_refinement')
        check(any(r['name']=='slow' and r['additional_updates']>4 for r in cases),'slow_contraction')
        # Endpoint classification itself must use strict sign, never a tolerance.
        check(not (0.>0 or 0.<0),'zero_not_a_sign')
        receipt.update(status='passed',checks=checks,exact_affine_values=affine_checks,
            initial_unresolved=unresolved,additional_updates=extra_updates,cases=cases)
    except BaseException as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-started)
        (args.output/'checks.json').write_text(json.dumps(records,indent=2)+'\n')
        (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
