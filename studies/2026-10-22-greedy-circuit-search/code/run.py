"""Fixed exact calculation. No learned model or statistical sample."""
import argparse, csv, hashlib, importlib.util, itertools, json, os, platform, resource, subprocess, sys, time
from pathlib import Path
from fractions import Fraction
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parents[1]
sys.path.insert(0,str(ROOT/'studies/2026-10-19-capability-gates/code'))
from common import check_stop

def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def table(path,rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def lock(commit):
    paths=[STUDY/'plan.md',STUDY/'sources.json',*sorted((STUDY/'code').glob('*.py')),
           ROOT/'studies/2026-10-19-capability-gates/code/common.py']
    for p in paths:
        rel=p.relative_to(ROOT).as_posix()
        assert subprocess.check_output(['git','show',commit+':'+rel],cwd=ROOT)==p.read_bytes(),rel

def launch(calculate):
    parser=argparse.ArgumentParser();parser.add_argument('--lock',required=True);args=parser.parse_args()
    lock(args.lock);check_stop(float('inf'))
    out=STUDY/'results';out.mkdir(exist_ok=False)
    start=time.monotonic();deadline=min(start+300,float(os.environ.get('MI_STAGE_DEADLINE','inf')))
    os.environ['MI_STAGE_DEADLINE']=str(deadline)
    meta={'public_lock':args.lock,'python':platform.python_version(),'platform':platform.platform(),
          'arithmetic':'Exact integers and rational numbers','trained_models':0,'status':'running'}
    try:
        summary=calculate(out,deadline);dump(out/'summary.json',summary)
        spec=importlib.util.spec_from_file_location('independent_audit',STUDY/'code/audit.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        check_stop(deadline);audit=module.audit(out,deadline);dump(out/'audit.json',audit)
        assert audit['passed'];check_stop(deadline);lock(args.lock);meta['status']='complete'
    except BaseException as exc:
        meta.update(status='failed',reason=repr(exc));raise
    finally:
        meta['seconds']=time.monotonic()-start
        meta['peak_rss_native_units']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        dump(out/'metadata.json',meta)
        dump(out/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'})
    print(json.dumps({'status':meta['status'],'seconds':meta['seconds'],'summary':summary}),flush=True)

import random

def cases():
    result=[]
    for k in range(2,7):
        result.append({'name':'singleton_chain_'+str(k),'n':k+1,'family_k':k,'routes':[[0],list(range(1,k+1))]})
    rng=random.Random(22102201)
    choices=[list(s) for k in range(1,5) for s in itertools.combinations(range(6),k)]
    choices.sort()
    for i in range(12):
        drawn=rng.sample(choices,6)
        routes=[s for s in drawn if not any(set(t)<set(s) for t in drawn)]
        result.append({'name':'fixed_random_'+str(i),'n':6,'family_k':None,'routes':sorted(routes)})
    return result

def output(x,keep,routes):
    hidden=[x if i in keep else 0 for i in range(7)]
    route_values=[int(all(hidden[i] for i in route)) for route in routes]
    return int(any(route_values))

def faithful(keep,routes):
    return all(output(x,keep,routes)==x for x in (0,1))

def calculate(out,deadline):
    models=cases();dump(out/'circuits.json',models)
    subsets=[];orders=[];summary=[]
    for case in models:
        check_stop(deadline);n=case['n'];routes=case['routes']
        assert faithful(set(range(n)),routes) and not faithful(set(),routes)
        good=[]
        for mask in range(1<<n):
            keep={i for i in range(n) if mask>>i&1}
            okay=faithful(keep,routes)
            subsets.append(dict(case=case['name'],mask=mask,size=len(keep),output_0=output(0,keep,routes),output_1=output(1,keep,routes),faithful=int(okay)))
            if okay:good.append(len(keep))
        optimum=min(good);sizes=[];bad=0
        for index,order in enumerate(itertools.permutations(range(n))):
            if index%128==0:check_stop(deadline)
            keep=set(range(n))
            for i in order:
                proposal=keep-{i}
                if faithful(proposal,routes):keep=proposal
            assert faithful(keep,routes) and all(not faithful(keep-{i},routes) for i in keep)
            mask=sum(1<<i for i in keep);size=len(keep);sizes.append(size);bad+=size>optimum
            orders.append(dict(case=case['name'],order=' '.join(map(str,order)),final_mask=mask,size=size,optimum=optimum,gap=size-optimum,queries=n))
        row=dict(case=case['name'],n=n,routes=len(routes),orders=len(sizes),optimum=optimum,
                 suboptimal_orders=bad,suboptimal_fraction=str(Fraction(bad,len(sizes))),
                 mean_size=str(Fraction(sum(sizes),len(sizes))),max_size=max(sizes),max_gap=max(sizes)-optimum)
        if case['family_k']:
            k=case['family_k'];assert Fraction(bad,len(sizes))==Fraction(1,k+1)
            assert Fraction(sum(sizes),len(sizes))==1+Fraction(k-1,k+1)
        summary.append(row)
    table(out/'subsets.csv',subsets);table(out/'orders.csv',orders);table(out/'case_summary.csv',summary)
    return {'cases':len(models),'subset_checks':len(subsets),'order_checks':len(orders),'case_results':summary,'scope':'Fixed constructed circuits; no population estimate.'}

if __name__=='__main__':
    import fcntl
    with (ROOT/'.local-state/research-compute.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        launch(calculate)
