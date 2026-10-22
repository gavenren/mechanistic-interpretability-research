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

DECODERS=[
 {'name':'coupled','D':[[1,0,1],[0,1,1]],'null':[-1,-1,1]},
 {'name':'duplicate','D':[[1,1,0],[0,0,1]],'null':[-1,1,0]},
 {'name':'silent','D':[[1,0,0],[0,1,0]],'null':[0,0,1]}]

def matvec(D,z):return tuple(sum(a*b for a,b in zip(row,z)) for row in D)
def canonical(name,x):return (x[0],0,x[1]) if name=='duplicate' else (x[0],x[1],0)

def calculate(out,deadline):
    dump(out/'decoders.json',DECODERS);rows=[];summaries=[]
    points=list(itertools.product((1,2),repeat=2));lifts=range(-2,3)
    for case in DECODERS:
        D=case['D'];v=case['null'];assert matvec(D,v)==(0,0)
        for mask in range(8):
            check_stop(deadline);bits=[mask>>i&1 for i in range(3)]
            dm=matvec(D,[b*n for b,n in zip(bits,v)])
            max_l1=max_read=max_nonnegative=max_nonnegative_read=0;nonnegative_count=0
            for x,y in itertools.product(points,repeat=2):
                cx,cy=canonical(case['name'],x),canonical(case['name'],y)
                reference=matvec(D,[a+b*(d-a) for a,d,b in zip(cx,cy,bits)])
                for tr,td in itertools.product(lifts,repeat=2):
                    z=[a+tr*n for a,n in zip(cx,v)];w=[a+td*n for a,n in zip(cy,v)]
                    assert matvec(D,z)==x and matvec(D,w)==y
                    edited=matvec(D,[a+b*(d-a) for a,d,b in zip(z,w,bits)])
                    delta=tuple(a-b for a,b in zip(edited,reference));l1=sum(abs(a) for a in delta);read=delta[0]-delta[1]
                    assert delta==tuple((td-tr)*a for a in dm)
                    if mask==0:assert edited==x
                    if mask==7:assert edited==y
                    if z==w:assert edited==x
                    nn=int(min(z+w)>=0);nonnegative_count+=nn
                    max_l1=max(max_l1,l1);max_read=max(max_read,abs(read))
                    if nn:max_nonnegative=max(max_nonnegative,l1);max_nonnegative_read=max(max_nonnegative_read,abs(read))
                    rows.append(dict(decoder=case['name'],mask=mask,x0=x[0],x1=x[1],y0=y[0],y1=y[1],receiver_lift=tr,donor_lift=td,
                                     nonnegative=nn,edited0=edited[0],edited1=edited[1],reference0=reference[0],reference1=reference[1],l1_discrepancy=l1,readout_discrepancy=read))
            summaries.append(dict(decoder=case['name'],mask=mask,activation_independent=int(dm==(0,0)),
                                  readout_independent=int(dm[0]-dm[1]==0),max_l1=max_l1,max_readout=max_read,
                                  nonnegative_cases=nonnegative_count,max_nonnegative_l1=max_nonnegative,max_nonnegative_readout=max_nonnegative_read))
    table(out/'edits.csv',rows);table(out/'mask_summary.csv',summaries)
    return {'decoders':3,'masks_per_decoder':8,'edit_cases':len(rows),'reconstruction_error':0,'mask_results':summaries,
            'scope':'Constructed linear codes, with no trained encoder or sparsity claim.'}

if __name__=='__main__':
    import fcntl
    with (ROOT/'.local-state/research-compute.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        launch(calculate)
