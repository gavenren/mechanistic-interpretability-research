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

def prediction(value):return 1 if value>=0 else -1

def calculate(out,deadline):
    raw=[];states=[];summaries=[]
    weights=[Fraction(k,4) for k in range(5)]
    for alpha,beta in itertools.product(range(1,5),range(5)):
        check_stop(deadline);correct=0
        for c,t,n in itertools.product((-1,1),repeat=3):
            score=alpha*c*t+beta*n;target=c*t;guess=prediction(score);correct+=guess==target
            states.append(dict(alpha=alpha,beta=beta,c=c,t=t,n=n,score=score,target=target,prediction=guess,correct=int(guess==target)))
        effects={}
        for edit in ('semantic','nuisance'):
            effects[edit]=[]
            for c,other in itertools.product((-1,1),repeat=2):
                t,n=(-1,other) if edit=='semantic' else (other,-1)
                before=alpha*c*t+beta*n
                new_t,new_n=(1,n) if edit=='semantic' else (t,1)
                after=alpha*c*new_t+beta*new_n;target=c*new_t
                row=dict(alpha=alpha,beta=beta,edit=edit,c=c,other=other,before=before,after=after,delta=after-before,
                         before_prediction=prediction(before),after_prediction=prediction(after),target=target,
                         correct_after=int(prediction(after)==target),answer_changed=int(prediction(after)!=prediction(before)))
                raw.append(row);effects[edit].append(row)
                assert (alpha*c*t+beta*n)-before==0
        for p in weights:
            values={}
            for edit in effects:
                rows=effects[edit]
                probs=[(p if r['c']==1 else 1-p)/2 for r in rows]
                mean=sum(q*r['delta'] for q,r in zip(probs,rows));mag=sum(q*abs(r['delta']) for q,r in zip(probs,rows))
                aligned=sum(q*r['c']*r['delta'] for q,r in zip(probs,rows))
                success=sum(q*r['correct_after'] for q,r in zip(probs,rows));changed=sum(q*r['answer_changed'] for q,r in zip(probs,rows))
                values[edit]=dict(mean=mean,abs_mean=abs(mean),mean_abs=mag,aligned=aligned,success=success,changed=changed)
            s=values['semantic'];n=values['nuisance']
            assert s['mean']==2*alpha*(2*p-1) and s['mean_abs']==2*alpha and s['aligned']==2*alpha
            assert n['mean']==2*beta and n['mean_abs']==2*beta
            summaries.append(dict(alpha=alpha,beta=beta,p=str(p),intact_accuracy=str(Fraction(correct,8)),capable=int(correct==8),
                 semantic_mean=str(s['mean']),semantic_abs_mean=str(s['abs_mean']),semantic_mean_abs=str(s['mean_abs']),
                 semantic_aligned=str(s['aligned']),semantic_success=str(s['success']),nuisance_mean=str(n['mean']),
                 nuisance_mean_abs=str(n['mean_abs']),nuisance_success=str(n['success']),nuisance_answer_change=str(n['changed']),
                 nuisance_ranks_above_semantic=int(n['abs_mean']>s['abs_mean'])))
    table(out/'states.csv',states);table(out/'effects.csv',raw);table(out/'mixtures.csv',summaries)
    balanced=[r for r in summaries if r['p']=='1/2']
    return {'constructed_models':20,'states':len(states),'individual_edits':len(raw),'mixtures':len(summaries),
            'capable_models':sum(r['capable'] for r in balanced),
            'capable_balanced_rank_reversals':sum(r['capable'] and r['nuisance_ranks_above_semantic'] for r in balanced),
            'balanced_case_results':balanced,'scope':'Exact constructed classifiers; known task alignment is extra information.'}

if __name__=='__main__':
    import fcntl
    with (ROOT/'.local-state/research-compute.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        launch(calculate)
