"""Exact, constructed donor-baseline study. No fitted neural network."""
from fractions import Fraction as F
from itertools import product
import csv, hashlib, json, os, platform, subprocess, sys, time, fcntl
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT / 'studies/2026-10-19-capability-gates/code'))
from common import check_stop

BETAS = (F(0), F(1, 2), F(1), F(2), F(4), F(8))
STATES = tuple(product((-1, 1), repeat=2))
METHODS = ('zero', 'mean', 'marginal', 'nuisance_matched', 'label_matched', 'opposite_label', 'coherent_opposite')


def native(h, beta):
    value = h[0] - 2
    return value + beta * (h[2] - value * h[1])


def reference(h, beta):
    # Expanded polynomial, separate from the implementation above.
    a, b, c = h
    return a - 2 + beta*c - beta*a*b + 2*beta*b


def configurations(s, n, method):
    if method == 'zero':
        return [(F(1), (F(0), F(n), F(s*n)), None)]
    if method == 'mean':
        return [(F(1), (F(2), F(n), F(s*n)), None)]
    if method == 'coherent_opposite':
        return [(F(1), (F(2-s), F(n), F(-s*n)), -s)]
    donors = [d for d in STATES if
              (method == 'marginal' or
               method == 'nuisance_matched' and d[1] == n or
               method == 'label_matched' and d[0] == s or
               method == 'opposite_label' and d[0] == -s)]
    return [(F(1, len(donors)), (F(2+ds), F(n), F(s*n)), ds) for ds, dn in donors]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def run(lock):
    start = time.monotonic(); deadline = start+300
    for p in [STUDY/'analysis-plan.md', STUDY/'sources.json',ROOT/'studies/2026-10-19-capability-gates/code/common.py', *sorted((STUDY/'code').glob('*.py'))]:
        assert p.read_bytes() == subprocess.check_output(['git', 'show', lock+':'+p.relative_to(ROOT).as_posix()], cwd=ROOT)
    (ROOT/'.local-state').mkdir(exist_ok=True)
    lease=(ROOT/'.local-state/research-compute.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out = STUDY/'results'; out.mkdir(exist_ok=False)
    rows=[]; summaries=[]; audits=0
    for beta in BETAS:
        check_stop(deadline)
        assert all(native((F(2+s),F(n),F(s*n)),beta)==s for s,n in STATES)
        for method in METHODS:
            accuracy=F(0); donor_accuracy=F(0); donor_mass=F(0); off_support=F(0); mean_abs_change=F(0)
            for s,n in STATES:
                for probability,h,donor_s in configurations(s,n,method):
                    weight=probability/4; score=native(h,beta); expected=reference(h,beta)
                    assert score==expected; audits+=1
                    pred=1 if score>=0 else -1
                    supported=bool(h in [(F(2+a),F(b),F(a*b)) for a,b in STATES])
                    accuracy+=weight*int(pred==s); off_support+=weight*int(not supported)
                    mean_abs_change+=weight*abs(score-s)
                    if donor_s is not None:
                        donor_accuracy+=weight*int(pred==donor_s); donor_mass+=weight
                    rows.append(dict(beta=str(beta),method=method,s=s,n=n,donor_s=donor_s,
                                     probability=str(weight),h1=str(h[0]),h2=str(h[1]),h3=str(h[2]),
                                     score=str(score),prediction=pred,on_support=supported))
            summaries.append(dict(beta=str(beta),method=method,original_accuracy=str(accuracy),
                                  donor_accuracy=str(donor_accuracy/donor_mass) if donor_mass else None,
                                  off_support_rate=str(off_support),mean_absolute_score_change=str(mean_abs_change)))
    with (out/'raw.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    # Independent saved-row audit reconstructs each score from the expanded equation.
    saved=list(csv.DictReader((out/'raw.csv').open()))
    aggregate={}
    for row in saved:
        check_stop(deadline)
        h=tuple(F(row[k]) for k in ('h1','h2','h3'))
        score=reference(h,F(row['beta']));prediction=1 if score>=0 else -1
        assert score==F(row['score']) and prediction==int(row['prediction'])
        supported=h[0] in (1,3) and h[1] in (-1,1) and h[2]==(h[0]-2)*h[1]
        assert str(supported)==row['on_support']
        key=(row['beta'],row['method']);weight=F(row['probability']);s=int(row['s'])
        a=aggregate.setdefault(key,dict(mass=F(0),accuracy=F(0),off_support=F(0),change=F(0),donor_accuracy=F(0),donor_mass=F(0)))
        a['mass']+=weight;a['accuracy']+=weight*int(prediction==s)
        a['off_support']+=weight*int(not supported);a['change']+=weight*abs(score-s)
        if row['donor_s']:
            a['donor_mass']+=weight;a['donor_accuracy']+=weight*int(prediction==int(row['donor_s']))
    summary_saved=list(csv.DictReader((out/'summary.csv').open()))
    for row in summary_saved:
        a=aggregate[(row['beta'],row['method'])]
        assert a['mass']==1 and a['accuracy']==F(row['original_accuracy'])
        assert a['off_support']==F(row['off_support_rate']) and a['change']==F(row['mean_absolute_score_change'])
        if a['donor_mass']:
            assert a['donor_accuracy']/a['donor_mass']==F(row['donor_accuracy'])
        else:
            assert row['donor_accuracy']==''
    compared=('original_accuracy','off_support_rate','mean_absolute_score_change')
    zero_equals_mean=all(all(next(r for r in summaries if r['beta']==str(beta) and r['method']=='zero')[k]
                             ==next(r for r in summaries if r['beta']==str(beta) and r['method']=='mean')[k]
                             for k in compared) for beta in BETAS)
    metrics=dict(status='complete',model_type='constructed polynomial hidden-state model',trained_models=0,
                 parameter_settings=len(BETAS),methods=len(METHODS),raw_rows=len(rows),summary_rows=len(summaries),
                 exact_reference_checks=audits,saved_row_replay=True,weighted_summary_replay=True,ordinary_accuracy='1',
                 zero_equals_mean=zero_equals_mean,public_lock=lock,seconds=time.monotonic()-start,
                 python=platform.python_version(),arithmetic='fractions.Fraction',threads=1,
                 plan_sha256=hashlib.sha256((STUDY/'analysis-plan.md').read_bytes()).hexdigest())
    dump(out/'metrics.json',metrics)
    dump(out/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(metrics))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True)
    args=p.parse_args()
    try:
        run(args.lock)
    except BaseException as exc:
        out=STUDY/'results'
        if out.exists():
            (out/'failure.json').write_text(json.dumps({'status':'failed','error':repr(exc),'public_lock':args.lock},indent=2)+'\n')
        raise
