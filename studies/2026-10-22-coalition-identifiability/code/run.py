"""Exact census of nonconstant monotone four-input Boolean functions."""
import csv, hashlib, itertools, json, platform, subprocess, sys, time, fcntl
from collections import defaultdict, Counter
from pathlib import Path
STUDY=Path(__file__).resolve().parents[1];ROOT=STUDY.parents[1]
sys.path.insert(0,str(ROOT/'studies/2026-10-19-capability-gates/code'))
from common import check_stop
N=4;FULL=15


def value(table,mask): return (table>>mask)&1
def subset(a,b): return (a&b)==a
def monotone(table):
    return all(value(table,m)<=value(table,m|(1<<i)) for m in range(16) for i in range(4) if not(m>>i&1))
def minimal(items): return sorted(m for m in items if not any(q!=m and subset(q,m) for q in items))
def signature(table):
    return ''.join(str(value(table,1<<i)) for i in range(4))+''.join(str(1-value(table,FULL^(1<<i))) for i in range(4))
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def run(lock):
    start=time.monotonic();deadline=start+300
    for p in [STUDY/'analysis-plan.md',STUDY/'sources.json',ROOT/'studies/2026-10-19-capability-gates/code/common.py',*sorted((STUDY/'code').glob('*.py'))]:
        assert p.read_bytes()==subprocess.check_output(['git','show',lock+':'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
    (ROOT/'.local-state').mkdir(exist_ok=True)
    lease=(ROOT/'.local-state/research-compute.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=STUDY/'results';out.mkdir(exist_ok=False)
    rows=[];groups=defaultdict(list);tables=[];coalitions=[]
    for table in range(1<<16):
        if table%512==0:check_stop(deadline)
        if value(table,0) or not value(table,FULL) or not monotone(table):continue
        tables.append(table)
        suff=[m for m in range(16) if value(table,m)]
        disabling=[m for m in range(16) if not value(table,FULL^m)]
        ms=minimal(suff);md=minimal(disabling);sig=signature(table);groups[sig].append(table)
        # Every disabling set must hit every minimal sufficient set, and conversely.
        for m in range(16):
            hit=all(bool(m&s) for s in ms)
            assert hit==(m in disabling)
            coalitions.append(dict(table=table,mask=m,size=m.bit_count(),sufficient=m in suff,
                                   disabling=m in disabling,minimal_sufficient=m in ms,minimal_disabling=m in md))
        rows.append(dict(table=table,signature=sig,min_sufficient_size=min(m.bit_count() for m in ms),
                         min_disabling_size=min(m.bit_count() for m in md),minimal_sufficient=ms,minimal_disabling=md,
                         no_singleton_sufficient=not any(x=='1' for x in sig[:4]),
                         no_singleton_disabling=not any(x=='1' for x in sig[4:])))
    # An independent generator enumerates antichains of nonempty input subsets.
    antichain_tables=set()
    for bits in range(1,1<<15):
        if bits%512==0:check_stop(deadline)
        candidates=[m for m in range(1,16) if bits>>(m-1)&1]
        if any(subset(a,b) or subset(b,a) for a,b in itertools.combinations(candidates,2)):continue
        table=sum(1<<m for m in range(16) if any(subset(a,m) for a in candidates))
        antichain_tables.add(table)
    assert set(tables)==antichain_tables
    for row in rows:row['signature_class_size']=len(groups[row['signature']])
    dump(out/'functions.json',rows)
    with (out/'coalitions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(coalitions[0]));w.writeheader();w.writerows(coalitions)
    dump(out/'signature_groups.json',{k:v for k,v in sorted(groups.items())})
    # Reconstruct each truth table solely from its saved minimal sufficient sets.
    for row in json.loads((out/'functions.json').read_text()):
        reconstructed=sum(1<<m for m in range(16) if any((s&m)==s for s in row['minimal_sufficient']))
        assert reconstructed==row['table']
    metrics=dict(status='complete',functions=len(tables),signature_classes=len(groups),
                 ambiguous_functions=sum(len(v) for v in groups.values() if len(v)>1),
                 unique_functions=sum(len(v)==1 for v in groups.values()),largest_signature_class=max(map(len,groups.values())),
                 no_singleton_sufficient=sum(r['no_singleton_sufficient'] for r in rows),
                 no_singleton_disabling=sum(r['no_singleton_disabling'] for r in rows),
                 neither_singleton_test_detects_any=sum(r['no_singleton_sufficient'] and r['no_singleton_disabling'] for r in rows),
                 coalition_rows=len(coalitions),duality_checks=len(coalitions),independent_antichain_match=True,
                 saved_truth_table_replay=True,trained_models=0,public_lock=lock,
                 seconds=time.monotonic()-start,python=platform.python_version(),threads=1,
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
