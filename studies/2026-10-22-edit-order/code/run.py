"""Exact ordered-projection audit on a fixed rational grid."""
from fractions import Fraction as F
from itertools import product
import csv, hashlib, json, platform, subprocess, sys, time, fcntl
from pathlib import Path
STUDY=Path(__file__).resolve().parents[1];ROOT=STUDY.parents[1]
sys.path.insert(0,str(ROOT/'studies/2026-10-19-capability-gates/code'))
from common import check_stop
VECTORS=((F(1),F(0)),(F(0),F(1)),(F(3,5),F(4,5)),(F(4,5),F(3,5)),(F(5,13),F(12,13)))
GRID=tuple(tuple(map(F,x)) for x in product((-1,0,1),repeat=2))


def add(a,b):return tuple(x+y for x,y in zip(a,b))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def scale(s,a):return tuple(s*x for x in a)
def project(u,x):return scale(dot(u,x),u)
def patch(u,x,a):return add(x,project(u,sub(a,x)))
def matrix(u):return ((u[0]*u[0],u[0]*u[1]),(u[1]*u[0],u[1]*u[1]))
def mv(m,x):return tuple(dot(row,x) for row in m)
def mm(a,b):return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)) for i in range(2))
def vstr(x):return ','.join(map(str,x))
def dump(p,o):p.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n')


def joint(u,v,x,a,b):
    ta,tb=dot(u,a),dot(v,b);det=u[0]*v[1]-u[1]*v[0]
    if det==0:
        # In the locked grid, the parallel case is exactly u=v=(1,0).
        assert u==v
        return patch(u,x,a) if ta==tb else None
    return ((ta*v[1]-u[1]*tb)/det,(u[0]*tb-ta*v[0])/det)


def run(lock):
    start=time.monotonic();deadline=start+300
    for p in [STUDY/'analysis-plan.md',STUDY/'sources.json',ROOT/'studies/2026-10-19-capability-gates/code/common.py',*sorted((STUDY/'code').glob('*.py'))]:
        assert p.read_bytes()==subprocess.check_output(['git','show',lock+':'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
    (ROOT/'.local-state').mkdir(exist_ok=True)
    lease=(ROOT/'.local-state/research-compute.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=STUDY/'results';out.mkdir(exist_ok=False)
    rows=[];summaries=[];u=VECTORS[0]
    for vi,v in enumerate(VECTORS):
        check_stop(deadline)
        assert dot(v,v)==1
        p,q=matrix(u),matrix(v);pq,qp=mm(p,q),mm(q,p)
        summary=dict(vector=vstr(v),rows=0,order_changes=0,same_donor_rows=0,same_donor_order_changes=0,
                     sequential_first_constraint_failures=0,joint_solutions=0,inconsistent_constraints=0,
                     max_gap_squared=F(0),max_joint_change_squared=F(0),projectors_commute=pq==qp)
        for x,a,b in product(GRID,repeat=3):
            check_stop(deadline)
            first=patch(v,patch(u,x,a),b);second=patch(u,patch(v,x,b),a)
            gap=sub(first,second)
            formula=add(sub(mv(qp,x),mv(pq,x)),sub(mv(pq,b),mv(qp,a)))
            assert gap==formula
            # No-op and repeated same edit are exact identity/idempotence controls.
            assert patch(u,x,x)==x and patch(v,patch(v,x,b),b)==patch(v,x,b)
            solution=joint(u,v,x,a,b)
            reverse=joint(v,u,x,b,a)
            assert solution==reverse
            if solution is not None:
                assert dot(u,solution)==dot(u,a) and dot(v,solution)==dot(v,b)
            same=a==b;changed=any(gap);gap2=dot(gap,gap)
            residual=dot(u,first)-dot(u,a)
            assert dot(v,first)==dot(v,b)
            summary['rows']+=1;summary['order_changes']+=changed
            summary['same_donor_rows']+=same;summary['same_donor_order_changes']+=same and changed
            summary['sequential_first_constraint_failures']+=bool(residual)
            summary['joint_solutions']+=solution is not None;summary['inconsistent_constraints']+=solution is None
            summary['max_gap_squared']=max(summary['max_gap_squared'],gap2)
            if solution is not None:summary['max_joint_change_squared']=max(summary['max_joint_change_squared'],dot(sub(solution,x),sub(solution,x)))
            rows.append(dict(vector_id=vi,vector=vstr(v),base=vstr(x),donor_a=vstr(a),donor_b=vstr(b),
                             same_donor=same,first_then_second=vstr(first),second_then_first=vstr(second),
                             gap=vstr(gap),gap_squared=str(gap2),first_constraint_residual=str(residual),
                             joint=vstr(solution) if solution is not None else 'inconsistent'))
        for k in ('max_gap_squared','max_joint_change_squared'):summary[k]=str(summary[k])
        summaries.append(summary)
    with (out/'raw.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Saved-row replay uses scalar formulas, independent of the patch function.
    for row in csv.DictReader((out/'raw.csv').open()):
        parse=lambda s:tuple(F(x) for x in s.split(','))
        v=parse(row['vector']);x=parse(row['base']);a=parse(row['donor_a']);b=parse(row['donor_b'])
        first0,first1=a[0],x[1]
        delta=v[0]*(b[0]-first0)+v[1]*(b[1]-first1)
        replay=(first0+v[0]*delta,first1+v[1]*delta)
        assert replay==parse(row['first_then_second'])
    dump(out/'summary.json',summaries)
    metrics=dict(status='complete',rows=len(rows),vector_pairs=len(VECTORS),trained_models=0,
                 exact_commutator_checks=len(rows),saved_row_replay=True,joint_order_check=True,
                 public_lock=lock,seconds=time.monotonic()-start,threads=1,python=platform.python_version(),
                 arithmetic='fractions.Fraction',plan_sha256=hashlib.sha256((STUDY/'analysis-plan.md').read_bytes()).hexdigest())
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
