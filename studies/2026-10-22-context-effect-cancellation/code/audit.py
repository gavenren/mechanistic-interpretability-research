"""Separate scalar/bitmask audit. Does not import the scientific runner."""
import csv,json,itertools,math,sys
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'studies/2026-10-19-capability-gates/code'))
from common import check_stop

def read(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))

def audit(out,deadline):
    raw=read(out/'effects.csv');state_rows=read(out/'states.csv');models={}
    def sign(z):return -1 if z<0 else 1
    for row in state_rows:
        a,b,c,t,n=[int(row[k]) for k in ('alpha','beta','c','t','n')]
        score=a*c*t+b*n;truth=c*t
        assert int(row['score'])==score and int(row['target'])==truth and int(row['prediction'])==sign(score)
        assert int(row['correct'])==int(sign(score)==truth)
        models.setdefault((a,b),[]).append(sign(score)==truth)
    effects={}
    for row in raw:
        a,b,c,o=[int(row[k]) for k in ('alpha','beta','c','other')]
        if row['edit']=='semantic':before=-a*c+b*o;after=a*c+b*o;target=c;delta=2*a*c
        else:before=a*c*o-b;after=a*c*o+b;target=c*o;delta=2*b
        assert [int(row[k]) for k in ('before','after','delta','target')]==[before,after,delta,target]
        assert int(row['after_prediction'])==sign(after) and int(row['before_prediction'])==sign(before)
        assert int(row['correct_after'])==int(sign(after)==target)
        assert int(row['answer_changed'])==int(sign(after)!=sign(before))
        effects.setdefault((a,b,row['edit']),[]).append(row)
    mixtures=read(out/'mixtures.csv')
    for row in mixtures:
        check_stop(deadline);a,b=int(row['alpha']),int(row['beta']);p=Fraction(row['p'])
        capable=all(models[(a,b)]);assert capable==(a>b)==bool(int(row['capable']))
        assert Fraction(row['intact_accuracy'])==Fraction(sum(models[(a,b)]),8)
        assert Fraction(row['semantic_mean'])==2*a*(2*p-1)
        assert Fraction(row['semantic_abs_mean'])==abs(2*a*(2*p-1))
        assert Fraction(row['semantic_mean_abs'])==2*a==Fraction(row['semantic_aligned'])
        assert Fraction(row['nuisance_mean'])==2*b==Fraction(row['nuisance_mean_abs'])
        assert Fraction(row['semantic_context_minus'])==-2*a and Fraction(row['semantic_context_plus'])==2*a
        assert Fraction(row['nuisance_context_minus'])==2*b==Fraction(row['nuisance_context_plus'])
        for edit in ('semantic','nuisance'):
            vals=effects[(a,b,edit)];success=sum((p if int(v['c'])==1 else 1-p)*Fraction(int(v['correct_after']),2) for v in vals)
            assert Fraction(row[edit+'_success'])==success
        changed=sum((p if int(v['c'])==1 else 1-p)*Fraction(int(v['answer_changed']),2) for v in effects[(a,b,'nuisance')])
        assert Fraction(row['nuisance_answer_change'])==changed
        assert bool(int(row['nuisance_ranks_above_semantic']))==(2*b>abs(2*a*(2*p-1)))
        if capable:assert Fraction(row['semantic_success'])==1 and changed==0
    summary=json.loads((out/'summary.json').read_text())
    balanced=[r for r in mixtures if Fraction(r['p'])==Fraction(1,2)]
    assert summary['capable_models']==sum(int(r['capable']) for r in balanced)
    assert summary['capable_balanced_rank_reversals']==sum(int(r['capable'])*int(r['nuisance_ranks_above_semantic']) for r in balanced)
    return {'passed':True,'state_rows':len(state_rows),'individual_edit_rows':len(raw),'mixture_rows':len(mixtures),
            'method':'Independent exact scalar equations, state truth, and rational mixture weights.'}
