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
    cases={x['name']:x for x in json.loads((out/'circuits.json').read_text())}
    routebits={name:[sum(2**i for i in r) for r in c['routes']] for name,c in cases.items()}
    def simulate(case,mask,x):
        route_values=[]
        for route in cases[case]['routes']:
            hidden_sum=sum(x*int(bool(mask&2**i)) for i in route)
            route_values.append(max(0,hidden_sum-len(route)+1))
        return 1-max(0,1-sum(route_values))
    subsets=read(out/'subsets.csv');scores={}
    for row in subsets:
        check_stop(deadline);name=row['case'];mask=int(row['mask']);y0=simulate(name,mask,0);y1=simulate(name,mask,1)
        okay=int(any(mask&r==r for r in routebits[name]))
        assert y0==0 and y1==okay
        assert int(row['output_0'])==y0 and int(row['output_1'])==y1 and int(row['faithful'])==okay
        assert int(row['size'])==mask.bit_count()
        scores[(name,mask)]=okay
    opt={name:min(mask.bit_count() for (case,mask),good in scores.items() if case==name and good) for name in cases}
    counts={name:[] for name in cases};families={name:[] for name in cases}
    orders=read(out/'orders.csv')
    for index,row in enumerate(orders):
        if index%128==0:check_stop(deadline)
        name=row['case'];order=list(map(int,row['order'].split()));n=cases[name]['n'];assert sorted(order)==list(range(n))
        mask=(1<<n)-1
        for i in order:
            candidate=mask&~(1<<i)
            if any(candidate&r==r for r in routebits[name]):mask=candidate
        assert mask==int(row['final_mask']) and mask.bit_count()==int(row['size'])
        assert opt[name]==int(row['optimum']) and int(row['gap'])==mask.bit_count()-opt[name]
        assert scores[(name,mask)] and all(not scores[(name,mask&~(1<<i))] for i in range(n) if mask>>i&1)
        assert int(row['queries'])==n;counts[name].append(mask.bit_count())
        k=cases[name]['family_k']
        if k is not None:
            assert mask.bit_count()==(k if order[0]==0 else 1)
    for row in read(out/'case_summary.csv'):
        name=row['case'];values=counts[name];n=cases[name]['n'];bad=sum(v>opt[name] for v in values)
        assert len(values)==math.factorial(n)==int(row['orders'])
        assert int(row['optimum'])==opt[name] and int(row['suboptimal_orders'])==bad
        assert Fraction(row['suboptimal_fraction'])==Fraction(bad,len(values))
        assert Fraction(row['mean_size'])==Fraction(sum(values),len(values))
        assert int(row['max_size'])==max(values) and int(row['max_gap'])==max(values)-opt[name]
        k=cases[name]['family_k']
        if k is not None:assert bad==math.factorial(k) and max(values)==k and opt[name]==1
    return {'passed':True,'subset_rows':len(subsets),'order_rows':len(orders),'method':'Independent integer masks and threshold-gate simulation; exact equality.'}
