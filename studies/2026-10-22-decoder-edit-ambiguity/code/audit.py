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
    groups={};rows=read(out/'edits.csv')
    for index,row in enumerate(rows):
        if index%128==0:check_stop(deadline)
        name=row['decoder'];m=int(row['mask']);b=[(m//(2**i))%2 for i in range(3)]
        x0,x1,y0,y1=[int(row[k]) for k in ('x0','x1','y0','y1')]
        tr,td=int(row['receiver_lift']),int(row['donor_lift'])
        if name=='coupled':
            r=(x0-tr,x1-tr,tr);d=(y0-td,y1-td,td);dm=(b[2]-b[0],b[2]-b[1]);reference=(x0+b[0]*(y0-x0),x1+b[1]*(y1-x1))
            def decode(z):return z[0]+z[2],z[1]+z[2]
        elif name=='duplicate':
            r=(x0-tr,tr,x1);d=(y0-td,td,y1);dm=(b[1]-b[0],0);reference=(x0+b[0]*(y0-x0),x1+b[2]*(y1-x1))
            def decode(z):return z[0]+z[1],z[2]
        else:
            assert name=='silent';r=(x0,x1,tr);d=(y0,y1,td);dm=(0,0);reference=(x0+b[0]*(y0-x0),x1+b[1]*(y1-x1))
            def decode(z):return z[0],z[1]
        assert decode(r)==(x0,x1) and decode(d)==(y0,y1)
        edited=decode(tuple(d[i] if b[i] else r[i] for i in range(3)))
        delta=((td-tr)*dm[0],(td-tr)*dm[1]);l1=abs(delta[0])+abs(delta[1]);readout=delta[0]-delta[1]
        assert edited==(reference[0]+delta[0],reference[1]+delta[1])
        assert edited==(int(row['edited0']),int(row['edited1']))
        assert reference==(int(row['reference0']),int(row['reference1']))
        assert l1==int(row['l1_discrepancy']) and readout==int(row['readout_discrepancy'])
        nn=int(all(v>=0 for v in r+d));assert nn==int(row['nonnegative'])
        groups.setdefault((name,m),[]).append((l1,abs(readout),nn,dm))
    for row in read(out/'mask_summary.csv'):
        values=groups[(row['decoder'],int(row['mask']))];dm=values[0][3]
        assert len(values)==400
        assert int(row['activation_independent'])==int(dm==(0,0))
        assert int(row['readout_independent'])==int(dm[0]==dm[1])
        assert int(row['max_l1'])==max(v[0] for v in values)
        assert int(row['max_readout'])==max(v[1] for v in values)
        nonnegative=[v for v in values if v[2]]
        assert int(row['nonnegative_cases'])==len(nonnegative)
        assert int(row['max_nonnegative_l1'])==max(v[0] for v in nonnegative)
        assert int(row['max_nonnegative_readout'])==max(v[1] for v in nonnegative)
        assert (max(v[0] for v in values)==0)==(dm==(0,0))
    assert len(rows)==9600
    return {'passed':True,'exact_edit_rows':len(rows),'masks':len(groups),'method':'Independent scalar decoding and selection equations; all reconstruction and edit comparisons exact.'}
