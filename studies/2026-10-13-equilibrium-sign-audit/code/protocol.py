"""Fixed effect classification and refinement; no data loading or training."""
import csv
import gzip
from pathlib import Path
import numpy as np
import intervals as iv
from model import mapping


def classify(lower, upper, step, cap=128):
    result=np.full(len(lower),'active',dtype='<U10')
    result[lower>0]='positive'
    result[upper<0]='negative'
    small=(result=='active')&(lower>=-1e-7)&(upper<=1e-7)
    result[small]='small'
    if step==cap:result[result=='active']='unresolved'
    return result


def sign_reversal(naive,status):
    return ((naive<0)&(status=='positive'))|((naive>0)&(status=='negative'))


def primary_counts(naive,lower,upper):
    nonzero=naive!=0
    unresolved=(lower<=0)&(upper>=0)
    denominator=int(nonzero.sum());numerator=int((nonzero&unresolved).sum())
    return dict(nonzero_ordinary=denominator,unresolved_nonzero=numerator,
        primary_fraction=numerator/denominator if denominator else None,
        effects=len(naive),ordinary_zero=int((~nonzero).sum()),
        unresolved_all=int(unresolved.sum()),unresolved_all_fraction=float(unresolved.mean()))


def bins(naive):
    value=np.abs(naive)
    return np.select([value==0,value<=1e-7,value<=1e-5,value<=1e-3],
        ['zero','(0,1e-7]','(1e-7,1e-5]','(1e-5,1e-3]'],default='above 1e-3')


def refine(base,edit,weight,injection_point,injection_interval,removed,readout,
           winner,runner,q_upper,out,control,tick,cap=128):
    """Freeze each pair at the first fixed stop rule. Save all checked bounds."""
    out=Path(out);n=len(base);base=base.copy();edit=edit.copy()
    status=np.full(n,'active',dtype='<U10');steps=np.zeros(n,dtype=np.int64)
    previous_lower=np.full(n,-np.inf);previous_upper=np.full(n,np.inf)
    initial=None;final={};calls=examples=cert_calls=cert_examples=0
    names=('lower','upper','radius','center_lower','center_upper',
           'coefficient_norm_upper','residual_base_upper','residual_edit_upper')
    final={key:np.zeros(n) for key in names}
    history_rows=0
    try:
        with gzip.open(out/'refinement.csv.gz','wt',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['pair','extra_updates',*names,'status'])
            for step in range(cap+1):
                tick();active=np.flatnonzero(status=='active')
                if not len(active):break
                values=iv.effect_interval(base[active],edit[active],weight,
                    tuple(v[active] for v in injection_interval),removed[active],readout,
                    winner[active],runner[active],q_upper)
                cert_calls+=1;cert_examples+=len(active)
                if initial is None:initial={key:value.copy() for key,value in values.items()}
                lo,hi=values['lower'],values['upper']
                overlap=np.maximum(lo,previous_lower[active])<=np.minimum(hi,previous_upper[active])
                if not overlap.all():
                    np.savez_compressed(out/'failed-intersection.npz',pair=active,
                        previous_lower=previous_lower[active],previous_upper=previous_upper[active],**values)
                control('successive_intersection',bool(overlap.all()),
                    dict(step=step,pairs=len(active),failed_pairs=active[~overlap].tolist()))
                previous_lower[active]=lo;previous_upper[active]=hi
                current=classify(lo,hi,step,cap)
                for k,pair in enumerate(active):
                    writer.writerow([int(pair),step,*[float(values[name][k]) for name in names],current[k]])
                handle.flush();history_rows+=len(active)
                status[active]=current;steps[active]=step
                for name in names:final[name][active]=values[name]
                live=active[current=='active']
                if len(live):
                    base[live]=mapping(base[live],weight,injection_point[live],np.full(len(live),-1))
                    edit[live]=mapping(edit[live],weight,injection_point[live],removed[live])
                    calls+=2;examples+=2*len(live)
                    error=float(np.max(np.abs(edit[live,removed[live]])))
                    control('refined_clamp_zero',error==0.,dict(step=step+1,error=error,pairs=len(live)))
        assert np.all(status!='active')
        np.savez_compressed(out/'refined.npz',base=base,edit=edit,status=status,
            extra_updates=steps,**final)
        return initial,dict(ordinary_map_calls=calls,ordinary_map_examples=examples,
            certificate_calls=cert_calls,certificate_examples=cert_examples,
            refinement_rows=history_rows),status,steps,final
    except BaseException:
        np.savez_compressed(out/'refinement-partial.npz',base=base,edit=edit,status=status,
            extra_updates=steps,**final)
        raise
