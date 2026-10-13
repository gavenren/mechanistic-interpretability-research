"""Independent saved-weight equilibrium audit; no scientific-module imports.

This replays evaluation and interval arithmetic. It does not repeat training.
All scientific calls require complete results from a public lock.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SEEDS = (401, 409, 419, 421)
FIELDS = ('lower', 'upper', 'radius', 'center_lower', 'center_upper',
          'coefficient_norm_upper', 'residual_base_upper', 'residual_edit_upper')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def weight_hash(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(key.encode())
        h.update(arrays[key].tobytes())
    return h.hexdigest()


def load_npz(path):
    with np.load(path) as a:
        return {k: a[k].copy() for k in a.files}


class Check:
    def __init__(self):
        self.count = 0
        self.max_errors = {}
        self.states = 0
        self.history_rows = 0
        self.effects = 0

    def eq(self, name, actual, expected):
        self.count += 1
        if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
            a, b = np.asarray(actual), np.asarray(expected)
            assert a.shape == b.shape, (name, a.shape, b.shape)
            assert np.array_equal(a, b), name
            if np.issubdtype(a.dtype, np.number):
                assert np.isfinite(a).all() and np.isfinite(b).all(), name
                self.max_errors[name] = max(self.max_errors.get(name, 0.), float(np.max(abs(a-b), initial=0.)))
        else:
            assert actual == expected, (name, actual, expected)

    def tree(self, name, a, b):
        if isinstance(a, dict):
            self.eq(name+' keys', set(a), set(b))
            for k in a:
                self.tree(name+'/'+str(k), a[k], b[k])
        elif isinstance(a, (list, tuple)):
            self.eq(name+' length', len(a), len(b))
            for k, (x,y) in enumerate(zip(a,b)):
                self.tree(name+'/'+str(k), x,y)
        else:
            self.eq(name,a,b)


def outward(x, upper):
    x = np.asarray(x, dtype=np.float64)
    assert np.isfinite(x).all()
    y = np.nextafter(x, np.inf if upper else -np.inf)
    assert np.isfinite(y).all()
    return y


def lower(x):
    return outward(x, False)


def upper(x):
    return outward(x, True)


def upper_sum(x):
    assert np.all(x >= 0.)
    total = np.zeros(x.shape[:-1])
    for column in np.moveaxis(x, -1, 0):
        total = upper(total+column)
    return total


def affine_bounds(x, matrix, bias):
    lo = np.broadcast_to(bias, (len(x), len(bias))).copy()
    hi = lo.copy()
    for j in range(x.shape[1]):
        product = x[:,j,None] * matrix[None,:,j]
        lo = lower(lo+lower(product))
        hi = upper(hi+upper(product))
    assert (lo<=hi).all()
    return lo,hi


def residual_bound(state, matrix, injection, removed):
    lo,hi = affine_bounds(state,matrix,np.zeros(matrix.shape[0]))
    lo = np.maximum(lower(lo+injection[0]),0.)
    hi = np.maximum(upper(hi+injection[1]),0.)
    rows = np.flatnonzero(removed>=0)
    lo[rows,removed[rows]] = hi[rows,removed[rows]] = 0.
    return np.maximum(abs(lower(lo-state)),abs(upper(hi-state))).max(axis=1)


def effect_bounds(base,edit,w,injection,units,v,winner,runner,q):
    rb = residual_bound(base,w,injection,np.full(len(base),-1))
    re = residual_bound(edit,w,injection,units)
    raw = v[winner]-v[runner]
    c0,c1 = lower(raw),upper(raw)
    norm = upper_sum(np.maximum(abs(c0),abs(c1)))
    denominator = lower(1.-q)
    assert denominator>0
    state_radius = upper(upper(rb+re)/denominator)
    radius = upper(norm*state_radius)
    d0,d1 = lower(base-edit),upper(base-edit)
    center_lo = np.zeros(len(base))
    center_hi = np.zeros(len(base))
    for j in range(base.shape[1]):
        products = np.array([c0[:,j]*d0[:,j],c0[:,j]*d1[:,j],c1[:,j]*d0[:,j],c1[:,j]*d1[:,j]])
        center_lo = lower(center_lo+lower(products).min(axis=0))
        center_hi = upper(center_hi+upper(products).max(axis=0))
    lo,hi = lower(center_lo-radius),upper(center_hi+radius)
    assert (lo<=hi).all()
    return dict(lower=lo,upper=hi,radius=radius,center_lower=center_lo,center_upper=center_hi,
                coefficient_norm_upper=norm,residual_base_upper=rb,residual_edit_upper=re)


def logit_bounds(state,w,injection,v,d,q):
    residual = residual_bound(state,w,injection,np.full(len(state),-1))
    error = upper(residual/lower(1.-q))
    radii = upper(error[:,None]*upper_sum(abs(v))[None,:])
    lo,hi = affine_bounds(state,v,d)
    return lower(lo-radii),upper(hi+radii),residual,error


class Solver:
    def __init__(self,w):
        self.w = w
        self.calls = self.examples = 0

    def advance(self,state,injection,removed):
        self.calls += 1
        self.examples += len(state)
        future = np.maximum(np.matmul(state,self.w.T)+injection,0.)
        rows = np.flatnonzero(removed>=0)
        future[rows,removed[rows]] = 0.
        assert np.isfinite(future).all()
        return future

    def solve(self,injection,removed,tolerance):
        state = np.zeros_like(injection)
        updates = np.zeros(len(state),dtype=np.int64)
        residual = np.empty(len(state))
        active = np.arange(len(state))
        for step in range(513):
            future = self.advance(state[active],injection[active],removed[active])
            error = abs(future-state[active]).max(axis=1)
            residual[active] = error
            keep = error>tolerance
            if not keep.any():
                return state,residual,updates
            assert step<512,'Independent ordinary solve missed fixed cap.'
            live = active[keep]
            state[live] = future[keep]
            updates[live] += 1
            active = live
        raise AssertionError('Unreachable solver branch')


def category(lo,hi,step):
    # Explicit priority: sign, then small magnitude, then cap.
    return np.array(['positive' if a>0 else 'negative' if b<0 else
                     'small' if a>=-1e-7 and b<=1e-7 else
                     'unresolved' if step==128 else 'active'
                     for a,b in zip(lo,hi)],dtype='<U10')


def bin_labels(naive):
    labels=[]
    for a in abs(naive):
        labels.append('zero' if a==0 else '(0,1e-7]' if a<=1e-7 else
                      '(1e-7,1e-5]' if a<=1e-5 else '(1e-5,1e-3]' if a<=1e-3 else 'above 1e-3')
    return np.array(labels)


def initial_weights(seed):
    torch.manual_seed(seed)
    inp=torch.nn.Linear(64,32,dtype=torch.float64)
    rec=torch.nn.Linear(32,32,bias=False,dtype=torch.float64)
    out=torch.nn.Linear(32,10,dtype=torch.float64)
    with torch.no_grad():
        rec.weight.div_(torch.maximum(torch.ones(32,1,dtype=torch.float64),rec.weight.abs().sum(1,keepdim=True)/.875))
    return {'input.weight':inp.weight.detach().numpy(),'input.bias':inp.bias.detach().numpy(),
            'recurrent.weight':rec.weight.detach().numpy(),'readout.weight':out.weight.detach().numpy(),
            'readout.bias':out.bias.detach().numpy()}


def audit_seed(seed,path,tx,ty,panel,check,control):
    weights=load_npz(path/'weights.npz')
    initial=load_npz(path/'initial-weights.npz')
    expected_initial=initial_weights(seed)
    check.eq('weight_keys',set(weights),set(expected_initial))
    for name,value in expected_initial.items():
        check.eq('initial_weight',initial[name],value)
        check.eq('weight_shape',weights[name].shape,value.shape)
        assert np.isfinite(weights[name]).all()
    check.eq('parameter_count',sum(a.size for a in weights.values()),3434)
    rng=np.random.default_rng(30000+seed)
    schedule=np.stack([rng.permutation(3823) for _ in range(40)])
    check.eq('training_schedule',np.load(path/'orders.npy'),schedule)
    with (path/'training.csv').open() as h:
        losses=list(csv.DictReader(h))
    check.eq('training_rows',len(losses),1200)
    for i,row in enumerate(losses):
        epoch,batch=divmod(i,30)
        check.eq('training_epoch',int(row['epoch']),epoch)
        check.eq('training_batch',int(row['batch']),batch)
        check.eq('training_examples',int(row['examples']),111 if batch==29 else 128)
        loss=float(row['loss']);assert math.isfinite(loss)
        control('finite_training_loss',dict(seed=seed,epoch=epoch,batch=batch,loss=loss))
        # This checks the saved assertion, not intermediate training weights.
        control('finite_parameters',dict(seed=seed,epoch=epoch,batch=batch))
    w,B,b,v,d=[weights[k] for k in ('recurrent.weight','input.weight','input.bias','readout.weight','readout.bias')]
    q=float(upper_sum(abs(w)).max());assert q<1
    control('contraction',dict(seed=seed,q_upper=q))
    solver=Solver(w);parts=[]
    for start in range(0,len(tx),128):
        x=tx[start:start+128];inj=x@B.T+b;n=len(x)
        state,resid,steps=solver.solve(inj,np.full(n,-1),1e-12)
        bound=affine_bounds(x,B,b)
        lo,hi,ru,er=logit_bounds(state,w,bound,v,d,q)
        first=solver.advance(np.zeros_like(state),inj,np.full(n,-1))@v.T+d
        expected=dict(state=state,logits=state@v.T+d,residual=resid,updates=steps,residual_upper=ru,
                      state_error_upper=er,lower=lo,upper=hi,zero_logits=np.broadcast_to(d,(n,10)).copy(),first_logits=first)
        check.tree('test_batch',expected,load_npz(path/f'test-batch-{start//128:02d}.npz'))
        parts.append(expected);check.states+=n
    test={key:np.concatenate([part[key] for part in parts]) for key in parts[0]}
    check.tree('test_combined',test,load_npz(path/'test.npz'))
    ranked=np.argsort(-test['logits'],axis=1,kind='stable');winner,runner=ranked[:,0],ranked[:,1]
    other=test['upper'].copy();other[np.arange(len(tx)),winner]=-np.inf
    certified=test['lower'][np.arange(len(tx)),winner]>other.max(axis=1)
    check.tree('test_classes',dict(winner=winner,runner=runner,certified=certified),load_npz(path/'test-classes.npz'))
    x=tx[panel];injection=x@B.T+b;states=[];residuals=[];counts=[]
    for condition in range(-1,32):
        state,resid,updates=solver.solve(injection,np.full(len(panel),condition),.001)
        check.tree('coarse_condition',dict(state=state,residual=resid,updates=updates),load_npz(path/f'coarse-{condition+1:02d}.npz'))
        states.append(state);residuals.append(resid);counts.append(updates);check.states+=len(state)
        if condition>=0:
            error=float(abs(state[:,condition]).max());assert error==0.
            control('coarse_clamp_zero',dict(seed=seed,condition=condition,error=error))
    repeated,rr,rc=solver.solve(injection,np.full(len(panel),-1),.001)
    check.eq('repeat_state',repeated,states[0]);check.eq('repeat_residual',rr,residuals[0]);check.eq('repeat_count',rc,counts[0])
    control('repeated_intact_solve',dict(seed=seed,error=0.))
    states,residuals,counts=np.array(states),np.array(residuals),np.array(counts)
    image=np.repeat(np.arange(120),32);unit=np.tile(np.arange(32),120)
    base=states[0,image].copy();edit=states[unit+1,image].copy()
    win,run=winner[panel][image],runner[panel][image]
    naive=np.sum((v[win]-v[run])*(base-edit),axis=1)
    inj_bounds=affine_bounds(x,B,b);pair_bounds=tuple(z[image] for z in inj_bounds)
    check.tree('coarse_all',dict(states=states,residuals=residuals,updates=counts,naive=naive,
                               panel_rows=panel,pair_image=image,pair_unit=unit,winner=win,runner=run,
                               injection_lower=inj_bounds[0],injection_upper=inj_bounds[1]),load_npz(path/'coarse.npz'))
    status=np.full(3840,'active',dtype='<U10');extra=np.zeros(3840,dtype=np.int64)
    final={key:np.zeros(3840) for key in FIELDS};previous_lo=np.full(3840,-np.inf);previous_hi=np.full(3840,np.inf)
    bound_calls=bound_examples=history_rows=0;initial_bounds=None
    with gzip.open(path/'refinement.csv.gz','rt',newline='') as h:
        history=csv.DictReader(h)
        check.eq('history_columns',history.fieldnames,['pair','extra_updates',*FIELDS,'status'])
        for step in range(129):
            active=np.flatnonzero(status=='active')
            if len(active)==0:break
            values=effect_bounds(base[active],edit[active],w,tuple(a[active] for a in pair_bounds),unit[active],v,win[active],run[active],q)
            bound_calls+=1;bound_examples+=len(active)
            if step==0:initial_bounds={k:a.copy() for k,a in values.items()}
            lo,hi=values['lower'],values['upper']
            overlap=np.maximum(lo,previous_lo[active])<=np.minimum(hi,previous_hi[active]);assert overlap.all()
            control('successive_intersection',dict(step=step,pairs=len(active),failed_pairs=[]))
            current=category(lo,hi,step)
            for j,pair in enumerate(active):
                saved=next(history)
                check.eq('history_pair',int(saved['pair']),int(pair));check.eq('history_step',int(saved['extra_updates']),step)
                check.eq('history_status',saved['status'],str(current[j]))
                for key in FIELDS:check.eq('history_'+key,float(saved[key]),float(values[key][j]))
            history_rows+=len(active)
            previous_lo[active]=lo;previous_hi[active]=hi
            status[active]=current;extra[active]=step
            for key in FIELDS:final[key][active]=values[key]
            live=active[current=='active']
            if len(live):
                base[live]=solver.advance(base[live],injection[image[live]],np.full(len(live),-1))
                edit[live]=solver.advance(edit[live],injection[image[live]],unit[live])
                error=float(abs(edit[live,unit[live]]).max());assert error==0.
                control('refined_clamp_zero',dict(step=step+1,error=error,pairs=len(live)))
        check.eq('history_exhausted',next(history,None),None)
    assert np.all(status!='active')
    check.tree('initial_bounds',initial_bounds,load_npz(path/'initial-bounds.npz'))
    check.tree('final_states_bounds',dict(base=base,edit=edit,status=status,extra_updates=extra,**final),load_npz(path/'refined.npz'))
    check.states+=2*3840;check.history_rows+=history_rows;check.effects+=3840
    whash=weight_hash(weights)
    control('unchanged_coefficients',dict(seed=seed,before=whash,after=whash))
    control('output_counts',dict(seed=seed,test_rows=len(tx),effects=3840))
    unresolved=(initial_bounds['lower']<=0)&(initial_bounds['upper']>=0)
    nonzero=naive!=0;denom=int(nonzero.sum());numer=int((nonzero&unresolved).sum())
    reversals=((naive<0)&(status=='positive'))|((naive>0)&(status=='negative'))
    labels=bin_labels(naive)
    with (path/'effects.csv').open() as h:
        effect_rows=list(csv.DictReader(h))
    check.eq('effect_rows',len(effect_rows),3840)
    for i,row in enumerate(effect_rows):
        vals=dict(pair=i,test_row=int(panel[image[i]]),unit=int(unit[i]),winner=int(win[i]),runner=int(run[i]),
                  ordinary_effect=float(naive[i]),bin=labels[i],initial_lower=float(initial_bounds['lower'][i]),
                  initial_upper=float(initial_bounds['upper'][i]),final_lower=float(final['lower'][i]),final_upper=float(final['upper'][i]),
                  final_status=status[i],extra_updates=int(extra[i]),certified_reversal=bool(reversals[i]))
        check.eq('effect_columns',set(row),set(vals))
        for key,value in vals.items():
            expected=str(value) if isinstance(value,(str,bool,np.str_)) else value
            actual=row[key] if isinstance(expected,str) else float(row[key])
            check.eq('effect_'+key,actual,expected)
    result=dict(seed=seed,q_upper=q,nonzero_ordinary=denom,unresolved_nonzero=numer,
                primary_fraction=numer/denom if denom else None,effects=3840,ordinary_zero=int((~nonzero).sum()),
                unresolved_all=int(unresolved.sum()),unresolved_all_fraction=float(unresolved.mean()),
                full_test_accuracy=float((winner==ty).mean()),panel_accuracy=float((winner[panel]==ty[panel]).mean()),
                ability_gate_passed=bool((winner==ty).mean()>=.8),certified_top_class=int(certified.sum()),
                zero_state_accuracy=float((test['zero_logits'].argmax(1)==ty).mean()),first_update_accuracy=float((test['first_logits'].argmax(1)==ty).mean()),
                zero_state_disagreement=float((test['zero_logits'].argmax(1)!=winner).mean()),first_update_disagreement=float((test['first_logits'].argmax(1)!=winner).mean()),
                final_counts={key:int((status==key).sum()) for key in ('positive','negative','small','unresolved')},
                certified_sign_reversals=int(reversals.sum()),extra_paired_updates=int(extra.sum()),ordinary_map_calls=solver.calls,
                ordinary_map_examples=solver.examples,certificate_calls=len(parts)+bound_calls,certificate_examples=len(tx)+bound_examples,
                refinement_rows=history_rows,certificate_map_passes=len(parts)+2*bound_calls,certificate_map_examples=len(tx)+2*bound_examples,
                full_logit_bound_requests=len(parts),effect_bound_requests=bound_calls,input_affine_enclosure_calls=len(parts)+1,
                input_affine_enclosure_examples=len(tx)+len(panel),initial_weight_sha256=weight_hash(initial),final_weight_sha256=whash,
                training_updates=1200,training_examples=40*3823,training_state_update_examples=32*40*3823)
    result['bins']=[]
    for label in ('zero','(0,1e-7]','(1e-7,1e-5]','(1e-5,1e-3]','above 1e-3'):
        mask=labels==label
        result['bins'].append(dict(bin=label,count=int(mask.sum()),ordinary_mean=float(naive[mask].mean()) if mask.any() else None,
            initial_width_mean=float((initial_bounds['upper'][mask]-initial_bounds['lower'][mask]).mean()) if mask.any() else None,
            final_width_mean=float((final['upper'][mask]-final['lower'][mask]).mean()) if mask.any() else None,
            final_counts={key:int(((status==key)&mask).sum()) for key in ('positive','negative','small','unresolved')}))
    check.tree('model_summary',result,read(path/'summary.json'))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();assert not args.output.exists(),'Use a fresh receipt path.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();c=Check();receipt=dict(status='running',audit_code_sha256=sha(__file__),training_repeated=False,
                                                  imported_study_modules=[],exact_comparison_tolerance=0)
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.mkldnn.enabled=False
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore')
        out=args.results;meta=read(out/'run-metadata.json')
        c.eq('run_complete',meta['status'],'complete');c.eq('scientific_run',meta['scientific_model_outputs_collected'],True)
        receipt['result_manifest_sha256']=sha(out/'manifest.json');receipt['public_lock']=meta['public_lock']
        for name,digest in read(out/'manifest.json').items():c.eq('result_hash',sha(out/name),digest)
        for name,digest in meta['code_sha256'].items():
            p=ROOT/name;rel=p.relative_to(REPO).as_posix()
            old=subprocess.check_output(['git','show',meta['public_lock']+':'+rel],cwd=REPO)
            c.eq('source_lock_bytes',old,p.read_bytes());c.eq('source_hash',sha(p),digest)
        peer=read(ROOT/'reviews/independent-interval-checks.json')
        c.eq('peer_rational_checks',peer['status'],'passed')
        c.eq('peer_interval_hash',peer['interval_code_sha256'],sha(ROOT/'code/intervals.py'))
        c.eq('peer_checker_hash',peer['checker_code_sha256'],sha(ROOT/'code/review_interval_checks.py'))
        receipt['separate_rational_receipt_sha256']=sha(ROOT/'reviews/independent-interval-checks.json')
        receipt['separate_rational_checks']=peer['check_count']
        ledger=read(ROOT/'inputs.json')
        for name,digest in ledger['data_hashes'].items():c.eq('raw_data_hash',sha(ROOT/'data'/name),digest)
        train=np.loadtxt(ROOT/'data/optdigits.tra',delimiter=',',dtype=np.int64)
        test=np.loadtxt(ROOT/'data/optdigits.tes',delimiter=',',dtype=np.int64)
        c.eq('training_data_shape',train.shape,(3823,65));c.eq('test_data_shape',test.shape,(1797,65))
        assert ((train[:,:64]>=0)&(train[:,:64]<=16)).all() and ((test[:,:64]>=0)&(test[:,:64]<=16)).all()
        panel=np.array(sorted(i for label in range(10) for i in np.flatnonzero(test[:,-1]==label)[:12]),dtype=np.int64)
        c.tree('input_panel',ledger['panel'],[dict(row=int(i),label=int(test[i,-1])) for i in panel])
        controls=[json.loads(line) for line in (out/'controls.jsonl').read_text().splitlines()]
        next_control=iter(controls)
        def control(name,evidence):
            actual=next(next_control)
            c.tree('control',actual,dict(name=name,passed=True,evidence=evidence))
        models=[audit_seed(seed,out/str(seed),test[:,:64]/16,test[:,-1],panel,c,control) for seed in SEEDS]
        control('complete_counts',dict(training_rows=4800,effects=15360))
        c.eq('control_exhausted',next(next_control,None),None)
        c.eq('control_count',len(controls),meta['control_rows']);c.eq('training_total',meta['training_rows'],4800)
        fractions=[x['primary_fraction'] for x in models]
        expected=dict(models=models,primary_mean=float(np.mean(fractions)) if all(v is not None for v in fractions) else None,
                      all_ability_gates_passed=all(x['ability_gate_passed'] for x in models))
        c.tree('full_summary',expected,read(out/'summary.json'))
        receipt.update(status='passed',primary_mean=expected['primary_mean'],primary_counts=[{k:m[k] for k in ('seed','nonzero_ordinary','unresolved_nonzero','primary_fraction')} for m in models],
                       replayed_counters={key:sum(m[key] for m in models) for key in ('ordinary_map_calls','ordinary_map_examples','certificate_calls','certificate_examples',
                                          'certificate_map_passes','certificate_map_examples','input_affine_enclosure_calls','input_affine_enclosure_examples','extra_paired_updates')},
                       control_rows=len(controls),training_records_checked=4800)
    except BaseException as error:
        receipt.update(status='failed',error=str(error),traceback=traceback.format_exc().replace(str(REPO),'<repo>'))
        raise
    finally:
        receipt.update(comparisons=c.count,max_errors=c.max_errors,checked_state_rows=c.states,checked_history_rows=c.history_rows,
                       checked_effect_rows=c.effects,elapsed_seconds=time.monotonic()-started)
        args.output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:receipt[k] for k in ('status','comparisons','checked_state_rows','checked_history_rows','checked_effect_rows','elapsed_seconds')}))


if __name__=='__main__':
    main()
