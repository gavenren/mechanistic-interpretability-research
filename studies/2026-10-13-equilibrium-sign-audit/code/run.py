#!/usr/bin/env python3
"""Fixed equilibrium sign audit. Scientific calls require an immutable lock."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import intervals as iv
from model import Model,optimizer,mapping,ordinary_solve
from protocol import refine,primary_counts,sign_reversal,bins

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(401,409,419,421)
LOCK_FILES=['analysis-plan.md','sources.md','inputs.json','requirements.txt',
    'data/optdigits.tra','data/optdigits.tes','code/intervals.py','code/check_intervals.py',
    'code/model.py','code/protocol.py','code/run.py','code/check_protocol.py','code/dummy_timing.py']
COST_DEFINITIONS={
    'training_updates':'One forward/backward/Adam update on one minibatch; each forward has 32 state updates.',
    'training_examples':'Minibatch sizes summed over all optimizer updates.',
    'training_state_update_examples':'32 times training_examples; backward work is additional and is included in wall time.',
    'ordinary_map_calls':'One batched floating-point ReLU recurrent map, including stopping-residual checks, no-op repeats, first-update diagnostics, and both refinement branches.',
    'ordinary_map_examples':'Rows summed over ordinary_map_calls. A final residual check is a call even when its candidate is not advanced.',
    'certificate_calls':'Bound requests: one full-logit enclosure or one paired effect enclosure. A paired request evaluates two residual maps.',
    'certificate_examples':'Rows summed over bound requests; a pair counts once here.',
    'certificate_map_passes':'One validated residual map per logit request and two per effect request.',
    'certificate_map_examples':'One row per logit request and two per paired effect request.',
    'input_affine_enclosure_calls':'One interval enclosure of Bx+b, outside validated residual maps.',
    'input_affine_enclosure_examples':'Rows summed over input affine enclosures.',
    'extra_paired_updates':'Sum of actual extra updates per image-unit pair; each updates two state vectors.',
    'wall_time':'Includes training, controls, evaluation, and saved output. The final cap check also covers the manifest.'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def lock_check(commit):
    repo=ROOT.parents[1]
    result=subprocess.run(['git','rev-parse',commit+'^{commit}'],cwd=repo,
        check=True,capture_output=True,text=True).stdout.strip()
    assert result==commit,'Use the full immutable lock hash.'
    for name in LOCK_FILES:
        path=ROOT/name;relative=path.relative_to(repo).as_posix()
        value=subprocess.run(['git','show',commit+':'+relative],cwd=repo,
            check=True,capture_output=True).stdout
        assert value==path.read_bytes(),relative


def setup():
    assert np.__version__=='2.3.5' and torch.__version__.split('+')[0]=='2.8.0'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.mkldnn.enabled=False
    np.seterr(over='raise',invalid='raise',divide='raise',under='ignore')


class Record:
    def __init__(self,out,cap=1200):
        self.out=Path(out);assert not self.out.exists(),'Use a fresh output directory.'
        self.out.mkdir(parents=True);self.start=time.monotonic();self.cap=cap
        self.controls=0;self.loss_rows=0
        self.handle=(self.out/'controls.jsonl').open('w')

    def tick(self):
        if time.monotonic()-self.start>self.cap:raise TimeoutError('Fixed run time limit reached.')

    def control(self,name,passed,evidence=None):
        self.handle.write(json.dumps(dict(name=name,passed=bool(passed),evidence=evidence),allow_nan=False)+'\n')
        self.handle.flush();self.controls+=1
        assert passed,name

    def close(self):self.handle.close()


def weight_hash(arrays):
    value=hashlib.sha256()
    for key,array in sorted(arrays.items()):value.update(key.encode());value.update(array.tobytes())
    return value.hexdigest()


def train(seed,x,y,out,record,epochs=40):
    model=Model(seed);opt=optimizer(model);rng=np.random.default_rng(30000+seed)
    orders=np.stack([rng.permutation(len(x)) for _ in range(epochs)])
    np.save(out/'orders.npy',orders);initial=model.arrays()
    np.savez_compressed(out/'initial-weights.npz',**initial)
    try:
        with (out/'training.csv').open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['epoch','batch','examples','loss'])
            for epoch,order in enumerate(orders):
                for batch,start in enumerate(range(0,len(x),128)):
                    record.tick();ids=order[start:start+128]
                    opt.zero_grad(set_to_none=True)
                    logits=model(x[ids]);loss=torch.nn.functional.cross_entropy(logits,y[ids])
                    scalar=float(loss.detach());writer.writerow([epoch,batch,len(ids),scalar]);handle.flush()
                    record.loss_rows+=1
                    record.control('finite_training_loss',np.isfinite(scalar),dict(seed=seed,epoch=epoch,batch=batch,loss=scalar))
                    loss.backward();opt.step();model.project()
                    record.control('finite_parameters',all(torch.isfinite(p).all().item() for p in model.parameters()),dict(seed=seed,epoch=epoch,batch=batch))
                np.savez_compressed(out/'current-weights.npz',**model.arrays())
    except BaseException:
        np.savez_compressed(out/'current-weights.npz',**model.arrays());raise
    final=model.arrays();np.savez_compressed(out/'weights.npz',**final)
    (out/'current-weights.npz').unlink()
    return final,dict(initial_weight_sha256=weight_hash(initial),final_weight_sha256=weight_hash(final),
        training_updates=epochs*((len(x)+127)//128),training_examples=epochs*len(x),
        training_state_update_examples=32*epochs*len(x))


def evaluate(seed,weights,test_x,test_y,panel,out,record):
    w=weights['recurrent.weight'];b=weights['input.bias'];B=weights['input.weight']
    v=weights['readout.weight'];d=weights['readout.bias'];before=weight_hash(weights)
    q=iv.contraction_upper(w);record.control('contraction',q<1,dict(seed=seed,q_upper=q))
    calls=examples=cert_calls=cert_examples=0
    test_parts=[]
    for start in range(0,len(test_x),128):
        record.tick();x=test_x[start:start+128];n=len(x);a=x@B.T+b;rem=np.full(n,-1)
        state,residual,updates,c,e=ordinary_solve(w,a,rem,1e-12,callback=record.tick)
        calls+=c;examples+=e;inj=iv.affine_point(x,B,b)
        logbounds,ru,er=iv.logits_interval(state,w,inj,v,d,q);cert_calls+=1;cert_examples+=n
        logits=state@v.T+d;first=mapping(np.zeros_like(state),w,a,rem)@v.T+d
        calls+=1;examples+=n
        part=dict(state=state,logits=logits,residual=residual,updates=updates,
            residual_upper=ru,state_error_upper=er,lower=logbounds[0],upper=logbounds[1],
            zero_logits=np.broadcast_to(d,logits.shape).copy(),first_logits=first)
        np.savez_compressed(out/f'test-batch-{start//128:02d}.npz',**part)
        test_parts.append(part)
    test={key:np.concatenate([part[key] for part in test_parts]) for key in test_parts[0]}
    np.savez_compressed(out/'test.npz',**test)
    order=np.argsort(-test['logits'],axis=1,kind='stable');winners=order[:,0];runners=order[:,1]
    upper_other=test['upper'].copy();upper_other[np.arange(len(test_x)),winners]=-np.inf
    certified=test['lower'][np.arange(len(test_x)),winners]>upper_other.max(1)
    np.savez_compressed(out/'test-classes.npz',winner=winners,runner=runners,certified=certified)
    x=test_x[panel];a=x@B.T+b;n=len(panel);states=[];residuals=[];counts=[]
    for condition in range(-1,len(w)):
        record.tick();removed=np.full(n,condition)
        state,residual,updates,c,e=ordinary_solve(w,a,removed,.001,callback=record.tick)
        calls+=c;examples+=e;states.append(state);residuals.append(residual);counts.append(updates)
        np.savez_compressed(out/f'coarse-{condition+1:02d}.npz',state=state,residual=residual,updates=updates)
        if condition>=0:
            error=float(abs(state[:,condition]).max())
            record.control('coarse_clamp_zero',error==0,dict(seed=seed,condition=condition,error=error))
    repeat,rr,rc,c,e=ordinary_solve(w,a,np.full(n,-1),.001,callback=record.tick)
    calls+=c;examples+=e
    error=float(abs(repeat-states[0]).max())
    record.control('repeated_intact_solve',error==0 and np.array_equal(rr,residuals[0]) and np.array_equal(rc,counts[0]),dict(seed=seed,error=error))
    states=np.stack(states);residuals=np.stack(residuals);counts=np.stack(counts)
    image=np.repeat(np.arange(n),len(w));unit=np.tile(np.arange(len(w)),n)
    base=states[0,image];edit=states[unit+1,image]
    win=winners[panel][image];run=runners[panel][image]
    naive=np.sum((v[win]-v[run])*(base-edit),axis=1)
    inj=iv.affine_point(x,B,b);pair_inj=tuple(z[image] for z in inj)
    np.savez_compressed(out/'coarse.npz',states=states,residuals=residuals,updates=counts,
        naive=naive,panel_rows=panel,pair_image=image,pair_unit=unit,winner=win,runner=run,
        injection_lower=inj[0],injection_upper=inj[1])
    initial,extra,status,steps,final=refine(base,edit,w,a[image],pair_inj,unit,v,win,run,q,out,
        record.control,record.tick)
    np.savez_compressed(out/'initial-bounds.npz',**initial)
    calls+=extra['ordinary_map_calls'];examples+=extra['ordinary_map_examples']
    cert_calls+=extra['certificate_calls'];cert_examples+=extra['certificate_examples']
    record.control('unchanged_coefficients',weight_hash(weights)==before,dict(seed=seed,before=before,after=weight_hash(weights)))
    record.control('output_counts',len(test['state'])==len(test_x) and len(naive)==n*len(w),dict(seed=seed,test_rows=len(test_x),effects=len(naive)))
    result=dict(seed=seed,q_upper=q,**primary_counts(naive,initial['lower'],initial['upper']),
        full_test_accuracy=float((winners==test_y).mean()),panel_accuracy=float((winners[panel]==test_y[panel]).mean()),
        ability_gate_passed=bool((winners==test_y).mean()>=.8),certified_top_class=int(certified.sum()),
        zero_state_accuracy=float((test['zero_logits'].argmax(1)==test_y).mean()),
        first_update_accuracy=float((test['first_logits'].argmax(1)==test_y).mean()),
        zero_state_disagreement=float((test['zero_logits'].argmax(1)!=winners).mean()),
        first_update_disagreement=float((test['first_logits'].argmax(1)!=winners).mean()),
        final_counts={name:int((status==name).sum()) for name in ('positive','negative','small','unresolved')},
        certified_sign_reversals=int(sign_reversal(naive,status).sum()),extra_paired_updates=int(steps.sum()),
        ordinary_map_calls=calls,ordinary_map_examples=examples,certificate_calls=cert_calls,
        certificate_examples=cert_examples,refinement_rows=extra['refinement_rows'],
        certificate_map_passes=len(test_parts)+2*extra['certificate_calls'],
        certificate_map_examples=len(test_x)+2*extra['certificate_examples'],
        full_logit_bound_requests=len(test_parts),effect_bound_requests=extra['certificate_calls'],
        input_affine_enclosure_calls=len(test_parts)+1,input_affine_enclosure_examples=len(test_x)+n)
    with (out/'effects.csv').open('w',newline='') as handle:
        keys=['pair','test_row','unit','winner','runner','ordinary_effect','bin','initial_lower','initial_upper',
            'final_lower','final_upper','final_status','extra_updates','certified_reversal']
        writer=csv.writer(handle);writer.writerow(keys);groups=bins(naive);reversals=sign_reversal(naive,status)
        for i in range(len(naive)):
            writer.writerow([i,int(panel[image[i]]),int(unit[i]),int(win[i]),int(run[i]),float(naive[i]),groups[i],
                float(initial['lower'][i]),float(initial['upper'][i]),float(final['lower'][i]),float(final['upper'][i]),
                status[i],int(steps[i]),bool(reversals[i])])
    result['bins']=[]
    for label in ('zero','(0,1e-7]','(1e-7,1e-5]','(1e-5,1e-3]','above 1e-3'):
        mask=groups==label
        result['bins'].append(dict(bin=label,count=int(mask.sum()),
            ordinary_mean=float(naive[mask].mean()) if mask.any() else None,
            initial_width_mean=float((initial['upper'][mask]-initial['lower'][mask]).mean()) if mask.any() else None,
            final_width_mean=float((final['upper'][mask]-final['lower'][mask]).mean()) if mask.any() else None,
            final_counts={name:int(((status==name)&mask).sum()) for name in ('positive','negative','small','unresolved')}))
    dump(out/'summary.json',result)
    return result


def manifest(out):
    dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name!='manifest.json'})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lock',required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    setup();lock_check(args.lock);record=Record(args.output);out=args.output
    metadata=dict(status='running',public_lock=args.lock,python=platform.python_version(),
        platform=platform.platform(),numpy=np.__version__,torch=torch.__version__,
        code_sha256={name:sha(ROOT/name) for name in LOCK_FILES},seeds=list(SEEDS),
        scientific_model_outputs_collected=False)
    dump(out/'run-metadata.json',metadata)
    dump(out/'cost-definitions.json',COST_DEFINITIONS)
    try:
        ledger=json.loads((ROOT/'inputs.json').read_text())
        for name,expected in ledger['data_hashes'].items():assert sha(ROOT/'data'/name)==expected
        train_data=np.loadtxt(ROOT/'data/optdigits.tra',delimiter=',',dtype=np.int64)
        test_data=np.loadtxt(ROOT/'data/optdigits.tes',delimiter=',',dtype=np.int64)
        assert train_data.shape==(3823,65) and test_data.shape==(1797,65)
        assert ((train_data[:,:64]>=0)&(train_data[:,:64]<=16)).all()
        assert ((test_data[:,:64]>=0)&(test_data[:,:64]<=16)).all()
        panel=np.array([item['row'] for item in ledger['panel']],dtype=np.int64)
        expected=sorted(i for digit in range(10) for i in np.flatnonzero(test_data[:,-1]==digit)[:12])
        assert panel.tolist()==expected
        assert [item['label'] for item in ledger['panel']]==test_data[panel,-1].tolist()
        x=torch.from_numpy(train_data[:,:64].astype(float)/16);y=torch.from_numpy(train_data[:,-1].copy())
        summaries=[]
        for seed in SEEDS:
            path=out/str(seed);path.mkdir()
            metadata['scientific_model_outputs_collected']=True;dump(out/'run-metadata.json',metadata)
            weights,training=train(seed,x,y,path,record)
            result=evaluate(seed,weights,test_data[:,:64]/16,test_data[:,-1],panel,path,record)
            result.update(training);dump(path/'summary.json',result);summaries.append(result)
            print(json.dumps(dict(seed=seed,status='complete',primary_fraction=result['primary_fraction'],
                accuracy=result['full_test_accuracy'],elapsed_seconds=time.monotonic()-record.start)),flush=True)
        fractions=[r['primary_fraction'] for r in summaries]
        summary=dict(models=summaries,primary_mean=float(np.mean(fractions)) if all(v is not None for v in fractions) else None,
            all_ability_gates_passed=all(r['ability_gate_passed'] for r in summaries))
        dump(out/'summary.json',summary)
        record.control('complete_counts',record.loss_rows==4800 and sum(r['effects'] for r in summaries)==15360,
            dict(training_rows=record.loss_rows,effects=sum(r['effects'] for r in summaries)))
        lock_check(args.lock);record.tick();metadata['status']='complete'
    except BaseException as error:
        metadata.update(status='failed',error_type=type(error).__name__,error=str(error))
        dump(out/'failure.json',metadata);raise
    finally:
        record.close();metadata.update(elapsed_seconds=time.monotonic()-record.start,
            control_rows=record.controls,training_rows=record.loss_rows)
        dump(out/'run-metadata.json',metadata);manifest(out)
        if time.monotonic()-record.start>record.cap and metadata['status']=='complete':
            metadata.update(status='failed',error_type='TimeoutError',error='Fixed run time limit reached during final records.')
            dump(out/'failure.json',metadata);dump(out/'run-metadata.json',metadata);manifest(out)
            raise TimeoutError(metadata['error'])


if __name__=='__main__':main()
