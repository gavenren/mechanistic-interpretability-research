#!/usr/bin/env python3
"""Paired, CPU-only SAE study. --check runs no model training."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SEEDS = range(6)
PS = (.01,.16)
AMPLITUDES = (.25,1.)
OBJECTIVES = ('local','downstream','balanced')
STEPS = 4000
BATCH = 128
LR = .005
K = 4
WIDTH = 24
TOL = 1e-12


def sha(data):
    return hashlib.sha256(data).hexdigest()


def array_hash(x):
    return sha(np.ascontiguousarray(x).tobytes())


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def write_csv(path,rows):
    rows=list(rows)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def sigmoid(logits):
    return np.exp(-np.logaddexp(0.,-logits))


def readout(h,u,a):
    return sigmoid(8*(h@u/a-.5))


def kl_from_logits(target,logits):
    entropy_term=target*np.log(np.maximum(target,np.finfo(float).tiny))+(1-target)*np.log(np.maximum(1-target,np.finfo(float).tiny))
    return entropy_term+np.logaddexp(0.,logits)-target*logits


def generate(seed):
    rng=np.random.default_rng(1000+seed)
    q,r=np.linalg.qr(rng.normal(size=(12,12)))
    q=q*np.where(np.diag(r)>=0,1.,-1.)[None,:]
    u=q[:,0]
    background_directions=rng.normal(size=(20,11))
    background_directions/=np.linalg.norm(background_directions,axis=1,keepdims=True)
    dictionary=background_directions@q[:,1:].T
    rng=np.random.default_rng(2000+seed)
    banks={}
    for name,n in (('train',10000),('validation',1024),('test',2048)):
        coefficient=(rng.random((n,20))<.1)*rng.uniform(.5,1.5,(n,20))
        banks[name]=coefficient@dictionary
    permutation=rng.permutation(10000)
    value_sets={name:{array_hash(row) for row in bank} for name,bank in banks.items()}
    overlaps={left+'__'+right:len(value_sets[left]&value_sets[right])
              for i,left in enumerate(banks) for right in list(banks)[i+1:]}
    record=dict(seed=seed,dictionary_seed=1000+seed,data_seed=2000+seed,
                initialization_seed=3000+seed,minibatch_seed=4000+seed,
                arrays={name:dict(shape=list(x.shape),sha256=array_hash(x)) for name,x in banks.items()},
                rare_direction_sha256=array_hash(u),dictionary_sha256=array_hash(dictionary),
                permutation_sha256=array_hash(permutation),
                positive_counts={'0.01':100,'0.16':1600},
                zero_background_counts={name:int(np.sum(np.all(x==0,axis=1))) for name,x in banks.items()},
                exact_cross_split_shared_unique_values=overlaps,
                sample_id_ranges={name:[name+':0',name+':'+str(len(x)-1)] for name,x in banks.items()})
    return banks,u,dictionary,permutation,record


def training_data(background,u,permutation,p,a):
    labels=np.zeros(len(background),dtype=np.int64)
    labels[permutation[:round(p*len(background))]]=1
    return background+a*labels[:,None]*u,labels


def initialize(seed):
    rng=np.random.default_rng(3000+seed)
    decoder=rng.normal(size=(WIDTH,12))
    decoder/=np.linalg.norm(decoder,axis=1,keepdims=True)
    return dict(Wenc=decoder.T.copy(),benc=np.zeros(WIDTH),
                Wdec=decoder.copy(),bdec=np.zeros(12))


def encode_decode(h,theta,mean):
    centered=h-mean
    pre=centered@theta['Wenc']+theta['benc']
    relu=np.maximum(pre,0.)
    keep=np.argsort(-relu,axis=1,kind='stable')[:,:K]
    gate=np.zeros_like(relu,dtype=bool)
    np.put_along_axis(gate,keep,True,axis=1)
    gate &= pre>0
    z=np.where(gate,relu,0.)
    reconstruction=z@theta['Wdec']+theta['bdec']+mean
    return reconstruction,z,pre,gate,centered


def loss_grad(h,labels,theta,mean,u,a,p,objective):
    reconstruction,z,pre,gate,centered=encode_decode(h,theta,mean)
    error=reconstruction-h
    rec=float(np.mean(np.sum(error*error,axis=1)))
    grad_reconstruction=2*error/len(h)
    loss=rec
    kl=0.
    if objective!='local':
        target=readout(h,u,a)
        logits=8*(reconstruction@u/a-.5)
        weight=np.ones(len(h)) if objective=='downstream' else np.where(labels==1,1/(2*p),1/(2*(1-p)))
        kl=float(np.mean(weight*kl_from_logits(target,logits)))
        loss+=.1*kl
        grad_reconstruction+=(.1*weight*(sigmoid(logits)-target)*8/a/len(h))[:,None]*u
    grad=dict(Wdec=z.T@grad_reconstruction,bdec=grad_reconstruction.sum(axis=0))
    dz=(grad_reconstruction@theta['Wdec'].T)*gate
    grad.update(Wenc=centered.T@dz,benc=dz.sum(axis=0))
    return loss,grad,rec,kl


def train(seed,h,labels,u,a,p,objective):
    theta=initialize(seed); mean=h.mean(axis=0)
    m={k:np.zeros_like(v) for k,v in theta.items()}
    v={k:np.zeros_like(x) for k,x in theta.items()}
    rng=np.random.default_rng(4000+seed)
    trace=[]
    for step in range(1,STEPS+1):
        ids=rng.integers(0,len(h),BATCH)
        loss,g,rec,kl=loss_grad(h[ids],labels[ids],theta,mean,u,a,p,objective)
        for name in theta:
            m[name]=.9*m[name]+.1*g[name]
            v[name]=.999*v[name]+.001*g[name]*g[name]
            theta[name]-=LR*(m[name]/(1-.9**step))/(np.sqrt(v[name]/(1-.999**step))+1e-8)
        norms=np.linalg.norm(theta['Wdec'],axis=1,keepdims=True)
        assert np.all(norms>1e-15)
        theta['Wdec']/=norms
        if step==1 or step%100==0:
            trace.append(dict(step=step,minibatch_loss_before_update=float(loss),
                              reconstruction_loss_before_update=rec,weighted_kl_before_update=kl))
    return theta,mean,trace


def pair_outputs(background,theta,mean,u,a):
    h0=background; h1=background+a*u
    rec0,z0,*_=encode_decode(h0,theta,mean)
    rec1,z1,*_=encode_decode(h1,theta,mean)
    delta=z1-z0
    contribution=delta*(theta['Wdec']@u)[None,:]
    # All 24 single-latent counterfactuals are evaluated, no test selection.
    native0=sigmoid(8*((h0@u)[:,None]/a+contribution/a-.5))
    native1=sigmoid(8*((h1@u)[:,None]/a-contribution/a-.5))
    recon0=sigmoid(8*((rec0@u)[:,None]/a+contribution/a-.5))
    recon1=sigmoid(8*((rec1@u)[:,None]/a-contribution/a-.5))
    true0=readout(h0,u,a); true1=readout(h1,u,a)
    native_error=.5*(np.abs(native0-true1[:,None])+np.abs(native1-true0[:,None]))
    recon_error=.5*(np.abs(recon0-true1[:,None])+np.abs(recon1-true0[:,None]))
    count0=(z0>0).sum(axis=1)[:,None]-(z0>0)+(z1>0)
    count1=(z1>0).sum(axis=1)[:,None]-(z1>0)+(z0>0)
    latent_rows=[]
    cosine=theta['Wdec']@u/np.linalg.norm(theta['Wdec'],axis=1)
    for j in range(WIDTH):
        latent_rows.append(dict(latent=j,native_error=float(native_error[:,j].mean()),
                               reconstruction_error=float(recon_error[:,j].mean()),
                               native_off_to_on_error=float(np.abs(native0[:,j]-true1).mean()),
                               native_on_to_off_error=float(np.abs(native1[:,j]-true0).mean()),
                               decoder_cosine=float(cosine[j]),
                               activation_rate_off=float(np.mean(z0[:,j]>0)),
                               activation_rate_on=float(np.mean(z1[:,j]>0)),
                               mean_activation_difference=float(delta[:,j].mean()),
                               patched_activity_mean=float(.5*(count0[:,j].mean()+count1[:,j].mean())),
                               patched_activity_max=int(max(count0[:,j].max(),count1[:,j].max()))))
    full_delta=(z1-z0)@theta['Wdec']
    full_native=.5*(np.abs(readout(h0+full_delta,u,a)-true1)+np.abs(readout(h1-full_delta,u,a)-true0))
    full_recon=.5*(np.abs(readout(rec1,u,a)-true1)+np.abs(readout(rec0,u,a)-true0))
    empty_native=.5*(np.abs(true0-true1)+np.abs(true1-true0))
    empty_recon=.5*(np.abs(readout(rec0,u,a)-true1)+np.abs(readout(rec1,u,a)-true0))
    baselines=dict(full_native_error=float(full_native.mean()),full_reconstruction_error=float(full_recon.mean()),
                   empty_native_error=float(empty_native.mean()),empty_reconstruction_error=float(empty_recon.mean()))
    arrays=dict(native_off_to_on=native0,native_on_to_off=native1,recon_off_to_on=recon0,recon_on_to_off=recon1,
                true_off=true0,true_on=true1,native_error=native_error,reconstruction_error=recon_error,
                patched_activity_off_to_on=count0,patched_activity_on_to_off=count1,
                full_native_error=full_native,full_reconstruction_error=full_recon,
                empty_native_error=empty_native,empty_reconstruction_error=empty_recon)
    return latent_rows,baselines,arrays


def same_label_control(background,theta,mean,u,a,latent):
    rows=[]
    arrays={}
    for label in (0,1):
        h=background+a*label*u
        rec,z,*_=encode_decode(h,theta,mean)
        donor=np.roll(z,-1,axis=0)
        difference=(donor[:,latent]-z[:,latent])[:,None]*theta['Wdec'][latent]
        native=readout(h+difference,u,a)
        reconstruction=readout(rec+difference,u,a)
        true=readout(h,u,a); rec_original=readout(rec,u,a)
        rows.append(dict(label=label,native_change=float(np.abs(native-true).mean()),
                         reconstruction_change=float(np.abs(reconstruction-rec_original).mean()),
                         reconstruction_true_error=float(np.abs(reconstruction-true).mean()),
                         patched_activity_mean=float(np.mean((z>0).sum(1)-(z[:,latent]>0)+(donor[:,latent]>0))),
                         patched_activity_max=int(np.max((z>0).sum(1)-(z[:,latent]>0)+(donor[:,latent]>0)))))
        arrays[f'same_label_{label}_native']=native
        arrays[f'same_label_{label}_reconstruction']=reconstruction
    return rows,arrays


def population_metrics(background,theta,mean,u,a,p):
    rows=[]
    for label in (0,1):
        h=background+a*label*u
        reconstruction,z,*_=encode_decode(h,theta,mean)
        true=readout(h,u,a); pred=readout(reconstruction,u,a)
        logits=8*(reconstruction@u/a-.5)
        rows.append(dict(label=label,reconstruction_mse=float(np.mean(np.sum((reconstruction-h)**2,axis=1))),
                         downstream_kl=float(kl_from_logits(true,logits).mean()),
                         probability_error=float(np.abs(pred-true).mean()),
                         classification_accuracy=float(np.mean((pred>=.5)==label)),
                         activity_mean=float(np.mean((z>0).sum(1)))))
    weighted={name:(1-p)*rows[0][name]+p*rows[1][name] for name in rows[0] if name!='label'}
    return rows,weighted


def execution_checks(theta,mean,u,a,background):
    h0=background; h1=background+a*u
    rec0,z0,*_=encode_decode(h0,theta,mean)
    rec1,z1,*_=encode_decode(h1,theta,mean)
    direct=rec0+(z1-z0)@theta['Wdec']
    full_err=float(np.max(np.abs(direct-rec1)))
    empty_err=float(np.max(np.abs((h0+0*z0@theta['Wdec'])-h0)))
    native_identity_err=float(np.max(np.abs(h0+(z0-z0)@theta['Wdec']-h0)))
    oracle_err=float(np.max(np.abs(readout(h0+a*u,u,a)-readout(h1,u,a))))
    assert max(full_err,empty_err,native_identity_err,oracle_err)<1e-10
    return dict(full_reconstruction_max_error=full_err,empty_native_max_error=empty_err,
                identity_native_max_error=native_identity_err,oracle_max_error=oracle_err)


def correctness_checks():
    banks,u,dictionary,permutation,record=generate(0)
    assert np.max(np.abs(dictionary@u))<1e-12
    for p in PS:
        for a in AMPLITUDES:
            h,labels=training_data(banks['train'],u,permutation,p,a)
            assert labels.sum()==round(10000*p)
            expected=sigmoid(8*(labels-.5))
            assert np.max(np.abs(readout(h,u,a)-expected))<1e-12
    p=.01; a=.25
    # Artificial continuous batch avoids all ReLU and TopK boundary kinks.
    rng=np.random.default_rng(8765)
    h=rng.normal(size=(13,12)); labels=(np.arange(13)%3==0).astype(int)
    theta=initialize(0); mean=h.mean(0)
    _,_,pre,gate,_=encode_decode(h,theta,mean)
    sorted_pre=np.sort(np.maximum(pre,0),axis=1)[:,::-1]
    assert np.min(np.abs(pre))>1e-5
    assert np.min(sorted_pre[:,3]-sorted_pre[:,4])>1e-5
    coordinates=[(name,ix) for name,value in theta.items() for ix in np.ndindex(value.shape)]
    # Include every parameter family, then a fixed random sample.
    chosen=[next((name,ix) for name,ix in coordinates if name==family) for family in theta]
    chosen += [coordinates[i] for i in rng.choice(len(coordinates),size=28,replace=False)]
    worst={}
    for objective in OBJECTIVES:
        _,grad,*_=loss_grad(h,labels,theta,mean,u,a,p,objective)
        maximum=0.
        for name,ix in chosen:
            original=theta[name][ix]
            theta[name][ix]=original+1e-6
            plus=loss_grad(h,labels,theta,mean,u,a,p,objective)[0]
            theta[name][ix]=original-1e-6
            minus=loss_grad(h,labels,theta,mean,u,a,p,objective)[0]
            theta[name][ix]=original
            numerical=(plus-minus)/2e-6
            assert np.isfinite(plus) and np.isfinite(minus) and np.isfinite(numerical)
            error=abs(numerical-grad[name][ix])/max(1,abs(numerical),abs(grad[name][ix]))
            maximum=max(maximum,float(error))
        assert maximum<2e-6,(objective,maximum)
        worst[objective]=maximum
    # This update is only a deterministic projection control; it is not a training run.
    changed={k:v.copy() for k,v in theta.items()}
    changed['Wdec']-=1e-4*grad['Wdec']
    changed['Wdec']/=np.linalg.norm(changed['Wdec'],axis=1,keepdims=True)
    norm_err=float(np.max(np.abs(np.linalg.norm(changed['Wdec'],axis=1)-1)))
    assert norm_err<1e-12
    controls=execution_checks(theta,mean,u,a,banks['validation'])
    # Check all held-out generator cells, including reverse oracle patches.
    max_readout_error=0.
    for a in AMPLITUDES:
        for name in ('validation','test'):
            b=banks[name]
            off=readout(b,u,a); on=readout(b+a*u,u,a)
            max_readout_error=max(max_readout_error,float(np.max(np.abs(off-sigmoid(-4.)))),
                                 float(np.max(np.abs(on-sigmoid(4.)))))
            assert np.max(np.abs(readout((b+a*u)-a*u,u,a)-off))<1e-12
    assert max_readout_error<1e-12
    # Draw identities are disjoint. Values may repeat, especially the all-zero atom.
    identity_sets=[{name+':'+str(i) for i in range(len(bank))} for name,bank in banks.items()]
    assert all(identity_sets[i].isdisjoint(identity_sets[j]) for i in range(3) for j in range(i+1,3))
    return dict(passed=True,gradient_coordinates_per_objective=len(chosen),
                max_scaled_gradient_errors=worst,decoder_norm_max_error=norm_err,
                generator_readout_max_error=max_readout_error,sample_identity_overlap=0,
                zero_background_counts=record['zero_background_counts'],
                exact_cross_split_shared_unique_values=record['exact_cross_split_shared_unique_values'],**controls)


def verify_lock(commit):
    if not commit: raise ValueError('--run requires --lock-commit')
    resolved=subprocess.check_output(['git','rev-parse',commit+'^{commit}'],cwd=REPO,text=True).strip()
    hashes={}
    for path in (ROOT/'analysis-plan.md',Path(__file__).resolve()):
        relative=path.relative_to(REPO).as_posix()
        locked=subprocess.check_output(['git','show',resolved+':'+relative],cwd=REPO)
        if locked!=path.read_bytes(): raise ValueError('File changed after lock: '+relative)
        hashes[relative]=sha(locked)
    return dict(commit=resolved,sha256=hashes)


def summarize(selected):
    factorial=[]; contrasts=[]
    for p in PS:
        for a in AMPLITUDES:
            for objective in OBJECTIVES:
                rows=[r for r in selected if r['p']==p and r['amplitude']==a and r['objective']==objective]
                metrics=('test_native_error','test_reconstruction_error','test_full_native_error',
                         'test_full_reconstruction_error','test_empty_native_error',
                         'natural_reconstruction_mse','natural_downstream_kl','same_label_native_change',
                         'natural_activity_mean')
                factorial.append(dict(p=p,amplitude=a,objective=objective,n=len(rows),
                                      **{key:float(np.mean([r[key] for r in rows])) for key in metrics}))
            for seed in SEEDS:
                rows={r['objective']:r for r in selected if r['p']==p and r['amplitude']==a and r['seed']==seed}
                contrasts.append(dict(seed=seed,p=p,amplitude=a,
                                      ordinary_minus_balanced=rows['downstream']['test_native_error']-rows['balanced']['test_native_error']))
    values=np.array([r['ordinary_minus_balanced'] for r in contrasts if r['p']==.01 and r['amplitude']==.25])
    rng=np.random.default_rng(20261013)
    boot=rng.choice(values,size=(10000,len(values)),replace=True).mean(1)
    primary=dict(n=len(values),values=values.tolist(),mean=float(values.mean()),median=float(np.median(values)),
                 minimum=float(values.min()),maximum=float(values.max()),ci95=np.quantile(boot,[.025,.975]).tolist())
    return factorial,contrasts,primary


def run(commit):
    lock=verify_lock(commit); started=time.monotonic(); checks=correctness_checks()
    results=ROOT/'results'
    if results.exists() and any(results.iterdir()): raise ValueError('Results already exist; preserve them and use a new checkout.')
    results.mkdir(parents=True,exist_ok=True)
    (ROOT/'models').mkdir(exist_ok=True); (results/'selected-test-outputs').mkdir()
    write_json(results/'lock.json',lock); write_json(results/'preflight-controls.json',checks)
    all_latents=[]; selected=[]; all_training=[]; all_population=[]; all_controls=[]; all_generators=[]; all_nuisance=[]
    for seed in SEEDS:
        banks,u,dictionary,permutation,record=generate(seed)
        all_generators.append(record)
        (ROOT/'data').mkdir(exist_ok=True)
        np.savez_compressed(ROOT/'data'/f'dictionary-seed-{seed}.npz',rare_direction=u,background_dictionary=dictionary)
        for p in PS:
            for a in AMPLITUDES:
                h,labels=training_data(banks['train'],u,permutation,p,a)
                for objective in OBJECTIVES:
                    begin=time.monotonic()
                    identity=dict(seed=seed,p=p,amplitude=a,objective=objective)
                    run_id=f'seed-{seed}_p-{p:g}_a-{a:g}_{objective}'
                    theta,mean,trace=train(seed,h,labels,u,a,p,objective)
                    np.savez_compressed(ROOT/'models'/(run_id+'.npz'),**theta,mean=mean)
                    all_training.extend([{**identity,**row} for row in trace])
                    controls=execution_checks(theta,mean,u,a,banks['test'])
                    all_controls.append({**identity,**controls})
                    validation,vbase,_=pair_outputs(banks['validation'],theta,mean,u,a)
                    best=min(row['native_error'] for row in validation)
                    chosen=next(row['latent'] for row in validation if row['native_error']<=best+TOL)
                    test,tbase,arrays=pair_outputs(banks['test'],theta,mean,u,a)
                    for split,rows in (('validation',validation),('test',test)):
                        all_latents.extend([{**identity,'split':split,**row} for row in rows])
                    nuisance,nuisance_arrays=same_label_control(banks['test'],theta,mean,u,a,chosen)
                    all_nuisance.extend([{**identity,'selected_latent':chosen,**row} for row in nuisance])
                    pop,natural=population_metrics(banks['test'],theta,mean,u,a,p)
                    all_population.extend([{**identity,**row} for row in pop])
                    training_code=encode_decode(h,theta,mean)[1]
                    row={**identity,'selected_latent':chosen,'training_rare_count':int(labels.sum()),
                         'validation_native_error':validation[chosen]['native_error'],
                         **{'test_'+k:v for k,v in test[chosen].items() if k!='latent'},
                         **{'test_'+k:v for k,v in tbase.items()},
                         **{'natural_'+k:v for k,v in natural.items()},
                         'same_label_native_change':float(np.mean([r['native_change'] for r in nuisance])),
                         'same_label_reconstruction_change':float(np.mean([r['reconstruction_change'] for r in nuisance])),
                         'same_label_reconstruction_true_error':float(np.mean([r['reconstruction_true_error'] for r in nuisance])),
                         'same_label_patched_activity_max':max(r['patched_activity_max'] for r in nuisance),
                         'dead_training_latents':int(np.sum(~np.any(training_code>0,axis=0))),
                         'training_activity_mean':float(np.mean((training_code>0).sum(1))),
                         'seconds':time.monotonic()-begin}
                    selected.append(row)
                    chosen_arrays={key:(value[:,chosen] if value.ndim==2 else value) for key,value in arrays.items()}
                    chosen_arrays.update(nuisance_arrays)
                    np.savez_compressed(results/'selected-test-outputs'/(run_id+'.npz'),**chosen_arrays)
                    write_csv(results/'selected-latents.csv',selected)
                    print(json.dumps(row),flush=True)
    write_json(ROOT/'data'/'generator-records.json',all_generators)
    write_csv(results/'all-latent-scores.csv',all_latents)
    write_csv(results/'training-trace.csv',all_training)
    write_csv(results/'population-metrics.csv',all_population)
    write_csv(results/'execution-controls.csv',all_controls)
    write_csv(results/'same-label-controls.csv',all_nuisance)
    factorial,contrasts,primary=summarize(selected)
    write_csv(results/'factorial-summary.csv',factorial); write_csv(results/'paired-contrasts.csv',contrasts)
    summary=dict(primary_cell=dict(p=.01,amplitude=.25),primary_contrast='downstream minus balanced; positive favours balanced',
                 primary=primary,factorial=factorial,runs=len(selected),elapsed_seconds=time.monotonic()-started,
                 environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),blas_threads=1),lock=lock)
    write_json(results/'summary.json',summary)
    assert verify_lock(commit)==lock
    files=sorted(list((ROOT/'data').rglob('*'))+list((ROOT/'models').rglob('*'))+list(results.rglob('*')))
    write_json(results/'manifest.json',{f.relative_to(ROOT).as_posix():sha(f.read_bytes()) for f in files if f.is_file() and f.name!='manifest.json'})
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true'); mode.add_argument('--run',action='store_true')
    parser.add_argument('--lock-commit'); args=parser.parse_args()
    if args.check: print(json.dumps(correctness_checks(),indent=2))
    else: run(args.lock_commit)


if __name__=='__main__': main()
