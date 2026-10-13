#!/usr/bin/env python3
"""Post-failure float64 evaluation with the original float32 training procedure."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(8))
ARCHS = ('direct', 'filtered')
SHIFTS = ((0,0), (-1,0), (1,0), (0,-1), (0,1))
CONTROL_SHIFTS = ((-4,0), (4,0), (0,-4), (0,4))
EPOCHS, BATCH, CAP = 10, 128, 1200
BOOT_SEED, BOOT_REPEATS = 2026101417, 10000
PACKAGES = {'numpy': '2.3.5', 'torch': '2.8.0'}
DATA_HASHES = {
 'optdigits.tra': 'e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd',
 'optdigits.tes': '6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8'}
TOL_LOGIT, TOL_FEATURE, TOL_PROB = 1e-10, 1e-10, 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')


def lock_check(commit):
    repo = ROOT.parents[1]
    got = subprocess.run(['git','rev-parse',commit+'^{commit}'], cwd=repo,
                         check=True,capture_output=True,text=True).stdout.strip()
    if got != commit:
        raise RuntimeError('Supply the full immutable public lock commit.')
    paths = [ROOT/'analysis-plan.md', ROOT/'followup-plan.md', ROOT/'code/run.py', Path(__file__).resolve(), ROOT/'requirements.txt']
    paths += [ROOT/'data'/name for name in DATA_HASHES]
    for path in paths:
        rel = path.relative_to(repo).as_posix()
        raw = subprocess.run(['git','show',commit+':'+rel],cwd=repo,
                             check=True,capture_output=True).stdout
        if raw != path.read_bytes():
            raise RuntimeError('Locked bytes differ: '+rel)


def data():
    out=[]
    for name, count in [('optdigits.tra',3823),('optdigits.tes',1797)]:
        path=ROOT/'data'/name
        assert sha(path)==DATA_HASHES[name]
        arr=np.loadtxt(path,delimiter=',',dtype=np.int64)
        assert arr.shape==(count,65)
        assert ((arr[:,:64]>=0)&(arr[:,:64]<=16)).all()
        assert ((arr[:,64]>=0)&(arr[:,64]<10)).all()
        x=torch.zeros(count,1,16,16,dtype=torch.float32)
        x[:,:,4:12,4:12]=torch.from_numpy(arr[:,:64].reshape(count,1,8,8)).float()/16
        out.append((x,torch.from_numpy(arr[:,64].copy())))
    return out


def shift(x, dy, dx):
    """Positive dy moves down; positive dx moves right. No interpolation."""
    return torch.roll(x,shifts=(int(dy),int(dx)),dims=(-2,-1))


def augment(x, shifts):
    n=x.shape[0]
    q=torch.arange(16)
    rows=(q[None,:]-torch.from_numpy(shifts[:,0]).long()[:,None])%16
    cols=(q[None,:]-torch.from_numpy(shifts[:,1]).long()[:,None])%16
    return x[torch.arange(n)[:,None,None],0,rows[:,:,None],cols[:,None,:]].unsqueeze(1)


class Net(nn.Module):
    def __init__(self, filtered, stride=2):
        super().__init__()
        self.filtered,self.stride=filtered,stride
        self.conv1=nn.Conv2d(1,8,3,padding=1,padding_mode='circular',bias=True)
        self.conv2=nn.Conv2d(8,16,3,padding=1,padding_mode='circular',bias=True)
        self.conv3=nn.Conv2d(16,32,3,padding=1,padding_mode='circular',bias=True)
        self.relu1=nn.ReLU()
        self.fc=nn.Linear(32,10,bias=True)
        v=torch.tensor([1.,2.,1.],dtype=torch.float32)
        self.register_buffer('kernel',torch.outer(v,v)[None,None]/16)

    def down(self,a):
        if self.filtered:
            a=F.conv2d(F.pad(a,(1,1,1,1),mode='circular'),
                       self.kernel.expand(a.shape[1],1,3,3),groups=a.shape[1])
        return a[:,:,::self.stride,::self.stride]

    def forward(self,x,channel=None,checker=None):
        a=self.relu1(self.conv1(x))
        if channel is not None:
            edited=a.clone()
            if channel>=0:
                edited[:,channel]=0
                if checker is not None:
                    checker('ablated_channel_zero',float(edited[:,channel].abs().max()),0.)
                    keep=[i for i in range(8) if i!=channel]
                    checker('untouched_channels',float((edited[:,keep]-a[:,keep]).abs().max()),0.)
            a=edited
        a=self.down(a)
        a=self.down(F.relu(self.conv2(a)))
        a=F.relu(self.conv3(a)).mean(dim=(-2,-1))
        return self.fc(a)


def init(seed,filtered,stride=2):
    torch.manual_seed(seed)
    return Net(filtered,stride)


def optimizer(model):
    return torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,
                            weight_decay=0,amsgrad=False,foreach=False,
                            maximize=False,capturable=False,differentiable=False,fused=False)


def schedule(seed,n):
    rng=np.random.default_rng(10000+seed)
    orders=[]; shifts=[]
    for _ in range(EPOCHS):
        orders.append(rng.permutation(n))
        shifts.append(rng.integers(-2,3,size=(n,2),dtype=np.int8))
    return np.stack(orders),np.stack(shifts)


def parameters_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode()); h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


def softmax(logits):
    z=logits.astype(np.float64)
    e=np.exp(z-z.max(axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)


class Table:
    def __init__(self,path):
        self.file=Path(path).open('w',newline=''); self.writer=None; self.count=0
    def add(self,row):
        if self.writer is None:
            self.writer=csv.DictWriter(self.file,fieldnames=list(row)); self.writer.writeheader()
        self.writer.writerow(row); self.file.flush(); self.count+=1
    def close(self): self.file.close()


class Run:
    def __init__(self,output,dummy):
        self.out=Path(output); self.out.mkdir(parents=True,exist_ok=False)
        self.started=time.monotonic(); self.dummy=dummy
        self.calls=self.examples=self.updates=0
        self.tags={'seed':-1,'architecture':'none','shift':'none','batch':-1,'channel':-1}
        self.controls=Table(self.out/'controls.csv')
        self.training=Table(self.out/'training.csv')
        self.maxima={}; self.active_model=None

    def budget(self):
        if time.monotonic()-self.started>CAP:
            raise TimeoutError('Fixed 20-minute run cap exceeded; preserve partial output.')

    def check(self,name,error,tol):
        error=float(error); ok=bool(np.isfinite(error) and error<=tol)
        self.maxima[name]=max(error,self.maxima.get(name,0.))
        self.controls.add(dict(**self.tags,name=name,error=error,tolerance=tol,passed=ok))
        if not ok: raise AssertionError(f'{name}: {error} > {tol}')

    def forward(self,model,x,channel=None):
        self.budget(); self.calls+=1; self.examples+=len(x)
        z=model(x,channel,checker=self.check)
        if not torch.isfinite(z).all(): raise FloatingPointError('Non-finite logits.')
        return z

    def train(self,model,x,y,orders,moves,seed,architecture):
        model.train(); opt=optimizer(model)
        for ep,order in enumerate(orders):
            for b,start in enumerate(range(0,len(x),BATCH)):
                self.budget()
                self.tags=dict(seed=seed,architecture=architecture,shift=f'train_epoch_{ep}',batch=b,channel=-1)
                ids=order[start:start+BATCH]
                xb=augment(x[ids],moves[ep,start:start+BATCH])
                opt.zero_grad(set_to_none=True)
                logits=self.forward(model,xb)
                loss=F.cross_entropy(logits,y[ids],reduction='mean')
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss.')
                loss.backward(); opt.step(); self.updates+=1
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError('Non-finite parameters.')
                self.training.add(dict(seed=seed,architecture=architecture,epoch=ep,
                    batch=b,n=len(ids),loss=float(loss.detach()),global_update=self.updates))
        model.eval()

    @torch.no_grad()
    def evaluate(self,model,x,seed,architecture,with_four=True):
        model.eval()
        partial=None
        if not self.dummy:
            partial=self.out/'logits'/f'{seed}-{architecture}.partial.npy'
            logits=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float64,shape=(5,9,len(x),10))
            logits[:]=np.nan; logits.flush()
        else:
            logits=np.empty((5,9,len(x),10),dtype=np.float64)
        for si,(dy,dx) in enumerate(SHIFTS):
            for start in range(0,len(x),BATCH):
                self.tags=dict(seed=seed,architecture=architecture,shift=f'{dy},{dx}',batch=start//BATCH,channel=-1)
                original=x[start:start+BATCH]; xb=shift(original,dy,dx)
                base=self.forward(model,xb)
                logits[si,0,start:start+len(xb)]=base.numpy()
                noop=self.forward(model,xb,-1)
                self.check('no_op_logits',float((noop-base).abs().max()),0.)
                af=model.relu1(model.conv1(xb))
                expected=shift(model.relu1(model.conv1(original)),dy,dx)
                self.check('first_features_shift',float((af-expected).abs().max()),TOL_FEATURE)
                for c in range(8):
                    self.tags['channel']=c
                    logits[si,c+1,start:start+len(xb)]=self.forward(model,xb,c).numpy()
                if partial is not None: logits.flush()
        if with_four:
            for dy,dx in CONTROL_SHIFTS:
                for start in range(0,len(x),BATCH):
                    xb=shift(x[start:start+BATCH],dy,dx)
                    self.tags=dict(seed=seed,architecture=architecture,shift=f'{dy},{dx}',batch=start//BATCH,channel=-1)
                    for j in range(9):
                        self.tags['channel']=j-1
                        z=self.forward(model,xb,None if j==0 else j-1).numpy()
                        err=np.max(np.abs(z-logits[0,j,start:start+len(xb)]))
                        self.check('four_pixel_logits',err,TOL_LOGIT)
        return logits

    def close(self): self.controls.close(); self.training.close()


def summarize(logits,labels,checker):
    p=softmax(logits)
    if not np.isfinite(p).all(): raise FloatingPointError('Non-finite probabilities.')
    checker('probability_row_sum',np.max(np.abs(p.sum(axis=-1)-1)),TOL_PROB)
    effect=p[:,0:1]-p[:,1:]
    error=.5*np.abs(effect[1:]-effect[0:1]).sum(axis=-1)
    base_shift=.5*np.abs(p[1:,0]-p[0:1,0]).sum(axis=-1)
    ablated_shift=.5*np.abs(p[1:,1:]-p[0:1,1:]).sum(axis=-1)
    bound=base_shift[:,None,:]+ablated_shift
    checker('triangle_inequality_excess',max(0.,float(np.max(error-bound))),TOL_PROB)
    magnitude=.5*np.abs(effect).sum(axis=-1)
    top=logits[:,0].argmax(axis=-1)
    correct=top==labels[None,:]
    subgroup=correct.all(axis=0)
    return dict(D=float(error.mean()),
        baseline_shift_tv=float(base_shift.mean()),ablated_shift_tv=float(ablated_shift.mean()),
        triangle_bound=float(bound.mean()),effect_size_original=float(magnitude[0].mean()),
        effect_size_all_shifts=float(magnitude.mean()),
        accuracy_original=float(correct[0].mean()),accuracy_shifted=float(correct[1:].mean()),
        top1_consistency=float((top[1:]==top[0:1]).mean()),
        correct_all_shifts_n=int(subgroup.sum()),
        correct_all_shifts_D=float(error[:,:,subgroup].mean()) if subgroup.any() else None,
        by_channel=[dict(channel=c,D=float(error[:,c].mean()),
                        effect_size_original=float(magnitude[0,c].mean()),
                        ablated_shift_tv=float(ablated_shift[:,c].mean())) for c in range(8)],
        by_shift=[dict(dy=dy,dx=dx,D=float(error[i-1].mean()),
                      accuracy=float(correct[i].mean()),baseline_shift_tv=float(base_shift[i-1].mean()),
                      ablated_shift_tv=float(ablated_shift[i-1].mean())) for i,(dy,dx) in enumerate(SHIFTS) if i],
        by_digit=[dict(digit=d,n=int((labels==d).sum()),D=float(error[:,:,labels==d].mean()),
                      accuracy_original=float(correct[0,labels==d].mean())) for d in range(10)])


def checkpoint(path,model):
    np.savez_compressed(path,**{k:v.detach().numpy() for k,v in model.state_dict().items()})


def dummy(run):
    gen=torch.Generator().manual_seed(8913)
    x=torch.randn(239,1,16,16,generator=gen)
    y=torch.arange(239)%10
    start=time.monotonic(); init_hashes=[]; train_seconds=[]; eval_seconds=[]
    for arch in ARCHS:
        model=init(9817,arch=='filtered')
        init_hashes.append(parameters_hash(model))
        # Random dummy data only. 15 epochs give 30 updates with batch sizes 128 and 111.
        orders,moves=schedule(8913,len(x))
        orders=np.tile(orders[:1],(15,1)); moves=np.tile(moves[:1],(15,1,1))
        run.tags=dict(seed=9817,architecture=arch,shift='training',batch=-1,channel=-1)
        ts=time.monotonic(); run.train(model,x,y,orders,moves,9817,arch)
        train_seconds.append(time.monotonic()-ts)
        ts=time.monotonic(); dummy_logits=run.evaluate(model.double(),x[:133].double(),9817,arch)
        eval_seconds.append(time.monotonic()-ts)
        run.tags=dict(seed=9817,architecture=arch,shift='all',batch=-1,channel=-1)
        summarize(dummy_logits,y[:133].numpy(),run.check)
    assert init_hashes[0]==init_hashes[1]
    # Independent scalar-roll check of the vector augmentation.
    moves=np.random.default_rng(7711).integers(-2,3,size=(len(x),2),dtype=np.int8)
    a=augment(x,moves)
    b=torch.cat([shift(x[i:i+1],*moves[i]) for i in range(len(x))])
    run.check('augmentation_matches_scalar_roll',float((a-b).abs().max()),0.)
    # Stride-one structural identity, without training or any data-set input.
    for filtered in (False,True):
        m=init(9819,filtered,stride=1).double().eval()
        run.tags=dict(seed=9819,architecture='stride1_filtered' if filtered else 'stride1_direct',shift='dummy',batch=0,channel=-1)
        with torch.no_grad():
            for c in [None,*range(8)]:
                reference=run.forward(m,x.double(),c)
                for dy,dx in SHIFTS[1:]+CONTROL_SHIFTS:
                    err=(run.forward(m,shift(x.double(),dy,dx),c)-reference).abs().max()
                    run.check('stride_one_logits',float(err),TOL_LOGIT)
    elapsed=time.monotonic()-start
    # A conservative arithmetic projection, not a measured scientific run time.
    return dict(scientific_model_outputs_collected=False,random_dummy_inputs=239,
        training_batch_sizes=[128,111],evaluation_batch_sizes=[128,5],
        dummy_updates=run.updates,calls=run.calls,examples=run.examples,
        elapsed_seconds=elapsed,initial_parameter_hashes_equal=True,
        control_maxima=run.maxima,control_rows=run.controls.count,
        projection_note='Twice the projection by exact forwarded-example ratios; this is an estimate.',
        dummy_train_seconds=train_seconds,dummy_evaluation_seconds=eval_seconds,
        projection_train_example_ratio=420530/(2*15*239),
        projection_evaluation_example_ratio=2472672/(2*133*86),
        projected_seconds_with_2x_safety=2*(sum(train_seconds)*420530/(2*15*239)
                                          +sum(eval_seconds)*2472672/(2*133*86)),
        scientific_training_updates=3300,scientific_evaluation_calls=20640)


ORIGINAL=ROOT/'results'
ORIGINAL_RESULTS_COMMIT='5d7a88d9e79d974f7e320359039e11e813d20b15'
ORIGINAL_MANIFEST_SHA='a4c9e610eb86d31fc163586c1de563eee3e6856a08460a5a9d94cd052b9f5e26'
ORIGINAL_TRAINING_SHA='e1afd92ec8939fc756ad2f4558546c2f5cc286a13287ebaab5cbf37eebb727cb'
REUSED={
 '0-direct.npz':'ac8be47aefc85db0c7414157025da5bedeae7a3572479f700c9d188c189e2a06',
 '0-filtered.npz':'e934d69bb24ba3b669408922db68eb34e0e96105628d7597a37d72a7ecd1cef5',
 '1-direct.npz':'3602fff9ff353f8a6dd0cbe1460cbd2fe2147b7768fe929652b1f56424340b76',
 '1-filtered.npz':'b0c870a6ca82df1110ca18d56e00e23eb05efabf06eb8a8d3212a53961f91d82',
 '2-direct.npz':'b6cc052463f641f3cb9eb1ecccf374de6610f384219de30d96902e1274c9ea8e'}
SCHEDULE_HASHES={
 'seed-0.npz':'9481caf447fdfaae2fe35670d7c283691eac2c0b2b57d92520243f196243e21c',
 'seed-1.npz':'eccdc3f33e57324920d095a1c2d0a5c4cf31f6688f139e1f2b468abf1cc132ec',
 'seed-2.npz':'7f01434ae3de01c6cb793b31e0a9dad515940bdb9eb059e2bb876d39c8ebeccf'}


def verify_reuse():
    assert sha(ORIGINAL/'manifest.json')==ORIGINAL_MANIFEST_SHA
    manifest=json.loads((ORIGINAL/'manifest.json').read_text())
    assert manifest['status']=='failed'
    for name,digest in manifest['files'].items(): assert sha(ORIGINAL/name)==digest
    repo=ROOT.parents[1]
    locked_paths=[ORIGINAL/'manifest.json',ORIGINAL/'training.csv']
    locked_paths += [ORIGINAL/'models'/name for name in REUSED]
    locked_paths += [ORIGINAL/'schedules'/name for name in SCHEDULE_HASHES]
    for path in locked_paths:
        rel=path.relative_to(repo).as_posix()
        blob=subprocess.run(['git','show',ORIGINAL_RESULTS_COMMIT+':'+rel],cwd=repo,
                            check=True,capture_output=True).stdout
        assert blob==path.read_bytes(),rel
    assert sha(ORIGINAL/'training.csv')==ORIGINAL_TRAINING_SHA
    rows=list(csv.DictReader((ORIGINAL/'training.csv').open()))
    assert len(rows)==1500
    report=[]
    for model_index,(name,digest) in enumerate(REUSED.items()):
        seed,arch=name[:-4].split('-');seed=int(seed)
        assert sha(ORIGINAL/'models'/name)==digest
        weights=dict(np.load(ORIGINAL/'models'/name))
        assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in weights.values())
        sub=[r for r in rows if int(r['seed'])==seed and r['architecture']==arch]
        assert len(sub)==300
        for i,r in enumerate(sub):
            assert int(r['epoch'])==i//30 and int(r['batch'])==i%30
            assert int(r['n'])==(111 if i%30==29 else 128)
            assert int(r['global_update'])==model_index*300+i+1
            assert np.isfinite(float(r['loss']))
        encoded=json.dumps(sub,sort_keys=True,separators=(',',':')).encode()
        report.append(dict(seed=seed,architecture=arch,checkpoint_sha256=digest,
                           training_rows=300,training_rows_sha256=hashlib.sha256(encoded).hexdigest()))
    for seed in (0,1,2):
        name=f'seed-{seed}.npz';assert sha(ORIGINAL/'schedules'/name)==SCHEDULE_HASHES[name]
        saved=dict(np.load(ORIGINAL/'schedules'/name));orders,moves=schedule(seed,3823)
        assert set(saved)=={'orders','shifts'}
        assert np.array_equal(saved['orders'],orders) and np.array_equal(saved['shifts'],moves)
    return dict(original_manifest_sha256=ORIGINAL_MANIFEST_SHA,original_results_commit=ORIGINAL_RESULTS_COMMIT,
                original_training_sha256=ORIGINAL_TRAINING_SHA,models=report,
                shared_schedule_sha256=SCHEDULE_HASHES,model_calls=0)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True)
    ap.add_argument('--lock-commit'); ap.add_argument('--dummy',action='store_true')
    args=ap.parse_args()
    for pkg,want in PACKAGES.items():
        if version(pkg)!=want: raise RuntimeError(f'{pkg} must be {want}')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.mkldnn.enabled=False
    if not args.dummy:
        if not args.lock_commit: raise RuntimeError('Scientific runs require --lock-commit.')
        lock_check(args.lock_commit)
    run=Run(args.output,args.dummy)
    metadata=dict(mode='dummy' if args.dummy else 'post-failure-followup',packages=PACKAGES,
        python=platform.python_version(),threads=2,device='cpu',training_dtype='float32',evaluation_dtype='float64',
        probability_dtype='numpy float64',mkldnn=False,deterministic=True,
        plan_sha256=sha(ROOT/'followup-plan.md'),original_plan_sha256=sha(ROOT/'analysis-plan.md'),runner_sha256=sha(__file__),
        data_hashes=DATA_HASHES,lock_commit=args.lock_commit)
    dump(run.out/'metadata.json',metadata)
    try:
        if args.dummy:
            result=dummy(run); result.update(metadata); dump(run.out/'dummy-preflight.json',result)
            print(json.dumps(result,indent=2))
        else:
            reuse=verify_reuse()
            dump(run.out/'reuse-verification.json',reuse)
            shutil.copy2(ORIGINAL/'training.csv',run.out/'reused-training.csv')
            (train_x,train_y),(test_x,test_y)=data()
            test_x=test_x.double()
            labels=test_y.numpy(); np.save(run.out/'test-labels.npy',labels)
            (run.out/'models').mkdir(); (run.out/'logits').mkdir(); (run.out/'schedules').mkdir()
            summaries=[]
            for seed in SEEDS:
                orders,moves=schedule(seed,len(train_x))
                schedule_path=run.out/'schedules'/f'seed-{seed}.npz'
                if seed in (0,1,2): shutil.copy2(ORIGINAL/'schedules'/f'seed-{seed}.npz',schedule_path)
                else: np.savez_compressed(schedule_path,orders=orders,shifts=moves)
                hashes=[]
                for arch in ARCHS:
                    model=init(seed,arch=='filtered'); run.active_model=model; hashes.append(parameters_hash(model))
                    run.tags=dict(seed=seed,architecture=arch,shift='training',batch=-1,channel=-1)
                    name=f'{seed}-{arch}.npz'
                    reused=name in REUSED
                    if reused:
                        weights=dict(np.load(ORIGINAL/'models'/name))
                        model.load_state_dict({k:torch.from_numpy(v) for k,v in weights.items()})
                        shutil.copy2(ORIGINAL/'models'/name,run.out/'models'/name)
                    else:
                        run.train(model,train_x,train_y,orders,moves,seed,arch)
                        checkpoint(run.out/'models'/name,model)
                    logits=run.evaluate(model.double(),test_x,seed,arch)
                    np.savez_compressed(run.out/'logits'/f'{seed}-{arch}.npz',logits=logits)
                    logits.flush()
                    (run.out/'logits'/f'{seed}-{arch}.partial.npy').unlink()
                    run.tags=dict(seed=seed,architecture=arch,shift='all',batch=-1,channel=-1)
                    row=dict(seed=seed,architecture=arch,reused_training=reused,initial_parameter_sha256=hashes[-1],**summarize(logits,labels,run.check))
                    summaries.append(row); dump(run.out/'model-summaries.json',summaries)
                    print(f'Complete seed {seed}, {arch}; all {len(test_x)} test images retained.',flush=True)
                if hashes[0]!=hashes[1]: raise AssertionError('Paired initial parameters differ.')
            diff=np.array([summaries[2*s]['D']-summaries[2*s+1]['D'] for s in SEEDS])
            rng=np.random.default_rng(BOOT_SEED)
            boot=diff[rng.integers(0,len(diff),size=(BOOT_REPEATS,len(diff)))].mean(axis=1)
            report=dict(status='post-failure follow-up; original primary comparison remains incomplete',primary='D_direct_minus_D_filtered',paired_seed_differences=diff.tolist(),
                mean=float(diff.mean()),percentile_95_interval=np.quantile(boot,[.025,.975],method='linear').tolist(),
                bootstrap_seed=BOOT_SEED,bootstrap_repeats=BOOT_REPEATS,
                control_maxima=run.maxima,control_rows=run.controls.count,
                calls=run.calls,examples=run.examples,new_training_updates=run.updates,reused_training_updates=1500,
                elapsed_seconds=time.monotonic()-run.started,
                interpretation='Seed interval conditional on this fixed data split and training design.')
            dump(run.out/'summary.json',report)
            assert run.updates==3300 and run.calls==23940 and run.examples==2893202
            assert len(summaries)==16 and run.controls.count==45632
            run.budget()
            lock_check(args.lock_commit)
            verify_reuse()
            run.budget()
        run.close()
        files={p.relative_to(run.out).as_posix():sha(p) for p in sorted(run.out.rglob('*')) if p.is_file()}
        dump(run.out/'manifest.json',dict(status='complete',files=files))
    except BaseException as exc:
        if run.active_model is not None: checkpoint(run.out/'partial-model-at-failure.npz',run.active_model)
        dump(run.out/'failure.json',dict(status='failed',type=type(exc).__name__,message=str(exc),
             calls=run.calls,examples=run.examples,updates=run.updates,
             elapsed_seconds=time.monotonic()-run.started,context=run.tags,control_maxima=run.maxima))
        run.close()
        files={p.relative_to(run.out).as_posix():sha(p) for p in sorted(run.out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
        dump(run.out/'manifest.json',dict(status='failed',files=files))
        raise

if __name__=='__main__': main()
