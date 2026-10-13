#!/usr/bin/env python3
"""Independent saved-state audit. Does not import or train the study runner."""
import argparse, csv, gzip, hashlib, itertools, json, signal, time, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

SEEDS=[1103,2207,3301,4409,5519,6607,7717,8803]
MASKS=[[0,6,14,15,17,20,23,25],[10,13,16,20,21,23,25,28],[3,6,8,11,17,19,20,26],[2,7,10,11,14,22,30,31],[3,8,14,20,22,25,27,30],[0,2,4,11,15,16,20,31],[1,2,3,4,5,19,27,28],[0,6,7,11,14,17,23,30]]
TOL=1e-10

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',newline='') as f: return list(csv.DictReader(f))
def key(r): return (int(r['seed']),r['schedule'],r['condition'],int(r['tail']))
def bykey(rs):
    d=defaultdict(list)
    for r in rs:d[key(r)].append(r)
    return d

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists(): raise FileExistsError('Use a fresh audit receipt path.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    def timeout(*_):raise TimeoutError('Audit reached its 20-minute cap.')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1200);start=time.monotonic()
    receipt=dict(status='running',runner_imported=False,training_replayed=False,scope='All saved evaluated checkpoints and path outputs; selection, summary, and record arithmetic. Training trajectories are not replayed.',tolerance=TOL,max_error=0.,checks=0,checkpoint_replays=0,intervention_rows_checked=0,audit_sha256=sha(__file__))
    def check(name,actual,expected,tol=TOL):
        x=np.asarray(actual,dtype=np.float64);y=np.asarray(expected,dtype=np.float64)
        if x.shape!=y.shape:raise AssertionError((name,'shape',x.shape,y.shape))
        err=float(np.max(np.abs(x-y))) if x.size else 0.
        if not np.isfinite(err) or err>tol:raise AssertionError((name,err,tol))
        receipt['max_error']=max(receipt['max_error'],err);receipt['checks']+=1
    def exact(name,actual,expected):
        if actual!=expected:raise AssertionError((name,actual,expected))
        receipt['checks']+=1
    try:
        root=a.results;run=read(root/'run-receipt.json');exact('run complete',run['status'],'complete');exact('updates',run['completed_updates'],120000)
        for name,item in read(root/'manifest.json').items():
            exact('hash '+name,sha(root/name),item['sha256']);exact('bytes '+name,(root/name).stat().st_size,item['bytes'])
        receipt['lock_commit']=run['lock_commit'];receipt['source_receipt_sha256']=sha(root/'run-receipt.json')
        sequence=read(root/'batch-sequences.json');x=torch.tensor(list(itertools.product((-1.,1.),repeat=5)),dtype=torch.float64);y=x[:,0]*x[:,1];donor=x.clone();donor[:,2]*=-1
        cal=(x[:,3]==x[:,4]);test=~cal;orient=-x[:,2]*y
        expected_batches={'A':torch.where(x[:,2]==y)[0].tolist(),'B':torch.where(x[:,2]!=y)[0].tolist(),'T':list(range(32))}
        expected_seq={'AB':['A']*500+['B']*500,'BA':['B']*500+['A']*500,'ALT':['A','B']*500}
        exact('batch arrays',sequence['batches'],expected_batches);exact('prefix sequences',sequence['prefix'],expected_seq);exact('tail sequence',sequence['tail'],['T']*2000);exact('random masks',read(root/'fixed-random-masks.json'),MASKS)
        for name,s in expected_seq.items():
            counts=np.zeros(32,dtype=int)
            for b in s:counts[expected_batches[b]]+=1
            check('exposure '+name,counts,np.full(32,500),0.);exact('stored exposures '+name,sequence['prefix_exposures'][name],counts.tolist())
        inputs=read(root/'inputs.json')
        for i,r in enumerate(inputs):
            check('input',r['input'],x[i].numpy(),0.);exact('label',r['y'],int(y[i]));exact('split',r['split'],'calibration' if cal[i] else 'test');check('donor',inputs[r['donor']]['input'],donor[i].numpy(),0.)
        intact=bykey(rows(root/'intact.csv'));interventions=bykey(rows(root/'interventions.csv.gz'));selection=bykey(rows(root/'selection.csv'));means=bykey(rows(root/'set-means.csv'));metrics=bykey(rows(root/'model-metrics.csv'));selected_means={}
        checkpoints=read(root/'checkpoints.json');exact('checkpoint count',len(checkpoints),176)
        for cp in checkpoints:exact('checkpoint hash',sha(root/cp['path']),cp['sha256'])
        for seed in SEEDS:
            torch.manual_seed(seed)
            layers=[torch.nn.Linear(5,32,dtype=torch.float64),torch.nn.Linear(32,32,dtype=torch.float64),torch.nn.Linear(32,1,dtype=torch.float64)]
            saved=np.load(root/'checkpoints'/f'{seed}-initial.npz')
            for name,layer in zip(('first','second','last'),layers):
                for p in ('weight','bias'):check('initialization',saved['weight.'+name+'.'+p],getattr(layer,p).detach().numpy(),0.)
            for schedule in ('AB','BA','ALT'):
                kept=np.load(root/'checkpoints'/f'{seed}-{schedule}-keep-tail0.npz');reset=np.load(root/'checkpoints'/f'{seed}-{schedule}-reset-tail0.npz')
                for n in kept.files:
                    if n.startswith('weight.'):check('fork weight',kept[n],reset[n],0.)
                    else:check('reset buffer',reset[n],np.zeros_like(reset[n]),0.)
        for cp in checkpoints:
            if 'tail' not in cp:continue
            k=key(cp);state={n:torch.from_numpy(z.copy()) for n,z in np.load(root/cp['path']).items()}
            def linear(z,name):return F.linear(z,state['weight.'+name+'.weight'],state['weight.'+name+'.bias'])
            def after(z):return linear(F.gelu(linear(F.gelu(z,approximate='none'),'second'),approximate='none'),'last').squeeze(-1)
            pre=linear(x,'first');dpre=linear(donor,'first');l=after(pre);dl=after(dpre);q=torch.sigmoid(y*l)
            check('analytic z edge',(dpre-pre).numpy(),(-2*x[:,2,None]*state['weight.first.weight'][:,2]).numpy())
            single=[];scores=[]
            for unit in range(32):
                z=pre.clone();z[:,unit]=dpre[:,unit];out=after(z);single.append(out);scores.append(float((torch.sigmoid(y*out)-q)[cal].abs().mean()))
            rank=sorted(range(32),key=lambda i:(-scores[i],i));chosen=sorted(rank[:8])
            exact('selection count',len(selection[k]),32)
            for r in selection[k]:
                u=int(r['unit']);check('calibration score',float(r['calibration_score']),scores[u]);exact('rank',int(r['rank']),rank.index(u)+1);exact('selected flag',r['selected']=='True',u in chosen)
            set_masks={('unit',str(i)):[i] for i in range(32)};set_masks.update({('selected','selected'):chosen,('complement','complement'):[i for i in range(32) if i not in chosen],('full','full'):list(range(32))});set_masks.update({('random',str(i)):m for i,m in enumerate(MASKS)})
            effects={}
            for (kind,sid),mask in set_masks.items():
                z=pre.clone();z[:,mask]=dpre[:,mask];out=single[int(sid)] if kind=='unit' else after(z);pc=torch.sigmoid(y*out)-q;mc=y*(out-l)
                effects[(kind,sid)]=(out,pc,orient*pc,mc,orient*mc)
            for r in interventions[k]:
                i=int(r['receiver']);effect=effects[(r['kind'],r['set_id'])]
                for field,v in zip(('patched_logit','probability_change','oriented_probability','margin_change','oriented_margin'),effect):check(field,float(r[field]),float(v[i]))
                receipt['intervention_rows_checked']+=1
            exact('intervention count per model',len(interventions[k]),43*32)
            for r in means[k]:
                which=cal if r['split']=='calibration' else test;out,pc,op,mc,om=effects[(r['kind'],r['set_id'])]
                for field,value in [('mean_oriented_probability',float(op[which].mean())),('mean_absolute_probability_change',float(pc[which].abs().mean())),('mean_oriented_margin',float(om[which].mean()))]:check(field,float(r[field]),value)
                if r['kind']=='selected' and r['split']=='test':selected_means[k]=float(op[test].mean())
            exact('set mean count',len(means[k]),86)
            flipped=x.clone();flipped[:,0]*=-1;fl=after(linear(flipped,'first'))
            for r in intact[k]:
                i=int(r['receiver']);expected={'logit':l[i],'correct_probability':q[i],'margin':y[i]*l[i],'donor_logit':dl[i],'x1_flip_logit':fl[i]}
                for field,v in expected.items():check(field,float(r[field]),float(v))
                exact('intact correctness',r['correct']=='True',bool((l[i]>0)==(y[i]>0)));exact('flipped correctness',r['x1_flip_correct']=='True',bool((fl[i]>0)==(-y[i]>0)))
            exact('intact count',len(intact[k]),32);exact('metric count',len(metrics[k]),1);r=metrics[k][0]
            values=dict(accuracy=float(((l>0)==(y>0)).double().mean()),cross_entropy=float(F.binary_cross_entropy_with_logits(l,(y+1)/2)),mean_correct_margin=float((y*l).mean()),x1_flip_accuracy=float(((fl>0)==(-y>0)).double().mean()))
            for field,value in values.items():check(field,float(r[field]),value)
            for prefix,field in [('weight.','weight_norm'),('momentum.','momentum_norm')]:
                values=[v for n,v in state.items() if n.startswith(prefix)];value=float(torch.sqrt(sum((v*v).sum() for v in values)));check(field,float(r[field]),value)
            receipt['checkpoint_replays']+=1
        exact('replayed checkpoints',receipt['checkpoint_replays'],144);exact('intervention rows',receipt['intervention_rows_checked'],198144)
        summary=read(root/'summary.json');differences=[]
        for r in summary['per_seed']:
            seed=r['seed'];get=lambda schedule,condition:selected_means[(seed,schedule,condition,2000)]
            reset=get('BA','reset')-get('AB','reset');kept=get('BA','keep')-get('AB','keep');differences.append(reset)
            for field,v in [('reset_BA_minus_AB',reset),('keep_BA_minus_AB',kept),('reset_minus_keep_contrast',reset-kept),('reset_ALT',get('ALT','reset')),('keep_ALT',get('ALT','keep'))]:check(field,r[field],v)
        d=np.array(differences);draws=np.random.default_rng(161803).integers(0,8,(10000,8));ci=np.quantile(d[draws].mean(1),[.025,.975],method='linear');p=summary['primary']
        for field,v in [('mean',d.mean()),('median',np.median(d)),('minimum',d.min()),('maximum',d.max())]:check('primary '+field,p[field],v)
        check('primary interval',p['descriptive_paired_seed_bootstrap_95'],ci);exact('negligibility',p['numerically_negligible'],bool(abs(d.mean())<1e-8))
        training=rows(root/'training.csv.gz');exact('training row count',len(training),120000);seen=set()
        for r in training:
            marker=(int(r['seed']),r['schedule'],r['condition'],r['stage'],int(r['step']))
            if marker in seen:raise AssertionError('Duplicate training row')
            seen.add(marker);expected=expected_seq[r['schedule']][int(r['step'])-1] if r['stage']=='prefix' else 'T';exact('training batch',r['batch'],expected)
            if not np.isfinite(float(r['pre_update_loss'])):raise AssertionError('Non-finite training loss')
        for r in rows(root/'controls.csv'):
            exact('reported control pass',r['passed'],'True')
            if float(r['error'])>float(r['tolerance']):raise AssertionError('Invalid reported control')
        receipt.update(status='passed',primary_mean=float(d.mean()),primary_interval=ci.tolist(),checkpoint_hashes_checked=len(checkpoints),training_rows_checked=len(training),seconds=time.monotonic()-start)
    except BaseException:
        receipt.update(status='failed',traceback=traceback.format_exc(),seconds=time.monotonic()-start);raise
    finally:
        signal.alarm(0);a.output.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(receipt,sort_keys=True))
if __name__=='__main__':main()
