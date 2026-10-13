#!/usr/bin/env python3
"""Separate saved-weight replay and arithmetic audit; never imports the runner."""
import argparse,csv,gzip,hashlib,itertools,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
SEEDS=[19121,19139,19157,19181]
TOL=1e-10

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def probability(z):return np.exp(-np.logaddexp(0.,-z))

class AuditNet(nn.Module):
 def __init__(self,w):
  super().__init__()
  self.net=nn.Sequential(nn.Linear(16,64,dtype=torch.float64),nn.ReLU(),nn.Linear(64,64,dtype=torch.float64),nn.ReLU(),nn.Linear(64,1,dtype=torch.float64))
  for index,name in [(0,'first'),(2,'second'),(4,'last')]:
   self.net[index].load_state_dict({k:torch.from_numpy(w[name+'.'+k]) for k in ('weight','bias')})
 def forward(self,x):return self.net(x).squeeze(-1)


def independent_inputs(ledger):
 ids=np.arange(65536,dtype=np.uint32)
 bits=((ids[:,None]>>np.arange(16))&1).astype(np.uint8)
 graphs=bits.reshape(-1,4,4); labels=np.zeros(len(ids),bool)
 for p in itertools.permutations(range(4)):labels|=graphs[:,range(4),p].all(1)
 hall=np.ones(len(ids),bool);pair_fail=np.zeros(len(ids),bool)
 for subset in range(1,16):
  r=[i for i in range(4) if subset&(1<<i)]
  fail=graphs[:,r,:].any(1).sum(1)<len(r);hall&=~fail
  if len(r)==2:pair_fail|=fail
 assert np.array_equal(hall,labels)
 degrees=np.concatenate([graphs.sum(2),graphs.sum(1)],1);noiso=(degrees>0).all(1)
 other=noiso&~labels&~pair_fail
 unions={phase:{cat:set() for cat in ('target','positive','hall3')} for phase in ('selection','test')}
 for entry in ledger['names']:
  a,b=entry['left_pair'];q=entry['right']
  target=noiso&(degrees[:,a]==1)&(degrees[:,b]==1)&graphs[:,a,q].astype(bool)&graphs[:,b,q].astype(bool)
  groups=[]
  for key in sorted(set(tuple(degrees[i]) for i in np.flatnonzero(target))):
   n=sum(key[:4])
   if n not in (6,7):continue
   same=(degrees==np.array(key)).all(1)
   cats={'target':np.flatnonzero(same&target).tolist(),'positive':np.flatnonzero(same&noiso&labels).tolist(),'hall3':np.flatnonzero(same&other).tolist()}
   if all(cats.values()):
    phase='selection' if n==6 else 'test';groups.append(dict(degree=list(map(int,key)),phase=phase,**cats))
    for cat,values in cats.items():unions[phase][cat].update(values)
  assert groups==entry['groups']
 assert {p:{c:sorted(v) for c,v in cats.items()} for p,cats in unions.items()}==ledger['category_unions']
 rng=np.random.default_rng(94301)
 random_masks=[sorted(rng.choice(64,4,replace=False).tolist()) for _ in range(8)]
 assert random_masks==ledger['random_masks']
 rng=np.random.default_rng(94210);epochs=np.array([rng.permutation(65536) for _ in range(64)],dtype=np.int32)
 for key,arr in [('graph_ids',ids),('labels',labels),('epoch_orders',epochs)]:
  expected={'dtype':arr.dtype.str,'shape':list(arr.shape),'sha256':hashlib.sha256(arr.tobytes()).hexdigest()}
  assert expected==ledger['arrays'][key]
 return bits.astype(np.float64),labels,noiso,epochs


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--results',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
 args=ap.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 started=time.monotonic();maxima={};checks=0;calls=0;examples=0;context={}
 receipt={'status':'running','audit_code_sha256':sha(__file__),'repeated_training':False,'tolerance':TOL,'dtype':'float64','threads':2}
 def close(a,b,label):
  nonlocal checks
  a=np.asarray(a);b=np.asarray(b)
  if a.shape!=b.shape:raise AssertionError(f'{label}: shape {a.shape} != {b.shape}')
  if not np.isfinite(a).all() or not np.isfinite(b).all():raise AssertionError(label+' nonfinite')
  err=float(np.max(np.abs(a-b))) if a.size else 0.
  maxima[label]=max(err,maxima.get(label,0.));checks+=1
  if err>TOL:raise AssertionError(f'{label}: {err} exceeds {TOL}')
 def forward(model,x):
  nonlocal calls,examples
  if time.monotonic()-started>1200:raise TimeoutError('Independent audit 20-minute cap')
  outputs=[]
  with torch.inference_mode():
   for i in range(0,len(x),512):
    z=model(x[i:i+512]);calls+=1;examples+=len(z);outputs.append(z.numpy())
  return np.concatenate(outputs)
 try:
  meta=json.loads((args.results/'run-metadata.json').read_text());assert meta['status']=='complete'
  manifest=json.loads((args.results/'manifest.json').read_text())
  for name,digest in manifest.items():assert sha(args.results/name)==digest,name
  repo=ROOT.parents[1]
  for name,digest in meta['locked_sha256'].items():
   path=ROOT/name;assert sha(path)==digest
   blob=subprocess.check_output(['git','show',meta['lock_commit']+':'+path.relative_to(repo).as_posix()],cwd=repo)
   assert blob==path.read_bytes()
  ledger=json.loads((ROOT/'inputs.json').read_text())
  raw,labels,noiso,epochs=independent_inputs(ledger)
  saved_data=dict(np.load(args.results/'data.npz'))
  assert np.array_equal(saved_data['graph_ids'],np.arange(65536,dtype=np.uint32))
  assert np.array_equal(saved_data['labels'],labels) and np.array_equal(saved_data['epoch_orders'],epochs)
  train=list(csv.DictReader(gzip.open(args.results/'training.csv.gz','rt')))
  assert len(train)==32768
  for j,seed in enumerate(SEEDS):
   sub=train[j*8192:(j+1)*8192]
   for k,row in enumerate(sub):
    assert int(row['seed'])==seed and int(row['update'])==k+1 and int(row['epoch'])==k//128 and int(row['first_in_epoch'])==(k%128)*512
    assert np.isfinite(float(row['loss']))
  controls=list(csv.DictReader((args.results/'controls.csv').open()))
  assert len(controls)==1188
  for row in controls:assert np.isfinite(float(row['max_error'])) and float(row['max_error'])<=TOL
  score_rows=list(csv.DictReader(gzip.open(args.results/'group-scores.csv.gz','rt')))
  assert len(score_rows)==64800
  score_lookup={(int(r['seed']),r['name'],r['phase'],int(r['mask_id']),int(r['degree_group'])):r for r in score_rows}
  assert len(score_lookup)==64800
  model_results=[];x=torch.from_numpy(raw)
  for seed in SEEDS:
   context={'seed':seed,'stage':'replay'}
   weights=dict(np.load(args.results/f'model-{seed}-step-8192.npz'))
   assert all(v.dtype==np.float64 and np.isfinite(v).all() for v in weights.values())
   model=AuditNet(weights).eval()
   intact=dict(np.load(args.results/f'intact-{seed}.npz'));ab=dict(np.load(args.results/f'ablations-{seed}.npz'))
   mask_file=json.loads((args.results/f'masks-{seed}.json').read_text())
   summary=json.loads((args.results/f'summary-{seed}.json').read_text())
   assert np.array_equal(intact['graph_ids'],np.arange(65536)) and np.array_equal(intact['labels'],labels)
   close(forward(model,x),intact['logits'],'full_natural_logits')
   graph_ids=np.array(sorted({g for e in ledger['names'] for group in e['groups'] for c in ('target','positive','hall3') for g in group[c]}))
   assert np.array_equal(ab['graph_ids'],graph_ids)
   lookup=np.full(65536,-1,int);lookup[graph_ids]=np.arange(len(graph_ids))
   close(ab['intact_logits'],intact['logits'][graph_ids],'saved_intact_subset')
   assert ab['logits'].shape==(98,3096) and ab['masks'].shape==(98,64) and ab['removed_norm'].shape==(98,3096)
   names=['empty','full']+[f'unit_{i}' for i in range(64)]+[f'random_{i}' for i in range(8)]
   masks=np.zeros((98,64),bool);masks[1]=True;masks[2:66]=np.eye(64,dtype=bool)
   for i,m in enumerate(ledger['random_masks']):masks[66+i,m]=True
   def statistic(output,group):
    e=[float((output-ab['intact_logits'])[lookup[group[c]]].mean()) for c in ('target','positive','hall3')]
    return e[0]-max([0.]+e[1:]),e
   for ni,entry in enumerate(ledger['names']):
    cal=[g for g in entry['groups'] if g['phase']=='selection']
    scores=np.array([np.mean([statistic(ab['logits'][2+u],g)[0] for g in cal]) for u in range(64)])
    rank=np.argsort(-scores,kind='stable').tolist();chosen=sorted(rank[:4])
    selection=mask_file['selections'][ni]
    assert selection['name']==entry['name'] and selection['ranking']==rank and selection['selected_units']==chosen
    close(scores,selection['calibration_unit_scores'],'selection_scores')
    masks[74+ni,chosen]=True;names.append('selected_'+entry['name'])
   assert np.array_equal(ab['masks'],masks) and mask_file['mask_names']==names
   subset=x[graph_ids]
   with torch.inference_mode():hidden=model.net[1](model.net[0](subset))
   for m,name in enumerate(names):
    context={'seed':seed,'stage':'replay_mask','mask':name}
    keep=torch.from_numpy((~masks[m]).astype(np.float64))
    hook=model.net[1].register_forward_hook(lambda module,inputs,output,keep=keep:output*keep)
    try:replayed=forward(model,subset)
    finally:hook.remove()
    close(replayed,ab['logits'][m],'intervention_logits')
    removed=torch.sqrt((hidden*torch.from_numpy(masks[m].astype(np.float64))).square().sum(-1)).numpy()
    close(removed,ab['removed_norm'][m],'removed_activation_norm')
   named=[]
   for ni,entry in enumerate(ledger['names']):
    phases={}
    for phase in ('selection','test'):
     groups=[g for g in entry['groups'] if g['phase']==phase];mean_scores={}
     for m in list(range(74))+[74+ni]:
      vals=[]
      for gi,g in enumerate(groups):
       value,e=statistic(ab['logits'][m],g);vals.append(value)
       row=score_lookup[(seed,entry['name'],phase,m,gi)]
       assert row['mask']==names[m] and row['degree']==','.join(map(str,g['degree']))
       want={'selectivity':value,'target_minus_positive':e[0]-e[1],'target_minus_hall3':e[0]-e[2]}
       for k,cat in enumerate(('target','positive','hall3')):
        idx=lookup[g[cat]];before=ab['intact_logits'][idx];after=ab['logits'][m,idx]
        want.update({cat+'_count':len(idx),cat+'_logit_change':e[k],cat+'_intact_logit':before.mean(),cat+'_ablated_logit':after.mean(),cat+'_probability_change':(probability(after)-probability(before)).mean(),cat+'_intact_acceptance':(before>=0).mean(),cat+'_ablated_acceptance':(after>=0).mean(),cat+'_removed_norm':ab['removed_norm'][m,idx].mean()})
       close([float(row[k]) for k in want],list(want.values()),'group_score_fields')
      mean_scores[names[m]]=float(np.mean(vals))
     selected=mean_scores['selected_'+entry['name']];random=np.mean([mean_scores[f'random_{i}'] for i in range(8)])
     got=summary['named_obstructions'][ni][phase]
     assert summary['named_obstructions'][ni]['name']==entry['name']
     close([selected,random,selected-random,mean_scores['full']],[got['selected'],got['random_mean'],got['selected_minus_random'],got['full_layer']],'name_summary')
     close([mean_scores[k] for k in mean_scores],[got['mask_scores'][k] for k in mean_scores],'all_mask_means')
     phases[phase]=(selected,float(random))
    named.append(phases['test'])
   correct=(intact['logits']>=0)==labels
   gates={'all_graphs':correct.mean(),'no_isolated_negatives':correct[noiso&~labels].mean()}
   for cat,gids in ledger['category_unions']['test'].items():gates['test_'+cat]=correct[gids].mean()
   close(list(gates.values()),[summary['ability'][k] for k in gates],'ability_gates')
   selected,random=np.mean(named,axis=0);primary=selected-random
   close([selected,random,primary],[summary['raw_selected_selectivity'],summary['random_selectivity'],summary['primary_difference']],'model_primary')
   ce=np.mean(np.logaddexp(0,intact['logits'])-labels*intact['logits'])
   close(ce,summary['all_graph_cross_entropy'],'cross_entropy')
   ability=all(v>=.99 for v in gates.values());interpretation=ability and selected>0 and primary>0
   assert summary['ability_passes']==ability and summary['per_model_interpretation_conditions_pass']==interpretation
   close([sum(v[0]>0 for v in named),min(v[0] for v in named),max(v[0] for v in named)],[summary['positive_raw_names'],*summary['raw_name_range']],'raw_name_summary')
   model_results.append(dict(seed=seed,selected=float(selected),random=float(random),primary=float(primary),ability=bool(ability),interpretation=bool(interpretation)))
  final=json.loads((args.results/'summary.json').read_text());diff=np.array([r['primary'] for r in model_results]);rng=np.random.default_rng(94302)
  boot=diff[rng.integers(0,4,(10000,4))].mean(1)
  close(np.quantile(boot,[.025,.975]),final['descriptive_seed_interval'],'seed_interval')
  close(diff.mean(),final['primary_mean_selected_minus_random'],'panel_primary')
  close([np.mean([r['selected'] for r in model_results]),np.mean([r['random'] for r in model_results])],[final['raw_selected_mean'],final['random_mean']],'panel_raw')
  assert final['all_models_pass_ability']==all(r['ability'] for r in model_results)
  assert final['panel_interpretation_conditions_pass']==all(r['interpretation'] for r in model_results)
  assert calls==3256 and examples==1475776
  receipt.update(status='passed',models=model_results,full_forward_calls=calls,forwarded_examples=examples,all_model_and_context_counts_verified=True,input_ledger_sha256=sha(ROOT/'inputs.json'),result_manifest_sha256=sha(args.results/'manifest.json'))
 except BaseException as error:
  receipt.update(status='failed',error_type=type(error).__name__,error=str(error),context=context,full_forward_calls=calls,forwarded_examples=examples)
  raise
 finally:
  receipt.update(elapsed_seconds=time.monotonic()-started,comparisons=checks,maximum_errors=maxima)
  write(args.output,receipt)
 print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
