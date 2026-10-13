#!/usr/bin/env python3
"""Fixed token-segmentation transfer study; remote lock required for outcomes."""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, re, signal, subprocess, time, traceback
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers.models.gpt_neox import modeling_gpt_neox as neox
ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'EleutherAI/pythia-14m'
REVISION = 'cf967c0a9a04383db6f7b1108d86b2962634b4ac'
MODEL_HASHES = {'config.json': 'f97f966a66c444890ed461fff2a51eefb15d74303df05b948124719f199b0b17', 'model.safetensors': '116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1', 'tokenizer.json': '870f4e2baa6b683221fa52004d5d6f40ab8c9d31961617304b78c910c2c3caf2', 'tokenizer_config.json': 'eee017c5bd133137f45907bd0a6e781e2ccd1a533734b7ed2a2f2f4446659809', 'special_tokens_map.json': '10b8c8852c1e1f70b54d9aff61728408c28971c0e97a6c5a7b2debbd1d3e9c0c', 'README.md': 'd1f2cf1d5181daedeaa70208ddd5cc5251867bde9acf6db7bb45a2265e25e163'}
STOCK = neox.eager_attention_forward

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def table(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def lock_check(commit):
    repo=ROOT.parents[1]
    actual=subprocess.run(['git','rev-parse',commit+'^{commit}'],cwd=repo,check=True,capture_output=True).stdout.decode().strip()
    if actual!=commit: raise RuntimeError('Use a full commit SHA.')
    for p in (ROOT/'analysis-plan.md',Path(__file__).resolve()):
        raw=subprocess.run(['git','show',commit+':'+p.relative_to(repo).as_posix()],cwd=repo,check=True,capture_output=True).stdout
        if raw!=p.read_bytes(): raise RuntimeError('File differs from lock: '+str(p))
def decode(t,ids): return t.decode(ids,clean_up_tokenization_spaces=False)

def make_inputs(t):
    text={i:decode(t,[i]) for i in range(len(t))}
    lookup={}
    for i,s in text.items(): lookup.setdefault(s,i)
    one=[i for i in range(1000,len(t)) if re.fullmatch(r' [a-z]{3,8}',text[i]) and t.encode(text[i],add_special_tokens=False)==[i]]
    old=set(one[:192]); candidates=[]
    for i in range(1000,len(t)):
        s=text[i]
        if i in old or not re.fullmatch(r' [a-z]{5,12}',s) or t.encode(s,add_special_tokens=False)!=[i]: continue
        cuts=[]
        for k in range(2,len(s)):
            if s[:k] in lookup and s[k:] in lookup:
                ids=[lookup[s[:k]],lookup[s[k:]]]
                if decode(t,ids)==s: cuts.append((k,ids))
        if len(cuts)<2: continue
        a,b=cuts[0],cuts[-1]
        if len(set(a[1]+b[1]))==4:
            candidates.append(dict(id=i,text=s,A=a[1],B=b[1],cuts=[a[0],b[0]],pieces_A=[text[x] for x in a[1]],pieces_B=[text[x] for x in b[1]]))
    count=len(candidates); remaining=candidates.copy(); groups=[]; rejected=[]
    for g in range(32):
        if not remaining: raise RuntimeError('Key pool too short.')
        k=remaining.pop(0); used=set(k['A']+k['B']); pick=None
        for j,l in enumerate(remaining):
            overlap=sorted(used.intersection(l['A']+l['B']))
            if not overlap: pick=j; break
            rejected.append(dict(group=g,first=k['id'],candidate=l['id'],overlap=overlap))
        if pick is None: raise RuntimeError('No disjoint second key.')
        groups.append(dict(group=g,K=k,L=remaining.pop(pick)))
    excluded=set(old)
    for g in groups:
        for k in ('K','L'): excluded.update([g[k]['id'],*g[k]['A'],*g[k]['B']])
    allowed=[i for i in one if i not in excluded]
    if len(allowed)<128: raise RuntimeError('Value/filler pool too short.')
    values,fillers=allowed[:64],allowed[64:128]
    prompts=[]; references=[]
    for g in groups:
        gi=g['group']; b,d=values[2*gi:2*gi+2]; g.update(B=b,D=d)
        fill=np.random.default_rng(2026101300+gi).permutation(fillers).tolist(); g['fillers']=fill[:16]
        for binding in range(2):
            kv,lv=(b,d) if binding==0 else (d,b)
            for order in range(2):
                pairs=[('K',kv),('L',lv)]
                if order: pairs.reverse()
                for gap in (0,8):
                    for query in ('K','L'):
                        target=kv if query=='K' else lv
                        family=dict(group=gi,binding=binding,order=order,gap=gap,query=query,B=b,D=d,target=target,sign=1 if target==b else -1,recent_sign=1 if pairs[1][1]==b else -1)
                        family_prompts=[]
                        for source in ('A','B'):
                            for qsplit in ('A','B'):
                                ids=[t.eos_token_id,*g[pairs[0][0]][source],pairs[0][1],*fill[:gap],*g[pairs[1][0]][source],pairs[1][1],*fill[8:8+gap],*g[query][qsplit]]
                                vp=[3,6+gap]; bp=vp[0] if pairs[0][1]==b else vp[1]; dp=vp[0] if pairs[0][1]==d else vp[1]
                                p=dict(family,prompt_id=len(prompts),cell=source+qsplit,matched=int(source==qsplit),input_ids=ids,text=decode(t,ids),utf8_hex=decode(t,ids).encode().hex(),readout=len(ids)-1,bpos=bp,dpos=dp)
                                assert len(ids)==9+2*gap and ids[bp]==b and ids[dp]==d
                                if source!=qsplit:
                                    qs=set(g[query][qsplit]); sources=set(g['K'][source]+g['L'][source])
                                    assert not qs.intersection(sources)
                                    assert not qs.intersection([b,d,*fill[:gap],*fill[8:8+gap]])
                                family_prompts.append(p); prompts.append(p)
                        assert len({p['utf8_hex'] for p in family_prompts})==1
                        assert len({(len(p['input_ids']),p['readout']) for p in family_prompts})==1
                        ids=[t.eos_token_id,g[pairs[0][0]]['id'],pairs[0][1],*fill[:gap],g[pairs[1][0]]['id'],pairs[1][1],*fill[8:8+gap],g[query]['id']]
                        assert decode(t,ids)==family_prompts[0]['text']
                        references.append(dict(family,prompt_id=len(references),cell='canonical',input_ids=ids,text=decode(t,ids),readout=len(ids)-1))
    assert len(prompts)==2048 and len(references)==512
    manifest=dict(candidate_count=count,old_excluded_ids=sorted(old),groups=groups,value_ids=values,filler_ids=fillers,rejected_pair_conflicts=rejected,token_text={str(i):text[i] for i in sorted(set(values+fillers))},split_prompt_count=len(prompts),canonical_count=len(references),all_byte_checks_passed=True)
    return prompts,references,manifest

def batches(records,size=32):
    for length in sorted({len(p['input_ids']) for p in records}):
        sub=[p for p in records if len(p['input_ids'])==length]
        for j in range(0,len(sub),size): yield sub[j:j+size]

class Edit:
    def __init__(self):
        self.mode=None; self.attention=None; self.norm=None
        self.errors=dict(untouched_output=0.,unchanged_attention=0.,ablated_vector=0.,future_attention=0.)
    def forward(self,module,query,key,value,attention_mask,scaling,dropout=0.,head_mask=None,**kwargs):
        out,w=STOCK(module,query,key,value,attention_mask,scaling,dropout=dropout,head_mask=head_mask,**kwargs)
        future=float(torch.triu(w,diagonal=1).abs().max()); self.errors['future_attention']=max(self.errors['future_attention'],future)
        assert future==0.
        if module._seg_layer!=3: return out,w
        self.attention=w[:,1,-1].detach().clone(); self.norm=out[:,-1,1].norm(dim=-1).detach().clone()
        if self.mode is None: return out,w
        original_weights=w.clone()
        edited=out.clone()
        if self.mode=='zero': edited[:,-1,1]=0.
        elif self.mode=='self': edited[:,-1,1]=out[:,-1,1]
        else: raise RuntimeError(self.mode)
        self.errors['unchanged_attention']=max(self.errors['unchanged_attention'],float((w-original_weights).abs().max()))
        diff=edited-out; diff[:,-1,1]=0.
        self.errors['untouched_output']=max(self.errors['untouched_output'],float(diff.abs().max()))
        if self.mode=='zero': self.errors['ablated_vector']=max(self.errors['ablated_vector'],float(edited[:,-1,1].abs().max()))
        assert all(v==0. for v in self.errors.values())
        return edited,w

def infer(model,records):
    ids=torch.tensor([p['input_ids'] for p in records],dtype=torch.long)
    with torch.inference_mode(): return model.embed_out(model.gpt_neox(ids,use_cache=False).last_hidden_state[:,-1])
def stats(logits,p):
    probs=logits.softmax(-1); target=p['target']; top=int(logits.argmax()); margin=float(logits[p['B']]-logits[p['D']]); score=p['sign']*margin
    return dict(logit_B=float(logits[p['B']]),logit_D=float(logits[p['D']]),margin=margin,score=score,recency=p['recent_sign']*margin,probability_B=float(probs[p['B']]),probability_D=float(probs[p['D']]),target_probability=float(probs[target]),target_rank=int((logits>logits[target]).sum())+1,top_token=top,top_probability=float(probs[top]),pair_correct=int(score>0),full_correct=int(top==target))
def load(path):
    for name,expected in MODEL_HASHES.items():
        if sha(path/name)!=expected: raise RuntimeError('Wrong model file: '+name)
    model=GPTNeoXForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
    model.requires_grad_(False)
    assert (model.config.num_hidden_layers,model.config.num_attention_heads,model.config.hidden_size)==(6,4,128)
    for i,l in enumerate(model.gpt_neox.layers): l.attention._seg_layer=i
    return model,AutoTokenizer.from_pretrained(path,local_files_only=True)

def controls(model,edit,records):
    neox.eager_attention_forward=STOCK; stock=infer(model,records)
    neox.eager_attention_forward=edit.forward; edit.mode=None; plain=infer(model,records)
    edit.mode='self'; noop=infer(model,records); edit.mode=None
    ids=torch.tensor([p['input_ids'] for p in records])
    with torch.inference_mode(): full=model(ids,use_cache=False).logits[:,-1]
    result=dict(stock_error=float((stock-plain).abs().max()),self_error=float((noop-plain).abs().max()),final_projection_error=float((full-plain).abs().max()))
    assert max(result.values())<=1e-5,result
    return result

def run_grid(model,edit,prompts,refs,out):
    rows=[]; refrows=[]; control_rows=[]; attn=np.zeros((len(prompts),25),dtype=np.float32); errors=dict(stock_error=0.,self_error=0.)
    for batch in batches(prompts):
        neox.eager_attention_forward=STOCK; stock=infer(model,batch)
        neox.eager_attention_forward=edit.forward; edit.mode=None; intact=infer(model,batch); attention=edit.attention.clone(); norms=edit.norm.clone()
        edit.mode='self'; noop=infer(model,batch)
        batch_errors=dict(stock_error=float((stock-intact).abs().max()),self_error=float((noop-intact).abs().max()))
        control_rows.append(dict(batch=len(control_rows),first_prompt=batch[0]['prompt_id'],last_prompt=batch[-1]['prompt_id'],length=len(batch[0]['input_ids']),count=len(batch),**batch_errors,**edit.errors))
        table(out/'batch-controls.csv',control_rows)
        for k,v in batch_errors.items(): errors[k]=max(errors[k],v)
        assert max(batch_errors.values())<=1e-5,batch_errors
        edit.mode='zero'; zero=infer(model,batch); edit.mode=None
        control_rows[-1].update(edit.errors); table(out/'batch-controls.csv',control_rows)
        for j,p in enumerate(batch):
            row={k:p[k] for k in ('prompt_id','group','binding','order','gap','query','cell','matched','B','D','target','sign','recent_sign')}
            a,z=stats(intact[j],p),stats(zero[j],p)
            row.update({'intact_'+k:v for k,v in a.items()}); row.update({'ablated_'+k:v for k,v in z.items()})
            row.update(effect=a['score']-z['score'],attention_B=float(attention[j,p['bpos']]),attention_D=float(attention[j,p['dpos']]),head_norm=float(norms[j]))
            rows.append(row); attn[p['prompt_id'],:len(p['input_ids'])]=attention[j].numpy()
        table(out/'scores.csv',sorted(rows,key=lambda r:r['prompt_id']))
    for batch in batches(refs):
        logits=infer(model,batch)
        for j,p in enumerate(batch):
            r={k:p[k] for k in ('prompt_id','group','binding','order','gap','query','B','D','target','sign','recent_sign')}; r.update(stats(logits[j],p)); refrows.append(r)
        table(out/'canonical-scores.csv',sorted(refrows,key=lambda r:r['prompt_id']))
    np.savez_compressed(out/'attention-rows.npz',attention=attn)
    return rows,refrows,errors

METRICS=('intact_score','effect','intact_pair_correct','intact_full_correct','intact_target_probability','intact_recency','ablated_score','ablated_pair_correct','ablated_full_correct')
def summary(rows,refs,out):
    grouped=[]; estimates={}; indices=np.random.default_rng(2026101320).integers(0,32,(10000,32))
    subsets=['all','matched','mismatched','AA','AB','BA','BB','gap0','gap8']
    def choose(r,name):
        if name=='all': return True
        if name=='matched': return r['matched']==1
        if name=='mismatched': return r['matched']==0
        if name.startswith('gap'): return r['gap']==int(name[3:])
        return r['cell']==name
    for subset in subsets:
        gr=[]
        for g in range(32):
            part=[r for r in rows if r['group']==g and choose(r,subset)]
            x=dict(group=g,subset=subset,count=len(part)); x.update({m:float(np.mean([r[m] for r in part])) for m in METRICS}); gr.append(x); grouped.append(x)
        estimates[subset]={}
        for m in METRICS:
            x=np.array([r[m] for r in gr]); low,high=np.quantile(x[indices].mean(1),[.025,.975]); estimates[subset][m]=dict(mean=float(x.mean()),lower=float(low),upper=float(high))
    for m in ('intact_score','effect'):
        matched=np.array([r[m] for r in grouped if r['subset']=='matched']); mismatch=np.array([r[m] for r in grouped if r['subset']=='mismatched']); x=matched-mismatch
        lo,hi=np.quantile(x[indices].mean(1),[.025,.975]); estimates.setdefault('matched_minus_mismatched',{})[m]=dict(mean=float(x.mean()),lower=float(lo),upper=float(hi))
    table(out/'group-scores.csv',grouped)
    canonical=[]
    for g in range(32):
        part=[r for r in refs if r['group']==g]; x=dict(group=g,count=len(part)); x.update({m:float(np.mean([r[m] for r in part])) for m in ('score','pair_correct','full_correct','target_probability','recency')}); canonical.append(x)
    table(out/'canonical-groups.csv',canonical)
    mm=estimates['mismatched']; pattern=mm['intact_score']['lower']>0 and mm['effect']['lower']>0
    gate=mm['intact_pair_correct']['mean']>=.75 and mm['intact_full_correct']['mean']>=.10 and mm['intact_score']['lower']>0
    dump(out/'summary.json',dict(seed=2026101320,bootstrap_draws=10000,clusters=32,estimates=estimates,primary_pattern_supported=pattern,capability_gate_passed=gate,causal_binding_interpretation_supported=pattern and gate,canonical_means={m:float(np.mean([r[m] for r in canonical])) for m in ('score','pair_correct','full_correct','target_probability','recency')}))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model-dir',type=Path,required=True); ap.add_argument('--lock-commit'); ap.add_argument('--dummy-preflight',action='store_true'); ap.add_argument('--token-only',action='store_true'); ap.add_argument('--output',type=Path,default=ROOT/'results'); args=ap.parse_args()
    if args.dummy_preflight and args.token_only: ap.error('Choose one preflight mode.')
    scientific=not (args.dummy_preflight or args.token_only)
    if scientific:
        if not args.lock_commit: ap.error('--lock-commit is required')
        lock_check(args.lock_commit)
    if args.output.exists() and any(args.output.iterdir()): raise RuntimeError('Output must be empty.')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.token_only:
        t=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True); prompts,refs,manifest=make_inputs(t); dump(args.output/'token-manifest.json',manifest); dump(args.output/'prompts.json',prompts); dump(args.output/'canonical-prompts.json',refs); print(json.dumps(dict(status='token checks passed',candidates=manifest['candidate_count'],prompts=len(prompts)))); return
    start=time.time(); torch.set_num_threads(2); torch.manual_seed(20261013); torch.use_deterministic_algorithms(True)
    def timeout(*unused): raise TimeoutError('Fixed 20-minute runtime limit reached.')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(1200)
    metadata=dict(model=MODEL_ID,revision=REVISION,model_hashes=MODEL_HASHES,lock_commit=args.lock_commit,plan_sha256=sha(ROOT/'analysis-plan.md'),runner_sha256=sha(Path(__file__)),platform=platform.platform(),python=platform.python_version(),packages={x:version(x) for x in ('torch','numpy','transformers','safetensors','tokenizers','huggingface-hub')},dtype='float32',cpu_threads=2,started_unix=start,status='running')
    edit=Edit()
    try:
        model,t=load(args.model_dir)
        if args.dummy_preflight:
            checks=[]
            for length in (9,25):
                records=[dict(input_ids=[t.eos_token_id,*range(101+25*j,100+length+25*j)]) for j in range(32)]
                checks.append(controls(model,edit,records)); edit.mode='zero'; infer(model,records); edit.mode=None
            dump(args.output/'dummy-preflight.json',dict(status='passed',scientific_model_outputs_collected=False,dummy_batch_size=32,dummy_lengths=[9,25],checks=checks,errors=edit.errors,plan_sha256=metadata['plan_sha256'],runner_sha256=metadata['runner_sha256'],packages=metadata['packages'],model_hashes=MODEL_HASHES,dtype='float32',cpu_threads=2)); print(json.dumps(dict(status='dummy passed',checks=checks,errors=edit.errors))); return
        dump(args.output/'run-metadata.json',metadata)
        prompts,refs,manifest=make_inputs(t); dump(args.output/'token-manifest.json',manifest); dump(args.output/'prompts.json',prompts); dump(args.output/'canonical-prompts.json',refs)
        neox.eager_attention_forward=edit.forward
        rows,refrows,errors=run_grid(model,edit,prompts,refs,args.output); summary(rows,refrows,args.output)
        lock_check(args.lock_commit); dump(args.output/'controls.json',dict(status='passed',errors=errors,local_errors=edit.errors,byte_checks_passed=True,no_op_prompt_count=len(prompts)))
        metadata.update(status='completed',elapsed_seconds=time.time()-start,split_prompts=len(rows),canonical_prompts=len(refrows)); dump(args.output/'run-metadata.json',metadata)
        dump(args.output/'manifest.json',dict(files={p.relative_to(args.output).as_posix():sha(p) for p in sorted(args.output.rglob('*')) if p.is_file()})); print(json.dumps(dict(status='completed',elapsed_seconds=metadata['elapsed_seconds'])))
    except Exception:
        metadata.update(status='failed',elapsed_seconds=time.time()-start); dump(args.output/'run-metadata.json',metadata); dump(args.output/'failure.json',dict(traceback=traceback.format_exc(),errors=edit.errors,per_batch_control_path='batch-controls.csv' if (args.output/'batch-controls.csv').exists() else None)); raise
    finally:
        neox.eager_attention_forward=STOCK; signal.alarm(0)
if __name__=='__main__': main()
