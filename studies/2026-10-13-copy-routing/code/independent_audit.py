#!/usr/bin/env python3
"""Independent table audit and selected-head replay; no runner import."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers.models.gpt_neox import modeling_gpt_neox as neox

STUDY = Path(__file__).resolve().parents[1]
STOCK = neox.eager_attention_forward
METRICS = ('baseline_score', 'attention_score', 'route_drop', 'value_drop',
           'joint_rescue_from_route', 'joint_rescue_from_value', 'joint_change_from_baseline')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline='') as file:
        return list(csv.DictReader(file))


class Audit:
    def __init__(self):
        self.errors, self.calls = {}, 0

    def check(self, name, actual, expected, tolerance=1e-12):
        difference = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        assert np.isfinite(difference), name
        self.errors[name] = max(difference, self.errors.get(name, 0.))
        assert difference <= tolerance, (name, difference, tolerance)


def inputs(tokenizer, results):
    manifest, prompts = js(results/'token-manifest.json'), js(results/'prompts.json')
    pool = []
    for tid in range(1000, len(tokenizer)):
        text = tokenizer.decode(tid)
        if re.fullmatch(r' [a-z]{3,8}', text) and tokenizer.encode(text, add_special_tokens=False) == [tid]:
            pool.append(tid)
        if len(pool) == 192:
            break
    assert pool == [r['id'] for r in manifest['eligible_pool']]
    shuffled = np.array(pool[:128])
    np.random.default_rng(20261013).shuffle(shuffled)
    assert shuffled.reshape(32, 4).tolist() == manifest['groups']
    assert pool[128:] == manifest['filler_pool'] and len(prompts) == 768
    for p in prompts:
        group, binding, order, gap, query = (p[x] for x in ('group','binding','order','gap','query'))
        a, b, c, d = manifest['groups'][group]
        assert [p[x] for x in ('A','B','C','D')] == [a,b,c,d]
        av, cv = ((b,d), (d,b))[binding]
        pairs = [(a,av),(c,cv)][::1 if order == 0 else -1]
        filler = np.random.default_rng(20261014+group).permutation(pool[128:])[:24].tolist()
        expected = [tokenizer.eos_token_id, *pairs[0], *filler[:gap], *pairs[1], *filler[12:12+gap], (a,c)[query]]
        assert expected == p['input_ids']
        assert p['split'] == ('calibration' if group < 16 else 'test')
        assert p['target'] == (av,cv)[query] and p['sign'] == (1 if p['target'] == b else -1)
        assert p['recent_sign'] == (1 if pairs[1][1] == b else -1)
        assert expected[p['bpos']] == b and expected[p['dpos']] == d
        assert prompts[p['donor_id']]['input_ids'][:-1] == expected[:-1]
        assert prompts[p['donor_id']]['input_ids'][-1] != expected[-1]
    assert {p['prompt_id'] for p in prompts} == set(range(768))
    return prompts


def tables(audit, results):
    manifest = js(results/'manifest.json')
    for name, digest in manifest['files'].items():
        assert sha(results/name) == digest, name
    metadata = js(results/'run-metadata.json')
    assert metadata['status'] == 'completed'
    assert metadata['intervention_rows'] == 9216 and metadata['intervention_forward_examples'] == 27648
    repo = STUDY.parents[1]
    for name, key in (('analysis-plan.md','plan_sha256'), ('code/run.py','runner_sha256')):
        path = STUDY/name
        assert sha(path) == metadata[key]
        assert subprocess.check_output(['git','show',metadata['lock_commit']+':'+path.relative_to(repo).as_posix()],cwd=repo) == path.read_bytes()
    prompts = js(results/'prompts.json')
    baselines = csv_rows(results/'baseline.csv')
    assert len(baselines) == 768
    base = {int(r['prompt_id']): r for r in baselines}
    attention = np.load(results/'attention-rows.npz')['attention']
    assert attention.shape == (768,6,4,30)
    for p in prompts:
        r = base[p['prompt_id']]
        margin = float(r['logit_B'])-float(r['logit_D'])
        audit.check('baseline_formulas',[margin,p['sign']*margin,p['recent_sign']*margin],
                    [float(r[x]) for x in ('margin','answer_score','recency_score')])
        assert int(r['target_is_top']) == (int(r['top_token']) == p['target'])
    for r in csv_rows(results/'baseline-groups.csv'):
        values = [base[p['prompt_id']] for p in prompts if p['group'] == int(r['group'])]
        assert len(values) == int(r['count']) == 24
        expected = [np.mean([float(x['answer_score']) for x in values]), np.mean([float(x['recency_score']) for x in values]),
                    np.mean([float(x['answer_score']) > 0 for x in values]), np.mean([int(x['target_is_top']) for x in values])]
        audit.check('baseline_groups',expected,[float(r[k]) for k in ('binding_score','recency_score','pair_accuracy','full_vocabulary_accuracy')])
    calibration = []
    for l in range(6):
        for h in range(4):
            value = np.mean([float(p['sign']*(attention[p['prompt_id'],l,h,p['bpos']]-attention[p['prompt_id'],l,h,p['dpos']])) for p in prompts if p['group'] < 16])
            calibration.append({'layer':l,'head':h,'calibration_score':float(value)})
    selection = js(results/'selected-head.json')
    for actual, expected in zip(calibration, selection['all_heads']):
        assert actual['layer'] == expected['layer'] and actual['head'] == expected['head']
        audit.check('head_calibration',actual['calibration_score'],expected['calibration_score'])
    chosen = sorted(calibration,key=lambda r:(-r['calibration_score'],r['layer'],r['head']))[0]
    assert chosen == selection['selected']
    rows = csv_rows(results/'interventions.csv')
    cells = {(int(r['prompt_id']),int(r['layer']),int(r['head'])):r for r in rows}
    assert len(rows) == len(cells) == 9216
    assert set(cells) == {(p['prompt_id'],l,h) for p in prompts if p['group'] >= 16 for l in range(6) for h in range(4)}
    for (pid,l,h), r in cells.items():
        p = prompts[pid]
        margins = {condition:float(r[condition+'_logit_B'])-float(r[condition+'_logit_D']) for condition in ('route','value','joint')}
        for condition, value in margins.items():
            audit.check('patch_margins',value,float(r[condition+'_margin']))
        b,s = float(base[pid]['margin']),p['sign']
        expected = [s*b,float(s*(attention[pid,l,h,p['bpos']]-attention[pid,l,h,p['dpos']])),s*(b-margins['route']),s*(b-margins['value']),
                    s*(margins['joint']-margins['route']),s*(margins['joint']-margins['value']),s*(margins['joint']-b)]
        audit.check('all_intervention_formulas',expected,[float(r[m]) for m in METRICS])
    for r in csv_rows(results/'head-summary.csv'):
        part = [x for x in rows if x['layer'] == r['layer'] and x['head'] == r['head'] and
                (r['subset'] == 'all' or x[r['subset']] == r['subset_value'])]
        assert len(part) == int(r['count'])
        audit.check('head_summary',[np.mean([float(x[m]) for x in part]) for m in METRICS],[float(r[m]) for m in METRICS])
    groups = csv_rows(results/'head-groups.csv')
    for r in groups:
        part = [x for x in rows if x['layer'] == r['layer'] and x['head'] == r['head'] and x['group'] == r['group']]
        assert len(part) == int(r['count']) == 24
        audit.check('head_groups',[np.mean([float(x[m]) for x in part]) for m in METRICS],[float(r[m]) for m in METRICS])
    selected_groups = [r for r in groups if int(r['layer']) == chosen['layer'] and int(r['head']) == chosen['head']]
    selected_groups.sort(key=lambda r:int(r['group']))
    sampled = np.random.default_rng(20261015).integers(0,16,(10000,16))
    bootstrap = js(results/'bootstrap.json')
    for metric in METRICS:
        values = np.array([float(r[metric]) for r in selected_groups])
        lower,upper = np.quantile(values[sampled].mean(1),[.025,.975])
        audit.check('bootstrap',[values.mean(),lower,upper],[bootstrap['intervals'][metric][k] for k in ('mean','lower','upper')])
    primary = ('baseline_score','attention_score','route_drop','value_drop','joint_rescue_from_route')
    assert bootstrap['primary_pattern_supported'] == all(bootstrap['intervals'][m]['lower'] > 0 for m in primary)
    control = js(results/'controls.json')
    assert control['status'] == 'passed'
    for key in ('stock_custom_logit_error','same_query_route_logit_error','double_value_swap_logit_error'):
        assert control[key] <= 1e-5
    for key in ('local_donor_vector','past_values','row_sum'):
        assert control['errors'][key] <= 2e-6
    for key in ('self_weight','untouched_attention','untouched_head_output'):
        assert control['errors'][key] == 0
    return prompts,base,attention,cells,chosen,metadata


class Replay:
    def __init__(self, audit, attention, chosen):
        self.audit,self.saved_attention,self.chosen = audit,attention,chosen
        self.batch,self.mode = [],None

    def hook(self,module,q,k,v,attention_mask,scaling,dropout=0.,head_mask=None,**kwargs):
        output,weights = STOCK(module,q,k,v,attention_mask,scaling,dropout=dropout,head_mask=head_mask,**kwargs)
        layer = module._audit_layer
        if self.mode is None:
            for i,p in enumerate(self.batch):
                self.audit.check('replayed_attention',weights[i,:,-1].numpy(),self.saved_attention[p['prompt_id'],layer,:,:len(p['input_ids'])],2e-6)
        if self.mode is None or layer != self.chosen['layer']:
            return output,weights
        head = self.chosen['head']
        modified = weights.clone()
        if self.mode in ('route','joint'):
            for i,p in enumerate(self.batch):
                donor = torch.from_numpy(self.saved_attention[p['donor_id'],layer,head,:len(p['input_ids'])])
                denominator = donor[:-1].sum()
                if denominator > 1e-12:
                    modified[i,head,-1,:-1] = (weights[i,head,-1,:-1].sum()/denominator)*donor[:-1]
        values = v.clone()
        if self.mode in ('value','joint'):
            for i,p in enumerate(self.batch):
                values[i,head,p['bpos']] = v[i,head,p['dpos']]
                values[i,head,p['dpos']] = v[i,head,p['bpos']]
        all_heads = torch.matmul(modified,values).transpose(1,2).contiguous()
        answer = output.clone()
        answer[:,-1,head] = all_heads[:,-1,head]
        return answer,modified

    @torch.inference_mode()
    def forward(self,model,batch,mode):
        self.batch,self.mode = batch,mode
        hidden = model.gpt_neox(torch.tensor([p['input_ids'] for p in batch]),use_cache=False).last_hidden_state
        logits = model.embed_out(hidden[:,-1])
        self.audit.calls += 1
        return logits.numpy()


def replay(audit,model_dir,prompts,base,attention,cells,chosen,metadata):
    assert metadata['revision'] == 'cf967c0a9a04383db6f7b1108d86b2962634b4ac'
    for name,digest in metadata['model_hashes'].items():
        assert sha(model_dir/name) == digest
    tokenizer = AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    model = GPTNeoXForCausalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
    model.requires_grad_(False)
    for i,block in enumerate(model.gpt_neox.layers):
        block.attention._audit_layer = i
    engine = Replay(audit,attention,chosen)
    neox.eager_attention_forward = engine.hook
    examples = 0
    for gap in (0,4,12):
        pool = [p for p in prompts if p['gap'] == gap]
        for start in range(0,len(pool),32):
            batch = pool[start:start+32]
            scores = engine.forward(model,batch,None)
            for i,p in enumerate(batch):
                r = base[p['prompt_id']]
                audit.check('replayed_baseline_logits',scores[i,[p['B'],p['D']]],[float(r['logit_B']),float(r['logit_D'])],1e-5)
                assert int(scores[i].argmax()) == int(r['top_token'])
        pool = [p for p in pool if p['group'] >= 16]
        for start in range(0,len(pool),32):
            batch = pool[start:start+32]
            for mode in ('route','value','joint'):
                scores = engine.forward(model,batch,mode)
                for i,p in enumerate(batch):
                    r = cells[p['prompt_id'],chosen['layer'],chosen['head']]
                    audit.check('replayed_'+mode+'_logits',scores[i,[p['B'],p['D']]],[float(r[mode+'_logit_B']),float(r[mode+'_logit_D'])],1e-5)
                    examples += 1
        print(json.dumps({'gap_complete':gap,'model_forward_calls':audit.calls,'intervention_examples':examples}),flush=True)
    assert examples == 1152 and audit.calls == 60
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--results',type=Path,default=STUDY/'results')
    parser.add_argument('--output',type=Path,default=STUDY/'reviews'/'independent-audit.json')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a new audit receipt path.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    audit,start = Audit(),time.monotonic()
    receipt = {'status':'failed','human_review':False,'imports_runner':False,'audit_sha256':sha(__file__)}
    try:
        prompts,base,attention,cells,chosen,metadata = tables(audit,args.results)
        receipt.update(lock_commit=metadata['lock_commit'],selected_head=chosen)
        tokenizer = replay(audit,args.model_dir,prompts,base,attention,cells,chosen,metadata)
        inputs(tokenizer,args.results)
        receipt.update(status='passed',table_rows=9216,baseline_rows=768,replayed_intervention_examples=1152,
                       replay_scope='All baseline attention rows and all test interventions for the preselected head.',
                       limit='Other heads have full arithmetic checks but no independent intervention replay.')
    except Exception as exc:
        receipt['failure'] = {'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        neox.eager_attention_forward = STOCK
        receipt.update(maximum_errors=audit.errors,model_forward_calls=audit.calls,elapsed_seconds=time.monotonic()-start)
        args.output.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
