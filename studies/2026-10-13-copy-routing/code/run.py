#!/usr/bin/env python3
"""Fixed copy-routing experiment. Scientific runs require a locked Git commit."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import time
import traceback
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def table(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def lock_check(commit):
    repo = ROOT.parents[1]
    actual = subprocess.run(['git', 'rev-parse', commit + '^{commit}'], cwd=repo,
                            check=True, capture_output=True).stdout.decode().strip()
    if actual != commit:
        raise RuntimeError('Use a full commit SHA for the lock.')
    for path in (ROOT / 'analysis-plan.md', Path(__file__).resolve()):
        rel = path.relative_to(repo).as_posix()
        saved = subprocess.run(['git', 'show', commit + ':' + rel], cwd=repo,
                               check=True, capture_output=True).stdout
        if saved != path.read_bytes():
            raise RuntimeError('Local bytes differ from lock: ' + rel)


def make_inputs(tokenizer):
    eligible = []
    for tid in range(1000, len(tokenizer)):
        word = tokenizer.decode([tid])
        if re.fullmatch(r' [a-z]{3,8}', word) and tokenizer.encode(word, add_special_tokens=False) == [tid]:
            eligible.append(tid)
        if len(eligible) == 192:
            break
    assert len(eligible) == 192
    words = np.array(eligible[:128], dtype=np.int64)
    np.random.default_rng(20261013).shuffle(words)
    groups = words.reshape(32, 4)
    fillers = np.array(eligible[128:], dtype=np.int64)
    prompts = []
    for group, ids4 in enumerate(groups):
        a, b, c, d = map(int, ids4)
        filler = np.random.default_rng(20261014 + group).permutation(fillers)[:24].tolist()
        for binding in range(2):
            av, cv = (b, d) if binding == 0 else (d, b)
            for order in range(2):
                pairs = [(a, av), (c, cv)]
                if order:
                    pairs.reverse()
                for gap in (0, 4, 12):
                    prefix = [tokenizer.eos_token_id, *pairs[0], *filler[:gap],
                              *pairs[1], *filler[12:12 + gap]]
                    value_positions = (2, 4 + gap)
                    bpos = value_positions[0] if pairs[0][1] == b else value_positions[1]
                    dpos = value_positions[0] if pairs[0][1] == d else value_positions[1]
                    recent = pairs[1][1]
                    for query, query_id in enumerate((a, c)):
                        target = av if query == 0 else cv
                        pid = len(prompts)
                        prompts.append(dict(prompt_id=pid, donor_id=pid + (1 if query == 0 else -1),
                                            group=group, split='calibration' if group < 16 else 'test',
                                            binding=binding, order=order, gap=gap, query=query,
                                            A=a, B=b, C=c, D=d, bpos=bpos, dpos=dpos,
                                            sign=1 if target == b else -1, target=target,
                                            recent_sign=1 if recent == b else -1,
                                            input_ids=prefix + [query_id]))
    assert len(prompts) == 768
    for p in prompts:
        donor = prompts[p['donor_id']]
        assert p['input_ids'][:-1] == donor['input_ids'][:-1]
        assert p['input_ids'][p['bpos']] == p['B']
        assert p['input_ids'][p['dpos']] == p['D']
    manifest = dict(eligible_pool=[dict(id=i, text=tokenizer.decode([i])) for i in eligible],
                    groups=groups.tolist(), filler_pool=fillers.tolist(), count=len(prompts))
    return prompts, manifest


def batches(prompts, batch_size=32):
    for length in sorted({len(p['input_ids']) for p in prompts}):
        pool = [p for p in prompts if len(p['input_ids']) == length]
        for start in range(0, len(pool), batch_size):
            yield pool[start:start + batch_size]


class AttentionChange:
    def __init__(self):
        self.records = []
        self.capture = False
        self.cache = {}
        self.layer = self.head = None
        self.mode = None
        self.same_donor = False
        self.errors = dict(row_sum=0., self_weight=0., untouched_head_output=0.,
                           untouched_attention=0., local_donor_vector=0., past_values=0.)
        self.zero_past_mass = 0

    def forward(self, module, query, key, value, attention_mask, scaling,
                dropout=0.0, head_mask=None, **kwargs):
        out, weights = STOCK(module, query, key, value, attention_mask, scaling,
                             dropout=dropout, head_mask=head_mask, **kwargs)
        layer = module._copy_layer
        if self.capture:
            for i, p in enumerate(self.records):
                self.cache[(p['prompt_id'], layer)] = (
                    weights[i, :, -1].detach().clone(), value[i].detach().clone(),
                    out[i, -1].detach().clone())
        if self.mode is None or self.layer != layer:
            return out, weights
        h = self.head
        original = weights[:, h, -1]
        row = original.clone()
        if self.mode in ('route', 'joint', 'same_route'):
            donor = torch.stack([self.cache[(p['prompt_id'] if self.same_donor else p['donor_id'], layer)][0][h]
                                 for p in self.records])
            source_mass = donor[:, :-1].sum(-1)
            receiver_mass = original[:, :-1].sum(-1)
            valid = source_mass > 1e-12
            self.zero_past_mass += int((~valid).sum())
            factor = receiver_mass[valid] / source_mass[valid]
            row[valid, :-1] = factor[:, None] * donor[valid, :-1]
        vectors = value[:, h].clone()
        if self.mode in ('value', 'joint', 'double_value'):
            idx = torch.arange(len(self.records))
            bp = torch.tensor([p['bpos'] for p in self.records])
            dp = torch.tensor([p['dpos'] for p in self.records])
            repetitions = 2 if self.mode == 'double_value' else 1
            for _ in range(repetitions):
                left = vectors[idx, bp].clone()
                vectors[idx, bp] = vectors[idx, dp]
                vectors[idx, dp] = left
        edited_weights = weights.clone()
        edited_weights[:, h, -1] = row
        # Preserve the stock batched matrix product shape. Only its selected
        # final-query result is used; earlier query outputs remain stock.
        temporary_values = value.clone()
        temporary_values[:, h] = vectors
        product = torch.matmul(edited_weights, temporary_values).transpose(1, 2).contiguous()
        edited_out = out.clone()
        edited_out[:, -1, h] = product[:, -1, h]
        delta_out = edited_out - out
        delta_out[:, -1, h] = 0
        delta_weights = edited_weights - weights
        delta_weights[:, h, -1] = 0
        self.errors['untouched_head_output'] = max(self.errors['untouched_head_output'], float(delta_out.abs().max()))
        self.errors['untouched_attention'] = max(self.errors['untouched_attention'], float(delta_weights.abs().max()))
        self.errors['row_sum'] = max(self.errors['row_sum'], float((row.sum(-1) - 1).abs().max()))
        self.errors['self_weight'] = max(self.errors['self_weight'], float((row[:, -1] - original[:, -1]).abs().max()))
        assert self.errors['row_sum'] <= 2e-6
        assert self.errors['self_weight'] == 0
        assert self.errors['untouched_head_output'] == 0
        assert self.errors['untouched_attention'] == 0
        return edited_out, edited_weights


def logits_for(model, change, records):
    change.records = records
    ids = torch.tensor([p['input_ids'] for p in records], dtype=torch.long)
    with torch.inference_mode():
        hidden = model.gpt_neox(ids, use_cache=False).last_hidden_state[:, -1]
        return model.embed_out(hidden)


def load_model(model_dir):
    for name, expected in MODEL_HASHES.items():
        if sha(model_dir / name) != expected:
            raise RuntimeError('Wrong model file: ' + name)
    model = GPTNeoXForCausalLM.from_pretrained(model_dir, local_files_only=True,
              dtype=torch.float32, attn_implementation='eager').eval()
    model.requires_grad_(False)
    assert model.config.num_hidden_layers == 6 and model.config.hidden_size == 128
    assert model.config.num_attention_heads == 4
    for i, layer in enumerate(model.gpt_neox.layers):
        layer.attention._copy_layer = i
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    return model, tokenizer


def baseline_run(model, change, prompts, output):
    rows, attention = [], np.zeros((768, 6, 4, 30), dtype=np.float32)
    baseline = {}
    change.capture = True
    for batch in batches(prompts):
        logits = logits_for(model, change, batch)
        probabilities = logits.softmax(-1)
        for i, p in enumerate(batch):
            lb, ld = float(logits[i, p['B']]), float(logits[i, p['D']])
            row = {k: p[k] for k in ('prompt_id', 'donor_id', 'group', 'split', 'binding', 'order', 'gap', 'query', 'sign', 'recent_sign')}
            row.update(logit_B=lb, logit_D=ld, margin=lb-ld, answer_score=p['sign']*(lb-ld),
                       recency_score=p['recent_sign']*(lb-ld), probability_B=float(probabilities[i,p['B']]),
                       probability_D=float(probabilities[i,p['D']]), top_token=int(logits[i].argmax()),
                       target=p['target'], target_is_top=int(int(logits[i].argmax()) == p['target']))
            rows.append(row)
            baseline[p['prompt_id']] = row
            for layer in range(6):
                weights = change.cache[(p['prompt_id'], layer)][0]
                attention[p['prompt_id'], layer, :, :weights.shape[-1]] = weights.numpy()
    change.capture = False
    rows.sort(key=lambda r: r['prompt_id'])
    table(output/'baseline.csv', rows)
    np.savez_compressed(output/'attention-rows.npz', attention=attention)
    grouped = []
    for group in range(32):
        group_rows = [r for r in rows if r['group'] == group]
        grouped.append(dict(group=group, split=group_rows[0]['split'], count=len(group_rows),
                            binding_score=float(np.mean([r['answer_score'] for r in group_rows])),
                            recency_score=float(np.mean([r['recency_score'] for r in group_rows])),
                            pair_accuracy=float(np.mean([r['answer_score'] > 0 for r in group_rows])),
                            full_vocabulary_accuracy=float(np.mean([r['target_is_top'] for r in group_rows]))))
    table(output/'baseline-groups.csv', grouped)
    return baseline, attention


def arithmetic_controls(model, change, prompts):
    errors = change.errors
    for p in prompts:
        for layer in range(6):
            row, values, head_out = change.cache[(p['prompt_id'],layer)]
            donor_row, donor_values, donor_out = change.cache[(p['donor_id'],layer)]
            reconstructed = torch.bmm(donor_row[:,None,:], donor_values).squeeze(1)
            errors['local_donor_vector'] = max(errors['local_donor_vector'], float((reconstructed-donor_out).abs().max()))
            errors['past_values'] = max(errors['past_values'], float((values[:,:-1]-donor_values[:,:-1]).abs().max()))
    assert errors['local_donor_vector'] <= 2e-6
    assert errors['past_values'] <= 2e-6
    controls = [p for p in prompts if p['binding']==0 and p['order']==0 and p['query']==0]
    stock_error = same_error = double_error = 0.
    for batch in batches(controls):
        change.mode = None
        neox.eager_attention_forward = STOCK
        stock_logits = logits_for(model,change,batch)
        neox.eager_attention_forward = change.forward
        plain = logits_for(model,change,batch)
        stock_error = max(stock_error,float((plain-stock_logits).abs().max()))
        for layer in range(6):
            for head in range(4):
                change.layer, change.head = layer, head
                change.mode, change.same_donor = 'same_route', True
                same = logits_for(model,change,batch)
                same_error = max(same_error,float((same-plain).abs().max()))
                change.mode, change.same_donor = 'double_value', False
                double = logits_for(model,change,batch)
                double_error = max(double_error,float((double-plain).abs().max()))
    change.mode = None
    assert stock_error <= 1e-5, stock_error
    assert same_error <= 1e-5, same_error
    assert double_error <= 1e-5, double_error
    return dict(stock_custom_logit_error=stock_error, same_query_route_logit_error=same_error,
                double_value_swap_logit_error=double_error, input_count=len(controls))


def attention_score(p, attention, layer, head):
    row = attention[p['prompt_id'],layer,head]
    return float(p['sign']*(row[p['bpos']]-row[p['dpos']]))


def selected_head(prompts, attention, output):
    calibration = [p for p in prompts if p['split']=='calibration']
    rows = [dict(layer=l, head=h, calibration_score=float(np.mean([attention_score(p,attention,l,h) for p in calibration])))
            for l in range(6) for h in range(4)]
    chosen = min(rows, key=lambda r: (-r['calibration_score'],r['layer'],r['head']))
    dump(output/'selected-head.json',dict(rule='maximum calibration binding-sensitive attention score',
                                         selected=chosen, all_heads=rows))
    return chosen


def intervene(model, change, prompts, baseline, attention, output):
    test = [p for p in prompts if p['split']=='test']
    rows = []
    for layer in range(6):
        for head in range(4):
            change.layer,change.head = layer,head
            for batch in batches(test):
                values = {}
                for mode in ('route','value','joint'):
                    change.mode = mode
                    logits = logits_for(model,change,batch)
                    values[mode] = [(float(logits[i,p['B']]),float(logits[i,p['D']])) for i,p in enumerate(batch)]
                for i,p in enumerate(batch):
                    base = baseline[p['prompt_id']]
                    row = {k:p[k] for k in ('prompt_id','donor_id','group','binding','order','gap','query','sign')}
                    row.update(layer=layer,head=head,baseline_margin=base['margin'],
                               baseline_score=base['answer_score'], attention_score=attention_score(p,attention,layer,head))
                    for mode in ('route','value','joint'):
                        lb,ld = values[mode][i]
                        row[mode+'_logit_B'],row[mode+'_logit_D'] = lb,ld
                        row[mode+'_margin'] = lb-ld
                    s = p['sign']
                    row.update(route_drop=s*(row['baseline_margin']-row['route_margin']),
                               value_drop=s*(row['baseline_margin']-row['value_margin']),
                               joint_rescue_from_route=s*(row['joint_margin']-row['route_margin']),
                               joint_rescue_from_value=s*(row['joint_margin']-row['value_margin']),
                               joint_change_from_baseline=s*(row['joint_margin']-row['baseline_margin']))
                    rows.append(row)
            print(json.dumps(dict(completed_layer=layer,completed_head=head,rows=len(rows))),flush=True)
            table(output/'interventions.csv',rows)
    change.mode = None
    return rows


METRICS = ('baseline_score','attention_score','route_drop','value_drop','joint_rescue_from_route',
           'joint_rescue_from_value','joint_change_from_baseline')


def summarize(rows, chosen, output):
    summary, groups = [], []
    for layer in range(6):
        for head in range(4):
            selected = [r for r in rows if r['layer']==layer and r['head']==head]
            for subset,value in [('all','all')]+[('gap',i) for i in (0,4,12)]+[('order',i) for i in (0,1)]:
                part = selected if subset=='all' else [r for r in selected if r[subset]==value]
                row = dict(layer=layer,head=head,subset=subset,subset_value=value,count=len(part))
                row.update({m:float(np.mean([r[m] for r in part])) for m in METRICS})
                summary.append(row)
            for group in range(16,32):
                part = [r for r in selected if r['group']==group]
                row = dict(layer=layer,head=head,group=group,count=len(part))
                row.update({m:float(np.mean([r[m] for r in part])) for m in METRICS})
                groups.append(row)
    table(output/'head-summary.csv',summary)
    table(output/'head-groups.csv',groups)
    selected = [r for r in groups if r['layer']==chosen['layer'] and r['head']==chosen['head']]
    indices = np.random.default_rng(20261015).integers(0,16,(10000,16))
    intervals = {}
    for metric in METRICS:
        x = np.array([r[metric] for r in selected])
        boot = x[indices].mean(1)
        lo,hi = np.quantile(boot,[.025,.975])
        intervals[metric] = dict(mean=float(x.mean()),lower=float(lo),upper=float(hi))
    primary = ('route_drop','value_drop','joint_rescue_from_route','baseline_score','attention_score')
    dump(output/'bootstrap.json',dict(head=chosen,seed=20261015,draws=10000,clusters=16,
                                      intervals=intervals,primary_pattern_supported=all(intervals[m]['lower']>0 for m in primary)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--lock-commit')
    parser.add_argument('--dummy-preflight',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    args = parser.parse_args()
    if not args.dummy_preflight:
        if not args.lock_commit:
            parser.error('--lock-commit is required for a scientific run')
        lock_check(args.lock_commit)
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError('Choose an empty output directory for each run.')
    args.output.mkdir(parents=True,exist_ok=True)
    start = time.time()
    torch.set_num_threads(2)
    torch.manual_seed(20261013)
    torch.use_deterministic_algorithms(True)
    model,tokenizer = load_model(args.model_dir)
    if args.dummy_preflight:
        # No study words, scores, or scientific effects are inspected here.
        change = AttentionChange()
        records = []
        for gap in (0,4,12):
            prefix = [tokenizer.eos_token_id,101,102,*range(110,110+gap),103,104,*range(130,130+gap)]
            for q in range(2):
                pid = len(records)
                records.append(dict(prompt_id=pid,donor_id=pid+(1 if q==0 else -1),
                                    binding=0,order=0,query=q,bpos=2,dpos=4+gap,
                                    input_ids=prefix+[101 if q==0 else 103]))
        neox.eager_attention_forward = change.forward
        try:
            change.capture = True
            for batch in batches(records):
                logits_for(model,change,batch)
            change.capture = False
            control = arithmetic_controls(model,change,records)
            dump(args.output/'dummy-preflight.json',dict(control,errors=change.errors,status='passed'))
            print(json.dumps(dict(control,errors=change.errors,status='passed')))
        finally:
            neox.eager_attention_forward = STOCK
        return
    prompts,manifest = make_inputs(tokenizer)
    dump(args.output/'token-manifest.json',manifest)
    dump(args.output/'prompts.json',prompts)
    metadata = dict(model=MODEL_ID,revision=REVISION,model_hashes=MODEL_HASHES,lock_commit=args.lock_commit,
                    plan_sha256=sha(ROOT/'analysis-plan.md'),runner_sha256=sha(Path(__file__)),
                    platform=platform.platform(),python=platform.python_version(),dtype='float32',
                    packages={name:version(name) for name in ('torch','numpy','transformers','safetensors','tokenizers','huggingface-hub')},
                    cpu_threads=2,started_unix=start,status='running')
    dump(args.output/'run-metadata.json',metadata)
    change = AttentionChange()
    neox.eager_attention_forward = change.forward
    try:
        baseline,attention = baseline_run(model,change,prompts,args.output)
        # No intervention outcome is consulted by the selection rule.
        chosen = selected_head(prompts,attention,args.output)
        control = arithmetic_controls(model,change,prompts)
        dump(args.output/'controls.json',dict(control,errors=change.errors,zero_past_mass=change.zero_past_mass,status='pre-run checks passed'))
        rows = intervene(model,change,prompts,baseline,attention,args.output)
        assert len(rows)==384*24
        summarize(rows,chosen,args.output)
        lock_check(args.lock_commit)
        dump(args.output/'controls.json',dict(control,errors=change.errors,zero_past_mass=change.zero_past_mass,status='passed'))
        metadata.update(status='completed',elapsed_seconds=time.time()-start,intervention_rows=len(rows),
                        intervention_forward_examples=len(rows)*3)
        dump(args.output/'run-metadata.json',metadata)
        dump(args.output/'manifest.json',dict(files={p.relative_to(args.output).as_posix():sha(p)
                                                    for p in sorted(args.output.rglob('*')) if p.is_file()}))
        print(json.dumps(dict(status='completed',elapsed_seconds=metadata['elapsed_seconds'])),flush=True)
    except Exception:
        metadata.update(status='failed',elapsed_seconds=time.time()-start)
        dump(args.output/'run-metadata.json',metadata)
        dump(args.output/'failure.json',dict(traceback=traceback.format_exc(),errors=change.errors))
        raise
    finally:
        neox.eager_attention_forward = STOCK


if __name__=='__main__':
    main()
