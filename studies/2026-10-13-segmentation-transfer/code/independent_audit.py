#!/usr/bin/env python3
"""Separate input/table audit and full head replay. Does not import the runner."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import re
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

STUDY = Path(__file__).resolve().parents[1]
REVISION = 'cf967c0a9a04383db6f7b1108d86b2962634b4ac'
LOGIT_TOL = 1e-5
PROBABILITY_TOL = 1e-7
METRICS = ('intact_score', 'effect', 'intact_pair_correct', 'intact_full_correct',
           'intact_target_probability', 'intact_recency', 'ablated_score',
           'ablated_pair_correct', 'ablated_full_correct')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as file:
        return list(csv.DictReader(file))


class Audit:
    def __init__(self):
        self.errors = {}
        self.diagnostics = {}
        self.calls = self.examples = 0
        self.deadline = time.monotonic() + 1200

    def check(self, name, actual, expected, tolerance=1e-12):
        error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        assert np.isfinite(error), name
        self.errors[name] = max(error, self.errors.get(name, 0.))
        if error > tolerance:
            raise AssertionError((name, error, tolerance))


def verify_files(results):
    manifest = js(results/'manifest.json')
    for name, digest in manifest['files'].items():
        assert sha(results/name) == digest, name
    metadata = js(results/'run-metadata.json')
    assert metadata['status'] == 'completed'
    assert metadata['revision'] == REVISION and metadata['dtype'] == 'float32'
    assert metadata['split_prompts'] == 2048 and metadata['canonical_prompts'] == 512
    repo = STUDY.parents[1]
    for name, key in (('analysis-plan.md', 'plan_sha256'), ('code/run.py', 'runner_sha256')):
        path = STUDY/name
        assert sha(path) == metadata[key]
        raw = subprocess.check_output(['git', 'show', metadata['lock_commit']+':'+path.relative_to(repo).as_posix()], cwd=repo)
        assert raw == path.read_bytes(), name
    for name, value in metadata['packages'].items():
        assert version(name) == value, name
    return metadata


def verify_inputs(tokenizer, results):
    manifest = js(results/'token-manifest.json')
    text = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(len(tokenizer))]
    smallest = {}
    for i, s in enumerate(text):
        if s not in smallest:
            smallest[s] = i
    eligible = [i for i in range(1000, len(text)) if re.fullmatch(r' [a-z]{3,8}', text[i])
                and tokenizer.encode(text[i], add_special_tokens=False) == [i]]
    old = set(eligible[:192])
    assert sorted(old) == manifest['old_excluded_ids']
    candidates = []
    for i in range(1000, len(text)):
        s = text[i]
        if i in old or not re.fullmatch(r' [a-z]{5,12}', s):
            continue
        if tokenizer.encode(s, add_special_tokens=False) != [i]:
            continue
        splits = []
        for cut in range(2, len(s)):
            left, right = smallest.get(s[:cut]), smallest.get(s[cut:])
            if left is not None and right is not None:
                pair = [left, right]
                if tokenizer.decode(pair, clean_up_tokenization_spaces=False) == s:
                    splits.append((cut, pair))
        if len(splits) >= 2 and len(set(splits[0][1]+splits[-1][1])) == 4:
            candidates.append(dict(id=i, text=s, A=splits[0][1], B=splits[-1][1],
                                   cuts=[splits[0][0], splits[-1][0]],
                                   pieces_A=[text[t] for t in splits[0][1]],
                                   pieces_B=[text[t] for t in splits[-1][1]]))
    assert len(candidates) == manifest['candidate_count']
    queue, groups, rejected = list(candidates), [], []
    for group in range(32):
        first = queue.pop(0)
        first_tokens = set(first['A']+first['B'])
        for i, second in enumerate(queue):
            overlap = sorted(first_tokens & set(second['A']+second['B']))
            if not overlap:
                groups.append(dict(group=group, K=first, L=queue.pop(i)))
                break
            rejected.append(dict(group=group, first=first['id'], candidate=second['id'], overlap=overlap))
        else:
            raise AssertionError('No second disjoint key.')
    forbidden = set(old)
    for group in groups:
        for key in ('K', 'L'):
            forbidden.update([group[key]['id'], *group[key]['A'], *group[key]['B']])
    pool = [tid for tid in eligible if tid not in forbidden]
    values, fillers = pool[:64], pool[64:128]
    assert values == manifest['value_ids'] and fillers == manifest['filler_ids']
    assert rejected == manifest['rejected_pair_conflicts']
    for i, group in enumerate(groups):
        group.update(B=values[2*i], D=values[2*i+1],
                     fillers=np.random.default_rng(2026101300+i).permutation(fillers)[:16].tolist())
    assert groups == manifest['groups']
    prompts, refs = js(results/'prompts.json'), js(results/'canonical-prompts.json')
    assert len(prompts) == 2048 and len(refs) == 512
    seen, canonical_seen, family_text = set(), set(), {}
    for collection, canonical in ((prompts, False), (refs, True)):
        for pid, p in enumerate(collection):
            assert p['prompt_id'] == pid
            group = groups[p['group']]
            b, d = group['B'], group['D']
            assignment = {'K': (b, d)[p['binding']], 'L': (d, b)[p['binding']]}
            order = ('K', 'L') if p['order'] == 0 else ('L', 'K')
            gap, query = p['gap'], p['query']
            source_split, query_split = (None, None) if canonical else tuple(p['cell'])
            key_ids = lambda key, split: [group[key]['id']] if split is None else group[key][split]
            ids = [tokenizer.eos_token_id]
            for index, key in enumerate(order):
                ids += key_ids(key, source_split) + [assignment[key]] + group['fillers'][8*index:8*index+gap]
            ids += key_ids(query, query_split)
            assert ids == p['input_ids']
            assert p['B'] == b and p['D'] == d and p['target'] == assignment[query]
            assert p['sign'] == (1 if assignment[query] == b else -1)
            assert p['recent_sign'] == (1 if assignment[order[1]] == b else -1)
            assert p['readout'] == len(ids)-1
            decoded = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
            assert decoded == p['text']
            family = (p['group'], p['binding'], p['order'], gap, query)
            if canonical:
                assert family not in canonical_seen
                canonical_seen.add(family)
                assert family_text[family] == decoded
            else:
                assert len(ids) == 9+2*gap
                assert p['utf8_hex'] == decoded.encode().hex()
                assert p['matched'] == int(source_split == query_split)
                assert ids[p['bpos']] == b and ids[p['dpos']] == d
                family_text.setdefault(family, decoded)
                assert family_text[family] == decoded
                key = family+(p['cell'],)
                assert key not in seen
                seen.add(key)
                if source_split != query_split:
                    query_ids = set(group[query][query_split])
                    assert not query_ids & set(group['K'][source_split]+group['L'][source_split]+[b,d]+group['fillers'])
    expected = set(itertools.product(range(32), range(2), range(2), (0,8), ('K','L')))
    assert canonical_seen == expected
    assert seen == {family+(cell,) for family in expected for cell in ('AA','AB','BA','BB')}
    return prompts, refs


def check_tables(audit, results, prompts, refs):
    scored = rows(results/'scores.csv')
    canonical = rows(results/'canonical-scores.csv')
    assert len(scored) == 2048 and len(canonical) == 512
    scored = {int(r['prompt_id']): r for r in scored}
    canonical = {int(r['prompt_id']): r for r in canonical}
    assert set(scored) == set(range(2048)) and set(canonical) == set(range(512))
    attention = np.load(results/'attention-rows.npz')['attention']
    assert attention.shape == (2048,25)
    for collection, saved, prefixes in ((prompts, scored, ('intact_', 'ablated_')), (refs, canonical, ('',))):
        for p in collection:
            r = saved[p['prompt_id']]
            for key in ('group','binding','order','gap','B','D','target','sign','recent_sign'):
                assert int(r[key]) == p[key]
            assert r['query'] == p['query']
            for prefix in prefixes:
                # The runner subtracts two float32 logits before it converts to Python float.
                margin = float(np.float32(r[prefix+'logit_B'])-np.float32(r[prefix+'logit_D']))
                score = p['sign']*margin
                audit.check('saved_score_formulas', [margin, score, p['recent_sign']*margin],
                            [float(r[prefix+k]) for k in ('margin','score','recency')])
                assert int(r[prefix+'pair_correct']) == (score > 0)
                assert int(r[prefix+'full_correct']) == (int(r[prefix+'top_token']) == p['target'])
            if len(prefixes) == 2:
                audit.check('saved_effect', float(r['intact_score'])-float(r['ablated_score']), float(r['effect']))
                audit.check('saved_attention_values', attention[p['prompt_id'], [p['bpos'],p['dpos']]],
                            [float(r['attention_B']),float(r['attention_D'])])
                assert np.all(attention[p['prompt_id'],len(p['input_ids']):] == 0)
    def member(r, subset):
        if subset == 'all': return True
        if subset == 'matched': return int(r['matched']) == 1
        if subset == 'mismatched': return int(r['matched']) == 0
        if subset.startswith('gap'): return int(r['gap']) == int(subset[3:])
        return r['cell'] == subset
    group_rows = rows(results/'group-scores.csv')
    group_values = {}
    assert len(group_rows) == 32*9
    for r in group_rows:
        group, subset = int(r['group']), r['subset']
        part = [x for x in scored.values() if int(x['group']) == group and member(x,subset)]
        assert len(part) == int(r['count'])
        expected = {metric: float(np.mean([float(x[metric]) for x in part])) for metric in METRICS}
        for metric, value in expected.items():
            audit.check('group_means', value, float(r[metric]))
            group_values[group,subset,metric] = value
    summary = js(results/'summary.json')
    assert summary['seed'] == 2026101320 and summary['bootstrap_draws'] == 10000 and summary['clusters'] == 32
    samples = np.random.default_rng(2026101320).integers(0,32,size=(10000,32))
    for subset, metrics in summary['estimates'].items():
        for metric, estimate in metrics.items():
            if subset == 'matched_minus_mismatched':
                values = np.array([group_values[g,'matched',metric]-group_values[g,'mismatched',metric] for g in range(32)])
            else:
                values = np.array([group_values[g,subset,metric] for g in range(32)])
            interval = np.percentile(values[samples].mean(axis=1),[2.5,97.5])
            audit.check('bootstrap', [values.mean(),*interval], [estimate[k] for k in ('mean','lower','upper')])
    mismatch = summary['estimates']['mismatched']
    pattern = mismatch['intact_score']['lower'] > 0 and mismatch['effect']['lower'] > 0
    gate = mismatch['intact_pair_correct']['mean'] >= .75 and mismatch['intact_full_correct']['mean'] >= .10 and mismatch['intact_score']['lower'] > 0
    assert summary['primary_pattern_supported'] == pattern
    assert summary['capability_gate_passed'] == gate
    assert summary['causal_binding_interpretation_supported'] == (pattern and gate)
    canonical_groups = rows(results/'canonical-groups.csv')
    assert len(canonical_groups) == 32
    for r in canonical_groups:
        part = [x for x in canonical.values() if x['group'] == r['group']]
        assert len(part) == int(r['count']) == 16
        for metric in ('score','pair_correct','full_correct','target_probability','recency'):
            audit.check('canonical_groups',np.mean([float(x[metric]) for x in part]),float(r[metric]))
    for metric, value in summary['canonical_means'].items():
        audit.check('canonical_means',np.mean([float(r[metric]) for r in canonical_groups]),value)
    control = js(results/'controls.json')
    assert control['status'] == 'passed' and control['byte_checks_passed']
    assert control['no_op_prompt_count'] == 2048
    assert max(control['errors'].values()) <= LOGIT_TOL
    assert all(value == 0 for value in control['local_errors'].values())
    batches = rows(results/'batch-controls.csv')
    assert len(batches) == 64 and sum(int(r['count']) for r in batches) == 2048
    for r in batches:
        assert max(float(r[k]) for k in ('stock_error','self_error')) <= LOGIT_TOL
        assert all(float(r[k]) == 0 for k in ('untouched_output','unchanged_attention','ablated_vector','future_attention'))
    return scored,canonical,attention


class Replay:
    """Edit the concatenated projection input, not the runner's attention return."""
    def __init__(self, model, audit):
        self.model,self.audit = model,audit
        self.ablated,self.original,self.hits = False,None,0
        self.handle = model.gpt_neox.layers[3].attention.dense.register_forward_pre_hook(self.project_input)

    def project_input(self, module, args):
        x = args[0]
        assert x.shape[-1] == 128
        self.original = x[:,-1,32:64].detach().clone()
        self.hits += 1
        if not self.ablated:
            return None
        edited = x.clone()
        edited[:,-1,32:64] = 0.
        difference = edited-x
        difference[:,-1,32:64] = 0.
        self.audit.check('independent_untouched_projection_input',difference.numpy(),0.,0.)
        return (edited,)+args[1:]

    @torch.inference_mode()
    def forward(self, batch, ablated=False, capture=True):
        if time.monotonic() > self.audit.deadline:
            raise TimeoutError('The independent audit reached its fixed 20-minute limit.')
        self.ablated,self.hits = ablated,0
        result = self.model.gpt_neox(torch.tensor([p['input_ids'] for p in batch]), use_cache=False, output_attentions=capture)
        output = self.model.embed_out(result.last_hidden_state[:,-1]).numpy()
        assert self.hits == 1
        self.audit.calls += 1
        self.audit.examples += len(batch)
        return output, result.attentions[3][:,1,-1].numpy() if capture else None, self.original.norm(dim=-1).numpy()


def compare_output(audit, logits, p, saved, prefix):
    audit.check('replayed_'+prefix+'logits',logits[[p['B'],p['D']]],
                [float(saved[prefix+'logit_B']),float(saved[prefix+'logit_D'])],LOGIT_TOL)
    target = p['target']
    top = int(np.argmax(logits))
    assert top == int(saved[prefix+'top_token'])
    assert int(np.sum(logits > logits[target]))+1 == int(saved[prefix+'target_rank'])
    # Replay the recorded definition. Keep the different float64 calculation
    # as a diagnostic; the initial failed audit of that path is preserved.
    probability = torch.from_numpy(logits).softmax(-1).numpy()
    audit.check('replayed_'+prefix+'probabilities',probability[[p['B'],p['D'],target,top]],
                [float(saved[prefix+k]) for k in ('probability_B','probability_D','target_probability','top_probability')],PROBABILITY_TOL)
    weights = np.exp(logits.astype(np.float64)-float(np.max(logits)))
    double_probability = weights/weights.sum()
    full_error = np.abs(probability.astype(np.float64)-double_probability)
    worst = int(np.argmax(full_error))
    old = audit.diagnostics.get('float64_probability_reference', {'maximum_error': -1.})
    if float(full_error[worst]) > old['maximum_error']:
        audit.diagnostics['float64_probability_reference'] = dict(maximum_error=float(full_error[worst]),
            prompt_id=p['prompt_id'],condition=prefix or 'canonical',token_id=worst,
            logit=float(logits[worst]),maximum_logit=float(np.max(logits)),
            float32_probability=float(probability[worst]),float64_probability=float(double_probability[worst]),
            float32_sum_in_float64=float(probability.astype(np.float64).sum()),
            float64_sum=float(double_probability.sum()))


def replay(audit, model_dir, metadata, prompts, refs, scored, canonical, attention):
    for name, digest in metadata['model_hashes'].items():
        assert sha(model_dir/name) == digest, name
    model = GPTNeoXForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32,
                                             attn_implementation='eager').eval()
    model.requires_grad_(False)
    assert (model.config.num_hidden_layers,model.config.num_attention_heads,model.config.hidden_size) == (6,4,128)
    engine = Replay(model,audit)
    try:
        for records, canonical_mode in ((prompts,False),(refs,True)):
            for length in sorted({len(p['input_ids']) for p in records}):
                pool = [p for p in records if len(p['input_ids']) == length]
                for start in range(0,len(pool),32):
                    batch = pool[start:start+32]
                    for ablated in ((False,) if canonical_mode else (False,True)):
                        outputs,weights,norms = engine.forward(batch,ablated,capture=not canonical_mode)
                        for i,p in enumerate(batch):
                            saved = canonical[p['prompt_id']] if canonical_mode else scored[p['prompt_id']]
                            prefix = '' if canonical_mode else ('ablated_' if ablated else 'intact_')
                            compare_output(audit,outputs[i],p,saved,prefix)
                            if not canonical_mode and not ablated:
                                audit.check('replayed_attention',weights[i],attention[p['prompt_id'],:length],LOGIT_TOL)
                                audit.check('replayed_head_norm',norms[i],float(saved['head_norm']),LOGIT_TOL)
                print(json.dumps(dict(canonical=canonical_mode,length=length,forward_calls=audit.calls)),flush=True)
    finally:
        engine.handle.remove()
    assert audit.calls == 144 and audit.examples == 4608


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
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    audit,started = Audit(),time.monotonic()
    receipt = dict(status='failed',imports_runner=False,human_review=False,audit_sha256=sha(__file__),
                   logit_tolerance=LOGIT_TOL,probability_tolerance=PROBABILITY_TOL,
                   initial_audit_failure_preserved=True,float64_probability_reference_is_diagnostic=True)
    try:
        metadata = verify_files(args.results)
        receipt['lock_commit'] = metadata['lock_commit']
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
        prompts,refs = verify_inputs(tokenizer,args.results)
        scored,canonical,attention = check_tables(audit,args.results,prompts,refs)
        replay(audit,args.model_dir,metadata,prompts,refs,scored,canonical,attention)
        verify_files(args.results)
        receipt.update(status='passed',split_prompts=2048,canonical_prompts=512,
                       replay_scope='All intact and ablated split prompts; all canonical references.',
                       independent_operator='Pre-hook on the concatenated input of block 3 attention output projection; zero final-position dimensions 32:64.')
    except Exception as exc:
        receipt['failure'] = dict(type=type(exc).__name__,message=str(exc))
        raise
    finally:
        receipt.update(maximum_errors=audit.errors,diagnostics=audit.diagnostics,model_forward_calls=audit.calls,
                       model_forward_examples=audit.examples,elapsed_seconds=time.monotonic()-started)
        args.output.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
