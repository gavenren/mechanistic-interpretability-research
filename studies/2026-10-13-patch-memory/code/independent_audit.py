#!/usr/bin/env python3
"""Separate table audit and full-prefix model replay. Does not import the runner."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers.models.gpt_neox import modeling_gpt_neox as neox

STUDY = Path(__file__).resolve().parents[1]
TOL = 1e-8
FRAMES = ('The entry is {w}. Copy the entry:', 'The code is {w}. Copy the code:',
          'The password is {w}. Copy the password:', 'The label is {w}. Copy the label:')
PAIRS = (('fox', 'wolf'), ('tree', 'leaf'), ('glass', 'plate'), ('rock', 'sand'),
         ('gold', 'silver'), ('door', 'wall'), ('road', 'path'), ('fire', 'wind'))
SUFFIX = [380, 1924, 588, 320, 10141, 969, 846, 436]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as file:
        return list(csv.DictReader(file))


def attention(module, q, k, v, attention_mask, scaling, dropout=0., head_mask=None, **kwargs):
    assert q.dtype == k.dtype == v.dtype == torch.float64
    scores = q @ k.transpose(-1, -2)
    scores = scores * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, :k.shape[-2]]
    probabilities = torch.softmax(scores, dim=-1)
    if head_mask is not None:
        probabilities = probabilities * head_mask
    assert dropout == 0 or not module.training
    return (probabilities @ v).transpose(1, 2).contiguous(), probabilities


class Audit:
    def __init__(self):
        self.errors = {}
        self.calls = 0

    def close(self, name, actual, expected, tolerance=TOL):
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        assert np.isfinite(error), name
        self.errors[name] = max(error, self.errors.get(name, 0.))
        assert error <= tolerance, (name, error, tolerance)

    @torch.inference_mode()
    def forward(self, model, tokens, patch=None, capture=False):
        handles, states = [], {}
        if patch is not None:
            layer, position, vector = patch
            def replace(module, inputs, output):
                hidden = output[0].clone()
                hidden[:, position, :] = torch.tensor(vector, dtype=torch.float64)
                return (hidden, *output[1:])
            handles.append(model.gpt_neox.layers[layer].register_forward_hook(replace))
        if capture:
            for layer, block in enumerate(model.gpt_neox.layers):
                def save(module, inputs, output, layer=layer):
                    states[layer] = output[0][0, -1].numpy().copy()
                handles.append(block.register_forward_hook(save))
        try:
            output = model(torch.tensor([tokens]), use_cache=False, logits_to_keep=1)
            self.calls += 1
            return output.logits[0, -1].numpy().copy(), states
        finally:
            for handle in handles:
                handle.remove()


def audit_tables(audit, results):
    manifest = read_json(results / 'manifest.json')
    assert manifest['status'] == 'complete'
    for name, entry in manifest['files'].items():
        assert digest(results / name) == entry['sha256'], name
        assert (results / name).stat().st_size == entry['bytes'], name
    metadata = read_json(results / 'run-metadata.json')
    assert metadata['status'] == 'complete' and metadata['model_forward_calls'] == 14592
    repo = STUDY.parents[1]
    for filename, key in (('analysis-plan.md', 'plan_sha256'), ('code/run.py', 'runner_sha256')):
        path = STUDY / filename
        assert digest(path) == metadata[key]
        recorded = subprocess.check_output(['git', 'show', metadata['lock_commit'] + ':' + path.relative_to(repo).as_posix()], cwd=repo)
        assert recorded == path.read_bytes()
    scores = read_csv(results / 'scores.csv')
    baseline = read_csv(results / 'baseline.csv')
    assert len(scores) == 3456 and len(baseline) == 576
    cells = {(int(r['case_id']), int(r['layer']), int(r['horizon'])): r for r in scores}
    base_cells = {(int(r['case_id']), int(r['horizon'])): r for r in baseline}
    assert set(cells) == {(c, l, h) for c in range(64) for l in range(6) for h in range(9)}
    assert set(base_cells) == {(c, h) for c in range(64) for h in range(9)}
    for filename in ('controls.csv', 'cache-controls.csv'):
        rows = read_csv(results / filename)
        assert all(r['passed'] == 'True' for r in rows), filename
    for r in read_csv(results / 'controls.csv'):
        assert float(r['tolerance']) == TOL and float(r['maximum_absolute_error']) <= TOL
    for r in read_csv(results / 'cache-controls.csv'):
        assert float(r['required_unchanged_maximum_change']) == 0
    hooks = read_csv(results / 'hook-calls.csv')
    assert len(hooks) == 4224 and all(int(r['hook_calls']) == 1 for r in hooks)
    for r in scores:
        tv = float(r['total_variation'])
        assert 0 <= tv <= 1
        assert (r['tv_numerically_negligible'] == 'True') == (tv <= TOL)
        assert r['split'] == ('calibration' if int(r['word_pair']) < 4 else 'test')
    for r in read_csv(results / 'group-summary.csv'):
        values = [float(x['total_variation']) for x in scores
                  if x['split'] == r['split'] and x['layer'] == r['layer'] and x['horizon'] == r['horizon']
                  and (r['group'] == 'all' or x[r['group']] == r['group_id'])]
        assert len(values) == int(r['n_cases'])
        for key, fn in (('mean_tv', np.mean), ('median_tv', np.median), ('minimum_tv', np.min), ('maximum_tv', np.max)):
            audit.close('group_' + key, fn(values), float(r[key]), 1e-14)
    selection = read_json(results / 'selection.json')
    choices = []
    for horizon in (0, 8):
        means = np.array([np.mean([float(r['total_variation']) for r in scores
                         if r['split'] == 'calibration' and int(r['layer']) == l and int(r['horizon']) == horizon]) for l in range(5)])
        audit.close('calibration_means', means, selection['calibration_mean_tv_by_layer'][str(horizon)], 1e-14)
        choices.append(int(np.flatnonzero(means.max() - means <= 1e-10)[0]))
        if horizon == 8:
            assert selection['delayed_calibration_choice_resolved'] == bool(means.max() > TOL)
    assert choices == [selection['immediate_selected_layer'], selection['delayed_selected_layer']]
    for row in selection['group_comparisons']:
        cases = [c for c in range(64) if cells[c, 0, 8]['split'] == 'test'
                 and (row['group'] == 'all' or int(cells[c, 0, 8][row['group']]) == row['group_id'])]
        first = np.mean([float(cells[c, choices[0], 8]['total_variation']) for c in cases])
        last = np.mean([float(cells[c, choices[1], 8]['total_variation']) for c in cases])
        audit.close('selected_test_tv', [first, last, last-first],
                    [row['immediate_selected_test_tv'], row['delayed_selected_test_tv'], row['delayed_minus_immediate']], 1e-14)
        uniform = np.mean([float(cells[c, l, 8]['total_variation']) for c in cases for l in range(5)])
        audit.close('uniform_mean', uniform, row['candidate_uniform_mean_tv'], 1e-14)
        assert len(cases) == row['n_cases']
    audit.close('primary', selection['group_comparisons'][0]['delayed_minus_immediate'], selection['primary_test_mean_difference'], 0.)
    assert selection['primary_numerically_negligible'] == (abs(selection['primary_test_mean_difference']) <= TOL)
    return cells, base_cells, metadata


def replay(audit, results, model_dir, cells, base_cells):
    receipt = read_json(results / 'model-receipt.json')
    assert receipt['model_id'] == 'EleutherAI/pythia-14m'
    assert receipt['revision'] == 'cf967c0a9a04383db6f7b1108d86b2962634b4ac'
    for name, entry in receipt['files'].items():
        assert digest(model_dir / name) == entry['sha256'], name
    neox.eager_attention_forward = attention
    model = GPTNeoXForCausalLM.from_pretrained(model_dir, local_files_only=True,
                 dtype=torch.float64, attn_implementation='eager').eval()
    model.requires_grad_(False)
    ledger = read_json(results / 'inputs.json')
    assert ledger['suffix_token_ids'] == SUFFIX
    prompts, cases = ledger['prompts'], ledger['cases']
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    vectors = np.load(results / 'post-block-vectors.npz')['post_block']
    assert vectors.shape == (64, 6, 128)
    for prompt in prompts:
        assert prompt['text'] == FRAMES[prompt['frame']].format(w=PAIRS[prompt['word_pair']][prompt['side']])
        assert tokenizer.encode(prompt['text'], add_special_tokens=False) == prompt['token_ids']
        word = PAIRS[prompt['word_pair']][prompt['side']]
        assert tokenizer.encode(' '+word, add_special_tokens=False) == [prompt['target_id']]
        _, states = audit.forward(model, prompt['token_ids'], capture=True)
        audit.close('saved_post_block_vectors', np.stack([states[l] for l in range(6)]), vectors[prompt['prompt_id']])
    for case in cases:
        c, rid, did = case['case_id'], case['receiver_id'], case['donor_id']
        tokens = prompts[rid]['token_ids']
        assert prompts[rid]['word_pair'] == prompts[did]['word_pair'] == case['word_pair']
        assert rid != did and prompts[rid]['frame'] == prompts[did]['frame'] == case['frame']
        rt, dt = case['receiver_token'], case['donor_token']
        for horizon in range(9):
            full_tokens = tokens + SUFFIX[:horizon]
            base, _ = audit.forward(model, full_tokens)
            published_base = base_cells[c, horizon]
            audit.close('baseline_target_logits', base[[rt, dt]], [float(published_base['receiver_logit']), float(published_base['donor_logit'])])
            assert int(base.argmax()) == int(published_base['top1_token'])
            p = torch.softmax(torch.from_numpy(base), dim=-1).numpy()
            for layer in range(6):
                changed, _ = audit.forward(model, full_tokens, patch=(layer, len(tokens)-1, vectors[did, layer]))
                row = cells[c, layer, horizon]
                q = torch.softmax(torch.from_numpy(changed), dim=-1).numpy()
                audit.close('full_prefix_tv', .5*np.abs(p-q).sum(), float(row['total_variation']))
                audit.close('patched_target_logits', changed[[rt, dt]], [float(row['receiver_logit']), float(row['donor_logit'])])
                audit.close('margin_change', changed[dt]-changed[rt]-base[dt]+base[rt], float(row['donor_minus_receiver_margin_change']))
                assert int(changed.argmax()) == int(row['top1_token'])
                assert (row['top1_changed'] == 'True') == (changed.argmax() != base.argmax())
                norm = np.linalg.norm(vectors[did, layer]-vectors[rid, layer])
                audit.close('residual_norm', norm, float(row['residual_change_l2']), 1e-12)
                audit.close('relative_norm', norm/np.linalg.norm(vectors[rid, layer]), float(row['residual_change_relative_l2']), 1e-12)
                if layer == 5 and horizon:
                    audit.close('last_layer_future_full_logits', changed, base)
        print(json.dumps({'audited_cases': c+1, 'model_forwards': audit.calls}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--results', type=Path, default=STUDY/'results')
    parser.add_argument('--output', type=Path, default=STUDY/'reviews'/'independent-audit.json')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a new audit receipt path.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    audit, start = Audit(), time.monotonic()
    receipt = {'status': 'failed', 'human_review': False, 'imports_runner': False,
               'control_tolerance': TOL, 'audit_sha256': digest(__file__)}
    try:
        cells, base, metadata = audit_tables(audit, args.results)
        receipt['lock_commit'] = metadata['lock_commit']
        replay(audit, args.results, args.model_dir, cells, base)
        assert audit.calls == 4096
        receipt.update(status='passed', score_rows=3456, baseline_rows=576,
                       replay='full prefixes without a key-value cache', condition='double arithmetic with shared float32 rotary tables')
    except Exception as exc:
        receipt['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        receipt.update(maximum_errors=audit.errors, model_forward_calls=audit.calls,
                       elapsed_seconds=time.monotonic()-start)
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
