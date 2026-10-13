#!/usr/bin/env python3
"""Fixed temporal patch benchmark. Scientific outcomes require a Git lock."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.gpt_neox import modeling_gpt_neox as neox

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'EleutherAI/pythia-14m'
REVISION = 'cf967c0a9a04383db6f7b1108d86b2962634b4ac'
MODEL_HASHES = {
    'config.json': 'f97f966a66c444890ed461fff2a51eefb15d74303df05b948124719f199b0b17',
    'model.safetensors': '116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1',
    'tokenizer.json': '870f4e2baa6b683221fa52004d5d6f40ab8c9d31961617304b78c910c2c3caf2',
    'tokenizer_config.json': 'eee017c5bd133137f45907bd0a6e781e2ccd1a533734b7ed2a2f2f4446659809',
    'special_tokens_map.json': '10b8c8852c1e1f70b54d9aff61728408c28971c0e97a6c5a7b2debbd1d3e9c0c',
    'README.md': 'd1f2cf1d5181daedeaa70208ddd5cc5251867bde9acf6db7bb45a2265e25e163',
}
FRAMES = ('The entry is {w}. Copy the entry:',
          'The code is {w}. Copy the code:',
          'The password is {w}. Copy the password:',
          'The label is {w}. Copy the label:')
PAIRS = (('fox', 'wolf'), ('tree', 'leaf'), ('glass', 'plate'), ('rock', 'sand'),
         ('gold', 'silver'), ('door', 'wall'), ('road', 'path'), ('fire', 'wind'))
TARGETS = ((30013, 25872), (5202, 10617), (5253, 5340), (5561, 7537),
           (5328, 9711), (3369, 3402), (3971, 1854), (3289, 5448))
SUFFIX_TEXT = ' The record will be checked again after this'
SUFFIX = [380, 1924, 588, 320, 10141, 969, 846, 436]
TOL = 1e-8
TIE = 1e-10
STOCK_ATTENTION = neox.eager_attention_forward


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(x):
    return hashlib.sha256(np.asarray(x).tobytes(order='C')).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def verify_lock(commit):
    repo = ROOT.parents[1]
    resolved = subprocess.run(['git', 'rev-parse', commit + '^{commit}'], cwd=repo,
                              check=True, capture_output=True).stdout.decode().strip()
    if resolved != commit:
        raise RuntimeError('Use the full immutable lock commit SHA.')
    for path in (ROOT / 'analysis-plan.md', Path(__file__).resolve()):
        relative = path.relative_to(repo).as_posix()
        saved = subprocess.run(['git', 'show', commit + ':' + relative], cwd=repo,
                               check=True, capture_output=True).stdout
        if saved != path.read_bytes():
            raise RuntimeError('Local file differs from lock: ' + relative)


def verify_model(model_dir):
    receipt = {}
    for name, expected in MODEL_HASHES.items():
        path = Path(model_dir) / name
        actual = sha(path)
        if actual != expected:
            raise RuntimeError('Model file hash mismatch: ' + name)
        receipt[name] = {'sha256': actual, 'bytes': path.stat().st_size}
    return receipt


class Table:
    """Append and flush each record so failed runs retain partial work."""
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open('w', newline='')
        self.writer = None
        self.rows = []

    def add(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()
        self.rows.append(row)

    def close(self):
        self.file.close()


def dtype_attention(module, query, key, value, attention_mask, scaling,
                    dropout=0.0, head_mask=None, **kwargs):
    """Pinned eager attention with only the softmax dtype changed."""
    assert query.dtype == key.dtype == value.dtype == torch.float64
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, :key.shape[-2]]
    weights = torch.nn.functional.softmax(weights, dim=-1, dtype=query.dtype)
    if head_mask is not None:
        weights = weights * head_mask
    weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value).transpose(1, 2).contiguous()
    assert output.dtype == torch.float64
    return output, weights


def load_model(model_dir):
    neox.eager_attention_forward = dtype_attention
    model = GPTNeoXForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float64,
        attn_implementation='eager').eval()
    model.requires_grad_(False)
    assert model.config.use_parallel_residual
    assert model.config.num_hidden_layers == 6 and model.config.hidden_size == 128
    assert model.config.rope_scaling is None and model.config.vocab_size == 50304
    assert all(p.dtype == torch.float64 for p in model.parameters())
    return model


def ledger(tokenizer):
    prompts, cases = [], []
    for frame, template in enumerate(FRAMES):
        for pair, words in enumerate(PAIRS):
            ids, pids = [], []
            split = 'calibration' if pair < 4 else 'test'
            for side, word in enumerate(words):
                assert tokenizer.encode(' ' + word, add_special_tokens=False) == [TARGETS[pair][side]]
                text = template.format(w=word)
                tokens = tokenizer.encode(text, add_special_tokens=False)
                pids.append(len(prompts))
                ids.append(tokens)
                prompts.append(dict(prompt_id=len(prompts), frame=frame, word_pair=pair,
                                    side=side, split=split, text=text, token_ids=tokens,
                                    target_id=TARGETS[pair][side]))
            assert len(ids[0]) == len(ids[1]) and ids[0][-1] == ids[1][-1]
            assert sum(a != b for a, b in zip(*ids)) == 1
            for side in (0, 1):
                cases.append(dict(case_id=len(cases), frame=frame, word_pair=pair,
                                  direction=side, split=split, receiver_id=pids[side],
                                  donor_id=pids[1-side], receiver_token=TARGETS[pair][side],
                                  donor_token=TARGETS[pair][1-side]))
    assert tokenizer.encode(SUFFIX_TEXT, add_special_tokens=False) == SUFFIX
    assert tokenizer.decode(SUFFIX) == SUFFIX_TEXT
    assert len(prompts) == len(cases) == 64
    return prompts, cases


def cache_copy(cache):
    return DynamicCache.from_legacy_cache(tuple(
        (key.clone(), value.clone()) for key, value in cache.to_legacy_cache()))


class Engine:
    def __init__(self, model, deadline, hook_table=None):
        self.model, self.deadline, self.hook_table = model, deadline, hook_table
        self.forwards = 0

    @torch.inference_mode()
    def forward(self, ids, use_cache=False, cache=None, patch=None, capture=False,
                tag=''):
        if time.monotonic() > self.deadline:
            raise TimeoutError('The fixed 20-minute scientific runtime limit was reached.')
        handles, vectors, hits = [], {}, []
        if patch is not None:
            layer, position, vector = patch
            assert 0 <= position < len(ids)
            assert cache is None, 'A hook must never be installed on an incremental suffix pass.'
            def replace(module, args, output):
                value = output[0].clone()
                value[0, position] = torch.as_tensor(vector, dtype=value.dtype)
                hits.append((value.shape[1], position))
                return (value,) + output[1:]
            handles.append(self.model.gpt_neox.layers[layer].register_forward_hook(replace))
        if capture:
            for layer, block in enumerate(self.model.gpt_neox.layers):
                def save(module, args, output, layer=layer):
                    vectors[layer] = output[0][0, -1].detach().numpy().copy()
                handles.append(block.register_forward_hook(save))
        try:
            result = self.model(input_ids=torch.tensor([ids]), use_cache=use_cache,
                                past_key_values=cache, logits_to_keep=1)
            self.forwards += 1
            logits = result.logits[0, -1].detach().numpy().copy()
            if logits.shape != (50304,) or not np.isfinite(logits).all():
                raise RuntimeError('Invalid full-vocabulary output.')
            if patch is not None:
                assert hits == [(len(ids), patch[1])]
                if self.hook_table is not None:
                    self.hook_table.add(dict(tag=tag, layer=patch[0], sequence_length=len(ids),
                                             patch_position=patch[1], hook_calls=len(hits)))
            return logits, result.past_key_values, vectors
        finally:
            for handle in handles:
                handle.remove()

    def suffix(self, cache, suffix=SUFFIX):
        fresh = cache_copy(cache)
        result = []
        for token in suffix:
            logits, fresh, _ = self.forward([token], use_cache=True, cache=fresh)
            result.append(logits)
        return result


def probability(logits):
    weights = np.exp(logits - np.max(logits))
    return weights / weights.sum()


def check(table, kind, case_id, layer, horizon, left, right, tolerance=TOL):
    error = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
    table.add(dict(kind=kind, case_id=case_id, layer=layer, horizon=horizon,
                   maximum_absolute_error=error, tolerance=tolerance, passed=error <= tolerance))


def cache_checks(table, case_id, layer, baseline, changed, kind):
    for cache_layer, (base_values, changed_values) in enumerate(zip(
            baseline.to_legacy_cache(), changed.to_legacy_cache())):
        for component, base, new in zip(('key', 'value'), base_values, changed_values):
            assert base.shape == new.shape and base.dtype == new.dtype == torch.float64
            delta = (new - base).abs()
            whole = float(delta.max())
            early = float(delta[:, :, :-1, :].max())
            last = float(delta[:, :, -1, :].max())
            required = whole if kind == 'self' or cache_layer <= layer else early
            table.add(dict(kind=kind, case_id=case_id, patch_layer=layer, cache_layer=cache_layer,
                           component=component, shape='x'.join(map(str, base.shape)),
                           maximum_change=whole, earlier_position_maximum_change=early,
                           final_position_maximum_change=last,
                           required_unchanged_maximum_change=required, passed=required == 0.0,
                           baseline_sha256=array_sha(base.numpy()), patched_sha256=array_sha(new.numpy())))


def summarize(rows, out):
    groups = {}
    for row in rows:
        for group, index in (('all', -1), ('frame', row['frame']), ('word_pair', row['word_pair'])):
            key = (row['split'], group, index, row['layer'], row['horizon'])
            groups.setdefault(key, []).append(row['total_variation'])
    table = Table(out / 'group-summary.csv')
    for (split, group, index, layer, horizon), values in sorted(groups.items()):
        table.add(dict(split=split, group=group, group_id=index, layer=layer, horizon=horizon,
                       n_cases=len(values), mean_tv=float(np.mean(values)), median_tv=float(np.median(values)),
                       minimum_tv=float(np.min(values)), maximum_tv=float(np.max(values))))
    table.close()
    calibration = {h: [float(np.mean([r['total_variation'] for r in rows
                        if r['split'] == 'calibration' and r['layer'] == layer and r['horizon'] == h]))
                       for layer in range(5)] for h in (0, 8)}
    def choose(values):
        maximum = max(values)
        return next(i for i, value in enumerate(values) if maximum - value <= TIE)
    immediate, delayed = choose(calibration[0]), choose(calibration[8])
    tests = [r for r in rows if r['split'] == 'test' and r['horizon'] == 8]
    differences = []
    for group, indices in (('all', [-1]), ('frame', range(4)), ('word_pair', range(4, 8))):
        for index in indices:
            included = [r for r in tests if group == 'all' or r[group] == index]
            first = np.mean([r['total_variation'] for r in included if r['layer'] == immediate])
            last = np.mean([r['total_variation'] for r in included if r['layer'] == delayed])
            uniform = np.mean([r['total_variation'] for r in included if r['layer'] < 5])
            differences.append(dict(group=group, group_id=index,
                                    n_cases=len([r for r in included if r['layer'] == 0]),
                                    immediate_selected_test_tv=float(first), delayed_selected_test_tv=float(last),
                                    delayed_minus_immediate=float(last-first), candidate_uniform_mean_tv=float(uniform)))
    primary = differences[0]['delayed_minus_immediate']
    result = dict(calibration_mean_tv_by_layer={str(k): v for k, v in calibration.items()},
                  candidate_layers=list(range(5)), immediate_selected_layer=immediate,
                  delayed_selected_layer=delayed, tie_tolerance=TIE, numerical_label_threshold=TOL,
                  delayed_calibration_choice_resolved=max(calibration[8]) > TOL,
                  primary_test_mean_difference=primary, primary_numerically_negligible=abs(primary) <= TOL,
                  group_comparisons=differences)
    dump(out / 'selection.json', result)
    return result


def preflight(model_dir):
    """No scientific model outputs: token checks plus separate dummy strings."""
    verify_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompts, cases = ledger(tokenizer)
    model = load_model(model_dir)
    engine = Engine(model, time.monotonic() + 180)
    ids = tokenizer.encode('A quiet test:', add_special_tokens=False)
    donor_ids = tokenizer.encode('A simple test:', add_special_tokens=False)
    dummy_suffix = tokenizer.encode(' one two', add_special_tokens=False)
    baseline, base_cache, base_vectors = engine.forward(ids, use_cache=True, capture=True)
    _, _, donor_vectors = engine.forward(donor_ids, capture=True)
    plain = [baseline] + engine.suffix(base_cache, dummy_suffix)
    max_error = 0.0
    for layer in range(6):
        patch = (layer, len(ids)-1, donor_vectors[layer])
        immediate, changed_cache, _ = engine.forward(ids, use_cache=True, patch=patch)
        changed = [immediate] + engine.suffix(changed_cache, dummy_suffix)
        self_logits, _, _ = engine.forward(ids, patch=(layer, len(ids)-1, base_vectors[layer]))
        assert np.max(np.abs(self_logits-baseline)) <= TOL
        for h in range(len(changed)):
            full, _, _ = engine.forward(ids + dummy_suffix[:h], patch=patch)
            max_error = max(max_error, float(np.max(np.abs(full-changed[h]))))
            if layer == 5 and h > 0:
                assert np.max(np.abs(changed[h]-plain[h])) <= TOL
    assert max_error <= TOL
    print(json.dumps(dict(status='preflight_passed', study_prompts_tokenized=len(prompts),
                          directed_cases=len(cases), dummy_forward_calls=engine.forwards,
                          dummy_cache_recompute_maximum_error=max_error,
                          scientific_model_outputs_collected=False)))


def run(model_dir, lock_commit, out):
    verify_lock(lock_commit)
    receipt = verify_model(model_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    tables = {name: Table(out / (name + '.csv')) for name in
              ('scores', 'baseline', 'controls', 'cache-controls', 'hook-calls')}
    engine = None
    status = 'failed'
    try:
        dump(out / 'model-receipt.json', dict(model_id=MODEL_ID, revision=REVISION, files=receipt))
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        prompts, cases = ledger(tokenizer)
        dump(out / 'inputs.json', dict(prompts=prompts, cases=cases, suffix_text=SUFFIX_TEXT,
                                       suffix_token_ids=SUFFIX, target_ids=TARGETS))
        model = load_model(model_dir)
        engine = Engine(model, started + 1200, tables['hook-calls'])
        prefills = []
        residuals = []
        for prompt in prompts:
            logits, cache, vectors = engine.forward(prompt['token_ids'], use_cache=True, capture=True)
            prefills.append((logits, cache))
            residuals.append(np.stack([vectors[layer] for layer in range(6)]))
        residuals = np.stack(residuals)
        np.savez_compressed(out / 'post-block-vectors.npz', post_block=residuals)
        for case in cases:
            c = case['case_id']
            receiver, donor = case['receiver_id'], case['donor_id']
            ids = prompts[receiver]['token_ids']
            position = len(ids) - 1
            baseline_logits, baseline_cache = prefills[receiver]
            baseline = [baseline_logits] + engine.suffix(baseline_cache)
            probs = [probability(logits) for logits in baseline]
            rt, dt = case['receiver_token'], case['donor_token']
            for h, logits in enumerate(baseline):
                tables['baseline'].add(dict(**case, horizon=h, receiver_logit=float(logits[rt]),
                                            donor_logit=float(logits[dt]), top1_token=int(np.argmax(logits)),
                                            receiver_target_is_top1=bool(np.argmax(logits) == rt),
                                            logits_sha256=array_sha(logits)))
                full, _, _ = engine.forward(ids + SUFFIX[:h])
                check(tables['controls'], 'baseline_recompute', c, -1, h, full, logits)
            for layer in range(6):
                donor_vector, receiver_vector = residuals[donor, layer], residuals[receiver, layer]
                patch = (layer, position, donor_vector)
                changed_logits, changed_cache, _ = engine.forward(
                    ids, use_cache=True, patch=patch, tag=f'patch:{c}:{layer}:0')
                cache_checks(tables['cache-controls'], c, layer, baseline_cache, changed_cache, 'patch')
                changed = [changed_logits] + engine.suffix(changed_cache)
                self_logits, self_cache, _ = engine.forward(
                    ids, use_cache=True, patch=(layer, position, receiver_vector), tag=f'self:{c}:{layer}:0')
                cache_checks(tables['cache-controls'], c, layer, baseline_cache, self_cache, 'self')
                self_trajectory = [self_logits] + engine.suffix(self_cache)
                reset_trajectory = engine.suffix(baseline_cache)
                residual_norm = float(np.linalg.norm(donor_vector - receiver_vector))
                relative_norm = residual_norm / float(np.linalg.norm(receiver_vector))
                for h, logits in enumerate(changed):
                    check(tables['controls'], 'self_patch', c, layer, h, self_trajectory[h], baseline[h])
                    if h > 0:
                        check(tables['controls'], 'cache_reset', c, layer, h, reset_trajectory[h-1], baseline[h])
                        if layer == 5:
                            check(tables['controls'], 'final_layer_future', c, layer, h, logits, baseline[h])
                    full, _, _ = engine.forward(ids + SUFFIX[:h], patch=patch,
                                                 tag=f'recompute:{c}:{layer}:{h}')
                    check(tables['controls'], 'patch_recompute', c, layer, h, full, logits)
                    score = float(0.5 * np.abs(probability(logits) - probs[h]).sum())
                    assert 0.0 <= score <= 1.0
                    baseline_margin = baseline[h][dt] - baseline[h][rt]
                    margin = logits[dt] - logits[rt]
                    tables['scores'].add(dict(**case, layer=layer, horizon=h, total_variation=score,
                                              tv_numerically_negligible=score <= TOL,
                                              receiver_logit=float(logits[rt]), donor_logit=float(logits[dt]),
                                              donor_minus_receiver_margin_change=float(margin-baseline_margin),
                                              top1_token=int(np.argmax(logits)),
                                              top1_changed=bool(np.argmax(logits) != np.argmax(baseline[h])),
                                              residual_change_l2=residual_norm,
                                              residual_change_relative_l2=relative_norm,
                                              logits_sha256=array_sha(logits)))
            print(json.dumps(dict(completed_cases=c+1, total_cases=64,
                                  elapsed_seconds=round(time.monotonic()-started, 2))), flush=True)
        assert len(tables['scores'].rows) == 64 * 6 * 9
        assert len(tables['baseline'].rows) == 64 * 9
        assert len(tables['hook-calls'].rows) == 64 * 6 * 11
        assert engine.forwards == 14592
        selection = summarize(tables['scores'].rows, out)
        failed_logits = [row for row in tables['controls'].rows if not row['passed']]
        failed_cache = [row for row in tables['cache-controls'].rows if not row['passed']]
        dump(out / 'control-summary.json', dict(logit_checks=len(tables['controls'].rows),
                                                failed_logit_checks=len(failed_logits),
                                                cache_checks=len(tables['cache-controls'].rows),
                                                failed_cache_checks=len(failed_cache),
                                                maximum_logit_error=max(r['maximum_absolute_error'] for r in tables['controls'].rows),
                                                tolerance=TOL, exact_cache_tolerance=0.0))
        if failed_logits or failed_cache:
            raise RuntimeError('A fixed control failed. Preserve the full run for a separate audit.')
        verify_lock(lock_commit)
        verify_model(model_dir)
        status = 'complete'
        print(json.dumps(dict(status=status, primary=selection['primary_test_mean_difference'])))
    except Exception as exc:
        dump(out / 'failure.json', dict(exception=type(exc).__name__, message=str(exc),
                                       completed_score_rows=len(tables['scores'].rows),
                                       fixed_tolerances_unchanged=True))
        raise
    finally:
        for table in tables.values():
            table.close()
        metadata = dict(status=status, lock_commit=lock_commit,
                        plan_sha256=sha(ROOT / 'analysis-plan.md'), runner_sha256=sha(__file__),
                        elapsed_seconds=time.monotonic()-started,
                        model_forward_calls=0 if engine is None else engine.forwards,
                        cpu_threads=torch.get_num_threads(), machine=platform.machine(),
                        system=platform.system(), python=platform.python_version(),
                        packages={name: version(name) for name in ('torch', 'transformers', 'numpy', 'safetensors')},
                        arithmetic='float64 operations with shared stock float32 rotary tables',
                        scientific_condition_count=1, human_review=False)
        dump(out / 'run-metadata.json', metadata)
        dump(out / 'manifest.json', dict(status=status, files={
            path.name: {'sha256': sha(path), 'bytes': path.stat().st_size}
            for path in sorted(out.iterdir()) if path.is_file() and path.name != 'manifest.json'}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--lock-commit')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.preflight:
        if args.lock_commit:
            parser.error('Preflight does not accept a scientific lock.')
        preflight(args.model_dir)
    else:
        if not args.lock_commit:
            parser.error('Scientific runs require --lock-commit.')
        run(args.model_dir, args.lock_commit, args.output)


if __name__ == '__main__':
    main()
