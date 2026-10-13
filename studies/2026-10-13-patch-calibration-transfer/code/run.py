#!/usr/bin/env python3
"""Fixed Pythia-14M patch-calibration study. No training or prompt selection."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers.models.gpt_neox import modeling_gpt_neox as neox

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'EleutherAI/pythia-14m'
REVISION = 'cf967c0a9a04383db6f7b1108d86b2962634b4ac'
MODEL_FILES = ('config.json', 'model.safetensors', 'tokenizer.json',
               'tokenizer_config.json', 'special_tokens_map.json', 'README.md')
MODEL_HASHES = {'config.json': 'f97f966a66c444890ed461fff2a51eefb15d74303df05b948124719f199b0b17', 'model.safetensors': '116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1', 'tokenizer.json': '870f4e2baa6b683221fa52004d5d6f40ab8c9d31961617304b78c910c2c3caf2', 'tokenizer_config.json': 'eee017c5bd133137f45907bd0a6e781e2ccd1a533734b7ed2a2f2f4446659809', 'special_tokens_map.json': '10b8c8852c1e1f70b54d9aff61728408c28971c0e97a6c5a7b2debbd1d3e9c0c', 'README.md': 'd1f2cf1d5181daedeaa70208ddd5cc5251867bde9acf6db7bb45a2265e25e163'}
FRAMES = (
    'The word is {w}. The word is',
    'The answer is {w}. The answer is',
    'The label is {w}. The label is',
    'The code is {w}. The code is',
    'Remember this word: {w}. The word was',
    'A card says {w}. The card says',
    'In this list, the item is {w}. The item is',
    'The name on the box is {w}. The name on the box is',
)
PAIRS = (('cat', 'dog'), ('book', 'chair'), ('apple', 'bread'), ('rain', 'snow'),
         ('red', 'blue'), ('king', 'queen'), ('car', 'train'), ('hand', 'foot'),
         ('bird', 'fish'), ('pen', 'cup'), ('milk', 'rice'), ('sun', 'moon'),
         ('green', 'white'), ('boy', 'girl'), ('bus', 'boat'), ('head', 'face'))
SPLITS = ('calibration', 'new_words', 'new_frames', 'new_words_and_frames')
CONDITIONS = ('float32', 'double_shared_rotary')
CELLS = ((0, 0), (1, 0), (0, 1), (1, 1))
STOCK_ATTENTION = neox.eager_attention_forward


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_lock(commit):
    """Require local study bytes to match the supplied immutable Git commit."""
    repo = ROOT.parents[1]
    resolved = subprocess.run(['git', 'rev-parse', commit + '^{commit}'], cwd=repo,
                              check=True, capture_output=True).stdout.decode().strip()
    if resolved != commit:
        raise RuntimeError('Lock must be a full commit SHA.')
    for path in (ROOT / 'analysis-plan.md', Path(__file__).resolve()):
        relative = path.relative_to(repo).as_posix()
        saved = subprocess.run(['git', 'show', commit + ':' + relative], cwd=repo,
                               check=True, capture_output=True).stdout
        if saved != path.read_bytes():
            raise RuntimeError('Local file differs from locked commit: ' + relative)


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


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


def load(model_dir, condition):
    neox.eager_attention_forward = STOCK_ATTENTION if condition == 'float32' else dtype_attention
    dtype = torch.float32 if condition == 'float32' else torch.float64
    model = GPTNeoXForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=dtype, attn_implementation='eager').eval()
    model.requires_grad_(False)
    assert model.config.use_parallel_residual
    assert model.config.num_hidden_layers == 6 and model.config.hidden_size == 128
    assert model.config.rope_scaling is None
    assert all(p.dtype == dtype for p in model.parameters())
    return model


def split_of(frame, pair):
    return SPLITS[(2 if frame >= 4 else 0) + (1 if pair >= 8 else 0)]


def prompt_ledger(tokenizer):
    prompts, directions = [], []
    for frame, template in enumerate(FRAMES):
        for pair, words in enumerate(PAIRS):
            pids = []
            targets = []
            tokens = []
            for side, word in enumerate(words):
                target = tokenizer.encode(' ' + word, add_special_tokens=False)
                assert len(target) == 1, (word, target)
                text = template.format(w=word)
                ids = tokenizer.encode(text, add_special_tokens=False)
                pid = len(prompts)
                pids.append(pid)
                targets.append(target[0])
                tokens.append(ids)
                prompts.append(dict(prompt_id=pid, frame=frame, word_pair=pair,
                                    side=side, split=split_of(frame, pair), word=word,
                                    text=text, token_ids=ids, target_id=target[0]))
            assert len(tokens[0]) == len(tokens[1])
            assert sum(a != b for a, b in zip(*tokens)) == 1
            assert tokens[0][-1] == tokens[1][-1]
            for receiver_side in (0, 1):
                directions.append(dict(case_id=len(directions), frame=frame, word_pair=pair,
                                       direction=receiver_side, split=split_of(frame, pair),
                                       receiver_id=pids[receiver_side], donor_id=pids[1-receiver_side],
                                       receiver_token=targets[receiver_side], donor_token=targets[1-receiver_side]))
    assert len(prompts) == len(directions) == 256
    return prompts, directions


@torch.inference_mode()
def forward(model, token_ids, capture=False, patch=None):
    """Patch branch module outputs; record only final-position values."""
    handles = []
    cache = {}
    if capture:
        def pre_hook(name):
            def hook(module, args):
                cache[name] = args[0][0, -1].detach().cpu().numpy().copy()
            return hook
        def post_hook(name, is_tuple):
            def hook(module, args, output):
                x = output[0] if is_tuple else output
                cache[name] = x[0, -1].detach().cpu().numpy().copy()
            return hook
        for layer, block in enumerate(model.gpt_neox.layers):
            handles.append(block.register_forward_pre_hook(pre_hook(f'skip_{layer}')))
            handles.append(block.register_forward_hook(post_hook(f'block_{layer}', True)))
            handles.append(block.attention.register_forward_hook(post_hook(f'attn_{layer}', True)))
            handles.append(block.mlp.register_forward_hook(post_hook(f'mlp_{layer}', False)))
    if patch is not None:
        layer, attention_value, mlp_value = patch
        block = model.gpt_neox.layers[layer]
        def replace(value, is_tuple):
            def hook(module, args, output):
                x = (output[0] if is_tuple else output).clone()
                x[0, -1] = torch.as_tensor(value, dtype=x.dtype, device=x.device)
                return (x,) + output[1:] if is_tuple else x
            return hook
        if attention_value is not None:
            handles.append(block.attention.register_forward_hook(replace(attention_value, True)))
        if mlp_value is not None:
            handles.append(block.mlp.register_forward_hook(replace(mlp_value, False)))
    def final_input(module, args):
        cache['final_residual'] = args[0][0, -1].detach().cpu().numpy().copy()
    handles.append(model.gpt_neox.final_layer_norm.register_forward_pre_hook(final_input))
    try:
        logits = model(input_ids=torch.tensor([token_ids]), use_cache=False).logits[0, -1]
        return logits.detach().cpu().numpy().copy(), cache
    finally:
        for handle in handles:
            handle.remove()


def margin_from_residual(x, gamma, beta, epsilon, rows, frozen_scale=None):
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=-1, keepdims=True)
    scale = np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + epsilon)
    if frozen_scale is not None:
        scale = frozen_scale
    out = (centered / scale * gamma + beta) @ rows.T
    return out[..., 1] - out[..., 0], np.squeeze(scale, axis=-1)


def score_summary(rows, corrections):
    groups = {}
    for row in rows:
        if row['split'] == 'calibration' or row['layer'] == 0:
            continue
        for key in [(row['split'], 'all', -1), (row['split'], 'layer', row['layer']),
                    (row['split'], 'frame', row['frame']), (row['split'], 'word_pair', row['word_pair'])]:
            groups.setdefault(key, []).append(row)
    summary = []
    def sign(x):
        return 0 if abs(x) <= 1e-6 else (1 if x > 0 else -1)
    for (split, group_type, group_id), items in groups.items():
        e0 = np.array([abs(r['interaction']) for r in items])
        e1 = np.array([abs(r['interaction'] - corrections[str(r['layer'])]) for r in items])
        pred0 = [r['attention_effect'] + r['mlp_effect'] for r in items]
        pred1 = [p + corrections[str(r['layer'])] for p, r in zip(pred0, items)]
        summary.append(dict(split=split, group_type=group_type, group_id=group_id, n_cells=len(items),
                            uncorrected_mae=float(e0.mean()), corrected_mae=float(e1.mean()),
                            mae_gain=float((e0-e1).mean()), median_absolute_interaction=float(np.median(e0)),
                            minimum_absolute_interaction=float(e0.min()), maximum_absolute_interaction=float(e0.max()),
                            improved_fraction=float(np.mean(e1 < e0)),
                            uncorrected_sign_error=float(np.mean([sign(p) != sign(r['joint_effect']) for p, r in zip(pred0, items)])),
                            corrected_sign_error=float(np.mean([sign(p) != sign(r['joint_effect']) for p, r in zip(pred1, items)]))))
    main = [r for r in rows if r['split'] == 'new_words_and_frames' and r['layer'] > 0]
    gain = np.zeros((4, 8, 2, 5))
    for r in main:
        gain[r['frame']-4, r['word_pair']-8, r['direction'], r['layer']-1] = (
            abs(r['interaction']) - abs(r['interaction']-corrections[str(r['layer'])]))
    rng = np.random.default_rng(20261013)
    draws = np.empty(10000)
    for i in range(len(draws)):
        f = rng.integers(0, 4, 4)
        p = rng.integers(0, 8, 8)
        draws[i] = gain[f[:, None], p[None, :]].mean()
    interval = dict(low=float(np.quantile(draws, .025)), high=float(np.quantile(draws, .975)),
                    repetitions=10000, seed=20261013,
                    scope='Reweighting fixed test frames and word pairs. Calibration correction stays fixed.')
    return summary, interval


def run_condition(model_dir, tokenizer, prompts, directions, condition, out):
    started = time.perf_counter()
    model = load(model_dir, condition)
    ndtype = np.float32 if condition == 'float32' else np.float64
    tolerance = 1e-4 if condition == 'float32' else 1e-9
    out.mkdir()
    branches = np.empty((256, 6, 2, 128), dtype=ndtype)
    skips = np.empty((256, 6, 128), dtype=ndtype)
    blocks = np.empty((256, 6, 128), dtype=ndtype)
    final_residuals = np.empty((256, 4, 128), dtype=ndtype)
    target_logits = np.empty((256, 6, 4, 2), dtype=ndtype)
    word_tokens = sorted({p['target_id'] for p in prompts})
    norm = model.gpt_neox.final_layer_norm
    gamma = norm.weight.detach().cpu().numpy().copy()
    beta = norm.bias.detach().cpu().numpy().copy()
    epsilon = float(norm.eps)
    readout = model.embed_out.weight.detach().cpu().numpy()[word_tokens].copy()
    token_rows = {tok: readout[i] for i, tok in enumerate(word_tokens)}
    baseline_logits = {}
    baseline_records, rows = [], []
    corrections = None
    controls = dict(max_layer0_mlp_difference=0.0, max_layer0_interaction=0.0,
                    max_last_layer_reconstruction_error=0.0,
                    max_frozen_denominator_interaction=0.0,
                    max_last_layer_additive_residual_error=0.0,
                    absolute_tolerance=tolerance)
    for split in SPLITS:
        for p in prompts:
            if p['split'] != split:
                continue
            pid = p['prompt_id']
            logits, cache = forward(model, p['token_ids'], capture=True)
            baseline_logits[pid] = logits
            for layer in range(6):
                branches[pid, layer, 0] = cache[f'attn_{layer}']
                branches[pid, layer, 1] = cache[f'mlp_{layer}']
                skips[pid, layer] = cache[f'skip_{layer}']
                blocks[pid, layer] = cache[f'block_{layer}']
            other = prompts[pid ^ 1]['target_id']
            baseline_records.append(dict(prompt_id=pid, split=split, frame=p['frame'],
                                          word_pair=p['word_pair'], target_logit=float(logits[p['target_id']]),
                                          alternative_logit=float(logits[other]),
                                          pairwise_target_preferred=bool(logits[p['target_id']] > logits[other]),
                                          target_top1=bool(int(np.argmax(logits)) == p['target_id']),
                                          top1_token_id=int(np.argmax(logits))))
        for case in directions:
            if case['split'] != split:
                continue
            cid, rid, did = case['case_id'], case['receiver_id'], case['donor_id']
            rt, dt = case['receiver_token'], case['donor_token']
            difference = float(np.max(np.abs(branches[rid, 0, 1] - branches[did, 0, 1])))
            controls['max_layer0_mlp_difference'] = max(controls['max_layer0_mlp_difference'], difference)
            for layer in range(6):
                target_logits[cid, layer, 0] = baseline_logits[rid][[rt, dt]]
                if layer == 5:
                    final_residuals[cid, 0] = blocks[rid, 5]
                for cell, (a, m) in enumerate(CELLS[1:], 1):
                    av = branches[did, layer, 0] if a else None
                    mv = branches[did, layer, 1] if m else None
                    logits, cache = forward(model, prompts[rid]['token_ids'], patch=(layer, av, mv))
                    target_logits[cid, layer, cell] = logits[[rt, dt]]
                    if layer == 5:
                        final_residuals[cid, cell] = cache['final_residual']
                raw = target_logits[cid, layer].astype(np.float64)
                scores = raw[:, 1] - raw[:, 0]
                attn, mlp, joint = scores[1:]-scores[0]
                interaction = joint - attn - mlp
                row = dict(case, layer=layer, condition=condition)
                for cell in range(4):
                    name = ''.join(map(str, CELLS[cell]))
                    row[f'receiver_logit_{name}'] = float(raw[cell, 0])
                    row[f'donor_logit_{name}'] = float(raw[cell, 1])
                    row[f'margin_{name}'] = float(scores[cell])
                row.update(attention_effect=float(attn), mlp_effect=float(mlp), joint_effect=float(joint),
                           interaction=float(interaction))
                rows.append(row)
                if layer == 0:
                    controls['max_layer0_interaction'] = max(controls['max_layer0_interaction'], abs(interaction))
                if layer == 5:
                    rr = np.stack([token_rows[rt], token_rows[dt]])
                    exact, scale = margin_from_residual(final_residuals[cid], gamma, beta, epsilon, rr)
                    frozen, _ = margin_from_residual(final_residuals[cid], gamma, beta, epsilon, rr, scale[0])
                    recerr = float(np.max(np.abs(exact-scores)))
                    fint = float(frozen[3]-frozen[1]-frozen[2]+frozen[0])
                    residual = final_residuals[cid].astype(np.float64)
                    rerr = float(np.max(np.abs(residual[3]-residual[1]-residual[2]+residual[0])))
                    for name, value in [('max_last_layer_reconstruction_error', recerr),
                                        ('max_frozen_denominator_interaction', abs(fint)),
                                        ('max_last_layer_additive_residual_error', rerr)]:
                        controls[name] = max(controls[name], value)
        if split == 'calibration':
            corrections = {str(layer): float(np.median([r['interaction'] for r in rows if r['layer'] == layer]))
                           for layer in range(6)}
            dump(out / 'calibration-corrections.json', dict(corrections=corrections,
                 estimator='median', n_directed_cases_per_layer=64,
                 status='Frozen before any held-out forward in this numerical condition.'))
        csv_write(out / 'four-cell-results.csv', rows)
        csv_write(out / 'intact-behavior.csv', baseline_records)
        print(json.dumps(dict(condition=condition, completed_split=split, seconds=time.perf_counter()-started)), flush=True)
        if time.perf_counter()-started > 1200:
            dump(out / 'failure.json', dict(reason='20-minute condition runtime cap reached', split=split))
            raise RuntimeError('Runtime cap reached. Preserve outputs; no dataset reduction.')
    assert corrections is not None and len(rows) == 1536 and len(baseline_records) == 256
    for row in rows:
        c = corrections[str(row['layer'])]
        row['calibration_correction'] = c
        row['uncorrected_prediction'] = row['attention_effect']+row['mlp_effect']
        row['corrected_prediction'] = row['uncorrected_prediction']+c
        row['uncorrected_absolute_error'] = abs(row['interaction'])
        row['corrected_absolute_error'] = abs(row['interaction']-c)
    csv_write(out / 'four-cell-results.csv', rows)
    csv_write(out / 'intact-behavior.csv', baseline_records)
    np.savez_compressed(out / 'activations.npz', branches=branches, skips=skips, blocks=blocks,
                        final_residuals=final_residuals, target_logits=target_logits,
                        gamma=gamma, beta=beta, epsilon=epsilon,
                        token_ids=np.array(word_tokens), unembedding_rows=readout)
    summaries, interval = score_summary(rows, corrections)
    csv_write(out / 'summary.csv', summaries)
    dump(out / 'bootstrap.json', interval)
    controls['passed'] = all(v <= tolerance for k, v in controls.items() if k.startswith('max_'))
    dump(out / 'controls.json', controls)
    dump(out / 'run-metadata.json', dict(condition=condition, elapsed_seconds=time.perf_counter()-started,
          precision='float32' if condition == 'float32' else 'float64 parameters and arithmetic; shared float32 rotary tables',
          prompt_count=256, directed_cases=256, layers=6, primary_layers=[1, 2, 3, 4, 5],
          forwards=256+256*6*3, model_id=MODEL_ID, revision=REVISION))
    if not controls['passed']:
        raise RuntimeError('Numerical control failed. Keep results; stop scientific interpretation.')
    del model
    return rows, summaries


def preflight(model_dir):
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompts, directions = prompt_ledger(tokenizer)
    checks = dict(tokenizer=dict(prompts=len(prompts), directions=len(directions),
                               equal_length_pairs=True, one_changed_token=True, equal_final_tokens=True))
    # These dummy strings are not study prompts. Do not inspect their scientific scores.
    texts = ('A small cat is here.', 'A small dog is here.')
    ids = [tokenizer.encode(s, add_special_tokens=False) for s in texts]
    assert len(ids[0]) == len(ids[1]) and ids[0][-1] == ids[1][-1]
    for condition in CONDITIONS:
        model = load(model_dir, condition)
        tolerance = 1e-4 if condition == 'float32' else 1e-9
        logs, caches = zip(*(forward(model, x, capture=True) for x in ids))
        noop, _ = forward(model, ids[0], patch=(3, caches[0]['attn_3'], caches[0]['mlp_3']))
        zero, _ = forward(model, ids[0], patch=(0, None, caches[1]['mlp_0']))
        max_noop = float(np.max(np.abs(noop.astype(float)-logs[0].astype(float))))
        max_zero = float(np.max(np.abs(zero.astype(float)-logs[0].astype(float))))
        norm = model.gpt_neox.final_layer_norm
        target_ids = [tokenizer.encode(' cat', add_special_tokens=False)[0], tokenizer.encode(' dog', add_special_tokens=False)[0]]
        readout = model.embed_out.weight.detach().cpu().numpy()[target_ids]
        gamma, beta = norm.weight.detach().cpu().numpy(), norm.bias.detach().cpu().numpy()
        residuals, margins = [], []
        for a, m in CELLS:
            log, cache = forward(model, ids[0], patch=(5, caches[1]['attn_5'] if a else None, caches[1]['mlp_5'] if m else None))
            residuals.append(cache['final_residual'])
            margins.append(float(log[target_ids[1]])-float(log[target_ids[0]]))
        residuals = np.array(residuals)
        exact, scales = margin_from_residual(residuals, gamma, beta, norm.eps, readout)
        frozen, _ = margin_from_residual(residuals, gamma, beta, norm.eps, readout, scales[0])
        controls = dict(noop_max_logit_error=max_noop, layer0_mlp_max_logit_error=max_zero,
                        last_norm_max_margin_error=float(np.max(np.abs(exact-margins))),
                        frozen_denominator_interaction=abs(float(frozen[3]-frozen[1]-frozen[2]+frozen[0])))
        assert max(controls.values()) <= tolerance, controls
        checks[condition] = controls
        del model
    checks['scope'] = 'Tokenizer checks and numerical controls on two dummy strings only; no scientific outcomes.'
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--lock-commit')
    args = parser.parse_args()
    for name, expected in MODEL_HASHES.items():
        if sha(args.model_dir/name) != expected:
            raise RuntimeError(f'Pinned model file hash mismatch: {name}')
    if args.preflight:
        print(json.dumps(preflight(args.model_dir), indent=2))
        return
    if not args.run or not args.lock_commit or len(args.lock_commit) != 40:
        parser.error('Use --preflight or --run --lock-commit FULL_SHA.')
    verify_lock(args.lock_commit)
    out = ROOT / 'results'
    if out.exists():
        raise RuntimeError('Results already exist. Do not overwrite or resume silently.')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    prompts, directions = prompt_ledger(tokenizer)
    out.mkdir()
    dump(out / 'prompts.json', prompts)
    dump(out / 'directions.json', directions)
    dump(out / 'provenance.json', dict(model_id=MODEL_ID, revision=REVISION,
         model_files={name: dict(sha256=sha(args.model_dir/name), bytes=(args.model_dir/name).stat().st_size) for name in MODEL_FILES},
         lock_commit=args.lock_commit, plan_sha256=sha(ROOT/'analysis-plan.md'), runner_sha256=sha(__file__),
         python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
         torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__, cpu_threads=2,
         packages={name: version(name) for name in ('torch', 'transformers', 'numpy', 'safetensors', 'huggingface-hub', 'tokenizers')},
         dependency_policy='Exact installed versions are recorded; requirements.txt pins the tested direct dependencies.'))
    primary, psum = run_condition(args.model_dir, tokenizer, prompts, directions, CONDITIONS[0], out/CONDITIONS[0])
    sensitivity, ssum = run_condition(args.model_dir, tokenizer, prompts, directions, CONDITIONS[1], out/CONDITIONS[1])
    difference = np.array([s['interaction']-p['interaction'] for p, s in zip(primary, sensitivity)])
    assert all((p['case_id'], p['layer']) == (s['case_id'], s['layer']) for p, s in zip(primary, sensitivity))
    primary_main = next(s for s in psum if s['split'] == 'new_words_and_frames' and s['group_type'] == 'all')
    sensitivity_main = next(s for s in ssum if s['split'] == 'new_words_and_frames' and s['group_type'] == 'all')
    def sign(x):
        return 0 if abs(x) <= 1e-6 else (1 if x > 0 else -1)
    dump(out / 'precision-comparison.json', dict(max_absolute_interaction_difference=float(np.max(np.abs(difference))),
          median_absolute_interaction_difference=float(np.median(np.abs(difference))),
          interaction_sign_change_count=sum(sign(p['interaction']) != sign(s['interaction']) for p, s in zip(primary, sensitivity)),
          number_of_layer_case_cells=len(primary), primary_main=primary_main, sensitivity_main=sensitivity_main,
          primary_mae_gain_change=sensitivity_main['mae_gain']-primary_main['mae_gain']))
    verify_lock(args.lock_commit)
    files = [p for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json']
    dump(out/'manifest.json', {str(p.relative_to(ROOT)): dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(files)})
    print(json.dumps(dict(status='complete', primary=primary_main, sensitivity=sensitivity_main)), flush=True)


if __name__ == '__main__':
    main()
