#!/usr/bin/env python3
"""Fixed position/access factorial. Scientific outputs require a public Git lock."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM
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
PACKAGES = {'numpy': '2.3.5', 'torch': '2.8.0', 'transformers': '4.57.6',
            'safetensors': '0.6.2', 'matplotlib': '3.10.8'}
LOGIT_TOL, STATE_TOL, ROW_TOL = 1e-8, 1e-10, 1e-12
GROUP_SEED, FILLER_SEED, BOOTSTRAP_SEED = 2026101401, 2026101402, 2026101403
GAPS = (0, 32, 128)
HEAD_LAYER, HEAD_INDEX = 3, 1
BATCH_SIZE, TIME_LIMIT = 4, 1200


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def verify_lock(commit):
    repo = ROOT.parents[1]
    resolved = subprocess.run(['git', 'rev-parse', commit + '^{commit}'], cwd=repo,
                              check=True, capture_output=True).stdout.decode().strip()
    if resolved != commit:
        raise RuntimeError('The lock must be a full commit SHA.')
    for path in (ROOT / 'analysis-plan.md', Path(__file__).resolve()):
        rel = path.relative_to(repo).as_posix()
        saved = subprocess.run(['git', 'show', commit + ':' + rel], cwd=repo,
                               check=True, capture_output=True).stdout
        if saved != path.read_bytes():
            raise RuntimeError('Local file differs from lock: ' + rel)


def verify_environment(model_dir):
    actual_versions = {name: version(name) for name in PACKAGES}
    if actual_versions != PACKAGES:
        raise RuntimeError('Package versions differ from the fixed requirements.')
    receipt = {}
    for name, expected in MODEL_HASHES.items():
        path = model_dir / name
        if sha(path) != expected:
            raise RuntimeError('Model file hash mismatch: ' + name)
        receipt[name] = dict(sha256=expected, bytes=path.stat().st_size)
    return dict(model_id=MODEL_ID, revision=REVISION, files=receipt, packages=actual_versions)


class Table:
    def __init__(self, path):
        self.file = Path(path).open('w', newline='')
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


def input_record(core, filler, gap, **labels):
    ids = core[:3] + filler[:gap] + core[3:5] + filler[128:128 + gap] + core[5:]
    core_positions = [0, 1, 2, 3 + gap, 4 + gap, 5 + 2 * gap]
    filler_positions = [p for p in range(len(ids)) if p not in core_positions]
    compressed = ([0., 1., 2.] + [2. + j / (gap + 1) for j in range(1, gap + 1)]
                  + [3., 4.] + [4. + j / (gap + 1) for j in range(1, gap + 1)] + [5.])
    alternate = filler[1:] + filler[:1]
    alt_ids = core[:3] + alternate[:gap] + core[3:5] + alternate[128:128 + gap] + core[5:]
    record = dict(**labels, gap=gap, input_ids=ids, alternate_ids=alt_ids,
                  core_ids=core, core_positions=core_positions, filler_positions=filler_positions,
                  native_positions=list(map(float, range(len(ids)))), compressed_positions=compressed)
    assert len(core) == 6 and len(ids) == 6 + 2 * gap and len(compressed) == len(ids)
    assert [ids[p] for p in core_positions] == core
    assert [compressed[p] for p in core_positions] == list(map(float, range(6)))
    assert all(compressed[i] < compressed[i+1] for i in range(len(ids)-1))
    assert all(ids[p] != alt_ids[p] for p in filler_positions)
    return record


def make_inputs(tokenizer):
    eligible = []
    for tid in range(1000, len(tokenizer)):
        word = tokenizer.decode([tid])
        if re.fullmatch(r' [a-z]{3,8}', word) and tokenizer.encode(word, add_special_tokens=False) == [tid]:
            eligible.append(tid)
        if len(eligible) == 576:
            break
    assert len(eligible) == 576
    words = np.array(eligible[192:320], dtype=np.int64)
    np.random.default_rng(GROUP_SEED).shuffle(words)
    groups = words.reshape(32, 4)
    prompts, cases = [], []
    for group, tokens in enumerate(groups):
        a, b, c, d = map(int, tokens)
        filler = np.random.default_rng(FILLER_SEED + group).permutation(eligible[320:576]).tolist()
        for binding in (0, 1):
            av, cv = (b, d) if binding == 0 else (d, b)
            for order in (0, 1):
                pairs = [(a, av), (c, cv)]
                if order:
                    pairs.reverse()
                for query, query_id in enumerate((a, c)):
                    target = av if query == 0 else cv
                    other = cv if query == 0 else av
                    core = [tokenizer.eos_token_id, *pairs[0], *pairs[1], query_id]
                    labels = dict(case_id=len(cases), group=group, binding=binding, order=order,
                                  query=query, target=target, other=other,
                                  target_core=core.index(target), other_core=core.index(other))
                    cases.append(labels)
                    for gap in GAPS:
                        prompts.append(input_record(core, filler, gap, input_id=len(prompts), **labels))
    assert len(cases) == 256 and len(prompts) == 768
    ledger = dict(eligible_pool=[dict(id=i, text=tokenizer.decode([i])) for i in eligible],
                  groups=groups.tolist(), filler_pool=eligible[320:576], cases=cases, inputs=prompts)
    return prompts, ledger


class Engine:
    def __init__(self, model, deadline, attention_table):
        self.model, self.deadline, self.table = model, deadline, attention_table
        self.forwards = self.examples = 0
        self.records, self.filler, self.core = [], [], []
        self.mode, self.hits, self.metrics, self.masses = None, [], {}, None
        for layer, block in enumerate(model.gpt_neox.layers):
            block.attention._position_access_layer = layer

    def attention(self, module, query, key, value, attention_mask, scaling,
                  dropout=0.0, head_mask=None, **kwargs):
        assert query.dtype == key.dtype == value.dtype == torch.float64
        weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, :, :key.shape[-2]]
        weights = torch.nn.functional.softmax(weights, dim=-1, dtype=query.dtype)
        if head_mask is not None:
            weights = weights * head_mask
        assert dropout == 0.0 and not module.training
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        layer = module._position_access_layer
        self.hits.append(layer)
        future = torch.triu(torch.ones(weights.shape[-2:], dtype=torch.bool), diagonal=1)
        self.metrics['future_weight'] = max(self.metrics['future_weight'], float(weights[:, :, future].abs().max()))
        self.metrics['row_sum'] = max(self.metrics['row_sum'], float((weights.sum(-1)-1).abs().max()))
        if self.blocked and self.filler:
            self.metrics['blocked_weight'] = max(self.metrics['blocked_weight'], float(weights[..., self.filler].abs().max()))
        if layer == HEAD_LAYER:
            rows = []
            for i, record in enumerate(self.records):
                row = weights[i, HEAD_INDEX, -1]
                rows.append([float(row[self.core[record['target_core']]]),
                             float(row[self.core[record['other_core']]]), float(row[0]),
                             float(row[self.filler].sum()) if self.filler else 0., float(row[-1])])
            self.masses = np.array(rows)
            if self.mode in ('ablated', 'noop'):
                changed = output.clone()
                if self.mode == 'ablated':
                    changed[:, -1, HEAD_INDEX] = 0.
                    self.metrics['ablated_output'] = float(changed[:, -1, HEAD_INDEX].abs().max())
                else:
                    changed[:, -1, HEAD_INDEX] = output[:, -1, HEAD_INDEX].clone()
                delta = changed - output
                delta[:, -1, HEAD_INDEX] = 0.
                self.metrics['untouched_output'] = float(delta.abs().max())
                output = changed
        return output, weights

    @torch.inference_mode()
    def forward(self, records, position, access, mode, variant='long', reference=False):
        if time.monotonic() > self.deadline:
            raise TimeoutError('The fixed 20-minute wall-clock limit was reached.')
        assert len(records) <= BATCH_SIZE
        self.records, self.mode = records, mode
        self.hits, self.masses = [], None
        self.metrics = {name: 0. for name in ('future_weight', 'row_sum', 'blocked_weight',
                                             'untouched_output', 'ablated_output')}
        first = records[0]
        assert all(r['core_positions'] == first['core_positions'] for r in records)
        coords = first['native_positions' if position else 'compressed_positions']
        self.core, self.filler = first['core_positions'], first['filler_positions']
        ids = [r['alternate_ids'] if variant == 'alternate' else r['input_ids'] for r in records]
        if variant == 'short':
            ids = [r['core_ids'] for r in records]
            coords = [coords[p] for p in self.core]
            self.core, self.filler = list(range(6)), []
        self.blocked = access == 0
        length = len(ids[0])
        mask = None
        if not reference:
            mask = torch.zeros((1, 1, length, length), dtype=torch.float64)
            future = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
            mask[0, 0].masked_fill_(future, -torch.inf)
            if self.blocked:
                mask[..., self.filler] = -torch.inf
        else:
            assert position == access == 1 and variant == 'long'
        vectors, handles = [], []
        for block in self.model.gpt_neox.layers:
            def save(module, args, output):
                vectors.append(output[0][:, self.core].detach().numpy().copy())
            handles.append(block.register_forward_hook(save))
        try:
            neox.eager_attention_forward = self.attention
            result = self.model(input_ids=torch.tensor(ids, dtype=torch.long), attention_mask=mask,
                                position_ids=None if reference else torch.tensor([coords], dtype=torch.float64),
                                use_cache=False, logits_to_keep=1)
            logits = result.logits[:, -1].detach().numpy().copy()
            states = np.stack(vectors, axis=1)
            if not np.isfinite(logits).all() or not np.isfinite(states).all():
                raise RuntimeError('Non-finite model output.')
            self.forwards += 1
            self.examples += len(records)
            assert self.hits == list(range(6)) and self.masses is not None
            passed = (self.metrics['row_sum'] <= ROW_TOL and all(v == 0. for k, v in self.metrics.items() if k != 'row_sum'))
            self.table.add(dict(forward=self.forwards, first_input=first['input_id'], batch_size=len(records),
                                gap=first['gap'], position=position, access=access, mode=mode,
                                variant=variant, reference=reference, attention_calls=len(self.hits),
                                **self.metrics, row_tolerance=ROW_TOL, passed=passed))
            return dict(logits=logits, states=states, masses=self.masses.copy())
        finally:
            for handle in handles:
                handle.remove()


def load_model(model_dir):
    model = GPTNeoXForCausalLM.from_pretrained(model_dir, local_files_only=True,
              dtype=torch.float64, attn_implementation='eager').eval()
    model.requires_grad_(False)
    assert model.config.use_parallel_residual and model.config.rope_scaling is None
    assert (model.config.num_hidden_layers, model.config.num_attention_heads,
            model.config.hidden_size, model.config.vocab_size) == (6, 4, 128, 50304)
    assert all(p.dtype == torch.float64 for p in model.parameters())
    return model


def compare(table, kind, records, position, access, mode, left, right):
    for i, record in enumerate(records):
        logit_error = float(np.max(np.abs(left['logits'][i] - right['logits'][i])))
        state_error = float(np.max(np.abs(left['states'][i] - right['states'][i])))
        table.add(dict(kind=kind, input_id=record['input_id'], gap=record['gap'],
                       position=position, access=access, mode=mode,
                       logit_error=logit_error, state_error=state_error,
                       logit_tolerance=LOGIT_TOL, state_tolerance=STATE_TOL,
                       passed=logit_error <= LOGIT_TOL and state_error <= STATE_TOL))


def score(table, records, position, access, mode, result):
    for i, record in enumerate(records):
        logits, masses = result['logits'][i], result['masses'][i]
        target, other = record['target'], record['other']
        top1 = int(np.argmax(logits))
        table.add(dict(**{k: record[k] for k in ('input_id', 'case_id', 'group', 'binding', 'order', 'query', 'gap')},
                       position=position, access=access, mode=mode, target=target, other=other,
                       target_logit=float(logits[target]), other_logit=float(logits[other]),
                       margin=float(logits[target]-logits[other]), target_preferred=bool(logits[target] > logits[other]),
                       top1_token=top1, target_is_top1=top1 == target,
                       target_attention=float(masses[0]), other_attention=float(masses[1]),
                       bos_attention=float(masses[2]), filler_attention=float(masses[3]),
                       query_attention=float(masses[4]), logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest()))


def collect(engine, inputs, tables):
    for gap in sorted({r['gap'] for r in inputs}):
        items = [r for r in inputs if r['gap'] == gap]
        for start in range(0, len(items), BATCH_SIZE):
            records = items[start:start+BATCH_SIZE]
            cells = {}
            for p in (0, 1):
                for a in (0, 1):
                    for mode in ('intact', 'ablated'):
                        result = engine.forward(records, p, a, mode)
                        cells[p, a, mode] = result
                        score(tables['scores'], records, p, a, mode, result)
                        if a == 0:
                            for variant, kind in (('short', 'short_equivalence'), ('alternate', 'filler_irrelevance')):
                                control = engine.forward(records, p, a, mode, variant=variant)
                                compare(tables['controls'], kind, records, p, a, mode, result, control)
                        if p == a == 1:
                            reference = engine.forward(records, p, a, mode, reference=True)
                            compare(tables['controls'], 'default_mask_reference', records, p, a, mode, result, reference)
                    noop = engine.forward(records, p, a, 'noop')
                    compare(tables['controls'], 'head_noop', records, p, a, 'intact', cells[p, a, 'intact'], noop)
            if gap == 0:
                for p in (0, 1):
                    for a in (0, 1):
                        for mode in ('intact', 'ablated'):
                            compare(tables['controls'], 'zero_gap', records, p, a, mode,
                                    cells[p, a, mode], cells[1, 1, mode])
            print(json.dumps(dict(gap=gap, completed_at_gap=min(start+BATCH_SIZE, len(items)),
                                  forward_calls=engine.forwards)), flush=True)


def summarize(rows, out):
    indexed = {(r['input_id'], r['position'], r['access'], r['mode']): r for r in rows}
    group_rows, cell_rows = [], []
    for group in range(32):
        for gap in GAPS:
            inputs = sorted({r['input_id'] for r in rows if r['group'] == group and r['gap'] == gap})
            assert len(inputs) == 8
            means = {}
            for p in (0, 1):
                for a in (0, 1):
                    for mode in ('intact', 'ablated'):
                        selected = [indexed[i, p, a, mode] for i in inputs]
                        means[p, a, mode] = float(np.mean([r['margin'] for r in selected]))
                        cell_rows.append(dict(group=group, gap=gap, position=p, access=a, mode=mode, n=8,
                                              **{key: float(np.mean([r[key] for r in selected])) for key in
                                                 ('margin', 'target_preferred', 'target_is_top1', 'target_attention',
                                                  'other_attention', 'bos_attention', 'filler_attention', 'query_attention')}))
            s = {(p, a): means[p, a, 'intact'] for p in (0, 1) for a in (0, 1)}
            group_rows.append(dict(group=group, gap=gap, n=8, native_readable=s[1, 1],
                                   native_blocked=s[1, 0], compressed_readable=s[0, 1], compressed_blocked=s[0, 0],
                                   access_repair=s[1, 0]-s[1, 1], position_repair=s[0, 1]-s[1, 1],
                                   primary_difference=s[1, 0]-s[0, 1],
                                   interaction=s[1, 1]-s[1, 0]-s[0, 1]+s[0, 0],
                                   **{f'head_effect_p{p}a{a}': means[p, a, 'intact']-means[p, a, 'ablated']
                                      for p in (0, 1) for a in (0, 1)}))
    for name, values in (('group-contrasts', group_rows), ('group-cell-summary', cell_rows)):
        table = Table(out / (name + '.csv'))
        for row in values:
            table.add(row)
        table.close()
    primary = np.array([r['primary_difference'] for r in group_rows if r['gap'] == 128])
    samples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 32, size=(10000, 32))
    interval = np.percentile(primary[samples].mean(axis=1), [2.5, 97.5]).tolist()
    cell_summary = []
    for gap in GAPS:
        for p in (0, 1):
            for a in (0, 1):
                for mode in ('intact', 'ablated'):
                    selected = [r for r in cell_rows if (r['gap'], r['position'], r['access'], r['mode']) == (gap, p, a, mode)]
                    cell_summary.append(dict(gap=gap, position=p, access=a, mode=mode, n=256,
                                              **{key: float(np.mean([r[key] for r in selected])) for key in
                                                 ('margin', 'target_preferred', 'target_is_top1', 'target_attention',
                                                  'other_attention', 'bos_attention', 'filler_attention', 'query_attention')}))
    short_mean = float(np.mean([r['native_readable'] for r in group_rows if r['gap'] == 0]))
    long_mean = float(np.mean([r['native_readable'] for r in group_rows if r['gap'] == 128]))
    summary = dict(primary=dict(contrast='native blocked minus compressed readable; intact; gap 128',
                                mean=float(primary.mean()), interval_95=interval, groups=32, cases=256,
                                bootstrap_draws=10000, bootstrap_seed=BOOTSTRAP_SEED,
                                directional_support=interval[0] > LOGIT_TOL,
                                reverse_support=interval[1] < -LOGIT_TOL),
                   native_long_minus_short=long_mean-short_mean,
                   native_mean_score_degrades=long_mean < short_mean,
                   cells=cell_summary, units='logits', conditional_inference=True, human_review=False)
    dump(out / 'summary.json', summary)
    return summary


def execute(model_dir, out, lock_commit=None, preflight=False):
    if not preflight:
        verify_lock(lock_commit)
    receipt = verify_environment(model_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    tables = {name: Table(out / (name + '.csv')) for name in ('scores', 'controls', 'attention-controls')}
    engine, status = None, 'failed'
    try:
        dump(out / 'model-receipt.json', receipt)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        inputs, ledger = make_inputs(tokenizer)
        if preflight:
            filler = list(range(101, 357))
            inputs = [input_record([0, 10+k, 20+k, 30+k, 40+k, 10+k], filler, gap, input_id=4*i+k,
                                   case_id=k, group=k, binding=0, order=0, query=0,
                                   target=20+k, other=40+k, target_core=2, other_core=4)
                      for i, gap in enumerate((0, 4, 128)) for k in range(4)]
            dump(out / 'inputs.json', dict(dummy_inputs=inputs, scientific_inputs_token_checked=len(ledger['inputs']),
                                          scientific_model_outputs_collected=False))
        else:
            dump(out / 'inputs.json', ledger)
        model = load_model(model_dir)
        engine = Engine(model, started+TIME_LIMIT, tables['attention-controls'])
        collect(engine, inputs, tables)
        expected_examples = 264 if preflight else 16896
        expected_rows = 96 if preflight else 6144
        assert engine.examples == expected_examples and len(tables['scores'].rows) == expected_rows
        failed = [r for name in ('controls', 'attention-controls') for r in tables[name].rows if not r['passed']]
        control_summary = dict(comparison_checks=len(tables['controls'].rows), attention_checks=len(tables['attention-controls'].rows),
                               failed_checks=len(failed), logit_tolerance=LOGIT_TOL, state_tolerance=STATE_TOL,
                               row_tolerance=ROW_TOL,
                               maximum_logit_error=max(r['logit_error'] for r in tables['controls'].rows),
                               maximum_state_error=max(r['state_error'] for r in tables['controls'].rows),
                               maximum_row_sum_error=max(r['row_sum'] for r in tables['attention-controls'].rows))
        dump(out / 'control-summary.json', control_summary)
        if failed:
            raise RuntimeError('A fixed control failed. Retain all outputs for a separate audit.')
        if not preflight:
            summarize(tables['scores'].rows, out)
            verify_lock(lock_commit)
        verify_environment(model_dir)
        if time.monotonic() > engine.deadline:
            raise TimeoutError('The fixed 20-minute wall-clock limit was reached.')
        status = 'preflight_passed' if preflight else 'complete'
    except Exception as exc:
        dump(out / 'failure.json', dict(exception=type(exc).__name__, message=str(exc),
                                       scored_rows=len(tables['scores'].rows), tolerance_unchanged=True))
        raise
    finally:
        for table in tables.values():
            table.close()
        metadata = dict(status=status, preflight=preflight, lock_commit=lock_commit,
                        scientific_model_outputs_collected=not preflight and bool(tables['scores'].rows),
                        plan_sha256=sha(ROOT / 'analysis-plan.md'), runner_sha256=sha(__file__),
                        elapsed_seconds=time.monotonic()-started,
                        model_forward_calls=0 if engine is None else engine.forwards,
                        model_forward_examples=0 if engine is None else engine.examples,
                        cpu_threads=torch.get_num_threads(), batch_size=BATCH_SIZE,
                        python=platform.python_version(), system=platform.system(), machine=platform.machine(),
                        packages={name: version(name) for name in PACKAGES},
                        arithmetic='double arithmetic with stock float32 rotary tables', human_review=False)
        dump(out / 'run-metadata.json', metadata)
        dump(out / 'manifest.json', dict(status=status, files={p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'}))
    print(json.dumps(dict(status=status, model_forward_examples=engine.examples)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--lock-commit')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    if args.preflight and args.lock_commit:
        parser.error('Preflight cannot use a scientific lock.')
    if not args.preflight and not args.lock_commit:
        parser.error('A scientific run requires --lock-commit.')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    execute(args.model_dir, args.output, args.lock_commit, args.preflight)


if __name__ == '__main__':
    main()
