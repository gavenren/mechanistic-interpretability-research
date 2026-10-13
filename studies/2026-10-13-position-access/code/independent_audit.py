#!/usr/bin/env python3
"""Independent input, table, and model replay audit. Does not import run.py."""
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
from transformers.models.gpt_neox import modeling_gpt_neox

ROOT = Path(__file__).resolve().parents[1]
HASHES = {
    'config.json': 'f97f966a66c444890ed461fff2a51eefb15d74303df05b948124719f199b0b17',
    'model.safetensors': '116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1',
    'tokenizer.json': '870f4e2baa6b683221fa52004d5d6f40ab8c9d31961617304b78c910c2c3caf2',
    'tokenizer_config.json': 'eee017c5bd133137f45907bd0a6e781e2ccd1a533734b7ed2a2f2f4446659809',
    'special_tokens_map.json': '10b8c8852c1e1f70b54d9aff61728408c28971c0e97a6c5a7b2debbd1d3e9c0c',
    'README.md': 'd1f2cf1d5181daedeaa70208ddd5cc5251867bde9acf6db7bb45a2265e25e163'}
LOGIT_TOL, STATE_TOL, ROW_TOL, ARITH_TOL = 1e-8, 1e-10, 1e-12, 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def boolean(value):
    assert value in ('True', 'False')
    return value == 'True'


def mean(values):
    return float(np.mean(list(values)))


class Audit:
    def __init__(self):
        self.maxima = {}
        self.counts = {}
        self.started = time.monotonic()

    def check(self, name, left, right, tolerance):
        difference = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
        self.maxima[name] = max(self.maxima.get(name, 0.), difference)
        self.counts[name] = self.counts.get(name, 0) + 1
        if not np.isfinite(difference) or difference > tolerance:
            raise AssertionError(f'{name}: {difference} exceeds {tolerance}')

    def budget(self):
        if time.monotonic() - self.started > 1200:
            raise TimeoutError('Independent audit exceeded its 20-minute cap.')


def reconstruct(tokenizer):
    pool = []
    for token in range(1000, len(tokenizer)):
        word = tokenizer.decode([token])
        if re.fullmatch(' [a-z]{3,8}', word) and tokenizer.encode(word, add_special_tokens=False) == [token]:
            pool.append(token)
            if len(pool) == 576:
                break
    assert len(pool) == 576
    selected = np.asarray(pool[192:320], dtype=np.int64)
    np.random.Generator(np.random.PCG64(2026101401)).shuffle(selected)
    groups = selected.reshape(32, 4).tolist()
    expected = []
    cases = []
    for group, (a, b, c, d) in enumerate(groups):
        filler = np.random.Generator(np.random.PCG64(2026101402 + group)).permutation(pool[320:]).tolist()
        for binding, order, query in itertools.product(range(2), repeat=3):
            mapping = {a: b if binding == 0 else d, c: d if binding == 0 else b}
            keys = [a, c] if order == 0 else [c, a]
            question = (a, c)[query]
            target = mapping[question]
            other = mapping[(c, a)[query]]
            core = [tokenizer.eos_token_id, keys[0], mapping[keys[0]], keys[1], mapping[keys[1]], question]
            case = dict(case_id=len(cases), group=group, binding=binding, order=order, query=query,
                        target=target, other=other, target_core=core.index(target), other_core=core.index(other))
            cases.append(case)
            for gap in (0, 32, 128):
                physical = [0, 1, 2, gap + 3, gap + 4, 2 * gap + 5]
                ids, positions = [], []
                alternate = filler[1:] + filler[:1]
                alt_ids = []
                for index, token in enumerate(core):
                    ids.append(token)
                    alt_ids.append(token)
                    positions.append(float(index))
                    if index in (2, 4):
                        start = 0 if index == 2 else 128
                        for j in range(gap):
                            ids.append(filler[start + j])
                            alt_ids.append(alternate[start + j])
                            positions.append(index + (j + 1) / (gap + 1))
                expected.append(dict(**case, input_id=len(expected), gap=gap, input_ids=ids,
                    alternate_ids=alt_ids, core_ids=core, core_positions=physical,
                    filler_positions=[i for i in range(len(ids)) if i not in physical],
                    native_positions=[float(i) for i in range(len(ids))], compressed_positions=positions))
    return dict(eligible_pool=[dict(id=i, text=tokenizer.decode([i])) for i in pool],
                groups=groups, filler_pool=pool[320:], cases=cases, inputs=expected)


def audit_tables(audit, result):
    raw = rows(result / 'scores.csv')
    assert len(raw) == 6144
    indexed = {}
    for row in raw:
        key = (int(row['input_id']), int(row['position']), int(row['access']), row['mode'])
        assert key not in indexed
        indexed[key] = row
        audit.check('raw_margin', float(row['margin']), float(row['target_logit']) - float(row['other_logit']), ARITH_TOL)
        assert boolean(row['target_preferred']) == (float(row['target_logit']) > float(row['other_logit']))
        assert boolean(row['target_is_top1']) == (int(row['target']) == int(row['top1_token']))
    assert set(indexed) == set(itertools.product(range(768), range(2), range(2), ('intact', 'ablated')))
    metrics = ['margin', 'target_preferred', 'target_is_top1', 'target_attention', 'other_attention',
               'bos_attention', 'filler_attention', 'query_attention']
    group_cells, group_contrasts = {}, {}
    for group, gap in itertools.product(range(32), (0, 32, 128)):
        cells = {}
        for p, access, mode in itertools.product(range(2), range(2), ('intact', 'ablated')):
            selected = [r for r in raw if (int(r['group']), int(r['gap']), int(r['position']), int(r['access']), r['mode']) == (group, gap, p, access, mode)]
            assert len(selected) == 8
            scores = {name: mean(boolean(r[name]) if name.startswith('target_') and name in ('target_preferred', 'target_is_top1') else float(r[name]) for r in selected) for name in metrics}
            group_cells[group, gap, p, access, mode] = scores
            cells[p, access, mode] = scores['margin']
        s00, s01 = cells[0, 0, 'intact'], cells[0, 1, 'intact']
        s10, s11 = cells[1, 0, 'intact'], cells[1, 1, 'intact']
        group_contrasts[group, gap] = dict(native_readable=s11, native_blocked=s10, compressed_readable=s01,
            compressed_blocked=s00, access_repair=s10-s11, position_repair=s01-s11,
            primary_difference=s10-s01, interaction=s11-s10-s01+s00,
            **{f'head_effect_p{p}a{access}': cells[p, access, 'intact']-cells[p, access, 'ablated'] for p, access in itertools.product(range(2), repeat=2)})
    group_saved = rows(result / 'group-contrasts.csv')
    assert len(group_saved) == 96
    for row in group_saved:
        key = int(row['group']), int(row['gap'])
        assert int(row['n']) == 8
        for name, value in group_contrasts[key].items():
            audit.check('group_contrast', float(row[name]), value, ARITH_TOL)
    cell_saved = rows(result / 'group-cell-summary.csv')
    assert len(cell_saved) == 768
    for row in cell_saved:
        key = int(row['group']), int(row['gap']), int(row['position']), int(row['access']), row['mode']
        assert int(row['n']) == 8
        for name, value in group_cells[key].items():
            audit.check('group_cell', float(row[name]), value, ARITH_TOL)
    summary = read(result / 'summary.json')
    primary = np.asarray([group_contrasts[g, 128]['primary_difference'] for g in range(32)])
    generator = np.random.Generator(np.random.PCG64(2026101403))
    draws = generator.integers(0, 32, size=(10000, 32))
    interval = np.quantile(primary[draws].mean(1), [.025, .975], method='linear')
    audit.check('primary_mean', summary['primary']['mean'], primary.mean(), ARITH_TOL)
    audit.check('primary_interval', summary['primary']['interval_95'], interval, ARITH_TOL)
    assert summary['primary']['directional_support'] == bool(interval[0] > 1e-8)
    assert summary['primary']['reverse_support'] == bool(interval[1] < -1e-8)
    long_short = mean(group_contrasts[g, 128]['native_readable'] for g in range(32)) - mean(group_contrasts[g, 0]['native_readable'] for g in range(32))
    audit.check('long_short', summary['native_long_minus_short'], long_short, ARITH_TOL)
    assert summary['native_mean_score_degrades'] == bool(long_short < 0)
    assert len(summary['cells']) == 24
    for row in summary['cells']:
        assert row['n'] == 256
        for name in metrics:
            value = mean(group_cells[g, row['gap'], row['position'], row['access'], row['mode']][name] for g in range(32))
            audit.check('cell_summary', row[name], value, ARITH_TOL)
    controls = rows(result / 'controls.csv')
    assert len(controls) == 12800
    for row in controls:
        assert boolean(row['passed'])
        assert float(row['logit_tolerance']) == LOGIT_TOL and float(row['state_tolerance']) == STATE_TOL
        assert float(row['logit_error']) <= LOGIT_TOL and float(row['state_error']) <= STATE_TOL
    attention = rows(result / 'attention-controls.csv')
    assert len(attention) == 4224
    for row in attention:
        assert boolean(row['passed']) and int(row['attention_calls']) == 6 and int(row['batch_size']) == 4
        assert float(row['row_sum']) <= ROW_TOL and float(row['row_tolerance']) == ROW_TOL
        assert all(float(row[key]) == 0 for key in ('future_weight', 'blocked_weight', 'untouched_output', 'ablated_output'))
    return indexed


class Replay:
    def __init__(self, model, audit):
        self.model, self.audit = model, audit
        self.calls = self.examples = self.hash_matches = 0
        self.records = []
        self.mode = None
        self.head_mass = None
        for layer, block in enumerate(model.gpt_neox.layers):
            block.attention._independent_layer = layer
        self.handle = model.gpt_neox.layers[3].attention.dense.register_forward_pre_hook(self.projection_hook)

    def projection_hook(self, module, args):
        if self.mode == 'ablated':
            edited = args[0].clone()
            edited[:, -1, 32:64] = 0
            difference = edited - args[0]
            difference[:, -1, 32:64] = 0
            self.audit.check('local_untouched', difference.numpy(), 0, 0)
            return (edited,)

    def attention(self, module, query, key, value, attention_mask, scaling, dropout=0., head_mask=None, **kwargs):
        # Construct the causal relation directly, without using the runner's mask.
        length = query.shape[-2]
        permitted = torch.arange(length)[:, None] >= torch.arange(length)[None, :]
        if self.blocked:
            permitted[:, self.filler] = False
        products = torch.matmul(query, key.transpose(-2, -1)) * scaling
        products = products.masked_fill(~permitted[None, None], -torch.inf)
        weights = torch.softmax(products, dim=-1)
        assert dropout == 0 and head_mask is None and weights.dtype == torch.float64
        self.audit.check('attention_row_sum', weights.sum(-1).numpy(), 1, ROW_TOL)
        self.audit.check('forbidden_weights', weights[:, :, ~permitted].numpy(), 0, 0)
        if module._independent_layer == 3:
            attention = weights[:, 1, -1]
            self.head_mass = [[float(attention[i, self.core[r['target_core']]]),
                float(attention[i, self.core[r['other_core']]]), float(attention[i, 0]),
                float(attention[i, self.filler].sum()), float(attention[i, -1])] for i, r in enumerate(self.records)]
        return torch.matmul(weights, value).transpose(1, 2).contiguous(), weights

    @torch.inference_mode()
    def forward(self, records, position, access, mode, short=False):
        self.audit.budget()
        self.records, self.mode = records, mode
        self.blocked = access == 0
        r = records[0]
        self.core = list(range(6)) if short else r['core_positions']
        self.filler = [] if short else r['filler_positions']
        coordinates = r['native_positions'] if position else r['compressed_positions']
        if short:
            coordinates = [coordinates[i] for i in r['core_positions']]
        token_ids = [item['core_ids'] if short else item['input_ids'] for item in records]
        vectors = []
        def capture(module, args, output):
            vectors.append(output[0][:, self.core].numpy().copy())
        hooks = [block.register_forward_hook(capture) for block in self.model.gpt_neox.layers]
        try:
            modeling_gpt_neox.eager_attention_forward = self.attention
            output = self.model(input_ids=torch.tensor(token_ids), position_ids=torch.tensor([coordinates], dtype=torch.float64),
                                use_cache=False, logits_to_keep=1)
            logits = output.logits[:, -1].numpy().copy()
            states = np.stack(vectors, axis=1)
            masses = np.asarray(self.head_mass)
            assert np.isfinite(logits).all() and np.isfinite(states).all()
            self.calls += 1
            self.examples += len(records)
            return logits, states, masses
        finally:
            for hook in hooks:
                hook.remove()


def execute(model_dir, result, audit):
    manifest = read(result / 'manifest.json')
    assert manifest['status'] == 'complete'
    for name, item in manifest['files'].items():
        assert sha(result / name) == item['sha256']
        assert (result / name).stat().st_size == item['bytes']
    metadata = read(result / 'run-metadata.json')
    assert metadata['status'] == 'complete' and not metadata['preflight']
    assert metadata['model_forward_examples'] == 16896 and metadata['model_forward_calls'] == 4224
    repo = ROOT.parents[1]
    for name, hash_key in [('analysis-plan.md', 'plan_sha256'), ('code/run.py', 'runner_sha256')]:
        path = ROOT / name
        locked = subprocess.run(['git', 'show', metadata['lock_commit'] + ':' + path.relative_to(repo).as_posix()], cwd=repo, check=True, capture_output=True).stdout
        assert locked == path.read_bytes() and sha(path) == metadata[hash_key]
    for name, expected in HASHES.items():
        assert sha(model_dir / name) == expected
    assert {name: version(name) for name in ('numpy', 'torch', 'transformers', 'safetensors')} == {
        'numpy': '2.3.5', 'torch': '2.8.0', 'transformers': '4.57.6', 'safetensors': '0.6.2'}
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    ledger = reconstruct(tokenizer)
    assert ledger == read(result / 'inputs.json')
    indexed = audit_tables(audit, result)
    model = GPTNeoXForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.float64,
                                             attn_implementation='eager').eval().requires_grad_(False)
    replay = Replay(model, audit)
    try:
        for gap in (0, 32, 128):
            records = [r for r in ledger['inputs'] if r['gap'] == gap]
            for offset in range(0, len(records), 4):
                batch = records[offset:offset+4]
                for position, access, mode in itertools.product(range(2), range(2), ('intact', 'ablated')):
                    logits, states, masses = replay.forward(batch, position, access, mode)
                    for index, record in enumerate(batch):
                        saved = indexed[record['input_id'], position, access, mode]
                        for key in ('case_id', 'group', 'binding', 'order', 'query', 'gap', 'target', 'other'):
                            assert int(saved[key]) == record[key]
                        audit.check('target_logit', logits[index, record['target']], float(saved['target_logit']), LOGIT_TOL)
                        audit.check('other_logit', logits[index, record['other']], float(saved['other_logit']), LOGIT_TOL)
                        assert int(np.argmax(logits[index])) == int(saved['top1_token'])
                        for column, name in enumerate(('target_attention', 'other_attention', 'bos_attention', 'filler_attention', 'query_attention')):
                            audit.check('head_attention', masses[index, column], float(saved[name]), ROW_TOL)
                        replay.hash_matches += hashlib.sha256(logits[index].tobytes()).hexdigest() == saved['logits_sha256']
                    if access == 0:
                        short_logits, short_states, _ = replay.forward(batch, position, access, mode, short=True)
                        audit.check('short_full_vocabulary', logits, short_logits, LOGIT_TOL)
                        audit.check('short_all_core_states', states, short_states, STATE_TOL)
            print(f'Audited gap {gap}; {replay.examples} model examples', flush=True)
        assert replay.calls == 2304 and replay.examples == 9216
    finally:
        replay.handle.remove()
    return dict(lock_commit=metadata['lock_commit'], model_calls=replay.calls, model_examples=replay.examples,
                scored_rows_replayed=6144, exact_full_vocabulary_hash_matches=replay.hash_matches,
                short_equivalence_examples=3072, reconstructed_inputs=768, human_review=False,
                arithmetic='double arithmetic with stock float32 rotary tables',
                scope='all input construction and saved arithmetic; all scored cells replayed; all blocked cells checked against short input; independent projection-input ablation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Audit output exists; choose a fresh path.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    audit = Audit()
    receipt = dict(status='failed', audit_sha256=sha(__file__), imports_runner=False)
    try:
        receipt.update(execute(args.model_dir, args.results, audit))
        audit.budget()
        receipt['status'] = 'passed'
    except Exception as error:
        receipt['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        receipt.update(maximum_errors=audit.maxima, check_counts=audit.counts, seconds=time.monotonic()-audit.started)
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
