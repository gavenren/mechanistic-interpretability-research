#!/usr/bin/env python3
"""Fixed all-unit flow ranking experiment. --preflight uses dummy inputs only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
LOCK_FILES = [HERE / 'analysis-plan.md', Path(__file__).resolve(),
              HERE / 'requirements.txt', HERE / 'sources.md']
SEEDS = [26501, 26519, 26539, 26561]
MEANS = np.array([[-2., -2.], [-2., 2.], [2., -2.], [2., 2.]])
WEIGHTS = np.array([.1, .2, .3, .4])
SD = .55
BATCH = 1024
TOL = 1e-9
CAP = 1200.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def finite(value):
    a = value.detach().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    assert np.isfinite(a).all(), 'Non-finite numeric value'


def lock_check(commit):
    assert len(commit) == 40, 'A full public lock SHA is required'
    resolved = subprocess.check_output(['git', 'rev-parse', commit + '^{commit}'], cwd=REPO, text=True).strip()
    assert resolved == commit
    out = {}
    for p in LOCK_FILES:
        rel = p.relative_to(REPO).as_posix()
        original = subprocess.check_output(['git', 'show', commit + ':' + rel], cwd=REPO)
        assert original == p.read_bytes(), 'Locked file changed: ' + rel
        out[rel] = sha(p)
    return out


class Counter:
    def __init__(self):
        self.calls = 0
        self.examples = 0
        self.updates = 0
        self.controls = 0

    def snapshot(self):
        return dict(mapping_calls=self.calls, input_examples=self.examples,
                    optimizer_updates=self.updates, control_rows=self.controls)


class Flow(nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.blocks = nn.ModuleList([nn.ModuleList([nn.Linear(1, 32), nn.Linear(32, 2)]) for _ in range(4)])
        self.double()
        with torch.no_grad():
            for block in self.blocks:
                block[1].weight.zero_()
                block[1].bias.zero_()
        assert sum(p.numel() for p in self.parameters()) == 520

    def map(self, x, inverse=False, condition=0, noop=False, counter=None, check_local=True):
        if counter is not None:
            counter.calls += 1
            counter.examples += len(x)
        ld = torch.zeros(len(x), dtype=x.dtype)
        local = 0.
        indices = range(3, -1, -1) if inverse else range(4)
        for b in indices:
            keep = b % 2
            change = 1 - keep
            hidden = torch.tanh(self.blocks[b][0](x[:, keep:keep + 1]))
            if condition and b == (condition - 1) // 32:
                u = (condition - 1) % 32
                before = hidden
                hidden = hidden.clone()
                hidden[:, u] = 0.
                if check_local:
                    untouched = torch.cat((hidden[:, :u] - before[:, :u], hidden[:, u + 1:] - before[:, u + 1:]), dim=1)
                    local = max(local, float(hidden[:, u].detach().abs().max()), float(untouched.detach().abs().max()))
            elif noop:
                hidden = hidden.clone()
            st = self.blocks[b][1](hidden)
            s = 1.5 * torch.tanh(st[:, 0])
            t = st[:, 1]
            v = (x[:, change] - t) * torch.exp(-s) if inverse else x[:, change] * torch.exp(s) + t
            x = torch.stack((x[:, keep], v), dim=1) if keep == 0 else torch.stack((v, x[:, keep]), dim=1)
            ld = ld - s if inverse else ld + s
        return x, ld, local


def base_logp(z):
    return -.5 * (z * z).sum(axis=-1) - math.log(2. * math.pi)


def true_logp(x):
    terms = np.log(WEIGHTS)[None, :] - math.log(2 * math.pi * SD * SD) - .5 * (((x[:, None, :] - MEANS) / SD) ** 2).sum(axis=-1)
    top = terms.max(axis=1)
    return top + np.log(np.exp(terms - top[:, None]).sum(axis=1))


def true_quadrants():
    positive = np.array([[.5 * (1 + math.erf(v / (SD * math.sqrt(2)))) for v in m] for m in MEANS])
    return np.array([sum(WEIGHTS[j] * (positive[j, 0] if q // 2 else 1 - positive[j, 0]) *
                         (positive[j, 1] if q % 2 else 1 - positive[j, 1]) for j in range(4)) for q in range(4)])


def quadrants(x):
    return 2 * (x[:, 0] >= 0).astype(np.int64) + (x[:, 1] >= 0).astype(np.int64)


def tv(a, b):
    return float(.5 * np.abs(np.asarray(a) - np.asarray(b)).sum())


def tau_b(x, y):
    a, b = np.triu_indices(len(x), 1)
    sx, sy = np.sign(np.asarray(x)[a] - np.asarray(x)[b]), np.sign(np.asarray(y)[a] - np.asarray(y)[b])
    denom = math.sqrt(float(np.count_nonzero(sx)) * float(np.count_nonzero(sy)))
    return None if denom == 0 else float((sx * sy).sum() / denom)


def select(scores, key):
    return min(range(1, 129), key=lambda c: (-scores[c][key], c))


def moments(x):
    return dict(count=len(x), mean=None if len(x) == 0 else x.mean(axis=0).tolist(),
                covariance=None if len(x) < 2 else np.cov(x, rowvar=False, ddof=1).tolist())


def data_inputs(dummy=False):
    out = {}
    if dummy:
        rng = np.random.Generator(np.random.PCG64(29001))
        for name, n in [('train_x', 4096), ('cal_x', 2048), ('test_x', 2048), ('cal_z', 2048), ('test_z', 2048)]:
            out[name] = rng.standard_normal((n, 2))
        out['batch_ids'] = rng.integers(0, 4096, size=(200, 256), dtype=np.int32)
    else:
        for name, n, seed in [('train_x', 16384, 26401), ('cal_x', 4096, 26402), ('test_x', 8192, 26403)]:
            rng = np.random.Generator(np.random.PCG64(seed))
            labels = rng.choice(4, size=n, p=WEIGHTS)
            out[name] = rng.standard_normal((n, 2)) * SD + MEANS[labels]
            out[name + '_components'] = labels
        out['cal_z'] = np.random.Generator(np.random.PCG64(26404)).standard_normal((16384, 2))
        out['test_z'] = np.random.Generator(np.random.PCG64(26405)).standard_normal((32768, 2))
        out['batch_ids'] = np.random.Generator(np.random.PCG64(26406)).integers(0, 16384, size=(4000, 256), dtype=np.int32)
    return out


class Run:
    def __init__(self, output, dummy):
        self.output, self.dummy = output, dummy
        output.mkdir(parents=True, exist_ok=False)
        self.start = time.perf_counter()
        self.counter = Counter()
        self.model = None
        self.pending_density = None
        self.current = {}
        self.f = (output / 'controls.csv').open('w', newline='')
        self.writer = csv.DictWriter(self.f, fieldnames=['seed', 'phase', 'condition', 'kind', 'start', 'count', 'coordinate_error', 'logdet_error', 'local_error', 'determinant_sign_min'])
        self.writer.writeheader()
        self.lossfile = (output / 'training.csv').open('w', newline='')
        self.losswriter = csv.DictWriter(self.lossfile, fieldnames=['seed', 'step', 'loss'])
        self.losswriter.writeheader()
        self.scorefile = (output / 'scores.jsonl').open('w')

    def timecheck(self):
        assert time.perf_counter() - self.start <= CAP, 'Fixed time cap exceeded'

    def control(self, row, tol=TOL):
        self.writer.writerow(row)
        self.f.flush()
        self.counter.controls += 1
        for key in ['coordinate_error', 'logdet_error', 'local_error']:
            if key in row:
                finite(row[key])
                assert row[key] <= (0. if key == 'local_error' else tol), 'Control failed: ' + str(row)
        if 'determinant_sign_min' in row:
            assert row['determinant_sign_min'] == 1., 'Non-positive Jacobian determinant'
        self.timecheck()

    def train(self, model, data, seed):
        opt = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8,
                               weight_decay=0., amsgrad=False, maximize=False, foreach=False,
                               fused=False, capturable=False, differentiable=False)
        np.savez_compressed(self.output / f'model-{seed}-step-0.npz', **{k: v.detach().numpy() for k, v in model.state_dict().items()})
        tx = torch.from_numpy(data['train_x'])
        start = time.perf_counter()
        for i, ids in enumerate(data['batch_ids'], 1):
            opt.zero_grad(set_to_none=True)
            z, ld, _ = model.map(tx[ids.astype(np.int64)], counter=self.counter, check_local=False)
            loss = -(base_logp(z) + ld).mean()
            finite(loss)
            loss.backward()
            opt.step()
            self.counter.updates += 1
            for p in model.parameters():
                finite(p)
            self.losswriter.writerow(dict(seed=seed, step=i, loss=float(loss.detach())))
            self.lossfile.flush()
            if i % 1000 == 0 or i == len(data['batch_ids']):
                np.savez_compressed(self.output / f'model-{seed}-step-{i}.npz', **{k: v.detach().numpy() for k, v in model.state_dict().items()})
            self.current = dict(seed=seed, stage='training', step=i)
            self.timecheck()
        return time.perf_counter() - start

    def evaluate_array(self, model, x, seed, phase, condition, inverse):
        ys, lds = [], []
        with torch.no_grad():
            for start in range(0, len(x), BATCH):
                xt = torch.from_numpy(x[start:start + BATCH])
                y, ld, local = model.map(xt, inverse, condition, counter=self.counter)
                back, revld, revlocal = model.map(y, not inverse, condition, counter=self.counter)
                finite(y); finite(ld); finite(back); finite(revld)
                self.control(dict(seed=seed, phase=phase, condition=condition, kind='roundtrip', start=start, count=len(xt),
                                  coordinate_error=float((back - xt).abs().max()), logdet_error=float((ld + revld).abs().max()), local_error=max(local, revlocal)))
                if condition == 0:
                    noopy, noopld, nooplocal = model.map(xt, inverse, condition, noop=True, counter=self.counter)
                    self.control(dict(seed=seed, phase=phase, condition=condition, kind='noop', start=start, count=len(xt),
                                      coordinate_error=float((noopy - y).abs().max()), logdet_error=float((noopld - ld).abs().max()), local_error=nooplocal), tol=0.)
                ys.append(y.numpy()); lds.append(ld.numpy())
                self.current = dict(seed=seed, stage=phase, condition=condition, batch_start=start)
        return np.concatenate(ys), np.concatenate(lds)

    def jacobian(self, model, x, seed, condition):
        xt = torch.from_numpy(x[:64].copy()).requires_grad_(True)
        y, ld, local = model.map(xt, condition=condition, counter=self.counter)
        rows = [torch.autograd.grad(y[:, k].sum(), xt, retain_graph=(k == 0))[0] for k in range(2)]
        jac = torch.stack(rows, dim=1)
        signs, logs = torch.linalg.slogdet(jac)
        finite(logs)
        self.control(dict(seed=seed, phase='cal_x', condition=condition, kind='jacobian', start=0, count=64,
                          logdet_error=float((logs - ld).detach().abs().max()), local_error=local,
                          determinant_sign_min=float(signs.min())))

    def phase(self, model, data, seed, phase, selectors=None):
        x, z = data[phase + '_x'], data[phase + '_z']
        density = np.full((129, len(x), 2), np.nan, dtype=np.float64)
        self.pending_density = (seed, phase, density)
        scores, baseline_generated, baseline_q, baseline_density = [], None, None, None
        selected_outputs = {}
        for condition in range(129):
            encoded, logdet = self.evaluate_array(model, x, seed, phase + '_x', condition, False)
            density[condition, :, 0] = base_logp(encoded)
            density[condition, :, 1] = logdet
            finite(density[condition])
            generated, inverse_ld = self.evaluate_array(model, z, seed, phase + '_z', condition, True)
            finite(inverse_ld)
            q = quadrants(generated)
            counts = np.bincount(q, minlength=4)
            frequencies = counts / len(z)
            assert counts.sum() == len(z) and np.all(frequencies >= 0) and abs(frequencies.sum() - 1.) <= 1e-12
            if condition == 0:
                baseline_generated = generated
                baseline_q = q
                baseline_density = density[0].copy()
            transitions = np.bincount(4 * baseline_q + q, minlength=16).reshape(4, 4)
            assert transitions.sum() == len(z)
            damage_parts = -(density[condition] - baseline_density).mean(axis=0)
            baseline_freq = np.bincount(baseline_q, minlength=4) / len(z)
            row = dict(seed=seed, phase=phase, condition=condition, block=None if condition == 0 else (condition - 1) // 32,
                       unit=None if condition == 0 else (condition - 1) % 32,
                       mean_nll=float(-density[condition].sum(axis=1).mean()), nll_damage=float(damage_parts.sum()),
                       gaussian_damage=float(damage_parts[0]), logjac_damage=float(damage_parts[1]),
                       quadrant_counts=counts.tolist(), quadrant_frequencies=frequencies.tolist(),
                       quadrant_changes=(frequencies - baseline_freq).tolist(), quadrant_tv=tv(frequencies, baseline_freq),
                       mean_squared_displacement=float(((generated - baseline_generated) ** 2).sum(axis=1).mean()),
                       crossing_fraction=float((q != baseline_q).mean()), crossing_count=int((q != baseline_q).sum()),
                       squared_displacement_sum=float(((generated - baseline_generated) ** 2).sum()), transitions=transitions.tolist())
            finite([row['mean_nll'], row['nll_damage'], row['gaussian_damage'], row['logjac_damage'], row['quadrant_tv'],
                    row['mean_squared_displacement'], row['crossing_fraction']])
            scores.append(row)
            self.scorefile.write(json.dumps(row, allow_nan=False) + '\n'); self.scorefile.flush()
            if phase == 'cal':
                self.jacobian(model, x, seed, condition)
            elif condition in {0, selectors['L'], selectors['M']}:
                selected_outputs['condition_' + str(condition)] = generated
            if condition % 32 == 31 or condition == 128:
                lo = condition // 32 * 32
                np.savez_compressed(self.output / f'density-{seed}-{phase}-{lo:03d}-{condition:03d}.npz',
                                    condition_ids=np.arange(lo, condition + 1), terms=density[lo:condition + 1])
            self.timecheck()
        self.pending_density = None
        if phase == 'test':
            np.savez_compressed(self.output / f'generated-{seed}-selected-test.npz', **selected_outputs)
        dump(self.output / f'scores-{seed}-{phase}.json', scores)
        return scores, baseline_generated

    def close(self):
        for f in (self.f, self.lossfile, self.scorefile):
            f.close()


def arithmetic_checks():
    points = np.array([[0., 0.], [-2., 2.], [1., -.7], [.4, .8]])
    separate = []
    for point in points:
        value = sum(float(w) / (2 * math.pi * SD * SD) * math.exp(-sum((float(a) - float(b)) ** 2 for a, b in zip(point, mean)) / (2 * SD * SD)) for w, mean in zip(WEIGHTS, MEANS))
        separate.append(math.log(value))
    assert np.max(np.abs(true_logp(points) - separate)) < 1e-12
    # Independent CDF quadrature identity: each axis has a product distribution per component.
    separate_q = []
    for q in range(4):
        total = 0.
        for weight, mean in zip(WEIGHTS, MEANS):
            probabilities = [.5 * math.erfc(-float(mean[d]) / (SD * math.sqrt(2))) if (q // 2 if d == 0 else q % 2)
                             else .5 * math.erfc(float(mean[d]) / (SD * math.sqrt(2))) for d in range(2)]
            total += float(weight) * math.prod(probabilities)
        separate_q.append(total)
    assert np.max(np.abs(true_quadrants() - separate_q)) < 1e-12
    assert abs(true_quadrants().sum() - 1) < 1e-12
    assert tau_b([1, 2, 3], [1, 2, 3]) == 1.
    assert tau_b([1, 2, 3], [3, 2, 1]) == -1.
    assert tau_b([1, 1, 1], [1, 2, 3]) is None
    assert abs(tau_b([1, 1, 2], [1, 2, 3]) - 2 / math.sqrt(6)) < 1e-12
    assert select([{'k': 0.} for _ in range(129)], 'k') == 1
    assert tv([.1, .2, .3, .4], [.1, .2, .3, .4]) == 0.
    return dict(mixture_density_max_error=float(np.max(np.abs(true_logp(points) - separate))),
                quadrant_probability_max_error=float(np.max(np.abs(true_quadrants() - separate_q))),
                tau_ties_sign_and_selector_ties='passed')


def dummy_witness():
    model = Flow(29003)
    x = torch.tensor([[-.6, .2], [.3, -.4], [.8, .9]], dtype=torch.float64)
    with torch.no_grad():
        y, ld, _ = model.map(x)
        assert torch.equal(x, y) and torch.equal(ld, torch.zeros_like(ld))
        for block in model.blocks:
            block[0].weight.fill_(.2); block[0].bias.fill_(.1)
            block[1].weight.fill_(.03); block[1].bias.fill_(.02)
        y, ld, _ = model.map(x)
        edited, eld, local = model.map(x, condition=1)
        back, bld, backlocal = model.map(edited, inverse=True, condition=1)
        effect = float((y - edited).abs().max())
        scale = float(ld.abs().max())
        inverse_error = float((back - x).abs().max())
        logdet_error = float((eld + bld).abs().max())
        assert effect > 1e-8 and scale > 1e-8 and max(inverse_error, logdet_error) < TOL and max(local, backlocal) == 0.
    xt = x.clone().requires_grad_(True)
    yt, analytic, _ = model.map(xt, condition=1)
    jac = torch.stack([torch.autograd.grad(yt[:, k].sum(), xt, retain_graph=k == 0)[0] for k in range(2)], dim=1)
    sign, logs = torch.linalg.slogdet(jac)
    jacerror = float((logs - analytic).detach().abs().max())
    assert (sign == 1).all() and jacerror < TOL
    return dict(identity_initialization='exact', nonzero_removal_effect=effect, nonzero_total_logscale=scale,
                coordinate_roundtrip_error=inverse_error, logdet_roundtrip_error=logdet_error, jacobian_error=jacerror,
                witness_mapping_calls=5, witness_examples=15)


def summary(seed, cal, test, generated, data, selectors):
    damage = [r['nll_damage'] for r in test[1:]]
    changes = [r['quadrant_tv'] for r in test[1:]]
    target = true_quadrants()
    excess = test[0]['mean_nll'] + float(true_logp(data['test_x']).mean())
    fit_tv = tv(test[0]['quadrant_frequencies'], target)
    q = quadrants(generated)
    ranked = lambda rows, key: sorted(range(1, 129), key=lambda c: (-rows[c][key], c))
    return dict(seed=seed, selectors=selectors, primary_difference=test[selectors['M']]['quadrant_tv'] - test[selectors['L']]['quadrant_tv'],
                selected_test={key: test[c] for key, c in selectors.items()},
                selector_agreement=selectors['L'] == selectors['M'],
                calibration_tau_b=tau_b([r['nll_damage'] for r in cal[1:]], [r['quadrant_tv'] for r in cal[1:]]),
                test_tau_b=tau_b(damage, changes),
                rankings={phase: {key: ranked(rows, key) for key in ['nll_damage', 'quadrant_tv']} for phase, rows in [('cal', cal), ('test', test)]},
                all_unit_test=dict(nll_damage_mean=float(np.mean(damage)), nll_damage_range=[min(damage), max(damage)],
                                   quadrant_tv_mean=float(np.mean(changes)), quadrant_tv_range=[min(changes), max(changes)]),
                fit=dict(test_excess_nll=excess, generated_quadrant_tv=fit_tv, gate=bool(excess <= .20 and fit_tv <= .05),
                         exact_target_quadrants=target.tolist(), generated=moments(generated),
                         generated_by_quadrant=[moments(generated[q == i]) for i in range(4)]))


def manifest(output, value):
    value['files'] = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ['manifest.json', 'failure-manifest.json']}
    dump(output / ('manifest.json' if value['status'] == 'complete' else 'failure-manifest.json'), value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock-commit')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    assert platform.python_version_tuple()[:2] == ('3', '12')
    assert np.__version__ == '2.3.5' and torch.__version__.split('+')[0] == '2.8.0'
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.mkldnn.enabled = False
    locked = {p.relative_to(REPO).as_posix(): sha(p) for p in LOCK_FILES} if args.preflight else lock_check(args.lock_commit)
    run = Run(args.output, args.preflight)
    meta = dict(lock_commit=args.lock_commit, locked_file_sha256=locked,
                python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                dtype='float64', initialization='PyTorch default float32 Linear initialization, then conversion to float64, then zero output Linear layers',
                threads=2, interop_threads=1, scientific_model_outputs_collected=not args.preflight)
    try:
        if args.preflight:
            meta['arithmetic_checks'] = arithmetic_checks()
            meta['dummy_witness'] = dummy_witness()
        data = data_inputs(args.preflight)
        np.savez_compressed(args.output / 'inputs.npz', **data)
        dump(args.output / 'input-hashes.json', {k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k, v in data.items()})
        summaries = []
        training_seconds = evaluation_seconds = 0.
        training_examples = evaluation_examples = 0
        for seed in ([29002] if args.preflight else SEEDS):
            model = Flow(seed); run.model = model
            before = run.counter.examples
            training_seconds += run.train(model, data, seed)
            training_examples += run.counter.examples - before
            before = run.counter.examples
            t0 = time.perf_counter()
            model.eval()
            cal, _ = run.phase(model, data, seed, 'cal')
            selectors = dict(L=select(cal, 'nll_damage'), M=select(cal, 'quadrant_tv'))
            dump(args.output / f'selectors-{seed}.json', dict(seed=seed, selectors=selectors,
                                                           test_model_calls_started=False, calibration_scores_sha256=sha(args.output / f'scores-{seed}-cal.json')))
            test, generated = run.phase(model, data, seed, 'test', selectors)
            if args.preflight:
                # The dummy law is standard normal. Do not apply scientific mixture fit gates.
                summaries.append(dict(seed=seed, selectors=selectors, dummy_only=True))
            else:
                summaries.append(summary(seed, cal, test, generated, data, selectors))
            evaluation_seconds += time.perf_counter() - t0
            evaluation_examples += run.counter.examples - before
            dump(args.output / 'model-summary.json', summaries)
        counts = run.counter.snapshot()
        if args.preflight:
            expected = dict(mapping_calls=2401, input_examples=2181184, optimizer_updates=200, control_rows=1169)
            # 129*8 main calls, the same count for round trips, 8 no-ops, 129 Jacobians, and 200 updates.
            projected = 2. * (training_seconds * 4096000 / training_examples + evaluation_seconds * 63684864 / evaluation_examples)
            meta['timing'] = dict(training_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
                                  training_examples=training_examples, evaluation_examples=evaluation_examples,
                                  full_run_projected_seconds_with_factor_two=projected)
            assert projected <= CAP, 'Dummy projection exceeds scientific time cap'
        else:
            expected = dict(mapping_calls=78676, input_examples=67780864, optimizer_updates=16000, control_rows=31716)
            differences = np.array([row['primary_difference'] for row in summaries])
            indices = np.random.Generator(np.random.PCG64(26407)).integers(0, 4, size=(10000, 4))
            resampled = differences[indices].mean(axis=1)
            np.savez_compressed(args.output / 'bootstrap.npz', model_indices=indices, resampled_means=resampled)
            dump(args.output / 'summary.json', dict(primary_model_values=differences.tolist(), primary_mean=float(differences.mean()),
                                                   descriptive_model_bootstrap_95=np.percentile(resampled, [2.5, 97.5], method='linear').tolist(),
                                                   all_fit_gates_pass=all(row['fit']['gate'] for row in summaries),
                                                   test_data=moments(data['test_x']),
                                                   test_data_by_quadrant=[moments(data['test_x'][quadrants(data['test_x']) == q]) for q in range(4)]))
            assert lock_check(args.lock_commit) == locked
        assert counts == expected, f'Count mismatch: {counts} != {expected}'
        run.timecheck()
        run.close()
        with (args.output / 'controls.csv').open() as f:
            rows = list(csv.DictReader(f))
        meta['control_maxima'] = {k: max(float(r[k]) for r in rows if r[k] != '')
                                  for k in ['coordinate_error', 'logdet_error', 'local_error']}
        assert len(rows) == counts['control_rows']
        meta.update(status='complete', counts=counts, elapsed_seconds=time.perf_counter() - run.start)
        manifest(args.output, meta)
        print(json.dumps({k: meta[k] for k in ['status', 'counts', 'elapsed_seconds']}))
    except BaseException as exc:
        run.close()
        if run.model is not None:
            np.savez_compressed(args.output / 'failure-current-model.npz', **{k: v.detach().numpy() for k, v in run.model.state_dict().items()})
        if run.pending_density is not None:
            seed, phase, density = run.pending_density
            completed = np.where(np.isfinite(density).all(axis=(1, 2)))[0]
            np.savez_compressed(args.output / f'failure-partial-density-{seed}-{phase}.npz', condition_ids=completed, terms=density[completed])
        meta.update(status='failed', error=str(exc), traceback=traceback.format_exc().replace(str(REPO), '<repo>'), counts=run.counter.snapshot(),
                    current=run.current, elapsed_seconds=time.perf_counter() - run.start)
        manifest(args.output, meta)
        raise


if __name__ == '__main__':
    main()
