"""Fixed flow task-ability jobs. The caller must verify the public plan lock."""
from __future__ import annotations
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import check_stop

MEANS = np.array([[-2., -2.], [-2., 2.], [2., -2.], [2., 2.]])
MIX = np.array([.1, .2, .3, .4])
SD = .55
UPDATES, TRAIN_BATCH, EVAL_BATCH = 16000, 256, 1024
TOL = 1e-9


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def guard(deadline, *values):
    check_stop(deadline)
    for value in values:
        if not torch.isfinite(value).all():
            raise FloatingPointError('Non-finite flow value.')


class Flow(nn.Module):
    def __init__(self, recipe, seed):
        super().__init__(); assert recipe in ('A', 'B'); torch.manual_seed(seed)
        width = 32 if recipe == 'A' else 64
        self.blocks = nn.ModuleList([nn.ModuleList([nn.Linear(1, width), nn.Linear(width, 2)]) for _ in range(4)])
        self.double()
        with torch.no_grad():
            for block in self.blocks:
                block[1].weight.zero_(); block[1].bias.zero_()
        assert sum(p.numel() for p in self.parameters()) <= 100000

    def map(self, x, inverse=False, noop=False):
        ld = torch.zeros(len(x), dtype=x.dtype)
        for b in (range(3, -1, -1) if inverse else range(4)):
            keep, change = b % 2, 1 - b % 2
            h = torch.tanh(self.blocks[b][0](x[:, keep:keep + 1]))
            if noop:
                h = h.clone()
            st = self.blocks[b][1](h); s, t = 1.5 * torch.tanh(st[:, 0]), st[:, 1]
            v = (x[:, change] - t) * torch.exp(-s) if inverse else x[:, change] * torch.exp(s) + t
            x = torch.stack((x[:, keep], v), 1) if keep == 0 else torch.stack((v, x[:, keep]), 1)
            ld = ld - s if inverse else ld + s
        return x, ld


def save_weights(path, model):
    np.savez_compressed(path, **{k: v.detach().numpy() for k, v in model.state_dict().items()})


def base_logp(z):
    return -.5 * (z * z).sum(-1) - math.log(2 * math.pi)


def true_logp(x):
    terms = np.log(MIX)[None, :] - math.log(2 * math.pi * SD * SD) - .5 * (((x[:, None, :] - MEANS) / SD) ** 2).sum(-1)
    top = terms.max(1)
    return top + np.log(np.exp(terms - top[:, None]).sum(1))


def target_quadrants():
    p = np.array([[.5 * (1 + math.erf(v / (SD * math.sqrt(2)))) for v in m] for m in MEANS])
    return np.array([sum(MIX[j] * (p[j, 0] if q // 2 else 1 - p[j, 0]) *
                         (p[j, 1] if q % 2 else 1 - p[j, 1]) for j in range(4)) for q in range(4)])


def inputs(split):
    assert split in ('development', 'final')
    # All models within a split share the same task arrays. Model seeds are the replicate units.
    base = 2026101910 if split == 'development' else 2026101920
    result = {}
    for name, count, offset in (('train_x', 16384, 1), ('eval_x', 8192, 2)):
        rng = np.random.default_rng(base + offset)
        components = rng.choice(4, size=count, p=MIX)
        result[name] = rng.standard_normal((count, 2)) * SD + MEANS[components]
        result[name + '_components'] = components
    result['eval_z'] = np.random.default_rng(base + 3).standard_normal((32768, 2))
    return result


def calculate_metrics(density, generated, eval_x):
    counts = np.bincount(2 * (generated[:, 0] >= 0).astype(np.int64) +
                         (generated[:, 1] >= 0).astype(np.int64), minlength=4)
    return dict(excess_nll=float((true_logp(eval_x) - density).mean()),
                quadrant_tv=float(.5 * np.abs(counts / len(generated) - target_quadrants()).sum()))


def controlled_map(model, values, inverse, deadline):
    outputs, determinants = [], []; max_roundtrip = 0.; max_ld = 0.
    with torch.no_grad():
        for offset in range(0, len(values), EVAL_BATCH):
            guard(deadline); x = torch.from_numpy(values[offset:offset + EVAL_BATCH])
            y, ld = model.map(x, inverse); back, reverse_ld = model.map(y, not inverse)
            noop, noop_ld = model.map(x, inverse, noop=True)
            guard(deadline, y, ld, back, reverse_ld, noop, noop_ld)
            max_roundtrip = max(max_roundtrip, float((back - x).abs().max()))
            max_ld = max(max_ld, float((ld + reverse_ld).abs().max()))
            assert max_roundtrip <= TOL and max_ld <= TOL
            assert torch.equal(y, noop) and torch.equal(ld, noop_ld)
            outputs.append(y.numpy()); determinants.append(ld.numpy())
    return np.concatenate(outputs), np.concatenate(determinants), dict(roundtrip_error=max_roundtrip, logdet_error=max_ld, no_op_error=0.)


def run_job(recipe: str, seed: int, split: str, out: Path, deadline: float) -> dict:
    assert recipe in ('A', 'B') and split in ('development', 'final')
    out = Path(out); out.mkdir(parents=True, exist_ok=False); start = time.monotonic()
    arrays = inputs(split); np.savez_compressed(out / 'inputs.npz', **arrays)
    metadata = dict(task='flow', recipe=recipe, seed=int(seed), split=split, updates=UPDATES,
                    training_batch=TRAIN_BATCH, evaluation_batch=EVAL_BATCH, dtype='float64',
                    input_seed_base=2026101910 if split == 'development' else 2026101920,
                    batch_seed=30000 + seed)
    dump(out / 'metadata.json', metadata)
    model = Flow(recipe, seed); save_weights(out / 'initial_weights.npz', model)
    opt = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8,
                           weight_decay=0., amsgrad=False, foreach=False, fused=False)
    x = torch.from_numpy(arrays['train_x']); rng = np.random.default_rng(30000 + seed)
    try:
        t = time.monotonic()
        with (out / 'trace.csv').open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['step', 'examples', 'loss'])
            for step in range(1, UPDATES + 1):
                guard(deadline); ids = rng.integers(0, len(x), size=TRAIN_BATCH, dtype=np.int32)
                opt.zero_grad(set_to_none=True); z, ld = model.map(x[ids.astype(np.int64)])
                loss = -(base_logp(z) + ld).mean(); guard(deadline, z, ld, loss)
                loss.backward(); guard(deadline, *(p.grad for p in model.parameters())); opt.step()
                guard(deadline, *model.parameters())
                writer.writerow([step, TRAIN_BATCH, float(loss.detach())]); stream.flush()
        train_seconds = time.monotonic() - t
        save_weights(out / 'final_weights.npz', model); t = time.monotonic()
        z, ld, density_control = controlled_map(model, arrays['eval_x'], False, deadline)
        generated, gen_ld, generation_control = controlled_map(model, arrays['eval_z'], True, deadline)
        density = base_logp(z) + ld
        np.savez_compressed(out / 'eval_raw.npz', encoded=z, logdet=ld, log_density=density,
                            generated=generated, generation_logdet=gen_ld)
        controls = dict(density=density_control, generation=generation_control)
        dump(out / 'controls.json', controls)
        metrics = calculate_metrics(density, generated, arrays['eval_x'])
        result = dict(**metadata, parameters=sum(p.numel() for p in model.parameters()),
                      metrics=metrics, eligible=metrics['quadrant_tv'] <= .05,
                      passed=metrics['excess_nll'] <= .20 and metrics['quadrant_tv'] <= .05,
                      selection_score=-metrics['excess_nll'], status='complete',
                      training_seconds=train_seconds, evaluation_seconds=time.monotonic() - t,
                      seconds=time.monotonic() - start)
        dump(out / 'metrics.json', result); return result
    except Exception as exc:
        save_weights(out / 'partial_weights.npz', model)
        dump(out / 'failure.json', dict(error=repr(exc), seconds=time.monotonic() - start, metadata=metadata))
        raise


def explicit_map(values, weights, inverse=False):
    """Saved-array affine-coupling equations, without the Flow class."""
    point = values.clone(); total = torch.zeros(len(point), dtype=torch.float64)
    for i in (list(range(4))[::-1] if inverse else range(4)):
        fixed, changed = i % 2, 1 - i % 2
        h = torch.tanh(F.linear(point[:, fixed:fixed + 1], weights[f'blocks.{i}.0.weight'], weights[f'blocks.{i}.0.bias']))
        raw = F.linear(h, weights[f'blocks.{i}.1.weight'], weights[f'blocks.{i}.1.bias'])
        scale, shift = 1.5 * torch.tanh(raw[:, 0]), raw[:, 1]
        old = point[:, changed]
        new = (old - shift) * torch.exp(-scale) if inverse else old * torch.exp(scale) + shift
        columns = [point[:, 0], point[:, 1]]; columns[changed] = new; point = torch.stack(columns, 1)
        total = total - scale if inverse else total + scale
    return point, total


def replay(out: Path, deadline: float = float('inf')) -> dict:
    out = Path(out); start = time.monotonic(); meta = json.loads((out / 'metadata.json').read_text())
    with np.load(out / 'inputs.npz') as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    regen = inputs(meta['split'])
    for k in arrays:
        np.testing.assert_array_equal(arrays[k], regen[k])
    with np.load(out / 'final_weights.npz') as saved:
        weights = {k: torch.from_numpy(saved[k].copy()) for k in saved.files}
    with np.load(out / 'eval_raw.npz') as saved:
        expected = {k: saved[k].copy() for k in saved.files}
    actual = {}; maximum = 0.
    with torch.no_grad():
        for name, reverse, result_key, ld_key in (('eval_x', False, 'encoded', 'logdet'),
                                                 ('eval_z', True, 'generated', 'generation_logdet')):
            pieces, dets = [], []
            for offset in range(0, len(arrays[name]), EVAL_BATCH):
                guard(deadline); x = torch.from_numpy(arrays[name][offset:offset + EVAL_BATCH])
                y, ld = explicit_map(x, weights, reverse); back, backward_ld = explicit_map(y, weights, not reverse)
                guard(deadline, y, ld, back, backward_ld)
                assert float((back - x).abs().max()) <= TOL and float((ld + backward_ld).abs().max()) <= TOL
                pieces.append(y.numpy()); dets.append(ld.numpy())
            actual[result_key] = np.concatenate(pieces); actual[ld_key] = np.concatenate(dets)
    actual['log_density'] = -.5 * (actual['encoded'] ** 2).sum(1) - math.log(2 * math.pi) + actual['logdet']
    for name in expected:
        error = float(np.abs(actual[name] - expected[name]).max()); maximum = max(maximum, error)
        assert error <= TOL, (name, error)
    # Independent log-sum and region calculations; no scientific metric helper.
    components = [math.log(w) - math.log(2 * math.pi * SD ** 2) - ((arrays['eval_x'] - mean) ** 2).sum(1) / (2 * SD ** 2)
                  for w, mean in zip(MIX, MEANS)]
    exact_density = np.logaddexp.reduce(np.array(components), axis=0)
    target = np.array([sum(w * math.prod(.5 * math.erfc((-1 if flag else 1) * m / (SD * math.sqrt(2)))
                         for m, flag in zip(mean, (q // 2, q % 2))) for w, mean in zip(MIX, MEANS)) for q in range(4)])
    region = (actual['generated'][:, 0] >= 0).astype(np.int64) * 2 + (actual['generated'][:, 1] >= 0).astype(np.int64)
    freq = np.array([np.count_nonzero(region == q) / len(region) for q in range(4)])
    metrics = dict(excess_nll=float(np.mean(exact_density - actual['log_density'])), quadrant_tv=float(np.abs(freq - target).sum() / 2))
    original = json.loads((out / 'metrics.json').read_text())
    for k, value in metrics.items():
        assert abs(value - original['metrics'][k]) <= 1e-12
    passed = metrics['excess_nll'] <= .20 and metrics['quadrant_tv'] <= .05
    assert passed == original['passed']; assert (metrics['quadrant_tv'] <= .05) == original['eligible']
    assert abs(-metrics['excess_nll'] - original['selection_score']) <= 1e-12
    result = dict(status='passed', passed=True, training_repeated=False, maximum_output_error=maximum,
                  tolerance=TOL, values_checked=sum(v.size for v in actual.values()), metrics=metrics,
                  gate_passed=passed, seconds=time.monotonic() - start)
    dump(out / 'replay.json', result); return result


def preflight(out: Path) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    model = Flow('B', 902); rng = np.random.default_rng(2026101903)
    x = torch.from_numpy(rng.standard_normal((TRAIN_BATCH, 2)))
    opt = torch.optim.Adam(model.parameters(), lr=.001, foreach=False, fused=False)
    with torch.no_grad():
        initial, initial_ld = model.map(x)
        assert torch.equal(initial, x) and torch.equal(initial_ld, torch.zeros(TRAIN_BATCH, dtype=torch.float64))
    t = time.monotonic()
    for _ in range(20):
        guard(float('inf')); opt.zero_grad(set_to_none=True); z, ld = model.map(x)
        loss = -(base_logp(z) + ld).mean(); loss.backward()
        guard(float('inf'), loss, *(p.grad for p in model.parameters())); opt.step()
        guard(float('inf'), *model.parameters())
    per_step = (time.monotonic() - t) / 20
    probe = x[:16].clone().requires_grad_(True); z, ld = model.map(probe)
    rows = [torch.autograd.grad(z[:, k].sum(), probe, retain_graph=(k == 0))[0] for k in range(2)]
    sign, logdet = torch.linalg.slogdet(torch.stack(rows, dim=1))
    jac_error = float((logdet - ld).detach().abs().max()); assert (sign == 1).all() and jac_error <= TOL
    values = rng.standard_normal((EVAL_BATCH, 2)); weights = {k: v.detach() for k, v in model.state_dict().items()}
    t = time.monotonic(); y, det, controls = controlled_map(model, values, False, float('inf'))
    _, _, inverse_controls = controlled_map(model, values, True, float('inf')); inference_pair = time.monotonic() - t
    with torch.no_grad():
        t = time.monotonic(); y2, det2 = explicit_map(torch.from_numpy(values), weights)
        back, backld = explicit_map(y2, weights, True); audit = time.monotonic() - t
        assert float(np.abs(y - y2.numpy()).max()) <= TOL and float(np.abs(det - det2.numpy()).max()) <= TOL
        assert float(np.abs(back.numpy() - values).max()) <= TOL and float((backld + det2).abs().max()) <= TOL
    # Use the slower cost for both density and generation chunks, then include replay.
    evaluate = (inference_pair + audit) * (math.ceil(8192 / EVAL_BATCH) + math.ceil(32768 / EVAL_BATCH))
    per_job = per_step * UPDATES + evaluate
    result = dict(status='passed', passed=True, task='flow', dummy_only=True, parameters_B=sum(p.numel() for p in model.parameters()),
                  timing_steps=20, training_step_seconds_B=per_step, jacobian_error=jac_error,
                  controls=controls, inverse_controls=inverse_controls, estimated_one_job_seconds=per_job,
                  estimated_eight_jobs_seconds=8 * per_job, safety_factor=4, projected_job_seconds=4 * per_job, safe_eight_jobs_seconds=32 * per_job,
                  seconds=time.monotonic() - started)
    dump(out / 'preflight.json', result); return result
