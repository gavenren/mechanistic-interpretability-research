"""Fixed digit task-ability jobs. The caller must verify the public plan lock."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import check_stop

STUDY = Path(__file__).resolve().parents[1]
DATA = STUDY.parent / '2026-10-13-shift-ablation' / 'data'
HASHES = {
    'optdigits.tra': 'e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd',
    'optdigits.tes': '6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8'}
SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
NAMES = ('original', 'up', 'down', 'left', 'right')
EPOCHS, BATCH = 40, 128
TOL = 1e-10


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def guard(deadline, *values):
    check_stop(deadline)
    if time.monotonic() >= deadline:
        raise TimeoutError('Experiment deadline reached.')
    for value in values:
        if not torch.isfinite(value).all():
            raise FloatingPointError('Non-finite digit value.')


class Net(nn.Module):
    def __init__(self, recipe, filtered):
        super().__init__()
        assert recipe in ('A', 'B')
        c = [8, 16, 32] if recipe == 'A' else [16, 32, 64]
        self.filtered = filtered
        self.conv1 = nn.Conv2d(1, c[0], 3, padding=1, padding_mode='circular')
        self.conv2 = nn.Conv2d(c[0], c[1], 3, padding=1, padding_mode='circular')
        self.conv3 = nn.Conv2d(c[1], c[2], 3, padding=1, padding_mode='circular')
        self.fc = nn.Linear(c[2], 10)
        v = torch.tensor([1., 2., 1.], dtype=torch.float32)
        self.register_buffer('kernel', torch.outer(v, v)[None, None] / 16)
        assert sum(p.numel() for p in self.parameters()) <= 100000

    def down(self, x):
        if self.filtered:
            x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='circular'),
                         self.kernel.expand(x.shape[1], 1, 3, 3), groups=x.shape[1])
        return x[:, :, ::2, ::2]

    def forward(self, x):
        x = self.down(F.relu(self.conv1(x)))
        x = self.down(F.relu(self.conv2(x)))
        return self.fc(F.relu(self.conv3(x)).mean((-2, -1)))


def save_weights(path, model):
    np.savez_compressed(path, **{k: v.detach().numpy() for k, v in model.state_dict().items()})


def explicit_forward(x, weights, filtered):
    """Saved-array calculation without the Net class or its forward/down methods."""
    for layer in (1, 2, 3):
        x = torch.relu(F.conv2d(F.pad(x, (1, 1, 1, 1), mode='circular'),
                               weights[f'conv{layer}.weight'], weights[f'conv{layer}.bias']))
        if layer < 3:
            if filtered:
                k = torch.tensor([[[[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]]], dtype=x.dtype) / 16
                x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='circular'),
                             k.repeat(x.shape[1], 1, 1, 1), groups=x.shape[1])
            x = x[:, :, 0::2, 0::2]
    return F.linear(x.mean((-2, -1)), weights['fc.weight'], weights['fc.bias'])


def inputs(split):
    assert split in ('development', 'final')
    tables = []
    # Loading test bytes does not evaluate them. No test-derived choice is made.
    for name, count in (('optdigits.tra', 3823), ('optdigits.tes', 1797)):
        raw = (DATA / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == HASHES[name], name
        table = np.loadtxt(DATA / name, delimiter=',', dtype=np.int64)
        assert table.shape == (count, 65)
        assert ((table[:, :64] >= 0) & (table[:, :64] <= 16)).all()
        assert ((table[:, -1] >= 0) & (table[:, -1] < 10)).all()
        tables.append(table)
    train, test = tables
    rng = np.random.default_rng(2026101901)
    validation = np.sort(np.concatenate([rng.permutation(np.flatnonzero(train[:, -1] == digit))[
        :int(math.floor(.2 * np.count_nonzero(train[:, -1] == digit)))] for digit in range(10)]))
    train_ids = np.setdiff1d(np.arange(len(train)), validation) if split == 'development' else np.arange(len(train))
    eval_ids = validation if split == 'development' else np.arange(len(test))
    ev = train[eval_ids] if split == 'development' else test
    def pixels(table):
        return np.pad(table[:, :64].reshape(-1, 1, 8, 8).astype(np.float32) / 16,
                      ((0, 0), (0, 0), (4, 4), (4, 4)))
    return dict(train_x=pixels(train[train_ids]), train_y=train[train_ids, -1],
                eval_x=pixels(ev), eval_y=ev[:, -1], train_ids=train_ids, eval_ids=eval_ids)


def augment(x, moves):
    q = torch.arange(16)
    rows = (q[None, :] - torch.from_numpy(moves[:, 0]).long()[:, None]) % 16
    cols = (q[None, :] - torch.from_numpy(moves[:, 1]).long()[:, None]) % 16
    return x[torch.arange(len(x))[:, None, None], 0, rows[:, :, None], cols[:, None, :]].unsqueeze(1)


def scores(logits, labels):
    metrics = {}
    for a, arch in enumerate(('direct', 'filtered')):
        for i, name in enumerate(NAMES):
            metrics[f'{arch}_{name}_accuracy'] = float((logits[a, i].argmax(-1) == labels).mean())
    passed = all(v >= (.95 if k.endswith('original_accuracy') else .90) for k, v in metrics.items())
    return metrics, passed


def run_job(recipe: str, seed: int, split: str, out: Path, deadline: float) -> dict:
    assert recipe in ('A', 'B') and split in ('development', 'final')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); data = inputs(split)
    np.savez_compressed(out / 'inputs.npz', **data)
    metadata = dict(task='digits', recipe=recipe, seed=int(seed), split=split, data_hashes=HASHES,
                    epochs=EPOCHS, batch=BATCH, training_dtype='float32', evaluation_dtype='float64',
                    architecture_order=['direct', 'filtered'], shifts=SHIFTS,
                    split_seed=2026101901, schedule_seed=10000 + seed)
    dump(out / 'metadata.json', metadata)
    x, y = torch.from_numpy(data['train_x']), torch.from_numpy(data['train_y'])
    evaluation = np.empty((2, 5, len(data['eval_y']), 10), dtype=np.float64)
    train_seconds = 0.; eval_seconds = 0.; controls = []; counts = {}
    model = None
    try:
        with (out / 'trace.csv').open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['architecture', 'epoch', 'batch', 'examples', 'loss'])
            for a, arch in enumerate(('direct', 'filtered')):
                torch.manual_seed(seed); model = Net(recipe, arch == 'filtered').float()
                counts[arch] = sum(p.numel() for p in model.parameters())
                save_weights(out / f'{arch}_initial.npz', model)
                opt = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8,
                                       weight_decay=0., amsgrad=False, foreach=False, fused=False)
                rng = np.random.default_rng(10000 + seed); t = time.monotonic()
                for ep in range(EPOCHS):
                    order = rng.permutation(len(x)); moves = rng.integers(-2, 3, size=(len(x), 2), dtype=np.int8)
                    for b, offset in enumerate(range(0, len(x), BATCH)):
                        guard(deadline); ids = order[offset:offset + BATCH]
                        xb = augment(x[ids], moves[offset:offset + len(ids)])
                        opt.zero_grad(set_to_none=True); z = model(xb); loss = F.cross_entropy(z, y[ids])
                        guard(deadline, z, loss); loss.backward()
                        guard(deadline, *(p.grad for p in model.parameters())); opt.step()
                        guard(deadline, *model.parameters())
                        writer.writerow([arch, ep, b, len(ids), float(loss.detach())]); stream.flush()
                train_seconds += time.monotonic() - t
                save_weights(out / f'{arch}_final.npz', model)
                model.eval().double(); t = time.monotonic()
                with torch.no_grad():
                    for si, shift in enumerate(SHIFTS):
                        for offset in range(0, len(data['eval_y']), BATCH):
                            guard(deadline)
                            xb = torch.from_numpy(data['eval_x'][offset:offset + BATCH]).double()
                            xb = torch.roll(xb, shifts=shift, dims=(-2, -1))
                            z = model(xb); guard(deadline, z)
                            noop = model(xb.clone()); err = float((z - noop).abs().max())
                            assert err == 0., ('no_op', err)
                            controls.append(dict(architecture=arch, shift=si, batch=offset // BATCH, no_op_error=err))
                            evaluation[a, si, offset:offset + len(xb)] = z.numpy()
                eval_seconds += time.monotonic() - t
        np.savez_compressed(out / 'eval_raw.npz', logits=evaluation, labels=data['eval_y'])
        dump(out / 'controls.json', controls)
        metrics, passed = scores(evaluation, data['eval_y'])
        result = dict(**metadata, metrics=metrics, passed=passed, eligible=True,
                      selection_score=min(metrics.values()), parameters=counts,
                      seconds=time.monotonic() - start, training_seconds=train_seconds,
                      evaluation_seconds=eval_seconds, status='complete')
        dump(out / 'metrics.json', result)
        return result
    except Exception as exc:
        if model is not None:
            save_weights(out / 'partial_weights.npz', model)
        dump(out / 'failure.json', dict(error=repr(exc), seconds=time.monotonic() - start, metadata=metadata))
        raise


def replay(out: Path, deadline: float = float('inf')) -> dict:
    out = Path(out); start = time.monotonic()
    metadata = json.loads((out / 'metadata.json').read_text())
    with np.load(out / 'inputs.npz') as saved:
        x = saved['eval_x'].copy(); labels = saved['eval_y'].copy()
        regenerated = inputs(metadata['split'])
        for name in saved.files:
            np.testing.assert_array_equal(saved[name], regenerated[name])
    with np.load(out / 'eval_raw.npz') as saved:
        expected = saved['logits'].copy(); np.testing.assert_array_equal(labels, saved['labels'])
    actual = np.empty_like(expected); maximum = 0.
    with torch.no_grad():
        for a, arch in enumerate(('direct', 'filtered')):
            with np.load(out / f'{arch}_final.npz') as saved:
                weights = {k: torch.from_numpy(saved[k].copy()).double() for k in saved.files}
            for si, (dy, dx) in enumerate(SHIFTS):
                rows = (torch.arange(16) - dy) % 16; cols = (torch.arange(16) - dx) % 16
                for offset in range(0, len(x), BATCH):
                    guard(deadline)
                    xb = torch.from_numpy(x[offset:offset + BATCH]).double()[:, :, rows[:, None], cols[None, :]]
                    z = explicit_forward(xb, weights, arch == 'filtered'); guard(deadline, z)
                    actual[a, si, offset:offset + len(xb)] = z.numpy()
            maximum = max(maximum, float(np.abs(actual[a] - expected[a]).max()))
    assert maximum <= TOL, ('independent_logits', maximum, TOL)
    # Recompute accuracy counts without the scientific scores helper.
    metrics = {f'{arch}_{name}_accuracy': float(np.count_nonzero(actual[a, i].argmax(1) == labels) / len(labels))
               for a, arch in enumerate(('direct', 'filtered')) for i, name in enumerate(NAMES)}
    original = json.loads((out / 'metrics.json').read_text())
    assert metrics == original['metrics']
    passed = all(value >= (.95 if name.endswith('original_accuracy') else .90) for name, value in metrics.items())
    assert passed == original['passed'] and min(metrics.values()) == original['selection_score']
    result = dict(status='passed', passed=True, training_repeated=False, maximum_logit_error=maximum,
                  tolerance=TOL, logits_checked=int(actual.size), metrics=metrics,
                  gate_passed=passed, seconds=time.monotonic() - start)
    dump(out / 'replay.json', result); return result


def preflight(out: Path) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); timings = []; counts = {}; errors = []
    rng = np.random.default_rng(2026101902)
    x = torch.from_numpy(rng.random((128, 1, 16, 16), dtype=np.float32))
    y = torch.from_numpy(rng.integers(0, 10, size=128, dtype=np.int64))
    for arch in ('direct', 'filtered'):
        torch.manual_seed(901); model = Net('B', arch == 'filtered').float()
        counts[arch] = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=.001, foreach=False, fused=False)
        t = time.monotonic()
        for _ in range(20):
            opt.zero_grad(set_to_none=True); loss = F.cross_entropy(model(x), y); loss.backward()
            guard(float('inf'), loss, *(p.grad for p in model.parameters())); opt.step()
            guard(float('inf'), *model.parameters())
        timings.append((time.monotonic() - t) / 20)
        model.double().eval(); weights = {k: v.detach() for k, v in model.state_dict().items()}
        with torch.no_grad():
            t = time.monotonic(); z = model(x.double()); inference = time.monotonic() - t
            t = time.monotonic(); z2 = explicit_forward(x.double(), weights, arch == 'filtered'); audit = time.monotonic() - t
            error = float((z - z2).abs().max()); assert error <= TOL
            assert z.shape == (128, 10); assert torch.equal(z, model(x.double().clone()))
            errors.append(dict(architecture=arch, forward_error=error, inference_seconds=inference, replay_seconds=audit))
    # Eight jobs = four development recipe/seed jobs plus four selected final seeds.
    # Each job has two architectures. Use final train/test sizes for all estimates.
    train = sum(timings) * EPOCHS * math.ceil(3823 / BATCH)
    evaluate = sum(v['inference_seconds'] * 2 + v['replay_seconds'] for v in errors) * 5 * math.ceil(1797 / BATCH)
    result = dict(status='passed', passed=True, task='digits', dummy_only=True, parameters_B=counts,
                  timing_steps_per_architecture=20, training_step_seconds_B=timings, controls=errors,
                  estimated_one_job_seconds=train + evaluate, estimated_eight_jobs_seconds=8 * (train + evaluate),
                  safety_factor=4, projected_job_seconds=4 * (train + evaluate), safe_eight_jobs_seconds=4 * 8 * (train + evaluate),
                  seconds=time.monotonic() - started)
    dump(out / 'preflight.json', result); return result
