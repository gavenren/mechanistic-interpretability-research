"""Experiment 1: exact finite matching task. No intervention study is run."""
from __future__ import annotations
import csv
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import check_stop

TOL = 1e-9
WIDTHS = {'A': 64, 'B': 128}
EPOCHS, BATCH = 256, 512


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def check(deadline, *tensors):
    check_stop(deadline)
    if time.monotonic() >= deadline:
        raise RuntimeError('Fixed deadline reached')
    if any(not torch.isfinite(t).all().item() for t in tensors):
        raise RuntimeError('Nonfinite value')


def data():
    # Reuse the original exact graph labels and fixed category ledger.
    path = Path(__file__).resolve().parents[2] / '2026-10-13-hall-obstruction/code/run.py'
    spec = importlib.util.spec_from_file_location('original_hall', path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    graphs, labels, no_isolated, _, ledger = old.data_and_ledger()
    categories = {'all_graphs': np.ones(65536, dtype=bool),
                  'no_isolated_negatives': no_isolated & ~labels}
    for kind, ids in ledger['category_unions']['test'].items():
        mask = np.zeros(65536, dtype=bool)
        mask[ids] = True
        categories['test_' + kind] = mask
    assert len(categories) == 5 and all(m.any() for m in categories.values())
    return graphs.reshape(-1, 16).astype(np.float64), labels, categories


class Model(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.first = nn.Linear(16, width, dtype=torch.float64)
        self.second = nn.Linear(width, width, dtype=torch.float64)
        self.last = nn.Linear(width, 1, dtype=torch.float64)

    def forward(self, x):
        return self.last(self.second(self.first(x).relu()).relu()).squeeze(-1)


def save(path, model):
    np.savez_compressed(path, **{k: v.detach().numpy() for k, v in model.state_dict().items()})


def explicit(x, weights):
    a = np.maximum(x @ weights['first.weight'].T + weights['first.bias'], 0)
    b = np.maximum(a @ weights['second.weight'].T + weights['second.bias'], 0)
    return (b @ weights['last.weight'].T + weights['last.bias']).squeeze(-1)


def run_job(recipe: str, seed: int, split: str, out: Path, deadline: float) -> dict:
    if recipe not in WIDTHS or split not in ('development', 'final'):
        raise ValueError('Unknown recipe or split')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    x, truth, categories = data()
    tx, ty = torch.from_numpy(x), torch.from_numpy(truth.astype(np.float64))
    torch.manual_seed(seed)
    model = Model(WIDTHS[recipe])
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters <= 100000
    save(out / 'initial.npz', model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    schedule_seed = 94210 if split == 'development' else 94211
    rng = np.random.default_rng(schedule_seed)
    train_started = time.monotonic()
    try:
        with (out / 'training.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['update', 'epoch', 'loss', 'elapsed_seconds'])
            step = 0
            for epoch in range(EPOCHS):
                order = rng.permutation(65536)
                for start in range(0, 65536, BATCH):
                    check(deadline, *model.parameters())
                    ids = order[start:start+BATCH]
                    loss = F.binary_cross_entropy_with_logits(model(tx[ids]), ty[ids])
                    check(deadline, loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    check(deadline, *(p.grad for p in model.parameters()))
                    optimizer.step()
                    check(deadline, *model.parameters())
                    step += 1
                    writer.writerow([step, epoch+1, loss.item(), time.monotonic()-train_started])
                stream.flush()
    except BaseException:
        save(out / 'partial.npz', model)
        raise
    train_seconds = time.monotonic() - train_started
    save(out / 'final.npz', model)
    eval_started = time.monotonic()
    with torch.no_grad():
        chunks = []
        for start in range(0, len(x), 512):
            check(deadline)
            value = model(tx[start:start+512])
            check(deadline, value)
            chunks.append(value.numpy())
    logits = np.concatenate(chunks)
    predictions = logits >= 0
    metrics = {name: float((predictions[mask] == truth[mask]).mean()) for name, mask in categories.items()}
    np.savez_compressed(out / 'evaluation.npz', graph_ids=np.arange(65536, dtype=np.uint32),
                        labels=truth, predictions=predictions, logits=logits,
                        **{'category_'+k: v for k, v in categories.items()})
    result = {'task': 'hall', 'recipe': recipe, 'seed': seed, 'split': split,
              'metrics': metrics, 'passed': all(v >= .99 for v in metrics.values()),
              'selection_score': min(metrics.values()), 'eligible': True,
              'seconds': time.monotonic()-started, 'train_seconds': train_seconds,
              'evaluation_seconds': time.monotonic()-eval_started, 'updates': step,
              'parameters': parameters, 'width': WIDTHS[recipe], 'precision': 'float64',
              'schedule_seed': schedule_seed, 'batch_size': BATCH,
              'category_counts': {k: int(v.sum()) for k, v in categories.items()},
              'data_scope': 'All 65,536 graph states are used for training and evaluation.'}
    dump(out / 'metrics.json', result)
    return result


def replay(out: Path, deadline: float = float('inf')) -> dict:
    started = time.monotonic()
    out = Path(out)
    w = dict(np.load(out / 'final.npz'))
    values = np.load(out / 'evaluation.npz')
    ids = values['graph_ids']
    x = ((ids[:, None] >> np.arange(16)) & 1).astype(np.float64)
    chunks = []
    for i in range(0, len(x), 512):
        check(deadline)
        chunks.append(explicit(x[i:i+512], w))
    logits = np.concatenate(chunks)
    error = float(np.max(np.abs(logits-values['logits'])))
    same = bool(np.array_equal(logits >= 0, values['predictions']))
    metrics = json.loads((out / 'metrics.json').read_text())['metrics']
    metrics_same = all(float(((logits >= 0)[values['category_'+k]] == values['labels'][values['category_'+k]]).mean()) == v
                       for k, v in metrics.items())
    return {'passed': bool(error <= TOL and same and metrics_same), 'max_logit_error': error,
            'tolerance': TOL, 'same_predictions': same, 'same_metrics': metrics_same,
            'rows': len(x), 'seconds': time.monotonic()-started,
            'method': 'NumPy explicit affine and ReLU equations; saved weights, no retraining.'}


def preflight(out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    check(float("inf"))
    torch.manual_seed(99101)
    model = Model(128)
    rng = np.random.default_rng(99102)
    # Continuous dummy inputs and random labels are outside the graph task.
    x = torch.from_numpy(rng.normal(size=(512, 16)))
    y = torch.from_numpy(rng.integers(0, 2, size=512).astype(np.float64))
    parameters = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=.002)
    began = time.monotonic()
    for _ in range(20):
        check(float("inf"))
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        check(float('inf'), loss, *(p.grad for p in model.parameters()))
        opt.step()
        check(float('inf'), *model.parameters())
    seconds = time.monotonic()-began
    with torch.no_grad():
        native = model(x).numpy()
        noop = model(x.clone()).numpy()
    weights = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    began = time.monotonic()
    direct = explicit(x.numpy(), weights)
    replay_seconds = time.monotonic()-began
    error = float(np.max(np.abs(native-direct)))
    # An empty graph has no matching. A diagonal and a full graph have a matching.
    import itertools
    known = [np.zeros((4,4), dtype=int), np.eye(4, dtype=int), np.ones((4,4), dtype=int)]
    oracle = [any(g[np.arange(4), p].all() for p in itertools.permutations(range(4))) for g in known]
    passed = bool(parameters <= 100000 and native.shape == (512,) and native.dtype == np.float64
                  and np.array_equal(native, noop) and error <= TOL and oracle == [False, True, True])
    projected = 4*(seconds/20*(256*128) + max(replay_seconds,.001)*128*2 + 10)
    result = {'passed': passed, 'parameters': parameters, 'dummy_steps': 20,
              'dummy_train_seconds': seconds, 'dummy_replay_seconds': replay_seconds,
              'max_replay_error': error, 'noop_exact': bool(np.array_equal(native, noop)),
              'oracle_sanity': oracle, 'projected_job_seconds': projected,
              'projected_eight_job_seconds': 8*projected,
              'projection': 'Four times measured larger-recipe update cost, eval/replay, and 10-second setup allowance.',
              'scientific_training_or_evaluation': False}
    dump(out / 'preflight.json', result)
    return result
