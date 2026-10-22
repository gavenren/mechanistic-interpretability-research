"""One locked timing revision. No scientific task is trained or evaluated."""
import argparse
import fcntl
import json
import os
import resource
import time
from pathlib import Path
import run as continuation
import numpy as np
import torch
from torch.nn import functional as F
import hall
from common import check_stop, dump

STUDY, ROOT = continuation.STUDY, continuation.ROOT


def measure(commit):
    continuation.verify_lock(commit, require_revised=False)
    failed = json.loads((STUDY / 'preparation/status.json').read_text())
    assert failed['status'] == 'stopped'
    assert failed['reason'] == "RuntimeError('Combined projected compute exceeds the unchanged cap')"
    assert failed['atomic_permit']['passed'] and all(x['passed'] for x in failed['tasks'].values())
    out = STUDY / 'revised-preparation'; out.mkdir(exist_ok=False)
    start = time.monotonic()
    prior_prep = failed['total_preparation_seconds']
    os.environ['MI_STAGE_DEADLINE'] = str(start + 3600 - prior_prep)
    report = dict(status='running', timing_method_lock=commit, prior_preparation_seconds=prior_prep,
                  earlier_projection_seconds=failed['conservative_stage1_seconds_with_prior_attempt'],
                  safety_factor=4, warmup_updates=20, timed_blocks=3, updates_per_block=100,
                  input_seed=99301, model_seed=99302, scientific_work=False,
                  hardware=continuation.original.hardware())
    try:
        check_stop(float('inf'))
        setup_start = time.monotonic()
        torch.manual_seed(99302)
        model = hall.Model(128)
        rng = np.random.default_rng(99301)
        x = torch.from_numpy(rng.normal(size=(512, 16)))
        y = torch.from_numpy(rng.integers(0, 2, size=512).astype(np.float64))
        opt = torch.optim.Adam(model.parameters(), lr=.002)
        def update():
            hall.check(float('inf'), *model.parameters())
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            hall.check(float('inf'), loss)
            opt.zero_grad(set_to_none=True); loss.backward()
            hall.check(float('inf'), *(p.grad for p in model.parameters()))
            opt.step(); hall.check(float('inf'), *model.parameters())
            return float(loss.detach())
        for _ in range(20):
            update()
        block_seconds, block_losses = [], []
        for _ in range(3):
            begun = time.monotonic()
            losses = [update() for _ in range(100)]
            block_seconds.append(time.monotonic() - begun)
            block_losses.append(losses)
        setup_and_benchmark = time.monotonic() - setup_start
        seconds_per_update = max(block_seconds) / 100
        old_hall = failed['tasks']['hall']
        old_eval = old_hall['projected_job_seconds']/4 - old_hall['dummy_train_seconds']/20*32768 - 10
        assert old_eval >= 0
        setup_allowance = max(10., setup_and_benchmark)
        hall_job = 4*(seconds_per_update*32768 + old_eval + setup_allowance)
        job_estimates = {task: (hall_job if task == 'hall' else row['projected_job_seconds'])
                         for task, row in failed['tasks'].items()}
        old_science = json.loads((continuation.OLD / 'results/status.json').read_text())['seconds']
        total = 8*sum(job_estimates.values()) + old_science
        with torch.no_grad():
            native = model(x).numpy()
            noop = model(x.clone()).numpy()
        weights = {k: v.detach().numpy() for k,v in model.state_dict().items()}
        explicit = hall.explicit(x.numpy(), weights)
        max_error = float(np.max(np.abs(explicit-native)))
        assert max_error <= hall.TOL and np.array_equal(native,noop)
        check_stop(float('inf'))
        report.update(block_seconds=block_seconds, maximum_seconds_per_update=seconds_per_update,
                      setup_and_benchmark_seconds=setup_and_benchmark, setup_allowance_seconds=setup_allowance,
                      old_evaluation_replay_seconds_per_job=old_eval, projected_job_seconds=job_estimates,
                      prior_scientific_seconds=old_science, combined_projected_scientific_seconds=total,
                      max_explicit_error=max_error, noop_exact=True, parameters=sum(p.numel() for p in model.parameters()))
        dump(out / 'dummy-losses.json', block_losses)
        if total > 14400:
            raise RuntimeError('Revised fixed timing estimate exceeds the unchanged cap')
        report['status'] = 'passed'
    except Exception as exc:
        import traceback
        report.update(status='stopped', reason=repr(exc), traceback=traceback.format_exc())
    finally:
        report['seconds'] = time.monotonic()-start
        report['total_preparation_seconds'] = prior_prep+report['seconds']
        report['peak_rss_native_units'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        dump(out / 'status.json', report)
        dump(out / 'manifest.json', continuation.original.files_manifest(out))
    print(json.dumps(report), flush=True)
    return report['status'] == 'passed'


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--lock',required=True); args=parser.parse_args()
    with (ROOT/'.local-state/research-compute.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        continuation.original.configure()
        raise SystemExit(0 if measure(args.lock) else 1)
