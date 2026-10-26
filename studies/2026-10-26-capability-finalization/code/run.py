"""Finish fixed Experiment 1 without changing old methods or output files."""
from __future__ import annotations
import argparse
import copy
import csv
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'
STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
OLD = STUDY.parent / '2026-10-19-capability-gates'
PREVIOUS = STUDY.parent / '2026-10-22-capability-completion'
SOURCE = '58251c651ef6397ee26ae0c0159d0f7632063191'
METHOD_SOURCE = 'c421ca112a0b421862934efb97bdecc0ebf323d3'
TASKS = ('hall', 'register', 'digits', 'flow')
DEV, FINAL = (8001, 8002), (8101, 8102, 8103, 8104)
PRIOR_SCIENTIFIC = 317.0650452510017
PRIOR_PREPARATION = 15.607510499994532
PRELOCK_CHECK_CHARGE = 60.0
CLOSING_RECEIPT_ALLOWANCE = 1.0
LOCK_DOCUMENTS = ('analysis-plan.md', 'sources.md', 'README.md')
REUSED_KEYS = [('development', 'A', 8001), ('development', 'A', 8002),
               ('development', 'B', 8001), ('development', 'B', 8002),
               ('final', 'A', 8101), ('final', 'A', 8102)]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def remaining_job_counts():
    # A digit job contains two distinct model trainings.
    return {'hall': 2, 'register': 8, 'digits': 8, 'flow': 8}


def remaining_graph_jobs():
    return [('hall', 'A', 8103, 'final'), ('hall', 'A', 8104, 'final')]


def selected_recipe(task, development):
    eligible = [r for r in ('A', 'B')
                if task != 'flow' or all(x['eligible'] for x in development[r])]
    if not eligible:
        return None
    def rank(recipe):
        rows = development[recipe]
        score = (sum(x['selection_score'] for x in rows) / len(rows)
                 if task == 'flow' else min(x['selection_score'] for x in rows))
        return (-score, sum(x['seconds'] for x in rows), recipe)
    return min(eligible, key=rank)


def remaining_seconds(stage):
    return (3600 - PRIOR_PREPARATION - PRELOCK_CHECK_CHARGE) if stage == 'prepare' else (14400 - PRIOR_SCIENTIFIC)


def verify_bytes(commit, path):
    path = Path(path)
    relative = path.relative_to(ROOT).as_posix()
    published = subprocess.check_output(['git', 'show', commit + ':' + relative], cwd=ROOT)
    if published != path.read_bytes():
        raise RuntimeError('Published file differs: ' + relative)


def verify_manifest(folder):
    for relative, expected in json.loads((folder / 'manifest.json').read_text()).items():
        if hashlib.sha256((folder / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Result hash differs: ' + str((folder / relative).relative_to(ROOT)))


def verify_sources():
    for folder in (OLD / 'code', PREVIOUS / 'code'):
        for path in sorted(folder.glob('*.py')):
            verify_bytes(METHOD_SOURCE if folder == OLD / 'code' else SOURCE, path)
    fixed_paths = [OLD / n for n in ('METHODS_SYMBOLIC.md', 'METHODS_NUMERIC.md', 'analysis-plan.md')]
    fixed_paths += [ROOT / n for n in ('studies/2026-10-13-hall-obstruction/code/run.py',
                                      'studies/2026-10-13-shift-ablation/data/optdigits.tra',
                                      'studies/2026-10-13-shift-ablation/data/optdigits.tes')]
    for path in fixed_paths:
        verify_bytes(METHOD_SOURCE, path)
    paths = [OLD / n for n in ('results/status.json', 'results/manifest.json')]
    paths += [PREVIOUS / n for n in ('analysis-plan.md', 'TIMING_REVISION.md',
                                    'results/status.json', 'results/manifest.json',
                                    'revised-preparation/status.json')]
    for path in paths:
        verify_bytes(SOURCE, path)
    verify_manifest(OLD / 'results'); verify_manifest(PREVIOUS / 'results')
    previous = json.loads((PREVIOUS / 'results/status.json').read_text())
    preparation = json.loads((PREVIOUS / 'revised-preparation/status.json').read_text())
    if abs(previous['total_stage1_seconds'] - PRIOR_SCIENTIFIC) > 1e-9:
        raise RuntimeError('Prior scientific accounting differs')
    if abs(preparation['total_preparation_seconds'] - PRIOR_PREPARATION) > 1e-9:
        raise RuntimeError('Prior preparation accounting differs')
    if previous['status'] != 'stopped' or previous['branches']['hall']['selected_recipe'] != 'A':
        raise RuntimeError('Unexpected prior task state')
    keys = [(r['split'], r['recipe'], r['seed']) for r in previous['jobs']]
    if keys != REUSED_KEYS or any(r['task'] != 'hall' for r in previous['jobs']):
        raise RuntimeError('Unexpected complete-model ledger')
    return previous


def verify_lock(commit, with_preparation=False):
    for path in sorted((STUDY / 'code').glob('*.py')):
        verify_bytes(commit, path)
    for name in LOCK_DOCUMENTS:
        verify_bytes(commit, STUDY / name)
    if with_preparation:
        folder = STUDY / 'preparation'
        verify_bytes(commit, folder / 'status.json')
        verify_bytes(commit, folder / 'manifest.json')
        verify_manifest(folder)
        preparation = json.loads((folder / 'status.json').read_text())
        if preparation['status'] != 'passed':
            raise RuntimeError('Preparation has not passed')
        # Preparation predates the result-containing commit. Its source lock must
        # have exactly the same scientific code and plan as the execution lock.
        verify_lock(preparation['plan_lock'], with_preparation=False)
        subprocess.run(['git', 'merge-base', '--is-ancestor', preparation['plan_lock'], commit],
                       cwd=ROOT, check=True, capture_output=True)
    return verify_sources()


def runtime():
    sys.path.insert(0, str(PREVIOUS / 'code'))
    sys.path.insert(0, str(OLD / 'code'))
    common = importlib.import_module('common')
    original_check = common.check_stop
    last_disk_check = [float('-inf')]
    def guarded_check(deadline):
        check_private_permit()
        try:
            original_check(deadline)
        except TimeoutError:
            raise
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            raise RuntimeError('Usage permit cannot be read or is invalid') from None
        now = time.monotonic()
        if now - last_disk_check[0] >= 5:
            if shutil.disk_usage(ROOT).free < 20 * 1024**3:
                raise RuntimeError('Less than 20 GiB free disk')
            last_disk_check[0] = now
    common.check_stop = guarded_check
    spec = importlib.util.spec_from_file_location('fixed_original_runner', OLD / 'code/run.py')
    original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    original.configure()
    modules = {task: importlib.import_module(task) for task in TASKS}
    return original, modules, guarded_check


def check_private_permit():
    # The existing atomic writer remains the only permit issuer. This reader
    # adds strict validation before the unchanged resource check runs.
    try:
        permit = json.loads(Path(os.environ['MI_USAGE_PERMIT']).read_text())
        expiry = permit['expires_at']
        now = time.time()
        valid = (permit['allowed'] is True and type(expiry) in (int, float)
                 and math.isfinite(expiry) and now < expiry <= now + 120)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError):
        raise RuntimeError('Usage permit cannot be read or is invalid') from None
    if not valid:
        raise RuntimeError('Usage permit is invalid, denied, expired, or too long')


def reuse_jobs(previous):
    rows = []
    for item in previous['jobs']:
        row = copy.deepcopy(item)
        source = row.get('source_path')
        if not source:
            source = (PREVIOUS / 'results/hall' / f"{row['split']}-{row['recipe']}-{row['seed']}").relative_to(ROOT).as_posix()
        row.update(reused=True, source_path=source)
        rows.append(row)
    return rows


def replay_reused(rows, hall, check, deadline, save=lambda: None):
    for row in rows:
        check(deadline)
        begun = time.monotonic()
        row['continuation_replay'] = hall.replay(ROOT / row['source_path'], deadline)
        row['continuation_replay_seconds'] = time.monotonic() - begun
        save()
        if not row['continuation_replay']['passed']:
            raise RuntimeError('A reused graph model failed replay')
    return rows


def hall_preparation(out, hall, check):
    import numpy as np
    import torch
    from torch.nn import functional as F
    new_output(out)
    begun = time.monotonic()
    torch.manual_seed(99302)
    model = hall.Model(128)
    rng = np.random.default_rng(99301)
    x = torch.from_numpy(rng.normal(size=(512, 16)))
    y = torch.from_numpy(rng.integers(0, 2, size=512).astype(np.float64))
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    def update():
        hall.check(float('inf'), *model.parameters())
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        hall.check(float('inf'), loss)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        hall.check(float('inf'), *(p.grad for p in model.parameters()))
        optimizer.step(); hall.check(float('inf'), *model.parameters())
        return float(loss.detach())
    for _ in range(20):
        update()
    blocks, losses = [], []
    for _ in range(3):
        started = time.monotonic()
        losses.append([update() for _ in range(100)])
        blocks.append(time.monotonic() - started)
    setup = time.monotonic() - begun
    with torch.no_grad():
        native = model(x).numpy(); noop = model(x.clone()).numpy()
    weights = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    started = time.monotonic(); direct = hall.explicit(x.numpy(), weights)
    replay = time.monotonic() - started
    error = float(np.max(np.abs(native - direct)))
    parameters = sum(p.numel() for p in model.parameters())
    import itertools
    known = [np.zeros((4, 4), dtype=int), np.eye(4, dtype=int), np.ones((4, 4), dtype=int)]
    oracle = [any(g[np.arange(4), p].all() for p in itertools.permutations(range(4))) for g in known]
    projected = 4 * (max(blocks) / 100 * 32768 + max(replay, .001) * 128 * 2 + max(10., setup))
    row = dict(passed=bool(parameters <= 100000 and native.shape == (512,) and
                          native.dtype == np.float64 and error <= hall.TOL and
                          np.array_equal(native, noop) and oracle == [False, True, True]),
               warmup_updates=20, timed_blocks=3, updates_per_block=100,
               block_seconds=blocks, maximum_seconds_per_update=max(blocks) / 100,
               setup_and_benchmark_seconds=setup, dummy_replay_seconds=replay,
               max_replay_error=error, tolerance=hall.TOL, output_shape=list(native.shape),
               output_dtype=str(native.dtype), noop_exact=bool(np.array_equal(native, noop)),
               oracle_sanity=oracle, parameters=parameters, projected_job_seconds=projected,
               safety_factor=4, scientific_training_or_evaluation=False)
    check(float('inf')); atomic_json(out / 'dummy-losses.json', losses)
    atomic_json(out / 'preflight.json', row)
    return row


def manifest(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file() and
            p.name != 'manifest.json' and '__pycache__' not in p.parts and not p.name.endswith('.tmp')}


def safe_error(exc):
    reason = str(exc).replace(str(ROOT), '[repository]')
    private_permit = os.environ.get('MI_USAGE_PERMIT')
    if private_permit:
        reason = reason.replace(private_permit, '[private permit]')
    return dict(error_type=type(exc).__name__, reason=reason)


def peak_memory_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == 'darwin' else peak * 1024


def close_receipts(out, report, started, stage):
    # First save all main results and scan their hashes. Then take the final
    # elapsed time and charge a reserved second for the final receipt writes.
    atomic_json(out / 'status.json', report)
    hashes = manifest(out)
    report['peak_memory_bytes'] = peak_memory_bytes()
    total_key = 'total_preparation_seconds' if stage == 'prepare' else 'total_stage1_seconds'
    prior = PRIOR_PREPARATION + PRELOCK_CHECK_CHARGE if stage == 'prepare' else PRIOR_SCIENTIFIC
    report['closing_receipt_allowance_seconds'] = CLOSING_RECEIPT_ALLOWANCE
    report['timing_boundary'] = ('Measured time includes main result writes and the full hash scan. The charged seconds '
                                 'add one reserved second for the final status write, its hash, and the manifest write. '
                                 'A measured closing-write overrun stops the run.')
    def write_closing_receipts():
        report['measured_seconds_before_closing_receipts'] = time.monotonic() - started
        report['seconds'] = report['measured_seconds_before_closing_receipts'] + CLOSING_RECEIPT_ALLOWANCE
        report[total_key] = prior + report['seconds']
        if report['seconds'] >= remaining_seconds(stage):
            report.update(status='stopped', error_type='TimeoutError', reason='Stage time limit reached during final saving')
        atomic_json(out / 'status.json', report)
        hashes['status.json'] = hashlib.sha256((out / 'status.json').read_bytes()).hexdigest()
        atomic_json(out / 'manifest.json', hashes)
    closing_started = time.monotonic()
    write_closing_receipts()
    closing_seconds = time.monotonic() - closing_started
    if closing_seconds > CLOSING_RECEIPT_ALLOWANCE:
        report.update(status='stopped', error_type='TimeoutError',
                      reason='Final receipt writes exceeded their reserved time',
                      closing_receipt_overrun_seconds=closing_seconds - CLOSING_RECEIPT_ALLOWANCE)
        # Charge the observed overrun and reserve one further second to record
        # the stop. This is a failure receipt, never a claim of a completed run.
        write_closing_receipts()


def prepare(commit):
    started = time.monotonic(); out = new_output(STUDY / 'preparation')
    os.environ['MI_STAGE_DEADLINE'] = str(started + remaining_seconds('prepare') - CLOSING_RECEIPT_ALLOWANCE)
    report = dict(status='running', plan_lock=commit, prior_preparation_seconds=PRIOR_PREPARATION,
                  prior_scientific_seconds=PRIOR_SCIENTIFIC,
                  prelock_check_charge_seconds=PRELOCK_CHECK_CHARGE,
                  prelock_check_accounting='Conservative bound for code checks, source checks, and repeated small tests.',
                  tasks={}, scientific_training=False)
    try:
        previous = verify_lock(commit)
        original, modules, check = runtime()
        check(float('inf')); report['hardware'] = original.hardware()
        from test_permit import check as permit_test
        report['atomic_permit'] = permit_test()
        check(float('inf'))
        report['tasks']['hall'] = hall_preparation(out / 'hall', modules['hall'], check)
        if not report['tasks']['hall']['passed']:
            raise RuntimeError('hall preparation failed')
        for task in TASKS[1:]:
            row = modules[task].preflight(out / task)
            report['tasks'][task] = row
            atomic_json(out / 'status.json', report)
            if not row['passed']:
                raise RuntimeError(task + ' preparation failed')
        replays = reuse_jobs(previous)
        replay_reused(replays, modules['hall'], check, float('inf'),
                      lambda: atomic_json(out / 'reused-model-replays.json', replays))
        replay_seconds = sum(row['continuation_replay_seconds'] for row in replays)
        atomic_json(out / 'reused-model-replays.json', replays)
        projected = sum(remaining_job_counts()[task] * report['tasks'][task]['projected_job_seconds'] for task in TASKS)
        setup_allowance = 4 * max(10., time.monotonic() - started)
        total = PRIOR_SCIENTIFIC + projected + 4 * replay_seconds + setup_allowance
        report.update(remaining_job_counts=remaining_job_counts(),
                      projected_remaining_job_seconds=projected,
                      measured_reused_replay_seconds=replay_seconds,
                      extra_replay_allowance_seconds=4 * replay_seconds,
                      continuation_setup_allowance_seconds=setup_allowance,
                      combined_projected_scientific_seconds=total, safety_factor=4,
                      measured_replay_accounting='Preparation only; the scientific estimate includes its repeated replay.')
        if total > 14400:
            raise RuntimeError('Projected combined scientific work exceeds four hours')
        check(float('inf')); report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='stopped', **safe_error(exc))
    finally:
        report['seconds'] = time.monotonic() - started
        report['total_preparation_seconds'] = PRIOR_PREPARATION + PRELOCK_CHECK_CHARGE + report['seconds']
        report['peak_memory_bytes'] = peak_memory_bytes()
        close_receipts(out, report, started, 'prepare')
    print(json.dumps(report), flush=True)
    return report['status'] == 'passed'


def save_results(out, report, started):
    report['seconds'] = time.monotonic() - started
    report['total_stage1_seconds'] = PRIOR_SCIENTIFIC + report['seconds']
    report['peak_memory_bytes'] = peak_memory_bytes()
    report['complete_distinct_models'] = sum(2 if r['task'] == 'digits' else 1 for r in report['jobs'])
    report['new_complete_model_trainings'] = sum((2 if r['task'] == 'digits' else 1) for r in report['jobs'] if not r.get('reused'))
    atomic_json(out / 'status.json', report)
    fields = ['task', 'split', 'recipe', 'seed', 'metric', 'value', 'model_gate_passed', 'reused']
    temporary = out / 'metrics.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in report['jobs']:
            for metric, value in row['metrics'].items():
                writer.writerow(dict(task=row['task'], split=row['split'], recipe=row['recipe'], seed=row['seed'],
                                     metric=metric, value=value, model_gate_passed=row['passed'], reused=row.get('reused', False)))
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, out / 'metrics.csv')


def experiment(commit):
    started = time.monotonic(); out = new_output(STUDY / 'results')
    deadline = started + remaining_seconds('experiment') - CLOSING_RECEIPT_ALLOWANCE
    os.environ['MI_STAGE_DEADLINE'] = str(deadline)
    report = dict(status='running', public_lock=commit, source_commit=SOURCE,
                  prior_scientific_seconds=PRIOR_SCIENTIFIC, jobs=[], branches={}, current_job=None,
                  retained_partial_training_invocations=2,
                  retained_partial_attempts=[
                    'studies/2026-10-19-capability-gates/results/hall/development-A-8002',
                    'studies/2026-10-22-capability-completion/results/hall/final-A-8103'],
                  restart_rule='One full restart of final graph seed 8103 from original initialization; no automatic retry.')
    try:
        previous = verify_lock(commit, with_preparation=True)
        original, modules, check = runtime(); check(deadline)
        report['hardware'] = original.hardware()
        report['jobs'] = reuse_jobs(previous)
        save_results(out, report, started)
        replay_reused(report['jobs'], modules['hall'], check, deadline,
                      lambda: save_results(out, report, started))
        report['branches']['hall'] = dict(selected_recipe='A', selection_reused=True, status='running',
                                          final_passes=[row['passed'] for row in report['jobs'] if row['split'] == 'final'])
        save_results(out, report, started)
        def job(task, recipe, seed, split):
            check(deadline)
            report['current_job'] = dict(task=task, recipe=recipe, seed=seed, split=split)
            save_results(out, report, started)
            path = out / task / f'{split}-{recipe}-{seed}'
            row = modules[task].run_job(recipe, seed, split, path, deadline)
            row['reused'] = False
            # Keep a completed training result even if its following replay stops.
            report['jobs'].append(row); save_results(out, report, started)
            check(deadline); begun = time.monotonic()
            row['replay'] = modules[task].replay(path, deadline)
            row['replay_seconds'] = time.monotonic() - begun
            save_results(out, report, started)
            if not row['replay']['passed']:
                raise RuntimeError(task + ' saved-weight replay failed')
            report['current_job'] = None; save_results(out, report, started)
            print(json.dumps(dict(task=task, split=split, recipe=recipe, seed=seed,
                                  passed=row['passed'], metrics=row['metrics'])), flush=True)
            return row
        for task, recipe, seed, split in remaining_graph_jobs():
            row = job(task, recipe, seed, split)
            report['branches']['hall']['final_passes'].append(row['passed'])
        report['branches']['hall']['status'] = 'passed' if all(report['branches']['hall']['final_passes']) else 'failed'
        save_results(out, report, started)
        for task in TASKS[1:]:
            development = {r: [job(task, r, seed, 'development') for seed in DEV] for r in ('A', 'B')}
            selected = selected_recipe(task, development)
            if selected is None:
                report['branches'][task] = dict(status='stopped_at_development', final_passes=[],
                                                reason='Neither recipe passed the development region-error rule')
                save_results(out, report, started); continue
            report['branches'][task] = dict(selected_recipe=selected, status='running', final_passes=[])
            save_results(out, report, started)
            for seed in FINAL:
                row = job(task, selected, seed, 'final')
                report['branches'][task]['final_passes'].append(row['passed'])
            report['branches'][task]['status'] = 'passed' if all(report['branches'][task]['final_passes']) else 'failed'
            save_results(out, report, started)
        verify_lock(commit, with_preparation=True); check(deadline)
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='stopped', **safe_error(exc))
        for branch in report['branches'].values():
            if branch['status'] == 'running':
                branch['status'] = 'incomplete'
    finally:
        save_results(out, report, started)
        close_receipts(out, report, started, 'experiment')
    print(json.dumps(report), flush=True)
    return report['status'] == 'complete'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'experiment'])
    parser.add_argument('--lock', required=True); args = parser.parse_args()
    (ROOT / '.local-state').mkdir(exist_ok=True)
    with (ROOT / '.local-state/research-compute.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(0 if (prepare(args.lock) if args.stage == 'prepare' else experiment(args.lock)) else 1)
