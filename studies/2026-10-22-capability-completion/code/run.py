"""Continue the fixed Experiment 1. Retain the earlier failed attempt."""
import argparse
import copy
import csv
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
OLD = STUDY.parent / '2026-10-19-capability-gates'
ORIGINAL = 'c421ca112a0b421862934efb97bdecc0ebf323d3'
sys.path.insert(0, str(OLD / 'code'))
from common import check_stop, dump
spec = importlib.util.spec_from_file_location('original_runner', OLD / 'code/run.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
TASKS, DEV, FINAL = original.TASKS, original.DEV, original.FINAL
from test_permit import check as permit_test


def verify_bytes(commit, path):
    rel = path.relative_to(ROOT).as_posix()
    if subprocess.check_output(['git', 'show', commit + ':' + rel], cwd=ROOT) != path.read_bytes():
        raise RuntimeError('Published source differs: ' + rel)


def verify_original():
    for p in sorted((OLD / 'code').glob('*.py')):
        verify_bytes(ORIGINAL, p)
    for name in ('METHODS_SYMBOLIC.md', 'METHODS_NUMERIC.md', 'analysis-plan.md',
                 'preparation/status.json', 'results/status.json', 'results/manifest.json'):
        verify_bytes(ORIGINAL, OLD / name)
    for rel in ('studies/2026-10-13-hall-obstruction/code/run.py',
                'studies/2026-10-13-shift-ablation/data/optdigits.tra',
                'studies/2026-10-13-shift-ablation/data/optdigits.tes'):
        verify_bytes(ORIGINAL, ROOT / rel)
    manifest = json.loads((OLD / 'results/manifest.json').read_text())
    for rel, expected in manifest.items():
        if hashlib.sha256((OLD / 'results' / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Prior result hash differs: ' + rel)


def verify_lock(commit, require_revised=True):
    for p in sorted((STUDY / 'code').glob('*.py')):
        verify_bytes(commit, p)
    for name in ('analysis-plan.md', 'sources.md', 'permit-check.json',
                 'preparation/status.json', 'preparation/manifest.json', 'TIMING_REVISION.md'):
        verify_bytes(commit, STUDY / name)
    if require_revised:
        for name in ('revised-preparation/status.json', 'revised-preparation/manifest.json'):
            verify_bytes(commit, STUDY / name)
    verify_original()


def prepare():
    verify_original()
    out = STUDY / 'preparation'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    old_preparation = json.loads((OLD / 'preparation/status.json').read_text())['seconds']
    first_permit_test = json.loads((STUDY / 'permit-check.json').read_text())['seconds']
    prior = old_preparation + first_permit_test
    os.environ['MI_STAGE_DEADLINE'] = str(start + 3600 - prior)
    report = dict(status='running', prior_preparation_seconds=prior,
                  original_preparation_seconds=old_preparation, initial_permit_test_seconds=first_permit_test,
                  hardware=original.hardware(), tasks={}, atomic_permit=permit_test())
    try:
        check_stop(float('inf'))
        if report['hardware']['free_disk_bytes'] < 20*1024**3:
            raise RuntimeError('Less than 20 GiB disk free')
        for task in TASKS:
            row = importlib.import_module(task).preflight(out / task)
            report['tasks'][task] = row
            dump(out / 'status.json', report)
            if not row['passed']:
                raise RuntimeError(task + ' preparation failed')
        # Keep the full 8-job/task estimate although one completed job is reused.
        projected = sum(8 * x['projected_job_seconds'] for x in report['tasks'].values())
        old_seconds = json.loads((OLD / 'results/status.json').read_text())['seconds']
        report['conservative_stage1_seconds_with_prior_attempt'] = projected + old_seconds
        if projected + old_seconds > 14400:
            raise RuntimeError('Combined projected compute exceeds the unchanged cap')
        check_stop(float('inf'))
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='stopped', reason=repr(exc))
    finally:
        report['seconds'] = time.monotonic() - start
        report['total_preparation_seconds'] = prior + report['seconds']
        report['peak_rss_native_units'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        dump(out / 'status.json', report)
        dump(out / 'manifest.json', original.files_manifest(out))
    print(json.dumps(report), flush=True)
    return report['status'] == 'passed'


def save(out, report):
    dump(out / 'status.json', report)
    fields = ['task', 'split', 'recipe', 'seed', 'metric', 'value', 'model_gate_passed', 'reused']
    with (out / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for job in report['jobs']:
            for metric, value in job['metrics'].items():
                writer.writerow(dict(task=job['task'], split=job['split'], recipe=job['recipe'],
                                     seed=job['seed'], metric=metric, value=value,
                                     model_gate_passed=job['passed'], reused=job.get('reused', False)))


def experiment(commit):
    verify_lock(commit)
    if json.loads((STUDY / 'revised-preparation/status.json').read_text())['status'] != 'passed':
        raise RuntimeError('Preparation did not pass')
    out = STUDY / 'results'; out.mkdir(exist_ok=False)
    previous = json.loads((OLD / 'results/status.json').read_text())
    start = time.monotonic()
    deadline = start + 14400 - previous['seconds']
    os.environ['MI_STAGE_DEADLINE'] = str(deadline)
    report = dict(status='running', public_lock=commit, original_commit=ORIGINAL,
                  hardware=original.hardware(), prior_attempt_seconds=previous['seconds'],
                  jobs=[], branches={}, current_job=None,
                  partial_attempt_retained='studies/2026-10-19-capability-gates/results/hall/development-A-8002',
                  partial_updates_recorded=13600,
                  partial_weights_may_include_one_unlogged_update=True)
    try:
        for task in TASKS:
            module = importlib.import_module(task)
            dev = {}
            for recipe in ('A', 'B'):
                dev[recipe] = []
                for seed in DEV:
                    check_stop(deadline)
                    report['current_job'] = dict(task=task, recipe=recipe, seed=seed, split='development')
                    save(out, report)
                    if (task, recipe, seed) == ('hall', 'A', 8001):
                        row = copy.deepcopy(previous['jobs'][0])
                        path = OLD / 'results/hall/development-A-8001'
                        row['reused'] = True
                        row['source_path'] = path.relative_to(ROOT).as_posix()
                    else:
                        path = out / task / f'development-{recipe}-{seed}'
                        row = module.run_job(recipe, seed, 'development', path, deadline)
                        row['reused'] = False
                    check_stop(deadline)
                    audit_start = time.monotonic()
                    row['replay'] = module.replay(path)
                    row['replay_seconds'] = time.monotonic() - audit_start
                    report['jobs'].append(row); dev[recipe].append(row)
                    report['current_job'] = None
                    save(out, report)
                    print(json.dumps(dict(task=task, split='development', recipe=recipe, seed=seed, passed=row['passed'], metrics=row['metrics'])), flush=True)
                    if not row['replay']['passed']:
                        raise RuntimeError(task + ' replay failed')
            eligible = [r for r in ('A', 'B') if task != 'flow' or all(x['eligible'] for x in dev[r])]
            if not eligible:
                report['branches'][task] = dict(status='stopped_at_development', reason='Neither recipe passed the region-error rule', final_passes=[])
                continue
            def rank(recipe):
                rows = dev[recipe]
                score = sum(x['selection_score'] for x in rows)/len(rows) if task == 'flow' else min(x['selection_score'] for x in rows)
                return (-score, sum(x['seconds'] for x in rows), recipe)
            selected = min(eligible, key=rank)
            report['branches'][task] = dict(selected_recipe=selected, status='running', final_passes=[])
            save(out, report)
            for seed in FINAL:
                check_stop(deadline)
                report['current_job'] = dict(task=task, recipe=selected, seed=seed, split='final')
                save(out, report)
                path = out / task / f'final-{selected}-{seed}'
                row = module.run_job(selected, seed, 'final', path, deadline)
                row['reused'] = False
                check_stop(deadline)
                audit_start = time.monotonic()
                row['replay'] = module.replay(path)
                row['replay_seconds'] = time.monotonic() - audit_start
                report['jobs'].append(row)
                report['branches'][task]['final_passes'].append(row['passed'])
                report['current_job'] = None
                save(out, report)
                print(json.dumps(dict(task=task, split='final', recipe=selected, seed=seed, passed=row['passed'], metrics=row['metrics'])), flush=True)
                if not row['replay']['passed']:
                    raise RuntimeError(task + ' replay failed')
            report['branches'][task]['status'] = 'passed' if all(report['branches'][task]['final_passes']) else 'failed'
        verify_lock(commit)
        report['status'] = 'complete'
    except Exception as exc:
        import traceback
        report.update(status='stopped', reason=repr(exc), traceback=traceback.format_exc())
    finally:
        report['seconds'] = time.monotonic() - start
        report['total_stage1_seconds'] = previous['seconds'] + report['seconds']
        report['peak_rss_native_units'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['complete_distinct_models'] = sum(2 if r['task'] == 'digits' else 1 for r in report['jobs'])
        report['new_complete_model_trainings'] = sum((2 if r['task'] == 'digits' else 1) for r in report['jobs'] if not r.get('reused'))
        report['retained_partial_training_invocations'] = 1
        save(out, report)
        dump(out / 'manifest.json', original.files_manifest(out))
    print(json.dumps(report), flush=True)
    return report['status'] == 'complete'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'experiment'])
    parser.add_argument('--lock'); args = parser.parse_args()
    if args.stage == 'experiment' and not args.lock:
        parser.error('--lock is required')
    with (ROOT / '.local-state/research-compute.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original.configure()
        raise SystemExit(0 if (prepare() if args.stage == 'prepare' else experiment(args.lock)) else 1)
