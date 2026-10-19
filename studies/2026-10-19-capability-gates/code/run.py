"""Serial runner for the fixed preparation and task-ability experiment."""
import argparse
import hashlib
import importlib
import json
import os
import platform
import resource
import shutil
import subprocess
import time
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
import torch
from common import check_stop, dump

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
TASKS = ('hall', 'register', 'digits', 'flow')
DEV = (8001, 8002)
FINAL = (8101, 8102, 8103, 8104)


def configure():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def files_manifest(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file()
            and '__pycache__' not in p.parts and p.name != 'manifest.json'}


def hardware():
    data = dict(platform=platform.platform(), machine=platform.machine(),
                python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                compute_threads=2, interop_threads=1,
                free_disk_bytes=shutil.disk_usage(ROOT).free,
                process_memory_cap_bytes=4 * 1024**3)
    if platform.system() == 'Darwin':
        for key in ('machdep.cpu.brand_string', 'hw.memsize'):
            data[key] = subprocess.check_output(['sysctl', '-n', key], text=True).strip()
        data['vm_stat'] = subprocess.check_output(['vm_stat'], text=True)
    return data


def verify_lock(commit):
    relative = STUDY.relative_to(ROOT).as_posix()
    for name in ('code', 'analysis-plan.md', 'METHODS_SYMBOLIC.md', 'METHODS_NUMERIC.md'):
        path = STUDY / name
        paths = sorted(path.glob('*.py')) if path.is_dir() else [path]
        for p in paths:
            original = subprocess.check_output(['git', 'show', commit + ':' + p.relative_to(ROOT).as_posix()], cwd=ROOT)
            if original != p.read_bytes():
                raise RuntimeError('Public lock differs: ' + str(p))
    for rel in ('studies/2026-10-13-hall-obstruction/code/run.py',
                'studies/2026-10-13-shift-ablation/data/optdigits.tra',
                'studies/2026-10-13-shift-ablation/data/optdigits.tes'):
        if subprocess.check_output(['git', 'show', commit + ':' + rel], cwd=ROOT) != (ROOT / rel).read_bytes():
            raise RuntimeError('Reused dependency differs: ' + rel)
    return commit


def prepare():
    out = STUDY / 'preparation'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    deadline = start + 3600
    os.environ['MI_STAGE_DEADLINE'] = str(deadline)
    report = dict(status='running', hardware=hardware(), tasks={})
    try:
        if report['hardware']['free_disk_bytes'] < 20 * 1024**3:
            raise RuntimeError('Less than 20 GiB of disk is free')
        for task in TASKS:
            check_stop(deadline)
            row = importlib.import_module(task).preflight(out / task)
            report['tasks'][task] = row
            dump(out / 'status.json', report)
            print(json.dumps(dict(preflight=task, result=row)), flush=True)
            if not row['passed']:
                raise RuntimeError(task + ' preflight failed')
        # Each task has four development jobs and four selected-recipe final jobs.
        # A digit job contains two models. Each module already adds a 4x factor.
        projected = sum(8 * row['projected_job_seconds'] for row in report['tasks'].values())
        report['projected_stage1_seconds'] = projected
        if projected > 14400:
            raise RuntimeError('Projected scientific work exceeds the four-hour cap')
        check_stop(deadline)
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'stopped'
        report['reason'] = str(exc)
    finally:
        report['seconds'] = time.monotonic() - start
        report['peak_rss_native_units'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        dump(out / 'status.json', report)
        dump(out / 'manifest.json', files_manifest(out))
    print(json.dumps(report), flush=True)
    return report['status'] == 'passed'


def experiment(commit):
    verify_lock(commit)
    prep = json.loads((STUDY / 'preparation/status.json').read_text())
    if prep['status'] != 'passed':
        raise RuntimeError('Preparation has not passed')
    out = STUDY / 'results'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    deadline = start + 14400
    os.environ['MI_STAGE_DEADLINE'] = str(deadline)
    report = dict(status='running', public_lock=commit, hardware=hardware(), jobs=[], branches={})
    try:
        for task in TASKS:
            module = importlib.import_module(task)
            dev = {}
            for recipe in ('A', 'B'):
                dev[recipe] = []
                for seed in DEV:
                    check_stop(deadline)
                    path = out / task / f'development-{recipe}-{seed}'
                    row = module.run_job(recipe, seed, 'development', path, deadline)
                    check_stop(deadline)
                    audit_start = time.monotonic()
                    audit = module.replay(path)
                    row['replay'] = audit
                    row['replay_seconds'] = time.monotonic() - audit_start
                    report['jobs'].append(row)
                    dev[recipe].append(row)
                    dump(out / 'status.json', report)
                    print(json.dumps(dict(job=task, recipe=recipe, seed=seed, split='development', metrics=row)), flush=True)
                    if not audit['passed']:
                        raise RuntimeError(task + ' saved-weight replay failed')
            eligible = [r for r in ('A', 'B') if task != 'flow' or all(x['eligible'] for x in dev[r])]
            if not eligible:
                report['branches'][task] = dict(status='stopped', reason='Neither flow recipe met the development region limit')
                continue
            def rank(recipe):
                rows = dev[recipe]
                score = (sum(x['selection_score'] for x in rows) / len(rows)
                         if task == 'flow' else min(x['selection_score'] for x in rows))
                return (-score, sum(x['seconds'] for x in rows), recipe)
            selected = min(eligible, key=rank)
            report['branches'][task] = dict(selected_recipe=selected, status='running', final_passes=[])
            dump(out / 'status.json', report)
            for seed in FINAL:
                check_stop(deadline)
                path = out / task / f'final-{selected}-{seed}'
                row = module.run_job(selected, seed, 'final', path, deadline)
                check_stop(deadline)
                audit_start = time.monotonic()
                audit = module.replay(path)
                row['replay'] = audit
                row['replay_seconds'] = time.monotonic() - audit_start
                report['jobs'].append(row)
                report['branches'][task]['final_passes'].append(row['passed'])
                dump(out / 'status.json', report)
                print(json.dumps(dict(job=task, recipe=selected, seed=seed, split='final', metrics=row)), flush=True)
                if not audit['passed']:
                    raise RuntimeError(task + ' saved-weight replay failed')
            report['branches'][task]['status'] = ('passed' if all(report['branches'][task]['final_passes']) else 'failed')
        verify_lock(commit)
        report['status'] = 'complete'
    except Exception as exc:
        report['status'] = 'stopped'
        report['reason'] = str(exc)
    finally:
        report['seconds'] = time.monotonic() - start
        report['peak_rss_native_units'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        dump(out / 'status.json', report)
        dump(out / 'manifest.json', files_manifest(out))
    print(json.dumps(report), flush=True)
    return report['status'] == 'complete'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'experiment'])
    parser.add_argument('--lock')
    args = parser.parse_args()
    configure()
    if args.stage == 'experiment' and not args.lock:
        parser.error('--lock is required')
    raise SystemExit(0 if (prepare() if args.stage == 'prepare' else experiment(args.lock)) else 1)
