"""Read saved results only. No training, model import, or forward calculation."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '2'

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
OLD = STUDY.parent / '2026-10-19-capability-gates'
PREVIOUS = STUDY.parent / '2026-10-22-capability-completion'
HALL_GROUPS = ('all_graphs', 'no_isolated_negatives', 'test_target', 'test_positive', 'test_hall3')
REGISTER_GROUPS = [('current_h0', 12), ('copy_first_h1', 13), ('write_first_h1', 13)] + [
    (f'{order}_h{h}', 12+h) for h in (2, 8, 32) for order in ('copy_write', 'write_copy')]
ARCHITECTURES = ('direct', 'filtered')
SHIFTS = ('original', 'up', 'down', 'left', 'right')
DEV, FINAL = (8001, 8002), (8101, 8102, 8103, 8104)
REUSED_KEYS = [('hall', 'development', r, s) for r in ('A', 'B') for s in DEV] + [
    ('hall', 'final', 'A', s) for s in (8101, 8102)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value * 1024


def guard(deadline):
    require(time.monotonic() < deadline, 'Audit time limit reached')
    require(peak_bytes() <= 4 * 1024**3, 'Audit memory limit reached')
    try:
        permit = read_json(Path(os.environ['MI_USAGE_PERMIT']))
        expiry = permit['expires_at']; now = time.time()
        valid = (permit['allowed'] is True and type(expiry) in (int, float)
                 and math.isfinite(expiry) and now < expiry <= now + 120)
    except (OSError, KeyError, TypeError, ValueError, OverflowError):
        valid = False
    require(valid, 'Private usage permit is missing or invalid')


def sha(path, deadline):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            guard(deadline); digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(folder, deadline):
    saved = read_json(folder / 'manifest.json')
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()
              and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    require(actual == set(saved), f'File list differs in {folder.name}')
    for relative, expected in saved.items():
        path = (folder / relative).resolve()
        require(path.is_relative_to(folder.resolve()), 'File list leaves its result directory')
        require(sha(path, deadline) == expected, f'File hash differs: {relative}')
    return {'files': len(saved), 'manifest_sha256': sha(folder / 'manifest.json', deadline)}


def key(row):
    return row['task'], row['split'], row['recipe'], row['seed']


def compare_metrics(actual, saved, tolerance=0.):
    require(set(actual) == set(saved), 'Metric names differ')
    for name, value in actual.items():
        require(math.isfinite(value) and math.isfinite(saved[name]), 'Nonfinite metric')
        require(abs(value-saved[name]) <= tolerance, f'Metric differs: {name}')


def check_trace(path, task, data, deadline):
    expected = {'hall': 32768, 'register': 6000, 'flow': 16000}
    filename = 'training.csv' if task in ('hall', 'register') else 'trace.csv'
    counts = {a: 0 for a in ARCHITECTURES} if task == 'digits' else {'model': 0}
    with (path / filename).open(newline='') as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if index % 512 == 0:
                guard(deadline)
            require(math.isfinite(float(row['loss'])), 'Nonfinite training loss')
            if task == 'digits':
                arch = row['architecture']; require(arch in counts, 'Unexpected digit architecture')
                batches = math.ceil(data['training_rows'] / 128)
                expected_arch = ARCHITECTURES[index // (40 * batches)] if index < 80*batches else None
                require(arch == expected_arch, 'Digit training architecture order differs')
                local = counts[arch]
                require(int(row['epoch']) == local // batches and int(row['batch']) == local % batches,
                        'Digit training epoch or batch differs')
                require(int(row['examples']) == min(128, data['training_rows']-128*(local % batches)),
                        'Digit batch size differs')
                counts[arch] += 1
            else:
                field = 'step' if task == 'flow' else 'update'
                require(int(row[field]) == index+1, 'Training update sequence differs')
                if task == 'hall':
                    require(int(row['epoch']) == index // 128 + 1, 'Graph epoch differs')
                if task == 'flow':
                    require(int(row['examples']) == 256, 'Flow training batch differs')
                counts['model'] += 1
    if task == 'digits':
        require(all(n == 40*math.ceil(data['training_rows']/128) for n in counts.values()),
                'Digit update count differs')
    else:
        require(counts['model'] == expected[task], 'Training update count differs')
    return counts


def raw_scores(path, row, np, deadline):
    task = row['task']; extra = {}; detail = {}
    guard(deadline)
    if task == 'hall':
        with np.load(path / 'evaluation.npz', allow_pickle=False) as raw:
            logits, labels, predictions = raw['logits'], raw['labels'], raw['predictions']
            require(logits.shape == labels.shape == predictions.shape == (65536,), 'Graph row count differs')
            require(np.isfinite(logits).all(), 'Nonfinite graph outputs')
            require(np.array_equal(raw['graph_ids'], np.arange(65536)), 'Graph IDs differ')
            require(np.array_equal(logits >= 0, predictions), 'Graph predictions differ from saved logits')
            metrics = {}
            for name in HALL_GROUPS:
                mask = raw['category_'+name]
                require(mask.shape == (65536,) and mask.dtype == np.bool_ and mask.any(), 'Graph category is invalid')
                require(int(mask.sum()) == row['category_counts'][name], 'Graph category count differs')
                metrics[name] = float(np.count_nonzero(predictions[mask] == labels[mask])/mask.sum())
            require(raw['category_all_graphs'].all(), 'All-graph category omits rows')
        passed = all(v >= .99 for v in metrics.values())
    elif task == 'register':
        with np.load(path / 'evaluation.npz', allow_pickle=False) as raw:
            pred, labels, logits = raw['predictions'], raw['labels'], raw['endpoint_logits']
            require(pred.shape == labels.shape == (9216, 44) and logits.shape == (9216, 16), 'Register shape differs')
            require(np.isfinite(logits).all(), 'Nonfinite register outputs')
            require(np.array_equal(raw['group_ids'], np.repeat(np.arange(9), 1024)), 'Register group IDs differ')
            metrics, supporting = {}, {}
            for group, (name, length) in enumerate(REGISTER_GROUPS):
                rows = slice(group*1024, (group+1)*1024)
                require(np.all(raw['lengths'][rows] == length), 'Register horizon differs')
                endpoint = logits[rows].argmax(1)
                require(np.array_equal(endpoint, pred[rows, length-1]), 'Register endpoints differ')
                metrics[name] = float(np.count_nonzero(endpoint == labels[rows, length-1])/1024)
                supporting[name] = float(np.all(pred[rows, 11:length] == labels[rows, 11:length], axis=1).mean())
            compare_metrics(supporting, row['complete_checkpoint_accuracy_supporting'])
        passed = all(v >= .99 for v in metrics.values())
    elif task == 'digits':
        with np.load(path / 'inputs.npz', allow_pickle=False) as inputs:
            extra['training_rows'] = len(inputs['train_y']); expected_labels = inputs['eval_y']
            require(extra['training_rows'] == len(inputs['train_ids']), 'Digit training IDs differ')
        with np.load(path / 'eval_raw.npz', allow_pickle=False) as raw:
            logits, labels = raw['logits'], raw['labels']
            require(np.array_equal(labels, expected_labels), 'Digit labels differ between saved files')
            require(logits.shape == (2, 5, len(labels), 10) and np.isfinite(logits).all(), 'Digit output shape or values differ')
            if row['split'] == 'final':
                require(extra['training_rows'] == 3823 and len(labels) == 1797, 'Final digit split sizes differ')
            else:
                require(extra['training_rows'] + len(labels) == 3823, 'Development digit split size differs')
            metrics = {f'{a}_{s}_accuracy': float(np.count_nonzero(logits[i,j].argmax(1) == labels)/len(labels))
                       for i,a in enumerate(ARCHITECTURES) for j,s in enumerate(SHIFTS)}
        for arch in ARCHITECTURES:
            detail[arch] = all(metrics[f'{arch}_{s}_accuracy'] >= (.95 if s == 'original' else .90) for s in SHIFTS)
        passed = all(detail.values())
    else:
        require(task == 'flow', 'Unknown task')
        with np.load(path / 'inputs.npz', allow_pickle=False) as inputs:
            x = inputs['eval_x']; require(x.shape == (8192, 2), 'Flow density row count differs')
        with np.load(path / 'eval_raw.npz', allow_pickle=False) as raw:
            density, generated = raw['log_density'], raw['generated']
            require(density.shape == (8192,) and generated.shape == (32768, 2), 'Flow output shape differs')
            require(np.isfinite(density).all() and np.isfinite(generated).all(), 'Nonfinite flow outputs')
            means = ((-2., -2.), (-2., 2.), (2., -2.), (2., 2.)); weights = (.1,.2,.3,.4); sd = .55
            components = np.array([math.log(w)-math.log(2*math.pi*sd*sd)-np.sum((x-m)**2, axis=1)/(2*sd*sd)
                                   for w,m in zip(weights, means)])
            target_log_density = np.logaddexp.reduce(components, axis=0)
            target = np.array([sum(w*math.prod(.5*math.erfc((-1 if flag else 1)*m/(sd*math.sqrt(2)))
                               for m,flag in zip(mean, (q//2, q%2))) for w,mean in zip(weights, means)) for q in range(4)])
            regions = 2*(generated[:,0] >= 0).astype(int)+(generated[:,1] >= 0).astype(int)
            frequencies = np.array([np.count_nonzero(regions == q)/len(regions) for q in range(4)])
            metrics = {'excess_nll': float(np.mean(target_log_density-density)),
                       'quadrant_tv': float(np.abs(frequencies-target).sum()/2)}
        passed = metrics['excess_nll'] <= .20 and metrics['quadrant_tv'] <= .05
    guard(deadline)
    return metrics, bool(passed), detail, extra


def choose_recipe(task, rows):
    recipes = {r: [x for x in rows if x['split'] == 'development' and x['recipe'] == r] for r in ('A','B')}
    eligible = [r for r in recipes if task != 'flow' or all(x['eligible'] for x in recipes[r])]
    if not eligible:
        return None
    def rank(recipe):
        group = recipes[recipe]
        score = sum(x['selection_score'] for x in group)/2 if task == 'flow' else min(x['selection_score'] for x in group)
        return -score, sum(x['seconds'] for x in group), recipe
    return min(eligible, key=rank)


def audit(report, deadline, np):
    require(report['status'] == 'complete' and report['current_job'] is None, 'Scientific run is not complete')
    hashes = {p.parent.name: verify_manifest(p, deadline) for p in (OLD/'results', PREVIOUS/'results', STUDY/'results')}
    jobs = report['jobs']; require(len({key(r) for r in jobs}) == len(jobs), 'Duplicate job')
    require([key(r) for r in jobs[:6]] == REUSED_KEYS, 'Reused jobs or order differ')
    require(all(r.get('reused') is True for r in jobs[:6]) and all(r.get('reused') is False for r in jobs[6:]), 'Reuse flags differ')
    table = []; checked = []
    for index,row in enumerate(jobs):
        guard(deadline)
        if index < 6:
            path = ROOT / row['source_path']
            expected_parent = OLD if index == 0 else PREVIOUS
            expected_path = expected_parent/'results/hall'/f"{row['split']}-{row['recipe']}-{row['seed']}"
            require(path == expected_path, 'Reused model path differs')
        else:
            path = STUDY/'results'/row['task']/f"{row['split']}-{row['recipe']}-{row['seed']}"
        saved = read_json(path/'metrics.json')
        require(key(saved) == key(row), 'Model identity differs')
        require(saved['seconds'] == row['seconds'], 'Original model runtime changed')
        if row['task'] in ('hall', 'register'):
            schedule = (94210 if row['split']=='development' else 94211) if row['task']=='hall' else (71202 if row['split']=='development' else 71208)
            updates = 32768 if row['task']=='hall' else 6000
            require(saved['updates'] == row['updates'] == updates, 'Recorded update count differs')
            require(saved['schedule_seed'] == row['schedule_seed'] == schedule, 'Recorded schedule seed differs')
        elif row['task'] == 'digits':
            require(saved['epochs'] == row['epochs'] == 40 and saved['batch'] == row['batch'] == 128,
                    'Digit training schedule differs')
            require(saved['schedule_seed'] == row['schedule_seed'] == 10000+row['seed'], 'Digit schedule seed differs')
        else:
            require(saved['updates'] == row['updates'] == 16000, 'Flow update count differs')
            require(saved['batch_seed'] == row['batch_seed'] == 30000+row['seed'], 'Flow batch seed differs')
        metrics, passed, detail, extra = raw_scores(path, saved, np, deadline)
        tolerance = 1e-12 if row['task'] == 'flow' else 0.
        compare_metrics(metrics, saved['metrics'], tolerance); compare_metrics(metrics, row['metrics'], tolerance)
        require(passed == saved['passed'] == row['passed'], 'Model gate decision differs')
        eligible = metrics['quadrant_tv'] <= .05 if row['task'] == 'flow' else True
        score = -metrics['excess_nll'] if row['task'] == 'flow' else min(metrics.values())
        require(eligible == saved['eligible'] == row['eligible'], 'Recipe eligibility differs')
        require(abs(score-row['selection_score']) <= tolerance and abs(score-saved['selection_score']) <= tolerance,
                'Recipe selection score differs')
        counts = check_trace(path, row['task'], extra, deadline)
        replay = row['continuation_replay'] if row.get('reused') else row['replay']
        require(replay['passed'] is True, 'Saved replay did not pass')
        for arch in (ARCHITECTURES if row['task'] == 'digits' else ('model',)):
            subset = {k:v for k,v in metrics.items() if row['task'] != 'digits' or k.startswith(arch+'_')}
            table.append(dict(task=row['task'], split=row['split'], recipe=row['recipe'], seed=row['seed'],
                              architecture=arch, reused=row['reused'], passed=detail.get(arch, passed),
                              updates=counts[arch], worst_accuracy=min(subset.values()) if row['task'] != 'flow' else '',
                              excess_nll=metrics.get('excess_nll',''), quadrant_tv=metrics.get('quadrant_tv','')))
        checked.append(dict(row, metrics=metrics, passed=passed, eligible=bool(eligible), selection_score=score))
    expected = list(REUSED_KEYS)+[('hall','final','A',s) for s in (8103,8104)]
    branches = {}
    for task in ('hall','register','digits','flow'):
        rows = [x for x in checked if x['task'] == task]
        chosen = choose_recipe(task, rows)
        require(chosen == 'A' if task == 'hall' else True, 'Reused graph selection differs')
        if task != 'hall':
            expected += [(task,'development',r,s) for r in ('A','B') for s in DEV]
            if chosen is not None:
                expected += [(task,'final',chosen,s) for s in FINAL]
        final = [x for x in rows if x['split'] == 'final']
        branch = report['branches'][task]
        if chosen is None:
            require(not final and branch['status'] == 'stopped_at_development', 'Ineligible flow branch ran final models')
            status = 'stopped_at_development'
        else:
            require(branch['selected_recipe'] == chosen, 'Selected recipe differs')
            require([x['seed'] for x in final] == list(FINAL), 'Final seed set differs')
            require(all(x['recipe'] == chosen for x in final), 'Final recipe differs')
            passes = [x['passed'] for x in final]
            require(branch['final_passes'] == passes, 'Final gate record differs')
            status = 'passed' if all(passes) else 'failed'
            require(branch['status'] == status, 'Final branch status differs')
        branches[task] = {'selected_recipe':chosen, 'status':status, 'complete_models':sum(2 if task=='digits' else 1 for _ in rows)}
    require([key(r) for r in jobs] == expected, 'Full task order, recipe, or seed set differs')
    require(report['complete_distinct_models'] == len(table), 'Complete model count differs')
    require(report['new_complete_model_trainings'] == len(table)-6, 'New training count differs')
    require(report['retained_partial_training_invocations'] == 2, 'Prior partial count differs')
    require(len(table) == (36 if branches['flow']['selected_recipe'] is None else 40), 'Total model count differs')
    return {'status':'passed', 'source_manifest_checks':hashes, 'jobs_checked':len(jobs), 'models_checked':len(table),
            'branches':branches, 'raw_gate_values_recalculated':True, 'training_repeated':False,
            'forward_or_replay_repeated':False, 'limit':'Saved outputs and traces only; this is not independent retraining.'}, table


def write_outputs(out, result, table, started, max_seconds):
    if table:
        with (out/'models.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    result['measured_seconds_before_closing_receipt'] = time.monotonic()-started
    result['charged_seconds'] = result['measured_seconds_before_closing_receipt']+1.
    result['closing_write_allowance_seconds'] = 1.
    result['peak_memory_bytes'] = peak_bytes()
    result['separate_conservative_stage_charge_seconds'] = 60.
    result['timing_limit'] = 'One second is reserved for the last audit receipt write. Main charges a separate conservative 60 seconds to Experiment 1.'
    if result['charged_seconds'] > max_seconds:
        result.update(status='stopped', error_type='TimeoutError', reason='Audit time limit reached during final saving')
    closing_started = time.monotonic()
    (out/'audit.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    closing = time.monotonic()-closing_started
    if closing > 1.:
        result.update(status='stopped', error_type='TimeoutError', reason='Audit closing write exceeded its reserved second',
                      measured_closing_overrun_seconds=closing-1., charged_seconds=time.monotonic()-started+1.)
        (out/'audit.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-seconds', type=float, default=60.)
    args = parser.parse_args(); require(1 < args.max_seconds <= 60, 'Audit limit must be above 1 and at most 60 seconds')
    started = time.monotonic(); deadline = started+args.max_seconds-1.
    with (ROOT/'.local-state/research-compute.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard(deadline)
        report = read_json(STUDY/'results/status.json')
        require(report['status'] == 'complete', 'Scientific run is not complete; no audit started')
        require(report['total_stage1_seconds']+60. <= 14400., 'The separate audit charge does not fit the stage cap')
        out = STUDY/'analysis/result-audit'; out.mkdir(exist_ok=False)
        result, table = {'status':'stopped'}, []
        try:
            import numpy as np
            result, table = audit(report, deadline, np)
            result['audit_code_sha256'] = sha(Path(__file__), deadline)
            result['scientific_charge_before_audit'] = report['total_stage1_seconds']
            guard(deadline)
        except Exception as exc:
            result.update(status='stopped', error_type=type(exc).__name__, reason=str(exc).replace(str(ROOT),'[repository]'))
        finally:
            write_outputs(out, result, table, started, args.max_seconds)
        print(json.dumps(result), flush=True)
        return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
