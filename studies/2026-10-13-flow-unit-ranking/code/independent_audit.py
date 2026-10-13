"""Independent saved-weight replay. Does not import the scientific runner.

The audit does not repeat training. Its analytic 2-by-2 Jacobian recurrence
is separate from the scientific runner's autograd determinant control.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
SEEDS = (26501, 26519, 26539, 26561)
TOL = 1e-9
MEANS = np.array([[-2., -2.], [-2., 2.], [2., -2.], [2., 2.]])
MIX = np.array([.1, .2, .3, .4])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class Checks:
    def __init__(self):
        self.count = 0
        self.maxima = {}
        self.mapping_calls = 0
        self.mapping_examples = 0
        self.density_values = 0
        self.generated_coordinates = 0
        self.score_rows = 0
        self.jacobian_points = 0

    def near(self, label, actual, expected, tolerance=TOL):
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.shape == b.shape, (label, a.shape, b.shape)
        assert np.isfinite(a).all() and np.isfinite(b).all(), label
        error = float(np.max(np.abs(a - b), initial=0.))
        self.maxima[label] = max(error, self.maxima.get(label, 0.))
        self.count += 1
        assert error <= tolerance, (label, error, tolerance)

    def same(self, label, actual, expected):
        self.count += 1
        assert actual == expected, (label, actual, expected)

    def tree(self, label, actual, expected):
        if isinstance(actual, dict):
            self.same(label + ':keys', set(actual), set(expected))
            for k in actual:
                self.tree(label + '/' + str(k), actual[k], expected[k])
        elif isinstance(actual, (list, tuple)):
            self.same(label + ':length', len(actual), len(expected))
            for i, (a, b) in enumerate(zip(actual, expected)):
                self.tree(label + '/' + str(i), a, b)
        elif actual is None or isinstance(actual, (str, bool)):
            self.same(label, actual, expected)
        else:
            self.near(label, actual, expected, 0. if isinstance(actual, int) else TOL)


class Replay:
    """Four coupling steps from stored arrays; no study model class is used."""
    def __init__(self, archive, check):
        self.c = check
        with np.load(archive) as saved:
            self.weights = [{
                'w': torch.from_numpy(saved[f'blocks.{i}.0.weight'].copy()),
                'b': torch.from_numpy(saved[f'blocks.{i}.0.bias'].copy()),
                'v': torch.from_numpy(saved[f'blocks.{i}.1.weight'].copy()),
                'd': torch.from_numpy(saved[f'blocks.{i}.1.bias'].copy()),
            } for i in range(4)]
        self.c.same('parameter_count', sum(v.numel() for w in self.weights for v in w.values()), 520)
        for w in self.weights:
            for v in w.values():
                assert v.dtype == torch.float64 and torch.isfinite(v).all()

    def transform(self, values, condition, reverse=False, jacobian=False):
        self.c.mapping_calls += 1
        self.c.mapping_examples += len(values)
        point = torch.from_numpy(np.array(values, copy=True))
        total = torch.zeros(len(point), dtype=torch.float64)
        deriv = torch.eye(2, dtype=torch.float64).expand(len(point), 2, 2).clone() if jacobian else None
        for index in (list(range(4))[::-1] if reverse else range(4)):
            w = self.weights[index]
            fixed, changed = index % 2, 1 - index % 2
            h = torch.tanh(F.linear(point[:, fixed:fixed + 1], w['w'], w['b']))
            dh = (1. - h*h) * w['w'][:, 0] if jacobian else None
            if condition and (condition - 1) // 32 == index:
                u = (condition - 1) % 32
                mask = torch.ones(32, dtype=torch.float64)
                mask[u] = 0.
                original = h
                h = h * mask
                self.c.near('edited_scalar_zero', h[:, u].numpy(), np.zeros(len(h)), 0.)
                keep = np.arange(32) != u
                self.c.near('untouched_hidden', h[:, keep].numpy(), original[:, keep].numpy(), 0.)
                if jacobian:
                    dh = dh * mask
            raw = F.linear(h, w['v'], w['d'])
            scale = 1.5 * torch.tanh(raw[:, 0])
            shift = raw[:, 1]
            old = point[:, changed]
            if reverse:
                new = (old - shift) * torch.exp(-scale)
                total = total - scale
            else:
                new = old * torch.exp(scale) + shift
                total = total + scale
            if jacobian:
                # Differentiate the conditioner and compose the local triangular
                # Jacobian with the input Jacobian. This does not use autograd.
                raw_deriv = dh @ w['v'].T
                ds = 1.5 * (1. - torch.tanh(raw[:, 0])**2) * raw_deriv[:, 0]
                dt = raw_deriv[:, 1]
                diagonal = torch.exp(-scale if reverse else scale)
                cross = -diagonal * dt - new * ds if reverse else old * diagonal * ds + dt
                next_deriv = deriv.clone()
                next_deriv[:, changed, :] = diagonal[:, None] * deriv[:, changed, :] + cross[:, None] * deriv[:, fixed, :]
                deriv = next_deriv
            columns = [point[:, 0], point[:, 1]]
            columns[changed] = new
            point = torch.stack(columns, dim=1)
        assert torch.isfinite(point).all() and torch.isfinite(total).all()
        if jacobian:
            determinant = deriv[:, 0, 0] * deriv[:, 1, 1] - deriv[:, 0, 1] * deriv[:, 1, 0]
            assert (determinant > 0).all()
            self.c.near('analytic_jacobian_logdet', torch.log(determinant).numpy(), total.numpy())
            self.c.jacobian_points += len(point)
        return point.numpy(), total.numpy()

    def full(self, values, condition, reverse=False):
        out, ld = [], []
        for start in range(0, len(values), 1024):
            x = values[start:start+1024]
            y, det = self.transform(x, condition, reverse)
            restored, inverse_det = self.transform(y, condition, not reverse)
            self.c.near('roundtrip_coordinates', restored, x)
            self.c.near('roundtrip_logdet', det + inverse_det, np.zeros(len(x)))
            out.append(y)
            ld.append(det)
        return np.concatenate(out), np.concatenate(ld)


def quadrant(x):
    return (x[:, 0] >= 0).astype(np.int64)*2 + (x[:, 1] >= 0).astype(np.int64)


def distribution_moments(x):
    return dict(count=len(x), mean=x.mean(axis=0).tolist() if len(x) else None,
                covariance=np.cov(x.T, ddof=1).tolist() if len(x) > 1 else None)


def target_quadrants():
    out = []
    for q in range(4):
        mass = 0.
        for weight, mean in zip(MIX, MEANS):
            mass += weight * math.prod(.5 * math.erfc((-1 if flag else 1)*m/(.55*math.sqrt(2)))
                                       for m, flag in zip(mean, (q//2, q%2)))
        out.append(mass)
    return np.array(out)


def exact_density(x):
    components = []
    for mean, weight in zip(MEANS, MIX):
        components.append(math.log(weight) - math.log(2*math.pi*.55**2)
                          - ((x-mean)**2).sum(axis=1)/(2*.55**2))
    return np.logaddexp.reduce(np.array(components), axis=0)


def tau(x, y):
    # Explicit pair count, rather than the runner's vectorized upper triangle.
    concordance = nonzero_x = nonzero_y = 0
    for i in range(len(x)):
        for j in range(i):
            a, b = int(x[i] > x[j])-int(x[i] < x[j]), int(y[i] > y[j])-int(y[i] < y[j])
            concordance += a*b
            nonzero_x += a != 0
            nonzero_y += b != 0
    return concordance / math.sqrt(nonzero_x*nonzero_y) if nonzero_x*nonzero_y else None


def fixed_inputs():
    arrays = {}
    for name, n, seed in [('train_x', 16384, 26401), ('cal_x', 4096, 26402), ('test_x', 8192, 26403)]:
        rng = np.random.Generator(np.random.PCG64(seed))
        labels = rng.choice(4, size=n, p=MIX)
        noise = rng.standard_normal((n, 2))
        arrays[name] = noise * .55 + MEANS[labels]
        arrays[name+'_components'] = labels
    for name, n, seed in [('cal_z', 16384, 26404), ('test_z', 32768, 26405)]:
        arrays[name] = np.random.Generator(np.random.PCG64(seed)).standard_normal((n, 2))
    arrays['batch_ids'] = np.random.Generator(np.random.PCG64(26406)).integers(0, 16384, size=(4000,256), dtype=np.int32)
    return arrays


def check_initialization(results, seed, c):
    torch.manual_seed(seed)
    expected = {}
    for i in range(4):
        # Reproduce constructor RNG consumption, but do not call the study model.
        first = torch.nn.Linear(1, 32)
        last = torch.nn.Linear(32, 2)
        for name, value in first.state_dict().items():
            expected[f'blocks.{i}.0.{name}'] = value.double().numpy()
        for name, value in last.state_dict().items():
            expected[f'blocks.{i}.1.{name}'] = np.zeros(value.shape, dtype=np.float64)
    with np.load(results/f'model-{seed}-step-0.npz') as saved:
        c.same('initial_parameter_keys', set(saved.files), set(expected))
        for name, a in expected.items():
            c.near('initial_parameters', saved[name], a, 0.)
    for step in (1000,2000,3000,4000):
        with np.load(results/f'model-{seed}-step-{step}.npz') as saved:
            c.same('checkpoint_parameter_keys', set(saved.files), set(expected))
            for key, value in expected.items():
                c.same('checkpoint_shape', saved[key].shape, value.shape)
                assert saved[key].dtype == np.float64 and np.isfinite(saved[key]).all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Use a fresh receipt path.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    c = Checks()
    receipt = dict(status='running', audit_code_sha256=sha(__file__), training_repeated=False,
                   scientific_runner_imported=False, tolerance=TOL)
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled = False
        r = args.results
        manifest = read(r/'manifest.json')
        c.same('result_complete', manifest['status'], 'complete')
        c.same('scientific_run', manifest['scientific_model_outputs_collected'], True)
        receipt['result_manifest_sha256'] = sha(r/'manifest.json')
        receipt['public_lock'] = manifest['lock_commit']
        for name, value in manifest['files'].items():
            c.same('result_file_hash', sha(r/name), value)
        for name, value in manifest['locked_file_sha256'].items():
            frozen = subprocess.check_output(['git','show',manifest['lock_commit']+':'+name], cwd=REPO)
            c.same('lock_bytes', frozen, (REPO/name).read_bytes())
            c.same('lock_hash', hashlib.sha256(frozen).hexdigest(), value)
        c.tree('run_counts', manifest['counts'], dict(mapping_calls=78676,input_examples=67780864,optimizer_updates=16000,control_rows=31716))
        inputs = fixed_inputs()
        hashes = read(r/'input-hashes.json')
        with np.load(r/'inputs.npz') as saved:
            c.same('input_keys', set(saved.files), set(inputs))
            for name, values in inputs.items():
                c.near('fixed_inputs', saved[name], values, 0.)
                c.same('input_dtype', str(saved[name].dtype), str(values.dtype))
                c.tree('input_hash_record', hashes[name], dict(shape=list(values.shape),dtype=str(values.dtype),sha256=hashlib.sha256(values.tobytes()).hexdigest()))
        with (r/'training.csv').open() as handle:
            training = list(csv.DictReader(handle))
        c.same('training_rows', len(training), 16000)
        for k, row in enumerate(training):
            c.same('training_seed', int(row['seed']), SEEDS[k//4000])
            c.same('training_step', int(row['step']), k%4000+1)
            assert math.isfinite(float(row['loss']))
        with (r/'controls.csv').open() as handle:
            controls = list(csv.DictReader(handle))
        c.same('controls_count', len(controls), 31716)
        for row in controls:
            for key in ('coordinate_error','logdet_error','local_error'):
                if row[key] != '':
                    value = float(row[key])
                    limit = 0. if row['kind']=='noop' or key=='local_error' else TOL
                    assert math.isfinite(value) and 0. <= value <= limit
            if row['determinant_sign_min'] != '':
                c.same('saved_jacobian_sign', float(row['determinant_sign_min']), 1.)
        model_summaries = read(r/'model-summary.json')
        c.same('model_seeds', [m['seed'] for m in model_summaries], list(SEEDS))
        raw_rows = [json.loads(line) for line in (r/'scores.jsonl').read_text().splitlines()]
        c.same('score_row_count', len(raw_rows), 4*2*129)
        primary = []
        all_fit = []
        for m, seed in zip(model_summaries, SEEDS):
            check_initialization(r, seed, c)
            net = Replay(r/f'model-{seed}-step-4000.npz', c)
            phases = {}
            baseline_test = None
            selected_receipt = read(r/f'selectors-{seed}.json')
            for phase in ('cal','test'):
                x, z = inputs[phase+'_x'], inputs[phase+'_z']
                saved_rows = read(r/f'scores-{seed}-{phase}.json')
                c.same('phase_conditions', [v['condition'] for v in saved_rows], list(range(129)))
                all_terms = np.empty((129, len(x), 2))
                seen = []
                for path in sorted(r.glob(f'density-{seed}-{phase}-*.npz')):
                    with np.load(path) as part:
                        ids = part['condition_ids']
                        seen.extend(ids.tolist())
                        all_terms[ids] = part['terms']
                c.same('density_conditions', seen, list(range(129)))
                baseline_points = baseline_bins = baseline_density = None
                computed = []
                for condition in range(129):
                    encoded, determinant = net.full(x, condition)
                    terms = np.stack((-np.log(2*np.pi)-.5*np.sum(encoded**2, axis=1), determinant), axis=-1)
                    c.near('density_array', terms, all_terms[condition])
                    c.density_values += terms.size
                    points, _ = net.full(z, condition, True)
                    c.generated_coordinates += points.size
                    bins = quadrant(points)
                    counts = np.bincount(bins, minlength=4)
                    freq = counts / len(z)
                    if condition == 0:
                        baseline_points, baseline_bins, baseline_density = points, bins, terms
                        if phase == 'test':
                            baseline_test = points
                    baseline_counts = np.bincount(baseline_bins, minlength=4)
                    transitions = np.zeros((4,4), dtype=np.int64)
                    np.add.at(transitions, (baseline_bins, bins), 1)
                    diff = freq-baseline_counts/len(z)
                    damage = np.mean(baseline_density-terms, axis=0)
                    displacement = np.square(points-baseline_points).sum(axis=1)
                    crossings = int(np.count_nonzero(bins != baseline_bins))
                    row = dict(seed=seed,phase=phase,condition=condition,block=(condition-1)//32 if condition else None,
                               unit=(condition-1)%32 if condition else None,mean_nll=float(-np.mean(terms.sum(axis=1))),
                               nll_damage=float(damage.sum()),gaussian_damage=float(damage[0]),logjac_damage=float(damage[1]),
                               quadrant_counts=counts.tolist(),quadrant_frequencies=freq.tolist(),quadrant_changes=diff.tolist(),
                               quadrant_tv=float(np.abs(diff).sum()/2),mean_squared_displacement=float(displacement.mean()),
                               crossing_fraction=crossings/len(z),crossing_count=crossings,
                               squared_displacement_sum=float(np.sum(np.square(points-baseline_points))),transitions=transitions.tolist())
                    c.tree('score', row, saved_rows[condition])
                    c.tree('jsonl_score', row, raw_rows[c.score_rows])
                    c.score_rows += 1
                    computed.append(row)
                    if phase == 'cal':
                        net.transform(x[:64], condition, jacobian=True)
                    elif condition in {0,*selected_receipt['selectors'].values()}:
                        with np.load(r/f'generated-{seed}-selected-test.npz') as selected_points:
                            c.near('saved_generated_points', points, selected_points[f'condition_{condition}'])
                phases[phase] = computed
                if phase == 'cal':
                    selectors = {name:sorted(range(1,129),key=lambda k:(-computed[k][metric],k))[0]
                                 for name,metric in [('L','nll_damage'),('M','quadrant_tv')]}
                    c.tree('selectors', selected_receipt, dict(seed=seed, selectors=selectors,test_model_calls_started=False,
                                                              calibration_scores_sha256=sha(r/f'scores-{seed}-cal.json')))
            cal,test = phases['cal'],phases['test']
            value = test[selectors['M']]['quadrant_tv']-test[selectors['L']]['quadrant_tv']
            primary.append(value)
            excess = test[0]['mean_nll'] + float(exact_density(inputs['test_x']).mean())
            fit_tv = float(np.abs(np.array(test[0]['quadrant_frequencies'])-target_quadrants()).sum()/2)
            passed = bool(excess<=.20 and fit_tv<=.05)
            all_fit.append(passed)
            damage = [v['nll_damage'] for v in test[1:]]
            changes = [v['quadrant_tv'] for v in test[1:]]
            expected = dict(seed=seed, selectors=selectors, primary_difference=value,
                            selected_test={k:test[v] for k,v in selectors.items()},selector_agreement=selectors['L']==selectors['M'],
                            calibration_tau_b=tau([v['nll_damage'] for v in cal[1:]],[v['quadrant_tv'] for v in cal[1:]]),
                            test_tau_b=tau(damage,changes),
                            rankings={phase:{metric:sorted(range(1,129),key=lambda k:(-rows[k][metric],k))
                                             for metric in ('nll_damage','quadrant_tv')} for phase,rows in phases.items()},
                            all_unit_test=dict(nll_damage_mean=float(np.mean(damage)),nll_damage_range=[min(damage),max(damage)],
                                               quadrant_tv_mean=float(np.mean(changes)),quadrant_tv_range=[min(changes),max(changes)]),
                            fit=dict(test_excess_nll=excess,generated_quadrant_tv=fit_tv,gate=passed,
                                     exact_target_quadrants=target_quadrants().tolist(),generated=distribution_moments(baseline_test),
                                     generated_by_quadrant=[distribution_moments(baseline_test[quadrant(baseline_test)==i]) for i in range(4)]))
            c.tree('model_summary', expected, m)
        indices = np.random.Generator(np.random.PCG64(26407)).integers(0,4,size=(10000,4))
        values = np.array(primary)[indices].mean(axis=1)
        with np.load(r/'bootstrap.npz') as stored:
            c.near('bootstrap_indices', stored['model_indices'], indices, 0.)
            c.near('bootstrap_values', stored['resampled_means'], values, 0.)
        expected_summary = dict(primary_model_values=primary,primary_mean=float(np.mean(primary)),
                                descriptive_model_bootstrap_95=np.percentile(values,[2.5,97.5],method='linear').tolist(),
                                all_fit_gates_pass=all(all_fit),test_data=distribution_moments(inputs['test_x']),
                                test_data_by_quadrant=[distribution_moments(inputs['test_x'][quadrant(inputs['test_x'])==q]) for q in range(4)])
        c.tree('summary', expected_summary, read(r/'summary.json'))
        receipt.update(status='passed', summary=expected_summary)
    except BaseException as error:
        receipt.update(status='failed',error=str(error),traceback=traceback.format_exc().replace(str(REPO),'<repo>'))
        raise
    finally:
        receipt.update(comparisons=c.count,max_errors=c.maxima,mapping_calls=c.mapping_calls,
                       mapping_examples=c.mapping_examples,density_values=c.density_values,
                       regenerated_coordinates=c.generated_coordinates,score_rows=c.score_rows,
                       analytic_jacobian_points=c.jacobian_points,elapsed_seconds=time.perf_counter()-started)
        write(args.output,receipt)
        print(json.dumps({k:receipt[k] for k in ('status','comparisons','mapping_calls','mapping_examples','elapsed_seconds')}))


if __name__ == '__main__':
    main()
