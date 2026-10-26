"""Small control-flow tests. No model, training, or scientific evaluation."""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('continuation_under_test', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


def previous_rows():
    rows = []
    for split, recipe, seed in runner.REUSED_KEYS:
        rows.append(dict(task='hall', split=split, recipe=recipe, seed=seed,
                         passed=True, eligible=True, selection_score=1.0,
                         seconds=seed / 1000, metrics={'all_graphs': 1.0}))
    rows[0]['source_path'] = 'old/results/hall/development-A-8001'
    return {'jobs': rows}


class DummyOriginal:
    @staticmethod
    def hardware():
        return {'dummy': True}


class DummyModule:
    def __init__(self, name, calls, fail_seed=None):
        self.name, self.calls, self.fail_seed = name, calls, fail_seed

    def replay(self, path, deadline=float('inf')):
        self.calls.append(('replay', self.name, str(path)))
        return {'passed': True, 'seconds': 0.0}

    def run_job(self, recipe, seed, split, path, deadline):
        self.calls.append(('train', self.name, recipe, seed, split))
        runner.new_output(path)
        if seed == self.fail_seed:
            (path / 'partial.mock').write_text('partial control state')
            raise RuntimeError('Forced dummy stop')
        return dict(task=self.name, recipe=recipe, seed=seed, split=split,
                    passed=True, eligible=True, selection_score=1.0, seconds=1.0,
                    metrics={'dummy': 1.0})


class RunnerTests(unittest.TestCase):
    def test_remaining_count_and_graph_order(self):
        self.assertEqual(runner.remaining_graph_jobs(), [('hall', 'A', 8103, 'final'), ('hall', 'A', 8104, 'final')])
        counts = runner.remaining_job_counts()
        self.assertEqual(sum(counts.values()), 26)
        self.assertEqual(sum(n * (2 if k == 'digits' else 1) for k, n in counts.items()), 34)

    def test_prior_accounting(self):
        self.assertAlmostEqual(runner.remaining_seconds('experiment'), 14082.934954748999)
        self.assertAlmostEqual(runner.remaining_seconds('prepare'), 3524.3924895000055)
        self.assertEqual(runner.PRELOCK_CHECK_CHARGE, 60.0)
        self.assertEqual(runner.CLOSING_RECEIPT_ALLOWANCE, 1.0)

    def test_classification_uses_worst_group_then_runtime(self):
        def row(score, seconds=1): return {'selection_score': score, 'seconds': seconds, 'eligible': True}
        self.assertEqual(runner.selected_recipe('register', {'A': [row(.99), row(.98)], 'B': [row(.985), row(.985)]}), 'B')
        self.assertEqual(runner.selected_recipe('digits', {'A': [row(1, 2), row(1, 2)], 'B': [row(1), row(1)]}), 'B')
        self.assertEqual(runner.selected_recipe('hall', {'A': [row(1), row(1)], 'B': [row(1), row(1)]}), 'A')

    def test_flow_requires_both_development_seeds(self):
        good = {'selection_score': -.1, 'seconds': 1, 'eligible': True}
        bad = {'selection_score': 0, 'seconds': 1, 'eligible': False}
        self.assertEqual(runner.selected_recipe('flow', {'A': [good, good], 'B': [good, bad]}), 'A')
        self.assertIsNone(runner.selected_recipe('flow', {'A': [good, bad], 'B': [bad, good]}))

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'results'; path.mkdir(); sentinel = path / 'keep.txt'; sentinel.write_text('keep')
            with self.assertRaises(FileExistsError): runner.new_output(path)
            self.assertEqual(sentinel.read_text(), 'keep')

    def test_denied_permit_stops(self):
        sys.path.insert(0, str(runner.OLD / 'code'))
        import common
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'private-permit.json'
            path.write_text(json.dumps({'allowed': False, 'expires_at': time.time() + 60}))
            with patch.dict(os.environ, {'MI_USAGE_PERMIT': str(path), 'MI_STAGE_DEADLINE': 'inf'}):
                with self.assertRaisesRegex(RuntimeError, 'Usage permit'):
                    common.check_stop(float('inf'))

    def test_strict_permit_rejects_bad_types_times_and_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'private-permit.json'
            with patch.dict(os.environ, {'MI_USAGE_PERMIT': str(path)}):
                for allowed, expiry in [('yes', time.time() + 60), (1, time.time() + 60),
                                        (True, float('nan')), (True, float('inf')), (True, True),
                                        (True, str(time.time() + 60)), (True, time.time() + 121),
                                        (True, time.time() - 1), (False, time.time() + 60)]:
                    path.write_text(json.dumps({'allowed': allowed, 'expires_at': expiry}))
                    with self.subTest(allowed=allowed, expiry=expiry):
                        with self.assertRaisesRegex(RuntimeError, 'Usage permit'):
                            runner.check_private_permit()
                for invalid in ('', '[]', '{}', 'true'):
                    path.write_text(invalid)
                    with self.assertRaisesRegex(RuntimeError, 'Usage permit'):
                        runner.check_private_permit()
                path.unlink()
                with self.assertRaisesRegex(RuntimeError, '^Usage permit cannot be read or is invalid$'):
                    runner.check_private_permit()
                path.write_text(json.dumps({'allowed': True, 'expires_at': time.time() + 60}))
                runner.check_private_permit()

    def run_mock_experiment(self, folder, fail_seed=None):
        root = Path(folder); study = root / 'study'; study.mkdir()
        previous = root / 'studies/2026-10-22-capability-completion'
        old_partial = previous / 'results/hall/final-A-8103'
        old_partial.mkdir(parents=True)
        (old_partial / 'partial.npz').write_bytes(b'Old weights without optimizer or random-generator state')
        (old_partial / 'loss.csv').write_text('22028,old value\n')
        calls = []
        modules = {name: DummyModule(name, calls, fail_seed if name == 'hall' else None) for name in runner.TASKS}
        with patch.object(runner, 'ROOT', root), patch.object(runner, 'STUDY', study), \
             patch.object(runner, 'PREVIOUS', previous), \
             patch.object(runner, 'verify_lock', return_value=previous_rows()), \
             patch.object(runner, 'runtime', return_value=(DummyOriginal(), modules, lambda deadline: None)), \
             patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            result = runner.experiment('dummy-lock')
        return result, calls, json.loads((study / 'results/status.json').read_text()), study

    def test_full_mock_order_and_no_retraining_completed_models(self):
        with tempfile.TemporaryDirectory() as folder:
            result, calls, record, study = self.run_mock_experiment(folder)
            self.assertTrue(result)
            trains = [c for c in calls if c[0] == 'train']
            self.assertEqual(len(trains), 26)
            self.assertEqual(trains[:2], [('train', 'hall', 'A', 8103, 'final'), ('train', 'hall', 'A', 8104, 'final')])
            self.assertEqual([c[1] for c in trains], ['hall'] * 2 + ['register'] * 8 + ['digits'] * 8 + ['flow'] * 8)
            self.assertEqual(record['complete_distinct_models'], 40)
            self.assertEqual(record['new_complete_model_trainings'], 34)
            self.assertEqual(record['retained_partial_training_invocations'], 2)
            self.assertEqual(record['branches']['hall']['final_passes'], [True] * 4)
            self.assertEqual([r['seconds'] for r in record['jobs'][:6]], [r['seconds'] for r in previous_rows()['jobs']])
            self.assertAlmostEqual(record['total_stage1_seconds'], runner.PRIOR_SCIENTIFIC + record['seconds'])
            self.assertAlmostEqual(record['seconds'], record['measured_seconds_before_closing_receipts'] + 1)
            self.assertTrue((study / 'results/manifest.json').exists())
            hashes = json.loads((study / 'results/manifest.json').read_text())
            self.assertEqual(hashes, runner.manifest(study / 'results'))
            self.assertIn('full hash scan', record['timing_boundary'])

    def test_restart_does_not_read_or_replace_old_partial_state(self):
        with tempfile.TemporaryDirectory() as folder:
            result, calls, record, study = self.run_mock_experiment(folder)
            self.assertTrue(result)
            old_partial = Path(folder) / 'studies/2026-10-22-capability-completion/results/hall/final-A-8103'
            self.assertEqual((old_partial / 'partial.npz').read_bytes(),
                             b'Old weights without optimizer or random-generator state')
            self.assertEqual((old_partial / 'loss.csv').read_text(), '22028,old value\n')
            self.assertTrue((study / 'results/hall/final-A-8103').is_dir())
            graph_jobs = [call for call in calls if call[:2] == ('train', 'hall')]
            self.assertEqual([call[3] for call in graph_jobs], [8103, 8104])
            self.assertIn('original initialization', record['restart_rule'])

    def test_private_permit_path_is_not_in_error_receipt(self):
        private = '/private/account/permit.json'
        with patch.dict(os.environ, {'MI_USAGE_PERMIT': private}):
            receipt = runner.safe_error(FileNotFoundError(private))
        self.assertNotIn(private, receipt['reason'])
        self.assertIn('[private permit]', receipt['reason'])

    def test_stop_keeps_partial_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            result, calls, record, study = self.run_mock_experiment(folder, fail_seed=8103)
            self.assertFalse(result)
            self.assertEqual([c for c in calls if c[0] == 'train'], [('train', 'hall', 'A', 8103, 'final')])
            self.assertEqual(record['current_job']['seed'], 8103)
            self.assertTrue((study / 'results/hall/final-A-8103/partial.mock').exists())
            self.assertEqual(record['new_complete_model_trainings'], 0)
            self.assertEqual(record['complete_distinct_models'], 6)
            self.assertEqual(record['branches']['hall']['status'], 'incomplete')
            self.assertEqual(record['status'], 'stopped')
            self.assertEqual(record['error_type'], 'RuntimeError')
            self.assertTrue((study / 'results/manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
