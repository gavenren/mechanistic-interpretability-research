"""Control tests with stub runtimes and a stub bounds branch.

No model, benchmark, or numerical dummy calculation runs. The journal test
imports the bounds module but replaces all interval and state-map arithmetic
with stubs. All receipts and source fixtures use a temporary directory.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bounds_control_check', HERE / 'run.py')
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)


class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.permit_path = self.root / 'private-permit.json'
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.dict(os.environ, {'MI_USAGE_PERMIT': str(self.permit_path)}))
        self.stack.enter_context(patch.object(run, 'peak_memory', return_value=0))
        self.stack.enter_context(patch.object(run.shutil, 'disk_usage', return_value=SimpleNamespace(free=24 * 1024**3)))
        self.stack.enter_context(patch.object(run.signal, 'signal'))
        self.stack.enter_context(patch.object(run.signal, 'setitimer'))
        self.permit(True, time.time() + 90)

    def tearDown(self):
        self.stack.close(); self.tmp.cleanup()

    def permit(self, allowed, expiry):
        self.permit_path.write_text(json.dumps({'allowed': allowed, 'expires_at': expiry}))

    def guard(self):
        return run.Guard(time.monotonic() + 60)

    def fixture(self):
        study = self.root / 'study'; (study / 'code').mkdir(parents=True)
        (study / 'code/stub.py').write_text('pass\n')
        (study / 'analysis-plan.md').write_text('A fixed test plan.\n')
        source = self.root / 'source.txt'; source.write_text('Fixed source bytes.\n')
        run.atomic_json(study / 'input-manifest.json', dict(source_commit='source-lock', files={'source.txt': run.digest(source)}))
        self.stack.enter_context(patch.object(run, 'ROOT', self.root))
        self.stack.enter_context(patch.object(run, 'STUDY', study))
        self.stack.enter_context(patch.object(run, 'SOURCE', 'source-lock'))
        self.stack.enter_context(patch.object(run, 'LOCK_DOCS', ('analysis-plan.md', 'input-manifest.json')))
        return study, source

    def fake_runtime(self):
        torch = SimpleNamespace(__version__='2.8.0', set_num_threads=lambda n: None,
            set_num_interop_threads=lambda n: None, use_deterministic_algorithms=lambda x: None,
            get_num_threads=lambda: 2, get_num_interop_threads=lambda: 1)
        self.stack.enter_context(patch.dict(sys.modules, {'torch': torch, 'numpy': SimpleNamespace(__version__='2.3.5')}))
        self.stack.enter_context(patch.object(run, 'version', return_value='4.57.6'))
        return dict(python=sys.version, torch='2.8.0', numpy='2.3.5', transformers='4.57.6', threads=2, interop_threads=1)

    def receipt(self, study, stage='experiment'):
        folder = study / ('preparation' if stage == 'prepare' else 'results')
        saved = json.loads((folder / 'status.json').read_text())
        self.assertEqual(run.manifest(folder), json.loads((folder / 'manifest.json').read_text()))
        self.assertNotIn(str(self.permit_path), json.dumps(saved))
        return saved

    def test_valid_permit_and_zero_language_tokens(self):
        guard = self.guard(); guard(); guard(tokens=0)
        self.assertEqual(guard.tokens, 0)
        with self.assertRaisesRegex(RuntimeError, 'token cap'): guard(tokens=1)
        for value in (-1, True, .5, '1'):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, 'Invalid token charge'):
                self.guard()(tokens=value)

    def test_strict_permit_types_and_times(self):
        cases = [(False, time.time() + 90), (1, time.time() + 90), ('yes', time.time() + 90),
                 (True, time.time() - 1), (True, time.time() + 121), (True, True),
                 (True, float('nan')), (True, float('inf')), (True, '123')]
        for allowed, expiry in cases:
            self.permit(allowed, expiry)
            with self.subTest(allowed=allowed, expiry=expiry), self.assertRaises(RuntimeError): self.guard()()

    def test_invalid_permits_hide_private_path(self):
        for text in ('', '{', '[]', '{}', 'true'):
            self.permit_path.write_text(text)
            with self.assertRaisesRegex(RuntimeError, '^Usage permit missing or invalid$'): self.guard()()
        self.permit_path.unlink()
        with self.assertRaisesRegex(RuntimeError, '^Usage permit missing or invalid$'): self.guard()()

    def test_time_memory_and_disk_caps(self):
        with self.assertRaises(TimeoutError): run.Guard(time.monotonic() - 1)()
        with patch.object(run, 'peak_memory', return_value=4 * 1024**3):
            run.Guard(time.monotonic() + 60, memory_gib=4)()
        with patch.object(run, 'peak_memory', return_value=4 * 1024**3 + 1):
            with self.assertRaisesRegex(RuntimeError, 'memory cap'):
                run.Guard(time.monotonic() + 60, memory_gib=4)()
        with patch.object(run.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 1024**3 - 1)):
            with self.assertRaisesRegex(RuntimeError, '20 GiB'): self.guard()()

    def test_alarm_preserves_timeout_and_clears_busy_flag(self):
        guard = run.Guard(time.monotonic() - 1)
        with self.assertRaises(TimeoutError): guard.alarm()
        self.assertFalse(guard.busy)

    def test_public_byte_mismatch_stops(self):
        _, source = self.fixture()
        with patch.object(run.subprocess, 'check_output', return_value=b'Other public bytes'):
            with self.assertRaisesRegex(RuntimeError, 'Public bytes differ'): run.verify_bytes('source-lock', source)

    def test_lock_covers_code_docs_and_source(self):
        study, source = self.fixture()
        with patch.object(run, 'verify_bytes') as check: run.verify_lock('plan-lock')
        self.assertEqual({(call.args[0], call.args[1]) for call in check.call_args_list},
                         {('plan-lock', study / 'code/stub.py'), ('plan-lock', study / 'analysis-plan.md'),
                          ('plan-lock', study / 'input-manifest.json'), ('source-lock', source)})

    def test_input_hash_and_source_commit_mismatch_stop(self):
        study, source = self.fixture(); source.write_text('Changed local source.\n')
        with patch.object(run, 'verify_bytes'):
            with self.assertRaisesRegex(RuntimeError, 'Input hash differs'): run.verify_lock('plan-lock')
            data = json.loads((study / 'input-manifest.json').read_text()); data['source_commit'] = 'other-lock'
            run.atomic_json(study / 'input-manifest.json', data)
            with self.assertRaisesRegex(RuntimeError, 'Wrong source commit'): run.verify_lock('plan-lock')

    def test_stopped_preparation_and_changed_manifest_block(self):
        study, _ = self.fixture(); folder = study / 'preparation'; folder.mkdir()
        run.atomic_json(folder / 'status.json', dict(status='stopped', plan_lock='plan-lock'))
        run.atomic_json(folder / 'manifest.json', run.manifest(folder))
        with patch.object(run, 'verify_bytes'), patch.object(run.subprocess, 'run'):
            with self.assertRaisesRegex(RuntimeError, 'Preparation did not pass'):
                run.verify_lock('execution-lock', preparation=True)
            run.atomic_json(folder / 'status.json', dict(status='passed', plan_lock='plan-lock'))
            with self.assertRaisesRegex(RuntimeError, 'Preparation manifest differs'):
                run.verify_lock('execution-lock', preparation=True)

    def test_passed_preparation_rechecks_parent_lock_and_ancestry(self):
        study, source = self.fixture(); folder = study / 'preparation'; folder.mkdir()
        run.atomic_json(folder / 'status.json', dict(status='passed', plan_lock='plan-lock'))
        run.atomic_json(folder / 'manifest.json', run.manifest(folder))
        with patch.object(run, 'verify_bytes') as check, patch.object(run.subprocess, 'run') as ancestry:
            result = run.verify_lock('execution-lock', preparation=True)
        self.assertEqual(result['status'], 'passed')
        self.assertIn(('plan-lock', study / 'code/stub.py'), [call.args for call in check.call_args_list])
        self.assertIn(('execution-lock', folder / 'status.json'), [call.args for call in check.call_args_list])
        self.assertIn(('source-lock', source), [call.args for call in check.call_args_list])
        self.assertEqual(ancestry.call_args.args[0], ['git', 'merge-base', '--is-ancestor', 'plan-lock', 'execution-lock'])

    def test_failed_initial_guard_saves_safe_receipt(self):
        study, _ = self.fixture(); self.permit(False, time.time() + 90)
        with patch.object(run, 'verify_lock') as lock, contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(run.execute('experiment', 'not-a-commit'))
        lock.assert_not_called(); receipt = self.receipt(study)
        self.assertEqual(receipt['status'], 'stopped'); self.assertEqual(receipt['forwarded_tokens'], 0)
        self.assertGreater(receipt['charged_seconds'], 0)

    def test_existing_output_is_not_replaced(self):
        study, _ = self.fixture(); folder = study / 'results'; folder.mkdir()
        (folder / 'keep.txt').write_text('Keep this result.\n')
        with self.assertRaises(FileExistsError): run.execute('experiment', 'not-a-commit')
        self.assertEqual((folder / 'keep.txt').read_text(), 'Keep this result.\n')

    def test_prepare_dispatch_projection_and_budget_deductions(self):
        study, _ = self.fixture(); self.fake_runtime(); calls = []
        def prepare(out, guard):
            guard(); calls.append((out, guard.deadline, guard.memory_bytes)); return dict(passed=True, projected_seconds=7.)
        branch = SimpleNamespace(prepare=prepare)
        with patch.object(run, 'verify_lock', return_value=None) as locks, \
             patch.object(run.importlib, 'import_module', return_value=branch) as imports, \
             patch.object(run.time, 'monotonic', return_value=100.), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(run.execute('prepare', 'plan-lock'))
        imports.assert_called_once_with('bounds')
        self.assertEqual(calls, [(study / 'preparation/bounds', 100. + run.PREP_CAP - run.PRIOR_PREP - run.PRELOCK_CHECK_CHARGE - 2, 4 * 1024**3)])
        self.assertEqual([call.kwargs for call in locks.call_args_list], [{'preparation': False}, {'preparation': False}])
        result = self.receipt(study, 'prepare')
        self.assertEqual(result['projected_experiment_seconds'], 7 + 120 + run.POSTRUN_AUDIT_RESERVE)
        self.assertEqual(result['total_preparation_seconds'], run.PRIOR_PREP + run.PRELOCK_CHECK_CHARGE + 2)

    def test_final_dispatch_uses_preparation_and_separate_charges(self):
        study, _ = self.fixture(); runtime = self.fake_runtime(); calls = []
        prep = dict(runtime=runtime, charged_seconds=10., forwarded_tokens=0)
        def final(out, guard):
            guard(); calls.append((out, guard.deadline, guard.memory_bytes)); return dict(passed=True)
        with patch.object(run, 'verify_lock', return_value=prep) as locks, \
             patch.object(run.importlib, 'import_module', return_value=SimpleNamespace(run=final)) as imports, \
             patch.object(run.time, 'monotonic', return_value=100.), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(run.execute('experiment', 'execution-lock'))
        imports.assert_called_once_with('bounds')
        self.assertEqual(calls, [(study / 'results/bounds', 100. + run.STAGE_CAP - run.POSTRUN_AUDIT_RESERVE - 2, 16 * 1024**3)])
        self.assertEqual([call.kwargs for call in locks.call_args_list], [{'preparation': True}, {'preparation': True}])
        result = self.receipt(study)
        self.assertEqual(result['total_preparation_seconds'], run.PRIOR_PREP + run.PRELOCK_CHECK_CHARGE + 10)
        self.assertEqual(result['stage5_charge_with_audit_reserve_seconds'], result['charged_seconds'] + run.POSTRUN_AUDIT_RESERVE)

    def test_roadmap_cap_blocks_before_branch_import(self):
        study, _ = self.fixture()
        prep = dict(charged_seconds=10., forwarded_tokens=0)
        minimum = run.PRIOR_PREP + run.PRELOCK_CHECK_CHARGE + 10 + run.PRIOR_SCIENCE + run.STAGE_CAP
        with patch.object(run, 'ROADMAP_CAP', minimum - 1), patch.object(run, 'verify_lock', return_value=prep), \
             patch.object(run.importlib, 'import_module') as imports, contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(run.execute('experiment', 'execution-lock'))
        imports.assert_not_called()
        self.assertIn('roadmap cap', self.receipt(study)['reason'])

    def test_prepare_roadmap_cap_reserves_remaining_preparation(self):
        study, _ = self.fixture()
        with patch.object(run, 'ROADMAP_CAP', run.PREP_CAP + run.PRIOR_SCIENCE - 1), \
             patch.object(run, 'verify_lock', return_value=None), \
             patch.object(run.importlib, 'import_module') as imports, contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(run.execute('prepare', 'plan-lock'))
        imports.assert_not_called(); self.assertIn('roadmap cap', self.receipt(study, 'prepare')['reason'])

    def test_excessive_projection_is_stopped(self):
        study, _ = self.fixture(); self.fake_runtime()
        branch = SimpleNamespace(prepare=lambda out, guard: dict(passed=True, projected_seconds=run.STAGE_CAP))
        with patch.object(run, 'verify_lock', return_value=None), \
             patch.object(run.importlib, 'import_module', return_value=branch), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(run.execute('prepare', 'plan-lock'))
        self.assertIn('projection exceeds', self.receipt(study, 'prepare')['reason'])

    def test_failed_branch_retains_record(self):
        study, _ = self.fixture(); runtime = self.fake_runtime()
        prep = dict(runtime=runtime, charged_seconds=10., forwarded_tokens=0)
        with patch.object(run, 'verify_lock', return_value=prep), \
             patch.object(run.importlib, 'import_module', return_value=SimpleNamespace(run=lambda out, guard: dict(passed=False, cause='stub failure'))), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(run.execute('experiment', 'execution-lock'))
        result = self.receipt(study)
        self.assertEqual(result['branches']['bounds']['cause'], 'stub failure')
        self.assertEqual(result['reason'], 'bounds checks failed')

    def test_branch_stop_retains_partial_and_timeout_cause(self):
        study, _ = self.fixture(); runtime = self.fake_runtime()
        prep = dict(runtime=runtime, charged_seconds=10., forwarded_tokens=0)
        def stopped(out, guard):
            out.mkdir(); (out / 'partial.json').write_text('{"stub_partial": true}\n')
            raise TimeoutError('Stub stage deadline reached')
        with patch.object(run, 'verify_lock', return_value=prep), \
             patch.object(run.importlib, 'import_module', return_value=SimpleNamespace(run=stopped)), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(run.execute('experiment', 'execution-lock'))
        result = self.receipt(study)
        self.assertEqual(result['error_type'], 'TimeoutError')
        self.assertEqual(result['reason'], 'Stub stage deadline reached')
        self.assertTrue((study / 'results/bounds/partial.json').exists())

    def test_paired_update_interruption_keeps_complete_journal(self):
        # Only the control loop, snapshots, and array indexing are real. No
        # model, legacy interval module, or numerical map is called.
        import numpy as np
        branch_spec = importlib.util.spec_from_file_location('bounds_journal_control_check', HERE / 'bounds.py')
        bounds = importlib.util.module_from_spec(branch_spec); branch_spec.loader.exec_module(bounds)
        base = np.zeros((1, 2)); edit = np.zeros((1, 2)); units = np.array([0])
        values = {name: np.array([value]) for name, value in dict(lower=-1., upper=1., radius=1.,
                  center_lower=0., center_upper=0., residual_base_upper=1., residual_edit_upper=1.).items()}
        calls = []
        def effect_stub(*args):
            return {name: value.copy() for name, value in values.items()}
        def map_stub(state, *args):
            calls.append('map')
            if len(calls) == 1: return np.ones_like(state)
            raise RuntimeError('Stub interruption during the edited-side update')
        journal = [{'context': {}}]
        with patch.object(bounds, '_map', side_effect=map_stub), \
             patch.object(bounds, '_legacy', side_effect=AssertionError('Legacy arithmetic must not run')) as legacy:
            with self.assertRaisesRegex(RuntimeError, 'Stub interruption'):
                bounds._refine('global', base, edit, np.zeros((2, 2)), np.zeros((1, 2)),
                               (np.zeros((1, 2)), np.zeros((1, 2))), units, np.zeros((2, 2)),
                               np.array([0]), np.array([1]), .5, None,
                               SimpleNamespace(effect_interval=effect_stub), lambda: None, journal)
        legacy.assert_not_called(); self.assertEqual(len(calls), 2)
        self.assertEqual(journal[0]['phase'], 'refinement')
        self.assertEqual(journal[0]['attempted_step'], 0)
        self.assertEqual(journal[0]['arrays']['base'].tolist(), [[0., 0.]])
        self.assertEqual(journal[0]['arrays']['edit'].tolist(), [[0., 0.]])
        self.assertEqual(journal[0]['arrays']['completed_paired_updates'].tolist(), [0])
        self.assertEqual(journal[0]['completed_history']['step'][0].tolist(), [0])
        self.assertEqual(journal[0]['completed_history']['lower'][0].tolist(), [-1.])
        self.assertEqual(journal[0]['completed_history']['upper'][0].tolist(), [1.])


if __name__ == '__main__': unittest.main()
