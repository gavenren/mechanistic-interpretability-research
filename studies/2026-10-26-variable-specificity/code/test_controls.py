"""Guard and publication-lock checks. All model modules are replaced by stubs.

No model, training, benchmark, scientific output, or numerical experiment runs.
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
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('specificity_controller_check', HERE / 'run.py')
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
        self.permit(True, time.time() + 90)

    def tearDown(self):
        self.stack.close(); self.tmp.cleanup()

    def permit(self, allowed, expiry):
        # Controlled invalid JSON numbers must reach the reader too.
        self.permit_path.write_text(json.dumps({'allowed': allowed, 'expires_at': expiry}))

    def guard(self):
        return run.Guard(time.monotonic() + 60)

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

    def test_missing_and_malformed_permits_hide_private_path(self):
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

    def fixture(self):
        study = self.root / 'study'; (study / 'code').mkdir(parents=True)
        (study / 'code/dummy.py').write_text('pass\n')
        (study / 'analysis-plan.md').write_text('A fixed dummy plan.\n')
        source = self.root / 'source.txt'; source.write_text('Fixed source bytes.\n')
        run.atomic_json(study / 'input-manifest.json', dict(source_commit='source-lock', files={'source.txt': run.digest(source)}))
        self.stack.enter_context(patch.object(run, 'ROOT', self.root))
        self.stack.enter_context(patch.object(run, 'STUDY', study))
        self.stack.enter_context(patch.object(run, 'SOURCE', 'source-lock'))
        self.stack.enter_context(patch.object(run, 'LOCK_DOCS', ('analysis-plan.md', 'input-manifest.json')))
        return study, source

    def test_public_byte_mismatch_stops(self):
        _, source = self.fixture()
        with patch.object(run.subprocess, 'check_output', return_value=b'Other public bytes'):
            with self.assertRaisesRegex(RuntimeError, 'Public bytes differ'): run.verify_bytes('source-lock', source)

    def test_lock_covers_code_docs_and_source(self):
        study, source = self.fixture()
        with patch.object(run, 'verify_bytes') as check:
            run.verify_lock('plan-lock')
        checked = {(call.args[0], call.args[1]) for call in check.call_args_list}
        self.assertEqual(checked, {('plan-lock', study / 'code/dummy.py'),
                                  ('plan-lock', study / 'analysis-plan.md'),
                                  ('plan-lock', study / 'input-manifest.json'), ('source-lock', source)})

    def test_input_hash_and_source_commit_mismatch_stop(self):
        study, source = self.fixture()
        source.write_text('Changed local source.\n')
        with patch.object(run, 'verify_bytes'):
            with self.assertRaisesRegex(RuntimeError, 'Input hash differs'): run.verify_lock('plan-lock')
            data = json.loads((study / 'input-manifest.json').read_text()); data['source_commit'] = 'other-lock'
            run.atomic_json(study / 'input-manifest.json', data)
            with self.assertRaisesRegex(RuntimeError, 'Wrong source commit'): run.verify_lock('plan-lock')

    def test_preparation_requires_pass_and_unchanged_manifest(self):
        study, _ = self.fixture(); folder = study / 'preparation'; folder.mkdir()
        run.atomic_json(folder / 'status.json', {'status': 'stopped', 'plan_lock': 'plan-lock'})
        run.atomic_json(folder / 'manifest.json', run.manifest(folder))
        with patch.object(run, 'verify_bytes'), patch.object(run.subprocess, 'run'):
            with self.assertRaisesRegex(RuntimeError, 'Preparation did not pass'):
                run.verify_lock('execution-lock', preparation=True)
            run.atomic_json(folder / 'status.json', {'status': 'passed', 'plan_lock': 'plan-lock'})
            with self.assertRaisesRegex(RuntimeError, 'Preparation manifest differs'):
                run.verify_lock('execution-lock', preparation=True)

    def test_training_ability_failure_blocks_edit_calibration(self):
        study, _ = self.fixture(); folder = study / 'training'; folder.mkdir()
        run.atomic_json(folder / 'status.json', dict(status='complete', plan_lock='train-lock',
                        branches={'models': {'passed': True, 'ability_gate_pass': False}}))
        run.atomic_json(folder / 'manifest.json', run.manifest(folder))
        with patch.object(run, 'verify_bytes'):
            with self.assertRaisesRegex(RuntimeError, 'full model panel'): run.verify_training('execution-lock')

    def calibration_fixture(self, wrong_member=False):
        study, _ = self.fixture(); folder = study / 'calibration'; folder.mkdir()
        dummy_names = {'edits/choices.json'} | {f'edits/seed-{seed}/rotation-step-{step}.npz'
                       for seed in range(91001, 91009) for step in (1000, 2000, 3000, 4000)}
        for name in dummy_names:
            path = folder / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('Dummy test bytes: ' + name).encode())
        run.atomic_json(folder / 'status.json', dict(status='complete', plan_lock='train-lock'))
        run.atomic_json(folder / 'edits/calibration-manifest.json', {'files': {}})
        run.atomic_json(folder / 'manifest.json', run.manifest(folder))
        with zipfile.ZipFile(folder / 'frozen.zip', 'w') as archive:
            for index, name in enumerate(sorted(run.frozen_names())):
                archive.writestr(name, b'Changed archive bytes' if wrong_member and index == 0 else (folder / name).read_bytes())
        return folder

    def test_public_frozen_archive_rejects_changed_member(self):
        self.calibration_fixture(wrong_member=True)
        with patch.object(run, 'verify_bytes'), patch.object(run, 'verify_lock'), patch.object(run, 'verify_training'):
            with self.assertRaisesRegex(RuntimeError, 'Public edit-calibration bytes differ'): run.verify_calibration('execution-lock')

    def test_calibration_rejects_extra_local_file(self):
        folder = self.calibration_fixture()
        (folder / 'edits/unlocked-file.txt').write_text('Unexpected local content')
        with patch.object(run, 'verify_bytes'):
            with self.assertRaisesRegex(RuntimeError, 'Calibration file membership differs'):
                run.verify_calibration('execution-lock')

    def test_frozen_archive_and_parent_lock_checks(self):
        self.calibration_fixture()
        with patch.object(run, 'verify_bytes'), patch.object(run, 'verify_lock') as plan, \
             patch.object(run, 'verify_training') as training, patch.object(run.subprocess, 'run') as ancestry:
            receipt = run.verify_calibration('execution-lock')
        self.assertEqual(receipt['status'], 'complete')
        plan.assert_called_once_with('train-lock', preparation=True)
        training.assert_called_once_with('train-lock')
        self.assertEqual(ancestry.call_args.args[0], ['git', 'merge-base', '--is-ancestor', 'train-lock', 'execution-lock'])

    def test_failed_initial_guard_retains_safe_receipt(self):
        study, _ = self.fixture(); self.permit(False, time.time() + 90)
        with contextlib.redirect_stdout(io.StringIO()): self.assertFalse(run.execute('experiment', 'not-a-commit'))
        receipt = json.loads((study / 'results/status.json').read_text())
        self.assertEqual(receipt['status'], 'stopped')
        self.assertEqual(receipt['forwarded_tokens'], 0)
        self.assertGreater(receipt['charged_seconds'], 0)
        self.assertNotIn(str(self.permit_path), json.dumps(receipt))
        self.assertEqual(run.manifest(study / 'results'), json.loads((study / 'results/manifest.json').read_text()))

    def test_existing_output_is_not_replaced(self):
        study, _ = self.fixture(); folder = study / 'results'; folder.mkdir()
        (folder / 'keep.txt').write_text('Keep this result.\n')
        with self.assertRaises(FileExistsError): run.execute('experiment', 'not-a-commit')
        self.assertEqual((folder / 'keep.txt').read_text(), 'Keep this result.\n')

    def fake_runtime(self):
        torch = SimpleNamespace(__version__='2.8.0', set_num_threads=lambda n: None,
                set_num_interop_threads=lambda n: None, use_deterministic_algorithms=lambda x: None,
                get_num_threads=lambda: 2, get_num_interop_threads=lambda: 1)
        numpy = SimpleNamespace(__version__='2.3.5')
        self.stack.enter_context(patch.dict(sys.modules, {'torch': torch, 'numpy': numpy}))
        self.stack.enter_context(patch.object(run, 'version', return_value='4.57.6'))
        return dict(python=sys.version, torch='2.8.0', numpy='2.3.5', transformers='4.57.6', threads=2, interop_threads=1)

    def test_mock_final_dispatch_and_prior_charge_are_exact(self):
        study, _ = self.fixture(); runtime = self.fake_runtime()
        calls = []
        def fake_run(out, guard, models_path, calibration_path):
            guard(); calls.append((out, models_path, calibration_path)); return {'passed': True}
        module = SimpleNamespace(run=fake_run)
        with patch.object(run, 'verify_lock', return_value={'runtime': runtime, 'charged_seconds': 10}), \
             patch.object(run, 'verify_training', return_value={'charged_seconds': 100}), \
             patch.object(run, 'verify_calibration', return_value={'charged_seconds': 200}), \
             patch.object(run.importlib, 'import_module', return_value=module), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(run.execute('experiment', 'execution-lock'))
        receipt = json.loads((study / 'results/status.json').read_text())
        self.assertEqual(calls, [(study / 'results/edits', study / 'training/models', study / 'calibration/edits')])
        self.assertAlmostEqual(receipt['stage2c_charge_seconds'], receipt['charged_seconds'] + 300)
        self.assertAlmostEqual(receipt['stage2c_charge_with_audit_reserve_seconds'], receipt['charged_seconds'] + 420)
        self.assertAlmostEqual(receipt['total_preparation_seconds'], run.PRIOR_PREP + run.PRELOCK_CHECK_CHARGE + 10)


if __name__ == '__main__': unittest.main()
