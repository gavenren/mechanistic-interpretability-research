"""Resource-control tests. No trained-model computation."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import run


class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = Path(self.tmp.name) / 'permit.json'
        self.env = patch.dict(os.environ, {'MI_USAGE_PERMIT': str(self.p)})
        self.env.start(); self.permit(True, time.time() + 90)

    def tearDown(self):
        self.env.stop(); self.tmp.cleanup()

    def permit(self, allowed, expiry):
        run.atomic_json(self.p, dict(allowed=allowed, expires_at=expiry))

    def guard(self):
        return run.Guard(time.monotonic() + 60)

    def test_token_charging(self):
        g = self.guard(); g(tokens=128); self.assertEqual(g.tokens, 128)
        g.tokens = run.TOKEN_CAP
        with self.assertRaises(RuntimeError): g(tokens=1)

    def test_invalid_permits(self):
        for allowed, expiry in ((False, time.time()+90), (1, time.time()+90),
                                (True, time.time()-1), (True, time.time()+180)):
            self.permit(allowed, expiry)
            with self.assertRaises(RuntimeError): self.guard()()
        self.p.write_text('{')
        with self.assertRaises(RuntimeError): self.guard()()
        self.p.unlink()
        with self.assertRaises(RuntimeError): self.guard()()

    def test_caps(self):
        with self.assertRaises(TimeoutError): run.Guard(time.monotonic()-1)()
        with patch.object(run, 'peak_memory', return_value=5*1024**3):
            with self.assertRaises(RuntimeError): run.Guard(time.monotonic()+60, memory_gib=4)()
        with patch.object(run.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 1})()):
            with self.assertRaises(RuntimeError): self.guard()()

    def test_failed_initial_guard_keeps_receipt(self):
        self.permit(False, time.time()+90)
        study = Path(self.tmp.name) / 'study'; study.mkdir()
        with patch.object(run, 'STUDY', study):
            self.assertFalse(run.execute('experiment', 'not-a-commit'))
        receipt = json.loads((study/'results/status.json').read_text())
        self.assertEqual(receipt['status'], 'stopped')
        self.assertEqual(receipt['forwarded_tokens'], 0)
        self.assertGreater(receipt['charged_seconds'], 0)
        self.assertEqual(run.manifest(study/'results'), json.loads((study/'results/manifest.json').read_text()))


if __name__ == '__main__': unittest.main()
