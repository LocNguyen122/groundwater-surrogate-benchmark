"""Terminal certification rejects incomplete or mismatched retrieval evidence."""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("terminal_certificate", ROOT / "scripts/certify_e1_terminal.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TerminalCertificateTests(unittest.TestCase):
    def record(self):
        c = json.loads((ROOT / "results/v7/E1_TERMINAL_CERTIFICATION_2026-10-08.json").read_text())
        return {"status": "FETCHED_AND_VERIFIED", "failures": c["failures"],
                "counts": c["arrays"], "task_manifest_sha256": c["task_manifest_sha256"]}

    def test_incomplete_array_is_rejected(self):
        r = self.record()
        r["counts"]["3485378"]["COMPLETED"] = 249
        with self.assertRaisesRegex(ValueError, "Incomplete array"):
            module.certify(ROOT, r)

    def test_manifest_mismatch_is_rejected(self):
        r = self.record()
        r["task_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Task manifest mismatch"):
            module.certify(ROOT, r)
