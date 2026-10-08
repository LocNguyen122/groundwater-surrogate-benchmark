"""The current numeric macro bundle is reproducible outside the package cwd."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class V7MacroInputs(unittest.TestCase):
    def test_portable_inputs_exist(self):
        config = ROOT / "scripts/v7_analysis/inputs_v7.json"
        values = json.loads(config.read_text())
        inputs = list(values["json"].values()) + [v for k, v in values.items() if k != "json"]
        for value in inputs:
            self.assertFalse(Path(value).is_absolute())
            path = (config.parent / value).resolve()
            self.assertIn(ROOT.resolve(), path.parents)
            self.assertTrue(path.is_file(), value)

    def test_generator_is_cwd_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "numbers.tex"
            run = subprocess.run([sys.executable, str(ROOT / "scripts/v7_analysis/make_numbers_v7.py"),
                                  str(ROOT / "scripts/v7_analysis/inputs_v7.json"), str(out)],
                                 cwd=temporary, capture_output=True, text=True, check=True)
            text = out.read_text(encoding="utf-8")
            self.assertIn("macros", run.stdout)
            self.assertGreater(text.count("\\newcommand"), 500)
            self.assertIn("\\newcommand{\\tLocoSixtyTwoHighPHolm}", text)
            self.assertIn("v7_poolT_best_terminal_2026-10-08.json", text)
            self.assertIn("\\newcommand{\\lastComp}", text)
            self.assertIn("v7_poolT_last_terminal_2026-10-08.json", text)
            self.assertNotIn("PENDING", text)


if __name__ == "__main__":
    unittest.main()
