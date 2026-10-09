"""Synthetic license-state checks, not grants of rights in research code."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("license_guard", ROOT / "scripts/verify_snapshot.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LicenseStatusTests(unittest.TestCase):
    def test_selected_is_not_effective(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(module.validate_license(Path(d),
                {"selected_original_code_license": "MIT", "open_source_license": None}))

    def test_mit_requires_authority_confirmation(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                module.validate_license(Path(d), {"open_source_license": "MIT"})

    def test_mit_requires_actual_license_file(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                module.validate_license(Path(d), {"open_source_license": "MIT",
                    "original_code_licensing_authority_confirmed": True})

    def test_license_extensionless_name_is_allowed(self):
        module.check_path(Path("LICENSE"))

    def test_unknown_grant_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                module.validate_license(Path(d), {"open_source_license": "unspecified-custom-terms"})


if __name__ == "__main__":
    unittest.main()
