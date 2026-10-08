import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("snapshot_check", ROOT / "scripts/verify_snapshot.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SnapshotBoundaryTests(unittest.TestCase):
    def test_raw_arrays_are_rejected(self):
        with self.assertRaises(ValueError):
            module.check_path(Path("results/predictions.npz"))

    def test_simulator_folder_is_rejected(self):
        with self.assertRaises(ValueError):
            module.check_path(Path("simulation/setup.py"))

    def test_safe_csv_is_permitted(self):
        module.check_path(Path("results/summary.csv"))
