import csv
import unittest
from pathlib import Path

from src.shared.data.transport_conditioning import load_transport_metadata


ROOT = Path(__file__).resolve().parents[1]


class TransportMetadataTests(unittest.TestCase):
    def test_bundled_registry_export_has_complete_unique_mapping(self):
        path = ROOT / "metadata" / "transport_parameters_162.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 162)
        self.assertEqual(len({row["param_folder"] for row in rows}), 162)
        self.assertEqual(
            {float(row["alpha_L"]) for row in rows},
            {0.62, 6.2, 62.0},
        )
        self.assertEqual(
            {float(row["alpha_T_ratio"]) for row in rows},
            {0.1, 1.0},
        )

    def test_default_loader_uses_bundled_registry_export(self):
        metadata = load_transport_metadata()
        self.assertEqual(len(metadata), 162)
        self.assertEqual(metadata["param_000"]["alpha_L"], 0.62)
        self.assertEqual(metadata["param_000"]["alpha_T_ratio"], 0.1)
        self.assertEqual(metadata["param_161"]["alpha_L"], 62.0)
        self.assertEqual(metadata["param_161"]["alpha_T_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
