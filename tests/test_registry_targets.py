import importlib
import unittest

from src.transport_surrogates.extensions.registry import EXPERIMENTS


PAPER_REGISTRY_KEYS = (
    "fno_baseline",
    "deeponet_baseline",
    "pix2pix_baseline",
    "cnn_baseline",
    "ms_tmo_baseline",
    "timecond_baseline",
    "fno_transport_conditioned",
    "deeponet_transport_conditioned",
    "pix2pix_transport_conditioned",
    "cnn_transport_conditioned",
    "ms_tmo_transport_conditioned",
    "timecond_transport_conditioned",
)


class RegistryImportTests(unittest.TestCase):
    def test_every_reported_family_target_imports(self):
        for key in PAPER_REGISTRY_KEYS:
            with self.subTest(key=key):
                spec = EXPERIMENTS[key]
                self.assertEqual(spec.launch.launch_type, "module")
                importlib.import_module(spec.launch.target)


if __name__ == "__main__":
    unittest.main()
