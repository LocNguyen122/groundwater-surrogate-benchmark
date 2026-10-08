"""Run documented CLI help only: no training, inference or restricted asset access."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = [
    ["scripts/v7_analysis/analyze_v7.py", "--help"],
    ["scripts/v7_analysis/regenerate_checkpoint.py", "--help"],
    ["scripts/build_checkpoint_sensitivity.py", "--help"],
    ["-m", "src.transport_surrogates.confirmatory_v5.train", "--help"],
    ["-m", "src.shared.eval.eval_transport_unified_logc_global_plumemask", "--help"],
]

if __name__ == "__main__":
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTHONPATH=str(ROOT))
    for args in COMMANDS:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=60)
        if result.returncode or "usage:" not in result.stdout.lower():
            raise RuntimeError("CLI help failed: " + " ".join(args) + "\n" + result.stderr)
        print("PASS:", " ".join(args))
