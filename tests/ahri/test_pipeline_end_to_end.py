"""
End-to-end pipeline integration test:
  generate tiny parquet -> train 2 steps -> evaluate -> assert results valid

This is the test that verifies the whole stack works on a fresh machine.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


def test_full_pipeline(tmp_path):
    """Run train_single_task.py as a subprocess on a tiny dataset.

    This is what cluster smoke tests look like — invoke the entry-point the
    way real jobs do."""
    data_root = tmp_path / "data"
    out_dir = tmp_path / "run"

    # generate splits' worth of tiny data inline (faster than subprocess for gen)
    from opentslm.ahri.generate import SPLIT_SIZES, generate_task
    SPLIT_SIZES["train"] = 8
    SPLIT_SIZES["val"] = 4
    SPLIT_SIZES["test"] = 8
    generate_task("1.2", data_root, overwrite=True)

    repo_root = Path(__file__).resolve().parents[2]
    env = {"PYTHONPATH": str(repo_root / "src"), "PATH": ":".join(["/usr/bin", "/bin", "/usr/local/bin"])}
    import os
    env.update({k: v for k, v in os.environ.items() if k.startswith(("LD_", "CUDA_", "HF_"))})

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "ahri" / "train_single_task.py"),
        "--task", "1.2",
        "--llm", "EleutherAI/pythia-70m",
        "--data", str(data_root),
        "--out", str(out_dir),
        "--epochs", "1",
        "--batch_size", "2",
        "--patience", "2",
        "--device", "cpu",
        "--max_new_tokens", "4",
        "--num_workers", "0",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"trainer failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert "test" in summary["results"]
    assert "accuracy" in summary["results"]["test"]
    assert "accuracy_in" in summary["results"]["test"]
    assert "accuracy_held" in summary["results"]["test"]
    assert 0.0 <= summary["results"]["test"]["accuracy"] <= 1.0
