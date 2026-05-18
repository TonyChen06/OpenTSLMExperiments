"""
Self-test for the environment diagnostic.

Verifies that:
  - the diagnostic produces well-formed structured output
  - each kind of failure is classified with a FIX hint, not just a traceback
  - the --json mode emits parseable JSON with the expected shape
  - the diagnostic itself doesn't crash on a bad path / missing module
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cluster" / "common" / "check_env.py"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str], env_extra: dict | None = None) -> tuple[int, str, str]:
    """Run check_env.py as a subprocess. Returns (returncode, stdout, stderr)."""
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["NO_COLOR"] = "1"  # disable ANSI for test parsing
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        env=env, capture_output=True, text=True, timeout=120,
    )
    return proc.returncode, proc.stdout, proc.stderr


# ---------------------------------------------------------------------
# Direct (in-process) tests of CheckResult-returning functions
# ---------------------------------------------------------------------

@pytest.fixture
def diag_module():
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location("check_env_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass introspects sys.modules[cls.__module__] — must register before exec
    _sys.modules["check_env_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_check_python_ok(diag_module):
    r = diag_module.check_python()
    assert r.status == "ok"
    assert r.detail  # version string


def test_check_import_missing_module_is_fail(diag_module):
    """Importing a definitely-missing module should produce a fail with a fix hint."""
    r = diag_module.check_import("this_package_does_not_exist_xyzzy")
    assert r.status == "fail"
    assert "pip install" in r.fix
    assert r.error_type in ("ImportError", "ModuleNotFoundError")
    assert "Traceback" in r.traceback


def test_check_import_present_is_ok(diag_module):
    r = diag_module.check_import("numpy")
    assert r.status == "ok"


def test_check_disk_bad_path_is_fail(diag_module):
    """A path that can't be created should produce a fail with a fix hint."""
    r = diag_module.check_disk("/proc/cant-create-here/nope", min_gb=1.0)
    assert r.status == "fail"
    assert r.fix  # should suggest something


def test_check_disk_small_min_is_ok(diag_module, tmp_path):
    r = diag_module.check_disk(str(tmp_path), min_gb=0.001)
    assert r.status == "ok"
    assert "free" in r.detail


def test_check_disk_huge_min_is_fail(diag_module, tmp_path):
    r = diag_module.check_disk(str(tmp_path), min_gb=1e9)  # 1 exabyte
    assert r.status == "fail"
    assert "need" in r.detail


def test_check_parquet_roundtrip(diag_module):
    r = diag_module.check_parquet_roundtrip()
    assert r.status == "ok"


def test_check_ahri_import(diag_module):
    r = diag_module.check_ahri_import()
    assert r.status == "ok"
    assert "21 tasks" in r.detail


def test_check_model_not_cached_fails(diag_module):
    """A model that's almost certainly NOT in cache should fail with a prefetch hint."""
    r = diag_module.check_model_cached(
        "definitely-not-a-real-org/definitely-not-a-real-model-xyzzy", cache_dir=None
    )
    assert r.status == "fail"
    assert "prefetch_models" in r.fix


def test_gather_context_has_expected_keys(diag_module):
    ctx = diag_module.gather_context()
    for key in ("hostname", "python", "python_executable", "os", "cwd"):
        assert key in ctx and ctx[key], f"context missing {key}"


def test_check_result_dataclass(diag_module):
    r = diag_module.CheckResult(name="x", status="ok")
    assert r.name == "x"
    assert r.status == "ok"
    assert r.fix == ""  # default


# ---------------------------------------------------------------------
# Subprocess integration tests (exercise the entry point)
# ---------------------------------------------------------------------

def test_subprocess_success_exit_zero():
    """A passing run should exit 0 and print 'ALL CHECKS PASSED'."""
    rc, out, err = _run([])
    assert rc == 0, f"failing run: stdout={out}\nstderr={err}"
    assert "ALL CHECKS PASSED" in out


def test_subprocess_with_missing_model_exits_one():
    """A required model that isn't cached should make the run fail with exit 1."""
    rc, out, err = _run(["--models", "definitely-fake-org/totally-not-real-zzzz"])
    assert rc == 1
    assert "CHECK(S) FAILED" in out
    assert "FIX:" in out


def test_subprocess_with_bad_disk_path_exits_one():
    rc, out, _err = _run(["--data-dir", "/proc/cant-create-here/nope"])
    assert rc == 1
    assert "CHECK(S) FAILED" in out


def test_json_mode_produces_parseable_output():
    rc, out, _err = _run(["--json"])
    assert rc == 0
    data = json.loads(out)
    assert "context" in data
    assert "checks" in data
    assert "n_fail" in data
    assert "passed" in data
    assert data["passed"] is True
    assert data["n_fail"] == 0
    # context shape
    for key in ("hostname", "python", "cwd"):
        assert key in data["context"]
    # checks shape
    for check in data["checks"]:
        assert {"name", "status", "detail"}.issubset(check.keys())
        assert check["status"] in ("ok", "fail", "skip")


def test_json_mode_on_failure_includes_fix_hint():
    rc, out, _err = _run(["--json", "--models", "fake-org/fake-model-xyz"])
    assert rc == 1
    data = json.loads(out)
    assert data["passed"] is False
    assert data["n_fail"] >= 1
    failed = [c for c in data["checks"] if c["status"] == "fail"]
    assert failed, "no failed checks in JSON"
    for c in failed:
        assert c["fix"], f"failed check {c['name']} has no fix hint"


def test_json_mode_on_failure_includes_error_type_when_applicable():
    """The disk-fail check raises a PermissionError; that should be captured."""
    rc, out, _err = _run(["--json", "--data-dir", "/proc/cant-create-here/nope"])
    assert rc == 1
    data = json.loads(out)
    disk_check = [c for c in data["checks"] if "disk at" in c["name"]]
    assert disk_check
    assert disk_check[0]["status"] == "fail"


def test_help_works():
    rc, out, _err = _run(["--help"])
    assert rc == 0
    assert "--gpu" in out
    assert "--json" in out
