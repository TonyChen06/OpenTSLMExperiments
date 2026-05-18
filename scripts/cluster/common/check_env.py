#!/usr/bin/env python3
"""
Ahri / PhysicsTSLM environment diagnostic.

Run this before launching anything heavy on a cluster. If it exits 0,
training will work. If it exits 1, the output tells you exactly what
broke and what to do about it.

The output has three parts:
  1. CONTEXT — node info, OS, Python, CUDA driver, repo state. Always
     printed at the top so when a failure is reported, the machine
     details come with it.
  2. CHECKS — one line per check: status (OK / FAIL / SKIP), summary,
     and on failure a "FIX:" hint that says specifically what to do.
  3. SUMMARY — final verdict + the failures grouped + suggested next
     action.

Usage:
    python scripts/cluster/common/check_env.py --gpu
    python scripts/cluster/common/check_env.py --gpu --json
    python scripts/cluster/common/check_env.py --gpu --models EleutherAI/pythia-160m

Exit codes:
    0 — all checks passed
    1 — at least one check failed
    2 — diagnostic itself crashed (rare; please report)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# ANSI colors — disabled if not a tty or --no-color
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s
def _green(s): return _c("32", s)
def _red(s):   return _c("31;1", s)
def _yellow(s): return _c("33", s)
def _bold(s):  return _c("1", s)
def _dim(s):   return _c("2", s)


# ---------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    status: str           # "ok" | "fail" | "skip"
    detail: str = ""
    fix: str = ""         # actionable hint shown on failure
    error_type: str = ""  # exception class name on fail
    traceback: str = ""   # full traceback on fail (for --verbose / json)


@dataclass
class Diagnostic:
    context: dict = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, r: CheckResult): self.checks.append(r)

    def n_fail(self) -> int:
        return sum(1 for c in self.checks if c.status == "fail")

    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "fail"]

    def to_dict(self) -> dict:
        return {
            "context": self.context,
            "checks": [c.__dict__ for c in self.checks],
            "n_fail": self.n_fail(),
            "passed": self.n_fail() == 0,
        }


# ---------------------------------------------------------------------
# Context discovery (always runs; no fail conditions)
# ---------------------------------------------------------------------

def _try(fn, default=""):
    try:
        return fn()
    except Exception:
        return default

def gather_context() -> dict:
    ctx = {
        "hostname": _try(socket.gethostname, "?"),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "os": _try(lambda: f"{platform.system()} {platform.release()}"),
        "platform": _try(platform.platform),
        "cwd": str(Path.cwd()),
        "user": os.environ.get("USER", "?"),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }
    # CUDA driver version (independent of torch's CUDA)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total,memory.free",
             "--format=csv,noheader"], stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        ctx["nvidia_smi"] = out
    except Exception:
        ctx["nvidia_smi"] = "not available"

    # repo state
    try:
        ctx["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        ctx["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
    except Exception:
        ctx["git_branch"] = ctx["git_commit"] = "?"

    return ctx


# ---------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------

MIN_PYTHON = (3, 10)

def check_python() -> CheckResult:
    v = sys.version_info
    if (v.major, v.minor) < MIN_PYTHON:
        return CheckResult(
            name="python version",
            status="fail",
            detail=f"{platform.python_version()} (need >={MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
            fix=f"Load a newer Python module. On Sherlock: `module load python/3.12.1`. "
                 f"Then recreate the venv: `python -m venv .venv && source .venv/bin/activate && pip install -e .`",
        )
    return CheckResult(name="python version", status="ok", detail=platform.python_version())


def check_import(pkg: str) -> CheckResult:
    try:
        m = importlib.import_module(pkg)
        return CheckResult(name=f"import {pkg}", status="ok", detail=getattr(m, "__version__", "?"))
    except ImportError as e:
        return CheckResult(
            name=f"import {pkg}",
            status="fail",
            detail=f"{pkg} not importable",
            fix=f"pip install {pkg}  (or rerun `pip install -e .` from the repo root)",
            error_type=type(e).__name__,
            traceback=traceback.format_exc(),
        )


def check_torch() -> CheckResult:
    try:
        import torch
        return CheckResult(
            name="torch",
            status="ok",
            detail=f"torch={torch.__version__} cuda_built={torch.version.cuda or 'cpu-only'}",
        )
    except ImportError as e:
        return CheckResult(
            name="torch",
            status="fail",
            detail="not importable",
            fix="pip install torch (use the official CUDA-matched wheel index if needed). "
                 "Check the version of CUDA driver shown in CONTEXT above to choose.",
            error_type=type(e).__name__,
            traceback=traceback.format_exc(),
        )


def check_cuda() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult(name="CUDA visible to torch", status="skip", detail="torch not available")
    if not torch.cuda.is_available():
        return CheckResult(
            name="CUDA visible to torch",
            status="fail",
            detail="torch.cuda.is_available() is False",
            fix=(
                "Three possible causes:\n"
                "    (a) running on a login node (login nodes typically have no GPU);\n"
                "    (b) wrong torch build (CPU-only wheel installed);\n"
                "    (c) CUDA driver version below what this torch needs.\n"
                "  Check nvidia-smi (see CONTEXT) and reinstall torch matching the driver."
            ),
        )
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    # CUDA runtime vs driver check
    try:
        runtime = torch.version.cuda
        # major version of driver from nvidia-smi
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip().splitlines()[0]
        driver = out
        detail = f"{n} GPU(s); torch CUDA {runtime}, driver {driver}; devices: {', '.join(names)}"
    except Exception:
        detail = f"{n} GPU(s): {', '.join(names)}"
    return CheckResult(name="CUDA visible to torch", status="ok", detail=detail)


def check_disk(path: str, min_gb: float) -> CheckResult:
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
        stat = shutil.disk_usage(p)
        free_gb = stat.free / 1e9
        if free_gb < min_gb:
            return CheckResult(
                name=f"disk at {path}",
                status="fail",
                detail=f"{free_gb:.1f} GB free, need >={min_gb} GB",
                fix=f"Move work to a larger filesystem (e.g. $SCRATCH on Sherlock) "
                     f"or free space at {path}.",
            )
        # writable test
        tmp = p / ".write_check"
        tmp.write_text("ok")
        tmp.unlink()
        return CheckResult(
            name=f"disk at {path}", status="ok",
            detail=f"{free_gb:.1f} GB free, writable",
        )
    except PermissionError as e:
        return CheckResult(
            name=f"disk at {path}", status="fail",
            detail=f"not writable: {e}",
            fix=f"Pick a path you own (e.g. $SCRATCH/ahri). Currently trying {path}.",
            error_type=type(e).__name__,
        )
    except Exception as e:
        return CheckResult(
            name=f"disk at {path}", status="fail",
            detail=str(e), error_type=type(e).__name__,
            fix=f"Inspect {path}: does the parent exist? is it on a working filesystem?",
        )


def check_parquet_roundtrip() -> CheckResult:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            t = pa.table({"x": [1, 2, 3], "y": [[1.0], [2.0], [3.0]]})
            pq.write_table(t, Path(d) / "t.parquet")
            t2 = pq.read_table(Path(d) / "t.parquet")
            assert t2.equals(t)
        return CheckResult(name="parquet roundtrip", status="ok", detail="rw ok")
    except Exception as e:
        return CheckResult(
            name="parquet roundtrip", status="fail",
            detail=str(e), error_type=type(e).__name__,
            traceback=traceback.format_exc(),
            fix="Reinstall pyarrow: `pip install --force-reinstall pyarrow`. "
                "If the platform has no wheel, try `pip install pyarrow --no-binary :all:` "
                "(needs Arrow C++ and CMake — usually easier to switch to a Python version with wheels).",
        )


def check_ahri_import() -> CheckResult:
    try:
        from opentslm.ahri.tasks import TASK_REGISTRY
        n = len(TASK_REGISTRY)
        if n != 21:
            return CheckResult(
                name="ahri import + task registry", status="fail",
                detail=f"registered {n} tasks, expected 21",
                fix="Tony: the task registry isn't loading all tiers — check tasks/__init__.py imports.",
            )
        return CheckResult(name="ahri import + task registry", status="ok", detail=f"{n} tasks registered")
    except Exception as e:
        return CheckResult(
            name="ahri import + task registry", status="fail",
            detail=str(e), error_type=type(e).__name__,
            traceback=traceback.format_exc(),
            fix="Make sure you're invoking with `PYTHONPATH=src` and the repo is checked out clean. "
                "If `from opentslm.ahri.waveforms import ...` works in a Python REPL but this check fails, paste the traceback to Tony.",
        )


def check_hf_cache(cache_dir: str | None) -> CheckResult:
    try:
        from huggingface_hub import constants
        resolved = cache_dir or constants.HF_HUB_CACHE
        p = Path(resolved)
        p.mkdir(parents=True, exist_ok=True)
        stat = shutil.disk_usage(p)
        return CheckResult(
            name="HF cache dir writable", status="ok",
            detail=f"{resolved} ({stat.free / 1e9:.1f} GB free)",
        )
    except Exception as e:
        return CheckResult(
            name="HF cache dir writable", status="fail",
            detail=str(e), error_type=type(e).__name__,
            fix="Set HF_HOME to a writable path on a large filesystem (on Sherlock: $SCRATCH/hf_cache).",
        )


def check_model_cached(repo_id: str, cache_dir: str | None) -> CheckResult:
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir(cache_dir=cache_dir)
        cached = {r.repo_id for r in cache.repos}
        if repo_id not in cached:
            return CheckResult(
                name=f"model cached: {repo_id}", status="fail",
                detail=f"not in HF cache",
                fix=f"Run on a node with internet (login node):\n"
                     f"     PYTHONPATH=src python scripts/cluster/common/prefetch_models.py --models {repo_id}\n"
                     f"  Then re-run this check.",
            )
        return CheckResult(name=f"model cached: {repo_id}", status="ok", detail="present")
    except Exception as e:
        return CheckResult(
            name=f"model cached: {repo_id}", status="fail",
            detail=str(e), error_type=type(e).__name__,
            fix="Check `huggingface_hub` install and HF_HOME path.",
        )


def check_ahri_dataset_cached(repo_id: str, cache_dir: str | None) -> CheckResult:
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir(cache_dir=cache_dir)
        cached = {r.repo_id for r in cache.repos if r.repo_type == "dataset"}
        if repo_id not in cached:
            return CheckResult(
                name=f"Ahri dataset cached: {repo_id}", status="fail",
                detail="not downloaded",
                fix=f"Download with:\n"
                     f"     export AHRI_DATA_ROOT=$(PYTHONPATH=src python -c \"from huggingface_hub import snapshot_download; import os; print(os.path.join(snapshot_download(repo_id='{repo_id}', repo_type='dataset', allow_patterns=['data/*/*']), 'data'))\")",
            )
        return CheckResult(name=f"Ahri dataset cached: {repo_id}", status="ok")
    except Exception as e:
        return CheckResult(
            name=f"Ahri dataset cached: {repo_id}", status="fail",
            detail=str(e), error_type=type(e).__name__,
            fix="Check `huggingface_hub` install and HF_HOME path.",
        )


def check_torch_distributed_init() -> CheckResult:
    try:
        import torch.distributed as dist
        # single-process gloo init — sanity check that the module works
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", init_method="env://")
        dist.destroy_process_group()
        return CheckResult(name="torch.distributed (gloo, 1 proc)", status="ok", detail="init+destroy ok")
    except Exception as e:
        return CheckResult(
            name="torch.distributed (gloo, 1 proc)", status="fail",
            detail=str(e), error_type=type(e).__name__,
            traceback=traceback.format_exc(),
            fix="Distributed init failed in single-proc gloo mode. "
                "If you weren't planning to run DDP, this can be ignored. "
                "If you were: check MASTER_ADDR/MASTER_PORT env vars aren't already bound.",
        )


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------

def _status_badge(status: str) -> str:
    return {
        "ok":   _green(" OK ".rjust(6)),
        "fail": _red(" FAIL".rjust(6)),
        "skip": _yellow(" SKIP".rjust(6)),
    }[status]


def print_context(ctx: dict):
    print("=" * 70)
    print(_bold("  Ahri / PhysicsTSLM Environment Diagnostic"))
    print("=" * 70)
    rows = [
        ("hostname",   ctx["hostname"]),
        ("user",       ctx["user"]),
        ("os",         ctx.get("os", "?")),
        ("python",     f"{ctx['python']}  ({ctx['python_executable']})"),
        ("cwd",        ctx["cwd"]),
        ("git",        f"branch={ctx.get('git_branch')} commit={ctx.get('git_commit')}"),
    ]
    if ctx.get("slurm_job"):
        rows.append(("slurm",
                     f"job={ctx['slurm_job']} node={ctx.get('slurm_node')} partition={ctx.get('slurm_partition')}"))
    nvsmi = ctx.get("nvidia_smi", "")
    if nvsmi and nvsmi != "not available":
        # condense multi-line nvidia-smi to first 4 GPUs
        lines = nvsmi.splitlines()
        rows.append(("nvidia-smi", lines[0]))
        for ln in lines[1:4]:
            rows.append(("", ln))
        if len(lines) > 4:
            rows.append(("", f"... ({len(lines)-4} more GPUs)"))
    else:
        rows.append(("nvidia-smi", _dim("not available")))
    for k, v in rows:
        print(f"  {k:12s}  {v}")
    print()


def print_check(r: CheckResult):
    detail = r.detail if r.detail else ""
    print(f"  [{_status_badge(r.status)}]  {r.name:34s}  {detail}")
    if r.status == "fail" and r.fix:
        # wrap fix to terminal width-ish
        for ln in textwrap.indent(textwrap.fill(r.fix, width=88), "         FIX: ").splitlines():
            print(ln)


def print_summary(diag: Diagnostic):
    print()
    print("=" * 70)
    fails = diag.failures()
    if not fails:
        print(_green(_bold("  ALL CHECKS PASSED — ready to train.")))
        print("=" * 70)
        return
    print(_red(_bold(f"  {len(fails)} CHECK(S) FAILED")))
    print()
    for r in fails:
        et = f" ({r.error_type})" if r.error_type else ""
        print(f"  - {r.name}{et}: {r.detail}")
    print()
    print("  Next steps:")
    print("    1. Try the FIX hints printed above each failure.")
    print("    2. If unsure, re-run with --json and paste the output to Tony:")
    print("         python scripts/cluster/common/check_env.py --gpu --json > diag.json")
    print("=" * 70)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_checks(args) -> Diagnostic:
    diag = Diagnostic(context=gather_context())
    diag.add(check_python())

    for pkg in ["numpy", "transformers", "huggingface_hub", "datasets", "pyarrow", "tqdm"]:
        diag.add(check_import(pkg))
    diag.add(check_torch())
    if args.gpu:
        diag.add(check_cuda())

    diag.add(check_parquet_roundtrip())
    diag.add(check_ahri_import())
    diag.add(check_hf_cache(args.hf_cache))
    if args.data_dir:
        diag.add(check_disk(args.data_dir, args.min_disk_gb))
    for mid in args.models or []:
        diag.add(check_model_cached(mid, args.hf_cache))
    if args.ahri_dataset:
        diag.add(check_ahri_dataset_cached(args.ahri_dataset, args.hf_cache))
    if args.check_distributed:
        diag.add(check_torch_distributed_init())

    return diag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true", help="Require CUDA visible to torch")
    ap.add_argument("--hf-cache", default=None, help="HF cache dir (else default)")
    ap.add_argument("--data-dir", default=None, help="Scratch path to verify writable + has space")
    ap.add_argument("--min-disk-gb", type=float, default=5.0)
    ap.add_argument("--models", nargs="*", default=[],
                    help="HF model IDs that must already be in HF cache")
    ap.add_argument("--ahri-dataset", default=None,
                    help="HF dataset repo id to check is cached (e.g. TonyChen06/AscendingHarmonicReasoningInstruction)")
    ap.add_argument("--check-distributed", action="store_true",
                    help="Try a single-process gloo init of torch.distributed")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON only to stdout; suppresses formatted output")
    args = ap.parse_args()

    try:
        diag = run_checks(args)
    except Exception:
        # Diagnostic crashed mid-run; this is a bug in this script.
        if args.json:
            print(json.dumps({"diagnostic_crash": True, "traceback": traceback.format_exc()}))
        else:
            print(_red("DIAGNOSTIC CRASHED — please send this to Tony:"))
            traceback.print_exc()
        sys.exit(2)

    if args.json:
        print(json.dumps(diag.to_dict(), indent=2, default=str))
    else:
        print_context(diag.context)
        for r in diag.checks:
            print_check(r)
        print_summary(diag)

    sys.exit(0 if diag.n_fail() == 0 else 1)


if __name__ == "__main__":
    main()
