"""Distributed helpers — semantics in single-process mode."""

from __future__ import annotations

import os

import pytest
import torch

from opentslm.ahri import distributed as d


def test_defaults_when_no_dist_env(monkeypatch):
    for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(var, raising=False)
    assert d.is_distributed() is False
    assert d.rank() == 0
    assert d.local_rank() == 0
    assert d.world_size() == 1
    assert d.is_main_rank() is True


def test_is_distributed_detects_world_size(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    assert d.is_distributed() is True
    assert d.rank() == 2
    assert d.world_size() == 4
    assert d.is_main_rank() is False


def test_setup_distributed_single_process_cpu(monkeypatch):
    """In single-process mode, setup_distributed should be a no-op except
    for returning the device."""
    for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(var, raising=False)
    dev = d.setup_distributed()
    if torch.cuda.is_available():
        assert dev.type == "cuda"
    else:
        assert dev.type == "cpu"


def test_all_reduce_mean_no_op_single_process():
    assert d.all_reduce_mean(3.14, torch.device("cpu")) == 3.14


def test_barrier_no_op_single_process():
    # Just ensure it doesn't crash
    d.barrier()
