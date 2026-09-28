from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..fd import project_bounds
from ..logging_utils import RunRecorder, TrackedOracle, run_log_path


STOCHASTIC_SOLVERS = {"zob_gda", "zob_sgda", "zominmax", "szo_conex"}


def beta_from_params(params: dict[str, Any]) -> float:
    return float(params["alpha"]) * float(params.get("beta_over_alpha", 0.1))


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def initialize_dual(m: int) -> np.ndarray:
    return np.zeros(int(m), dtype=float)


def project_dual(y: np.ndarray, ybar: float) -> np.ndarray:
    if y.size == 0:
        return y
    return np.minimum(np.maximum(y, 0.0), float(ybar))


def prepare_run(
    oracle,
    solver_name: str,
    seed: int,
    eps_feas: float,
    record_every: int,
):
    recorder = RunRecorder(
        problem=getattr(oracle, "problem_name", "UNKNOWN"),
        solver=solver_name,
        seed=seed,
        eps_feas=eps_feas,
        record_every=record_every,
    )
    return TrackedOracle(oracle, recorder), recorder


def finalize_run(
    recorder: RunRecorder,
    out_dir: str | Path | None,
    problem: str,
    solver_name: str,
    seed: int,
) -> RunRecorder:
    if not recorder.rows or recorder.rows[-1].get("status") != "time_limit":
        recorder.mark_status("completed")
    if out_dir is not None:
        log_path = run_log_path(out_dir, problem, solver_name, seed)
        recorder.to_csv(log_path)
        recorder.save_incumbent(log_path.with_suffix(".incumbent.json"))
    return recorder


def initial_x(oracle) -> np.ndarray:
    return project_bounds(np.asarray(oracle.x0, dtype=float).copy(), oracle.bl, oracle.bu)


def remaining_budget(oracle, budget: int) -> int:
    return int(budget) - int(oracle.query_count)


def max_time_seconds_from_params(params: dict[str, Any]) -> int:
    value = params.get("max_time_seconds", os.environ.get("CUTEST_MAX_TIME_SECONDS", 0))
    return max(0, int(float(value or 0)))


def flush_every_from_params(params: dict[str, Any]) -> int:
    value = params.get("flush_every", os.environ.get("CUTEST_FLUSH_EVERY", 0))
    return max(0, int(float(value or 0)))


def maybe_flush_run(
    recorder: RunRecorder,
    out_dir: str | Path | None,
    problem: str,
    solver_name: str,
    seed: int,
    oracle,
    flush_every: int,
    last_flush_query: int,
) -> int:
    if out_dir is None or flush_every <= 0:
        return last_flush_query
    query = int(oracle.query_count)
    if query - int(last_flush_query) < flush_every:
        return last_flush_query
    log_path = run_log_path(out_dir, problem, solver_name, seed)
    recorder.to_csv(log_path)
    recorder.save_incumbent(log_path.with_suffix(".incumbent.json"))
    return query


def time_limit_reached(recorder: RunRecorder, max_time_seconds: int) -> bool:
    if max_time_seconds <= 0:
        return False
    return (time.perf_counter() - recorder.start) >= float(max_time_seconds)


def should_continue(
    oracle,
    budget: int,
    min_remaining: int,
    recorder: RunRecorder,
    max_time_seconds: int,
) -> bool:
    if remaining_budget(oracle, budget) <= min_remaining:
        return False
    if time_limit_reached(recorder, max_time_seconds):
        recorder.mark_status("time_limit")
        return False
    return True
