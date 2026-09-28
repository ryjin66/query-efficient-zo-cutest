from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import eval_block_rule
from ..fd import finite_difference_lagrangian, project_bounds, radius_at
from .common import (
    beta_from_params,
    finalize_run,
    flush_every_from_params,
    initial_x,
    initialize_dual,
    make_rng,
    max_time_seconds_from_params,
    maybe_flush_run,
    prepare_run,
    project_dual,
    should_continue,
)


def run(
    oracle,
    params: dict[str, Any],
    budget: int,
    seed: int,
    eps_feas: float,
    out_dir: str | Path | None = None,
    record_every: int = 1,
):
    solver_name = "zob_gda"
    oracle, recorder = prepare_run(oracle, solver_name, seed, eps_feas, record_every)
    rng = make_rng(seed)
    x = initial_x(oracle)
    y = initialize_dual(oracle.num_ineq)
    alpha = float(params["alpha"])
    beta = beta_from_params(params)
    ybar = float(params.get("ybar", 100.0))
    block_size = eval_block_rule(params.get("block_rule", "ceil(sqrt(n))"), oracle.n)
    r0 = float(params.get("r0", 1.0e-3))
    decay = float(params.get("radius_decay", 0.5))
    max_time_seconds = max_time_seconds_from_params(params)
    flush_every = flush_every_from_params(params)
    last_flush_query = 0

    k = 0
    while should_continue(oracle, budget, block_size + 1, recorder, max_time_seconds):
        indices = rng.choice(oracle.n, size=block_size, replace=False)
        f_base, g_base, grad = finite_difference_lagrangian(
            oracle, x, y, indices, radius_at(r0, decay, k), k
        )
        del f_base
        x_next = x.copy()
        x_next[indices] -= alpha * grad[indices]
        x = project_bounds(x_next, oracle.bl, oracle.bu)
        # The dual step uses constraints at the pre-update primal point.
        y = project_dual(y + beta * g_base, ybar)
        k += 1
        last_flush_query = maybe_flush_run(
            recorder,
            out_dir,
            oracle.problem_name,
            solver_name,
            seed,
            oracle,
            flush_every,
            last_flush_query,
        )

    return finalize_run(recorder, out_dir, oracle.problem_name, solver_name, seed)
