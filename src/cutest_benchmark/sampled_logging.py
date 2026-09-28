from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .constraints import violation


class RunRecorder:
    def __init__(
        self,
        problem: str,
        solver: str,
        seed: int,
        eps_feas: float,
        record_every: int = 1,
    ):
        self.problem = problem
        self.solver = solver
        self.seed = int(seed)
        self.eps_feas = float(eps_feas)
        self.record_every = max(1, int(record_every))
        self.start = time.perf_counter()
        self.rows: list[dict[str, Any]] = []
        self.best_feasible_f = float("inf")
        self.best_violation = float("inf")
        self.best_x: np.ndarray | None = None
        self.best_violation_x: np.ndarray | None = None

    def record(
        self,
        query: int,
        iteration: int,
        f_norm: float,
        g_norm: np.ndarray,
        x: np.ndarray,
        status: str = "eval",
    ) -> None:
        v = violation(g_norm)
        if v < self.best_violation:
            self.best_violation = v
            self.best_violation_x = np.asarray(x, dtype=float).copy()
        if v <= self.eps_feas and f_norm < self.best_feasible_f:
            self.best_feasible_f = float(f_norm)
            self.best_x = np.asarray(x, dtype=float).copy()

        if query % self.record_every != 0 and query != 1:
            return
        elapsed = time.perf_counter() - self.start
        self.rows.append(
            {
                "problem": self.problem,
                "solver": self.solver,
                "seed": self.seed,
                "query": int(query),
                "iter": int(iteration),
                "f_norm": float(f_norm),
                "violation": float(v),
                "best_feasible_f": float(self.best_feasible_f),
                "best_violation": float(self.best_violation),
                "x_norm": float(np.linalg.norm(x)),
                "elapsed_seconds": float(elapsed),
                "status": status,
            }
        )

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "problem",
            "solver",
            "seed",
            "query",
            "iter",
            "f_norm",
            "violation",
            "best_feasible_f",
            "best_violation",
            "x_norm",
            "elapsed_seconds",
            "status",
        ]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def save_incumbent(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "problem": self.problem,
            "solver": self.solver,
            "seed": self.seed,
            "best_feasible_f": self.best_feasible_f,
            "best_violation": self.best_violation,
            "best_x": None if self.best_x is None else self.best_x.tolist(),
            "best_violation_x": None
            if self.best_violation_x is None
            else self.best_violation_x.tolist(),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)


class TrackedOracle:
    def __init__(self, oracle, recorder: RunRecorder):
        self._oracle = oracle
        self._recorder = recorder

    def __getattr__(self, name: str):
        return getattr(self._oracle, name)

    def eval(self, x: np.ndarray, context: dict[str, Any] | None = None):
        context = context or {}
        f, g = self._oracle.eval(x, context=context)
        self._recorder.record(
            query=self._oracle.query_count,
            iteration=int(context.get("iter", -1)),
            f_norm=f,
            g_norm=g,
            x=x,
            status=str(context.get("candidate", "eval")),
        )
        return f, g


def run_log_path(out_dir: str | Path, problem: str, solver: str, seed: int) -> Path:
    safe_problem = problem.replace("/", "_")
    safe_solver = solver.replace("/", "_")
    return Path(out_dir) / f"{safe_problem}__{safe_solver}__seed{seed}.csv"
