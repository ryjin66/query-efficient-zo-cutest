"""Joint-query recording and budget enforcement for reference algorithms."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np

from cutest_benchmark.constraints import violation
from cutest_benchmark.fd import project_bounds
from cutest_benchmark.logging_utils import RunRecorder


class RunStopped(Exception):
    pass


class QueryRecorder(RunRecorder):
    def to_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            return
        temporary = path.with_name(path.name+'.tmp')
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        temporary.replace(path)


class EvaluationContext:
    def __init__(self, oracle, solver, seed, budget, time_cap=1800, out=None, record_every=50):
        self.oracle = oracle
        self.budget = int(budget)
        self.time_cap = float(time_cap)
        self.out = Path(out) if out is not None else None
        self.recorder = QueryRecorder(oracle.problem_name, solver, seed, 1e-4, record_every)
        self.status = 'completed'
        self.last_flush = 0
        self.x = project_bounds(oracle.x0, oracle.bl, oracle.bu)
        self.y = np.zeros(oracle.num_ineq)
        self.iterations = 0
        self.outer_iterations = 0
        self.error = None

    @property
    def remaining(self):
        return self.budget-self.oracle.query_count

    def check(self, required=1):
        if self.remaining < required:
            self.status = 'query_limit'
            raise RunStopped()
        if self.time_cap > 0 and time.perf_counter()-self.recorder.start >= self.time_cap:
            self.status = 'time_limit'
            raise RunStopped()

    def evaluate(self, x, candidate='base', iteration=0):
        self.check()
        x = np.asarray(x, dtype=float)
        if not np.isfinite(x).all():
            self.status = 'numerical_failure'
            self.error = 'Nonfinite candidate point'
            raise RunStopped()
        query_before = self.oracle.query_count
        try:
            f, g = self.oracle.eval(x)
        except (ArithmeticError, ValueError) as error:
            self.oracle.query_count = query_before+1
            f, g = np.nan, np.full(self.oracle.num_ineq, np.inf)
            self.error = str(error)
        finite = bool(np.isfinite(f) and np.isfinite(g).all())
        in_box = bool(np.all(x >= self.oracle.bl) and np.all(x <= self.oracle.bu))
        # Unprojected smoothing queries cannot become box-feasible incumbents.
        scoring_g = g if in_box and finite else np.array([np.inf])
        self.recorder.record(self.oracle.query_count, iteration, f, scoring_g, x, candidate)
        self.recorder._last_snapshot.update(constraint_violation=violation(g) if finite else np.inf, box_feasible=in_box)
        if not finite:
            self.status = 'oracle_domain_failure'
            self.error = self.error or 'The unprojected reference-method query has nonfinite function values'
            raise RunStopped()
        if self.out is not None and self.oracle.query_count-self.last_flush >= 2500:
            self.recorder.to_csv(self.out/'trace.csv')
            self.last_flush = self.oracle.query_count
        return float(f), np.asarray(g, dtype=float)

    def finish(self):
        self.recorder.mark_status(self.status)
        result = dict(status=self.status, queries=self.oracle.query_count, iterations=self.iterations,
                      outer_iterations=self.outer_iterations, error=self.error,
                      x=self.x.tolist(), y=self.y.tolist(), best_feasible_f=self.recorder.best_feasible_f,
                      best_violation=self.recorder.best_violation,
                      elapsed_seconds=time.perf_counter()-self.recorder.start)
        if self.out is not None:
            self.out.mkdir(parents=True, exist_ok=True)
            self.recorder.to_csv(self.out/'trace.csv')
            self.recorder.save_incumbent(self.out/'incumbent.json')
            (self.out/'result.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
        return result, self.recorder


def lagrangian(fg, y):
    f, g = fg
    return float(f+np.dot(g, y))


def project_dual(y, upper):
    return np.clip(y, 0., upper)


def coordinate_gradient(context, x, y, radius, base, iteration):
    value = lagrangian(base, y)
    gradient = np.empty(x.size)
    for i in range(x.size):
        point = x.copy()
        point[i] += radius
        fg = context.evaluate(point, 'coordinate_fd', iteration)
        gradient[i] = (lagrangian(fg, y)-value)/radius
    return gradient


def sphere_gradient(context, x, y, radius, directions, base, rng, iteration):
    value = lagrangian(base, y)
    gradient = np.zeros(x.size)
    for _ in range(directions):
        direction = rng.normal(size=x.size)
        direction /= np.linalg.norm(direction)
        fg = context.evaluate(x+radius*direction, 'sphere_fd', iteration)
        gradient += x.size*(lagrangian(fg,y)-value)*direction/radius
    return gradient/directions
