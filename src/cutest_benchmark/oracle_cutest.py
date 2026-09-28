from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constraints import convert_cutest_constraints_to_inequalities, violation


class CutestUnavailableError(RuntimeError):
    pass


@dataclass
class OracleMetadata:
    name: str
    n: int
    m_raw: int
    m_ineq: int
    initial_f_raw: float
    initial_violation: float
    sf: float
    finite_lower_bounds: int
    finite_upper_bounds: int


class CutestOracle:
    """Value-only CUTEst oracle with normalization and query counting."""

    def __init__(self, problem_name: str, drop_fixed_variables: bool = True):
        try:
            import pycutest  # type: ignore
        except ModuleNotFoundError as exc:
            raise CutestUnavailableError(
                "PyCUTEst is required for CUTEst runs. Install CUTEst/PyCUTEst "
                "and rerun the benchmark scripts."
            ) from exc

        self.problem_name = problem_name
        self.prob = pycutest.import_problem(
            problem_name, drop_fixed_variables=drop_fixed_variables
        )
        self.n = int(getattr(self.prob, "n"))
        self.x0 = np.asarray(getattr(self.prob, "x0"), dtype=float).reshape(-1)
        self.bl = self._array_attr("bl", self.n, -np.inf)
        self.bu = self._array_attr("bu", self.n, np.inf)
        self.cl = self._optional_array_attr("cl")
        self.cu = self._optional_array_attr("cu")
        self.query_count = 0

        f0, g0 = self.raw_eval(self.x0)
        self.sf = max(1.0, abs(float(f0)))
        self.sg = np.maximum(1.0, np.abs(g0)) if g0.size else np.empty(0)
        self.num_ineq = int(g0.size)
        self.initial_f = float(f0 / self.sf)
        self.initial_violation = violation(self._normalize_g(g0))
        self.metadata = OracleMetadata(
            name=problem_name,
            n=self.n,
            m_raw=int(getattr(self.prob, "m", 0) or 0),
            m_ineq=self.num_ineq,
            initial_f_raw=float(f0),
            initial_violation=self.initial_violation,
            sf=float(self.sf),
            finite_lower_bounds=int(np.isfinite(self.bl).sum()),
            finite_upper_bounds=int(np.isfinite(self.bu).sum()),
        )

    def _array_attr(self, name: str, size: int, default: float) -> np.ndarray:
        val = getattr(self.prob, name, None)
        if val is None:
            return np.full(size, default, dtype=float)
        arr = np.asarray(val, dtype=float).reshape(-1)
        if arr.size != size:
            return np.full(size, default, dtype=float)
        return arr

    def _optional_array_attr(self, name: str) -> np.ndarray | None:
        val = getattr(self.prob, name, None)
        if val is None:
            return None
        return np.asarray(val, dtype=float).reshape(-1)

    def raw_eval(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=float).reshape(-1)
        f_raw = self.prob.obj(x, gradient=False)
        if isinstance(f_raw, tuple):
            f_raw = f_raw[0]
        m_raw = int(getattr(self.prob, "m", 0) or 0)
        if m_raw > 0:
            c_raw = self.prob.cons(x, gradient=False)
            if isinstance(c_raw, tuple):
                c_raw = c_raw[0]
            c_raw = np.asarray(c_raw, dtype=float).reshape(-1)
            g_raw, _ = convert_cutest_constraints_to_inequalities(c_raw, self.cl, self.cu)
        else:
            g_raw = np.empty(0, dtype=float)
        return float(f_raw), g_raw

    def _normalize_g(self, g_raw: np.ndarray) -> np.ndarray:
        if g_raw.size == 0:
            return np.empty(0, dtype=float)
        return np.asarray(g_raw, dtype=float) / self.sg

    def eval(self, x: np.ndarray, context: dict[str, Any] | None = None) -> tuple[float, np.ndarray]:
        del context
        f_raw, g_raw = self.raw_eval(x)
        self.query_count += 1
        return float(f_raw / self.sf), self._normalize_g(g_raw)

    def reset_count(self) -> None:
        self.query_count = 0


class ToyConstrainedOracle:
    """Small deterministic oracle used for local pretests without PyCUTEst."""

    def __init__(self, n: int = 4):
        self.problem_name = "TOY_NONCONVEX"
        self.n = int(n)
        self.x0 = np.linspace(-0.8, 0.8, self.n)
        self.bl = -2.0 * np.ones(self.n)
        self.bu = 2.0 * np.ones(self.n)
        self.query_count = 0
        self.num_ineq = 2
        f0, g0 = self.raw_eval(self.x0)
        self.sf = max(1.0, abs(f0))
        self.sg = np.maximum(1.0, np.abs(g0))
        self.initial_f = f0 / self.sf
        self.initial_violation = violation(g0 / self.sg)
        self.metadata = OracleMetadata(
            name=self.problem_name,
            n=self.n,
            m_raw=2,
            m_ineq=2,
            initial_f_raw=float(f0),
            initial_violation=self.initial_violation,
            sf=float(self.sf),
            finite_lower_bounds=self.n,
            finite_upper_bounds=self.n,
        )

    def raw_eval(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=float).reshape(-1)
        target = np.linspace(-0.4, 0.4, self.n)
        f = float(np.sum((x - target) ** 2) + 0.15 * np.sin(3.0 * x[0]))
        g = np.array(
            [
                np.sum(x**2) - 1.5,
                0.25 + 0.5 * x[0] - x[-1],
            ],
            dtype=float,
        )
        return f, g

    def eval(self, x: np.ndarray, context: dict[str, Any] | None = None) -> tuple[float, np.ndarray]:
        del context
        f, g = self.raw_eval(x)
        self.query_count += 1
        return float(f / self.sf), g / self.sg

    def reset_count(self) -> None:
        self.query_count = 0
