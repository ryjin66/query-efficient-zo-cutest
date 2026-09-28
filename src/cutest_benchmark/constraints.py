from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InequalityMap:
    source_index: int
    sense: str
    bound: float


def finite_or_none(value: float) -> float | None:
    value = float(value)
    if np.isfinite(value):
        return value
    return None


def convert_cutest_constraints_to_inequalities(
    c: np.ndarray,
    cl: np.ndarray | None,
    cu: np.ndarray | None,
) -> tuple[np.ndarray, list[InequalityMap]]:
    """Convert CUTEst cl <= c(x) <= cu constraints to g(x) <= 0.

    Equality constraints are represented by both upper and lower inequalities.
    The Core-30 config avoids equality constraints, but supporting this keeps
    the oracle robust for future supplementary runs.
    """

    c = np.asarray(c, dtype=float).reshape(-1)
    if c.size == 0:
        return np.empty(0, dtype=float), []

    if cl is None:
        cl_arr = np.full(c.shape, -np.inf)
    else:
        cl_arr = np.asarray(cl, dtype=float).reshape(-1)
    if cu is None:
        cu_arr = np.full(c.shape, np.inf)
    else:
        cu_arr = np.asarray(cu, dtype=float).reshape(-1)

    if cl_arr.size != c.size or cu_arr.size != c.size:
        raise ValueError("Constraint value and bound arrays have incompatible sizes.")

    values: list[float] = []
    mapping: list[InequalityMap] = []
    for idx, ci in enumerate(c):
        upper = finite_or_none(cu_arr[idx])
        lower = finite_or_none(cl_arr[idx])
        if upper is not None:
            values.append(float(ci - upper))
            mapping.append(InequalityMap(idx, "upper", upper))
        if lower is not None:
            values.append(float(lower - ci))
            mapping.append(InequalityMap(idx, "lower", lower))
    return np.asarray(values, dtype=float), mapping


def violation(g: np.ndarray) -> float:
    g = np.asarray(g, dtype=float).reshape(-1)
    if g.size == 0:
        return 0.0
    return float(np.max(np.maximum(g, 0.0)))
