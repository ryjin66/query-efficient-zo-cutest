from __future__ import annotations

import numpy as np


def project_bounds(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(np.asarray(x, dtype=float), lower), upper)


def radius_at(r0: float, decay: float, iteration: int) -> float:
    return float(r0 / ((iteration + 1) ** decay))


def lagrangian_value(f: float, g: np.ndarray, y: np.ndarray) -> float:
    if g.size == 0:
        return float(f)
    return float(f + np.dot(y, g))


def coordinate_step(
    x: np.ndarray,
    idx: int,
    r: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    scale = max(1.0, abs(float(x[idx])))
    step = float(r * scale)
    if step <= 0.0:
        return 0.0
    if x[idx] + step <= upper[idx]:
        return step
    if x[idx] - step >= lower[idx]:
        return -step
    return 0.0


def finite_difference_lagrangian(
    oracle,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    r: float,
    iteration: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    indices = np.asarray(indices, dtype=int).reshape(-1)
    # Reuse one base evaluation across all selected coordinate differences.
    f_base, g_base = oracle.eval(x, context={"iter": iteration, "candidate": "base"})
    l_base = lagrangian_value(f_base, g_base, y)
    grad = np.zeros_like(x)

    for idx in indices:
        step = coordinate_step(x, int(idx), r, oracle.bl, oracle.bu)
        if step == 0.0:
            continue
        xp = x.copy()
        xp[idx] += step
        fp, gp = oracle.eval(xp, context={"iter": iteration, "candidate": "fd"})
        grad[idx] = (lagrangian_value(fp, gp, y) - l_base) / step
    return f_base, g_base, grad


def bounded_random_direction(
    rng: np.random.Generator,
    x: np.ndarray,
    r: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    direction = rng.normal(size=x.size)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        direction[0] = 1.0
        norm = 1.0
    direction = direction / norm
    scaled_direction = direction * np.maximum(1.0, np.abs(x))
    xp = project_bounds(x + r * scaled_direction, lower, upper)
    actual = xp - x
    if np.linalg.norm(actual) == 0.0:
        return xp, np.zeros_like(x)
    return xp, actual


def random_direction_lagrangian_gradient(
    oracle,
    x: np.ndarray,
    y: np.ndarray,
    r: float,
    rng: np.random.Generator,
    iteration: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    f_base, g_base = oracle.eval(x, context={"iter": iteration, "candidate": "base"})
    l_base = lagrangian_value(f_base, g_base, y)
    xp, actual = bounded_random_direction(rng, x, r, oracle.bl, oracle.bu)
    if not np.any(actual):
        return f_base, g_base, np.zeros_like(x)
    fp, gp = oracle.eval(xp, context={"iter": iteration, "candidate": "random_fd"})
    diff = lagrangian_value(fp, gp, y) - l_base
    denom = float(np.dot(actual, actual))
    grad = diff * actual / max(denom, 1.0e-30)
    return f_base, g_base, grad
