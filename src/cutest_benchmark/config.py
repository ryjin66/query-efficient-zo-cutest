from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def query_budget(n: int, budget_cfg: dict[str, Any]) -> int:
    if "fixed" in budget_cfg:
        return int(budget_cfg["fixed"])
    if budget_cfg.get("rule") == "fixed":
        return int(budget_cfg["value"])
    multiplier = int(budget_cfg.get("multiplier", 1000))
    floor = int(budget_cfg.get("floor", 50000))
    cap = int(budget_cfg.get("cap", 1000000))
    return int(min(cap, max(floor, multiplier * (int(n) + 1))))


def eval_block_rule(rule: str | int | float, n: int) -> int:
    if isinstance(rule, (int, float)):
        return max(1, min(int(math.ceil(float(rule))), n))
    text = str(rule).strip().lower().replace(" ", "")
    if text == "n":
        value = n
    elif text == "1":
        value = 1
    elif text.startswith("min(") and text.endswith(",n)"):
        value = int(float(text[4:-3]))
    elif text == "ceil(sqrt(n))":
        value = int(math.ceil(math.sqrt(n)))
    elif text.startswith("ceil(") and text.endswith("*n)"):
        value = int(math.ceil(float(text[5:-3]) * n))
    else:
        raise ValueError(f"Unsupported block-size rule: {rule!r}")
    return max(1, min(value, n))


def solver_names_from_params(params_cfg: dict[str, Any]) -> list[str]:
    solvers = params_cfg.get("solvers", {})
    if not isinstance(solvers, dict):
        raise ValueError("Parameter file must contain a 'solvers' mapping.")
    return list(solvers)
