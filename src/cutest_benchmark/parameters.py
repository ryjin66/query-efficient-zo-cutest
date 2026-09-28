"""Uniform parameter formulas used by the retained formal comparison."""
import math


def select_parameters(solver: str, d_x: int, query_budget: int) -> dict:
    if solver not in ("zob_gda", "zob_sgda", "zoagp", "zominmax", "szo_conex", "nomad_default"):
        raise ValueError("Unknown solver")
    if isinstance(d_x, bool) or not isinstance(d_x, int) or d_x < 1:
        raise ValueError("Dimension must be a positive integer")
    if isinstance(query_budget, bool) or not isinstance(query_budget, int) or query_budget < 2:
        raise ValueError("Query budget must be an integer >= 2")
    if solver == 'nomad_default':
        return {}
    if solver == 'zoagp':
        return dict(alpha=.3, beta_over_alpha=.3, dual_regularization=.1,
                    r0=.003, radius_decay=.5, ybar=100.)
    if solver == 'zominmax':
        return dict(alpha=.1, beta_over_alpha=.03, directions=1, r0=.001, ybar=100.)
    if solver == 'szo_conex':
        return dict(alpha=.03, beta_over_alpha=.3, r0=.01, theta=1.)
    block = (math.ceil(math.sqrt(d_x)) if solver == 'zob_gda'
             else min(d_x, max(1, math.ceil(.75*math.sqrt(d_x)))))
    params = dict(alpha=.08 if solver == 'zob_gda' else .1,
                  beta_over_alpha=.625 if solver == 'zob_gda' else .5,
                  block_rule=block, r0=.003, radius_decay=.5, ybar=100.)
    if solver == "zob_sgda":
        alpha = params["alpha"]
        beta = alpha * params["beta_over_alpha"]
        blocks = d_x / params["block_rule"]
        horizon = max(1, query_budget // (params["block_rule"] + 1))
        # Solve the coupled gamma cap and alpha*p = N*gamma.
        gamma = min(1 / 40, 1 / math.sqrt(horizon * blocks),
                    math.sqrt(alpha / (768 * blocks * beta)))
        params.update(gamma=gamma, p=blocks * gamma / alpha)
    return params
