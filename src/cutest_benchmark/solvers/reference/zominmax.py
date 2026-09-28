"""ZO-Min-Max, Algorithm 1, with its one-sided black-box dual step."""
import numpy as np

from cutest_benchmark.fd import project_bounds
from .common import EvaluationContext, RunStopped, sphere_gradient, project_dual


def run(oracle, params, budget, seed, time_cap=1800, out=None, record_every=50):
    context = EvaluationContext(oracle, 'zominmax', seed, budget, time_cap, out, record_every)
    rng = np.random.default_rng(seed)
    alpha = float(params['alpha'])
    beta = alpha*float(params['beta_over_alpha'])
    radius, directions = float(params['r0']), int(params['directions'])
    upper = float(params['ybar'])
    try:
        base = context.evaluate(context.x)
        while True:
            context.check(directions+1)
            gradient = sphere_gradient(context, context.x, context.y, radius, directions,
                                       base, rng, context.iterations)
            x = project_bounds(context.x-alpha*gradient, oracle.bl, oracle.bu)
            base = context.evaluate(x, 'primal_iterate', context.iterations)
            context.y = project_dual(context.y+beta*base[1], upper)
            context.x = x
            context.iterations += 1
    except RunStopped:
        return context.finish()
