"""ZO-AGP: coordinate differences and alternating regularized projections."""
import math

from cutest_benchmark.fd import project_bounds
from .common import EvaluationContext, RunStopped, coordinate_gradient, project_dual


def step_parameters(params, iteration):
    t = iteration+1
    alpha = float(params['alpha'])*101./(100.+math.sqrt(t))
    regularization = float(params.get('dual_regularization', .1))/(t**.25)
    radius = float(params['r0'])/(t**float(params['radius_decay']))
    return alpha, regularization, radius


def run(oracle, params, budget, seed, time_cap=1800, out=None, record_every=50):
    context = EvaluationContext(oracle, 'zoagp', seed, budget, time_cap, out, record_every)
    beta = float(params['alpha'])*float(params['beta_over_alpha'])
    upper = float(params['ybar'])
    try:
        base = context.evaluate(context.x)
        while True:
            context.check(oracle.n+1)
            alpha, penalty, radius = step_parameters(params, context.iterations)
            gradient = coordinate_gradient(context, context.x, context.y, radius, base, context.iterations)
            x = project_bounds(context.x-alpha*gradient, oracle.bl, oracle.bu)
            # For the affine Lagrangian, the coordinate dual differences equal g(x).
            base = context.evaluate(x, 'primal_iterate', context.iterations)
            context.y = project_dual(context.y+beta*(base[1]-penalty*context.y), upper)
            context.x = x
            context.iterations += 1
    except RunStopped:
        return context.finish()
