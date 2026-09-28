"""Single-loop SZO-ConEx with independent Gaussian estimates."""
import numpy as np

from cutest_benchmark.fd import project_bounds
from .common import EvaluationContext, RunStopped


def constraint_model(context, x, radius, base, rng, iteration):
    m = context.oracle.num_ineq
    values = np.empty(m)
    gradients = np.empty((m, x.size))
    base_constraints = base[1]
    for i in range(m):
        direction = rng.normal(size=x.size)
        point = x+radius*direction
        fg = context.evaluate(point, 'constraint_model_fd', iteration)
        value = fg[1][i]
        values[i] = value
        gradients[i] = (value-base_constraints[i])*direction/radius
    return values, gradients


def primal_gradient(context, x, y, radius, base, rng, iteration):
    objective, constraints = base
    direction = rng.normal(size=x.size)
    point = x+radius*direction
    fg = context.evaluate(point, 'objective_fd', iteration)
    gradient = (fg[0]-objective)*direction/radius
    # Each component uses its own direction, independent of the constraint model.
    for i in range(context.oracle.num_ineq):
        direction = rng.normal(size=x.size)
        point = x+radius*direction
        fg = context.evaluate(point, 'constraint_primal_fd', iteration)
        value = fg[1][i]
        gradient += y[i]*(value-constraints[i])*direction/radius
    return gradient


def run(oracle, params, budget, seed, time_cap=1800, out=None, record_every=50):
    context = EvaluationContext(oracle, 'szo_conex', seed, budget, time_cap, out, record_every)
    rng = np.random.default_rng(seed)
    alpha = float(params['alpha'])
    beta = alpha*float(params['beta_over_alpha'])
    radius = float(params['r0'])
    theta = float(params.get('theta', 1.))
    average = np.zeros(oracle.n)
    average_evaluated = False
    try:
        base = context.evaluate(context.x)
        while True:
            # Reserve one query for the returned weighted mean.
            context.check(2*oracle.num_ineq+3)
            values, jacobian = constraint_model(context, context.x, radius, base, rng, context.iterations)
            if context.iterations == 0:
                current_model = values.copy()
                previous_model = current_model.copy()
            signal = (1.+theta)*current_model-theta*previous_model
            context.y = np.maximum(0., context.y+beta*signal)
            gradient = primal_gradient(context, context.x, context.y, radius, base, rng, context.iterations)
            next_x = project_bounds(context.x-alpha*gradient, oracle.bl, oracle.bu)
            previous_model = current_model
            current_model = values+jacobian@(next_x-context.x)
            base = context.evaluate(next_x, 'primal_iterate', context.iterations)
            context.x = next_x
            context.iterations += 1
            average += (context.x-average)/context.iterations
    except RunStopped:
        pass
    if context.iterations:
        context.x = average
        if context.status == 'query_limit':
            try:
                context.evaluate(context.x, 'returned_average', context.iterations)
                average_evaluated = True
            except RunStopped:
                pass
    result, recorder = context.finish()
    result['average_evaluated'] = average_evaluated
    return result, recorder
