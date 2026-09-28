"""Independent high-precision formulas for flagged CUTEst saved points."""
from __future__ import annotations

import math

import numpy as np

import mpmath as mp


def raw_cresc4(point, precision=100):
    with mp.workdps(precision):
        v, w, d, a, t, r = [mp.mpf(float(value)) for value in point]
        distance, radius1, radius2 = a*d, a*d+r, d+r
        c2x, c2y = v+distance*mp.cos(t), w+distance*mp.sin(t)
        constraints = []
        for x, y in [(1, 0), (0, 1), (0, -1), (mp.mpf('.5'), 0)]:
            constraints.extend([(c2x-x)**2+(c2y-y)**2-radius2**2,
                                (v-x)**2+(w-y)**2-radius1**2])
        if min(distance, radius1, radius2) <= 0:
            objective = mp.nan
        else:
            h = (distance**2+radius1**2-radius2**2)/(2*radius1*distance)
            ell = -(distance**2-radius1**2+radius2**2)/(2*radius2*distance)
            if abs(h) > 1 or abs(ell) > 1:
                objective = mp.nan
            else:
                angle1, angle2 = mp.acos(h), mp.acos(ell)
                objective = radius2**2*angle2-radius1**2*angle1+radius2*distance*mp.sin(angle2)
        return objective, constraints


def raw_problem(problem, point, precision=100):
    if problem == 'CRESC4':
        return raw_cresc4(point, precision)
    with mp.workdps(precision):
        x = [mp.mpf(float(value)) for value in point]
        u = x[-1]
        if problem == 'MADSEN':
            v, w, _ = x
            q = v*v+w*w+v*w
            return u, [u-q, u+q, u-mp.sin(v), u+mp.sin(v), u-mp.cos(w), u+mp.cos(w)]
        if problem == 'GIGOMEZ2':
            v, w, _ = x
            return u, [u-v*v-w**4, u-(2-v)**2-(2-w)**2, u-2*mp.exp(w-v)]
        if problem == 'MAKELA2':
            v, w, _ = x
            q = v*v+w*w-u
            return u, [q, q-40*v-10*w+40, q-10*v-20*w+60]
        if problem == 'MAKELA3':
            return u, [v*v-u for v in x[:-1]]
        if problem == 'CANTILVR':
            return mp.mpf(.0624)*sum(x), [sum(c/v**3 for c, v in zip([61, 37, 19, 7, 1], x))-1]
        if problem == 'POLAK6':
            a, b, c, d, _ = x
            e1 = (a-(d+1)**4)**2
            e2 = (b-(a-(d+1)**4)**4)**2
            e3, e4 = (d+1)**4, (a-(d+1)**4)**4
            return u, [
                e1+e2+2*c*c+d*d+5*e3+5*e4-5*a-5*b-21*c+7*d-u,
                11*e1+11*e2+12*c*c+11*d*d-5*e3+15*e4+5*a-15*b-11*c-3*d-80-u,
                11*e1+21*e2+12*c*c+21*d*d+15*e3+5*e4-15*a-5*b-21*c-3*d-100-u,
                11*e1+11*e2+12*c*c+d*d-15*e3+15*e4+15*a-15*b-21*c-3*d-50-u]
        if problem == 'POLAK3':
            weights = [mp.mpf(float(format(1/j, '.10e'))) for j in range(1, 12)]
            return u, [sum(w*mp.exp((v-mp.sin(i+2*j))**2)
                           for j, (v, w) in enumerate(zip(x[:-1], weights), 1))-u for i in range(10)]
        if problem == 'MADSSCHJ':
            v = x[:-1]
            total = sum(v)
            base = [u-total+a+1 for a in v]
            constraints = [base[0]-v[0]**2]
            for i in range(1, len(v)-1):
                constraints.extend([base[i]-v[i]**2, base[i]-2*v[i]**2])
            constraints.append(base[-1]-v[-1]**2)
            return u, constraints
        if problem == 'HALDMADS':
            # Preserve the decoded OUTSDIF constants, including its near-zero grid point.
            ys = [mp.mpf(float(f'{(i-10)/10:.10e}')) for i in range(21)]
            ys[10] = mp.mpf(float('-1.3877787808e-16'))
            exponentials = [mp.mpf(float(format(math.exp((i-10)/10), '.10e'))) for i in range(21)]
            a, b, c, d, e, _ = x
            constraints = []
            for y, ey in zip(ys, exponentials):
                ratio = (a+y*b)/(1+c*y+d*y*y+e*y**3)
                constraints.extend([ratio-ey-u, -ratio+ey-u])
            return u, constraints
        raise ValueError('No independent formula for ' + problem)


def evaluate(problem, point, oracle, precision=100):
    with mp.workdps(precision):
        objective, constraints = raw_problem(problem, point, precision)
        residuals = []
        for i, value in enumerate(constraints):
            for bound, sign in [(oracle.cu[i], 1), (oracle.cl[i], -1)]:
                if math.isfinite(bound):
                    residuals.append(sign*(value-mp.mpf(float(bound))))
        normalized = [value/mp.mpf(float(scale)) for value, scale in zip(residuals, oracle.sg)]
        violation = max([mp.mpf(0), *normalized])
        in_box = bool(np.all(np.asarray(point) >= oracle.bl) and np.all(np.asarray(point) <= oracle.bu))
        return dict(f_norm=float(objective/mp.mpf(float(oracle.sf))),
                    violation=float(violation), feasible=bool(in_box and mp.isfinite(objective)
                    and violation <= mp.mpf(float(1e-4))), precision=precision,
                    raw_constraints=[mp.nstr(value, 70) for value in constraints])


def cresc4(point, oracle, precision=100):
    return evaluate('CRESC4', point, oracle, precision)


def verify_formula(oracle, problem='CRESC4'):
    points = ([oracle.x0, np.array([-.75, 0., .5, 1.5, 0., .5]),
               np.array([-2., .4, .7, 2.3, .4, .8])] if problem == 'CRESC4'
              else [oracle.x0, np.full(oracle.n, .1), np.full(oracle.n, .3)])
    for point in points:
        f, c = raw_problem(problem, point)
        raw_f = oracle.prob.obj(point, gradient=False)
        raw_c = oracle.prob.cons(point, gradient=False)
        np.testing.assert_allclose(float(f), raw_f, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose([float(value) for value in c], raw_c, rtol=1e-10, atol=1e-10)
