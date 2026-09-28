import math
import unittest
from unittest.mock import patch

import numpy as np

from cutest_benchmark.solvers.reference.common import EvaluationContext, coordinate_gradient, sphere_gradient
from cutest_benchmark.solvers.reference import zoagp, zominmax, szo_conex


class LinearOracle:
    problem_name = 'LINEAR_TEST'
    def __init__(self, n=2, m=1):
        self.n, self.num_ineq = n, m
        self.x0 = np.ones(n)
        self.bl, self.bu = np.full(n,-100.), np.full(n,100.)
        self.query_count = 0
        self.points = []
    def eval(self, x):
        self.query_count += 1
        self.points.append(np.asarray(x).copy())
        return float(np.sum(x)), np.full(self.num_ineq,np.sum(x)-1.)


class FixedRNG:
    def __init__(self, values):
        self.values = iter(values)
    def normal(self, size):
        value = np.asarray(next(self.values), dtype=float)
        assert len(value)==size
        return value
    def integers(self, count):
        return count-1


class ReferenceTests(unittest.TestCase):
    def test_agp_uses_new_primal_point_for_dual(self):
        oracle = LinearOracle()
        params = dict(alpha=.1,beta_over_alpha=1.,ybar=100.,r0=.01,radius_decay=.5)
        result, _ = zoagp.run(oracle,params,4,101,time_cap=0,record_every=1)
        np.testing.assert_allclose(result['x'],[.9,.9])
        np.testing.assert_allclose(result['y'],[.08])
        self.assertEqual(result['queries'],4)

    def test_agp_regularization_and_learning_rate_schedule(self):
        params = dict(alpha=.1,r0=.01,radius_decay=.5,dual_regularization=.1)
        first = zoagp.step_parameters(params,0)
        second = zoagp.step_parameters(params,15)
        np.testing.assert_allclose(first,[.1,.1,.01])
        np.testing.assert_allclose(second,[.1*101/104,.05,.0025])
        oracle = LinearOracle()
        run_params = dict(params,beta_over_alpha=1.,ybar=100.)
        result,_ = zoagp.run(oracle,run_params,7,101,time_cap=0,record_every=1)
        alpha2,penalty2,_ = zoagp.step_parameters(params,1)
        x2 = .9-alpha2*1.08
        y2 = .08+.1*(2*x2-1-penalty2*.08)
        np.testing.assert_allclose(result['x'],[x2,x2])
        np.testing.assert_allclose(result['y'],[y2])

    def test_coordinates_are_unscaled_forward_differences(self):
        oracle=LinearOracle()
        oracle.bu=np.ones(2)
        ctx=EvaluationContext(oracle,'zoagp',101,10,time_cap=0,record_every=1)
        base=ctx.evaluate(ctx.x)
        gradient=coordinate_gradient(ctx,ctx.x,np.zeros(1),.1,base,0)
        np.testing.assert_allclose(gradient,[1.,1.])
        np.testing.assert_allclose(oracle.points[1],[1.1,1.])
        self.assertFalse(ctx.recorder.rows[1]['box_feasible'])
        self.assertTrue(math.isinf(ctx.recorder.rows[1]['violation']))

    def test_sphere_estimator_contains_dimension_factor(self):
        oracle=LinearOracle()
        ctx=EvaluationContext(oracle,'zominmax',101,10,time_cap=0,record_every=1)
        base=ctx.evaluate(ctx.x)
        gradient=sphere_gradient(ctx,ctx.x,np.zeros(1),.1,2,base,
                                 FixedRNG([[1,0],[0,1]]),0)
        np.testing.assert_allclose(gradient,[1.,1.])
        self.assertEqual(oracle.query_count,3)

    def test_minmax_alternates_and_uses_cached_base(self):
        oracle=LinearOracle()
        params=dict(alpha=.1,beta_over_alpha=1.,ybar=100.,r0=.01,directions=1)
        with patch.object(zominmax.np.random,'default_rng',return_value=FixedRNG([[1,0]])):
            result,_=zominmax.run(oracle,params,3,101,time_cap=0,record_every=1)
        np.testing.assert_allclose(result['x'],[.8,1.])
        np.testing.assert_allclose(result['y'],[.08])
        self.assertEqual(result['queries'],3)

    def test_gaussian_model_and_independent_component_gradients(self):
        oracle=LinearOracle(n=1)
        ctx=EvaluationContext(oracle,'szo_conex',101,10,time_cap=0,record_every=1)
        base=ctx.evaluate(ctx.x)
        values,jac=szo_conex.constraint_model(ctx,ctx.x,.1,base,FixedRNG([[2.]]),0)
        np.testing.assert_allclose(values,[.2])
        np.testing.assert_allclose(jac,[[4.]])
        gradient=szo_conex.primal_gradient(ctx,ctx.x,np.array([2.]),.1,base,
                                          FixedRNG([[1.],[3.]]),0)
        np.testing.assert_allclose(gradient,[19.])
        self.assertEqual(oracle.query_count,4)

    def test_conex_dual_first_unbounded_and_initial_extrapolation(self):
        oracle=LinearOracle(n=1)
        params=dict(alpha=.1,beta_over_alpha=1.,r0=.1,theta=1.)
        with patch.object(szo_conex.np.random,'default_rng',return_value=FixedRNG([[2.],[1.],[3.]])):
            result,_=szo_conex.run(oracle,params,6,101,time_cap=0,record_every=1)
        y=.1*.2
        gradient=1+y*9
        np.testing.assert_allclose(result['x'],[1-.1*gradient])
        np.testing.assert_allclose(result['y'],[y])
        self.assertEqual(result['iterations'],1)
        self.assertEqual(result['outer_iterations'],0)
        self.assertTrue(result['average_evaluated'])
        self.assertEqual(result['queries'],6)

    def test_conex_linearized_extrapolation_and_returned_mean(self):
        oracle=LinearOracle(n=1)
        params=dict(alpha=.1,beta_over_alpha=1.,r0=.1,theta=1.)
        directions=FixedRNG([[2.],[1.],[3.],[2.],[1.],[3.]])
        with patch.object(szo_conex.np.random,'default_rng',return_value=directions):
            result,_=szo_conex.run(oracle,params,10,101,time_cap=0,record_every=1)
        y1=.1*.2
        x1=1-.1*(1+9*y1)
        ell1=.2+4*(x1-1)
        y2=max(0.,y1+.1*(2*ell1-.2))
        x2=x1-.1*(1+9*y2)
        np.testing.assert_allclose(result['y'],[y2])
        np.testing.assert_allclose(result['x'],[(x1+x2)/2])
        self.assertEqual(result['queries'],10)

    def test_partial_budget_does_not_fake_completed_conex_step(self):
        oracle=LinearOracle(n=2,m=4)
        params=dict(alpha=.1,beta_over_alpha=1.,r0=.1)
        result,_=szo_conex.run(oracle,params,5,101,time_cap=0,record_every=1)
        self.assertEqual(result['iterations'],0)
        self.assertEqual(result['queries'],1)
        self.assertEqual(result['status'],'query_limit')

    def test_conex_has_no_finite_dual_upper_bound(self):
        oracle=LinearOracle(n=1)
        oracle.x0=np.array([50.])
        params=dict(alpha=.1,beta_over_alpha=100.,r0=.1,theta=1.)
        with patch.object(szo_conex.np.random,'default_rng',return_value=FixedRNG([[1.],[1.],[1.]])):
            result,_=szo_conex.run(oracle,params,6,101,time_cap=0,record_every=1)
        self.assertGreater(result['y'][0],100.)

    def test_conex_does_not_query_returned_mean_after_timeout(self):
        oracle=LinearOracle(n=1)
        params=dict(alpha=.1,beta_over_alpha=1.,r0=.1,theta=1.)
        original=EvaluationContext.check
        def timed_check(context,required=1):
            if context.iterations == 1:
                from cutest_benchmark.solvers.reference.common import RunStopped
                context.status='time_limit'
                raise RunStopped()
            original(context,required)
        with patch.object(EvaluationContext,'check',timed_check):
            result,_=szo_conex.run(oracle,params,20,101,time_cap=0,record_every=1)
        self.assertEqual(result['iterations'],1)
        self.assertEqual(result['queries'],5)
        self.assertFalse(result['average_evaluated'])
        self.assertEqual(result['status'],'time_limit')

    def test_nonfinite_evaluation_is_counted_and_recorded(self):
        oracle=LinearOracle(n=1)
        original=oracle.eval
        def invalid(x):
            f,g=original(x)
            return (np.nan,g) if oracle.query_count==2 else (f,g)
        oracle.eval=invalid
        params=dict(alpha=.1,beta_over_alpha=1.,r0=.1,radius_decay=.5,ybar=100.)
        result,recorder=zoagp.run(oracle,params,5,101,time_cap=0,record_every=1)
        self.assertEqual(result['queries'],2)
        self.assertEqual(recorder.rows[-1]['query'],2)
        self.assertEqual(result['status'],'oracle_domain_failure')


if __name__=='__main__':
    unittest.main()
