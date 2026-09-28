import ast
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from cutest_benchmark import fd
from cutest_benchmark.constraints import convert_cutest_constraints_to_inequalities
from cutest_benchmark.logging_utils import RunRecorder
from cutest_benchmark.oracle_cutest import ToyConstrainedOracle
from cutest_benchmark.parameters import select_parameters
from cutest_benchmark.recording import SampledRunRecorder
from cutest_benchmark.runner import tasks_for
from cutest_benchmark.solvers import common, nomad, zob_gda, zob_sgda

ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def test_frozen_executable_source(self):
        sources = json.loads((ROOT/'config/source_versions.json').read_text())['core_files']
        for item in sources:
            with self.subTest(path=item['path']):
                tree = ast.dump(ast.parse((ROOT/item['path']).read_text()))
                self.assertEqual(hashlib.sha256(tree.encode()).hexdigest(), item['executable_ast_sha256'])

    def test_uniform_rules_match_all_resolved_entries(self):
        for task in tasks_for(ROOT, seeds=[1]):
            actual = select_parameters(task['solver'], task['n'], task['query_budget'])
            self.assertEqual(actual, task['params'])
            if task['solver'] == 'zob_sgda':
                n = task['n']/actual['block_rule']
                k = max(1, task['query_budget']//(actual['block_rule']+1))
                beta = actual['alpha']*actual['beta_over_alpha']
                expected = min(1/40, 1/np.sqrt(k*n), 1/(768*actual['p']*beta))
                self.assertAlmostEqual(actual['gamma'], expected, places=14)
                self.assertAlmostEqual(actual['alpha']*actual['p'], n*actual['gamma'], places=14)
                self.assertLess(actual['gamma'], 1/36)

    def test_full_task_grid(self):
        tasks = tasks_for(ROOT)
        self.assertEqual(len(tasks), 1530)
        self.assertEqual(len({(r['problem'],r['solver'],r['seed']) for r in tasks}), 1530)
        for task in tasks:
            self.assertEqual(task['solver_max_time_seconds'], 1800)
            self.assertEqual(task['outer_timeout_seconds'], 1920)
            if task['solver'] == 'nomad_default':
                self.assertEqual(task['solver_max_time_seconds'], 1800)
                self.assertEqual(task['outer_timeout_seconds'], 1920)
            if task['solver'] in ('zoagp', 'zominmax', 'szo_conex'):
                self.assertEqual(task['solver_max_time_seconds'], 1800)
                self.assertEqual(task['outer_timeout_seconds'], 1920)
                self.assertEqual(task['recording'], 'terminal')
            self.assertEqual(task['scoring_time_seconds'], 1800)

    def test_uniform_rule_has_no_problem_or_dimension_regimes(self):
        for solver in ('zob_gda', 'zob_sgda', 'zoagp', 'zominmax', 'szo_conex'):
            for n in (1, 2, 19, 20, 21, 100, 1000):
                params = select_parameters(solver, n, 50000)
                expected = select_parameters(solver, 2, 50000)
                for key in ('alpha', 'beta_over_alpha', 'r0', 'radius_decay', 'ybar', 'theta'):
                    self.assertEqual(params.get(key), expected.get(key))
                if solver == 'zob_gda':
                    self.assertEqual(params['block_rule'], int(np.ceil(np.sqrt(n))))
                elif solver == 'zob_sgda':
                    self.assertEqual(params['block_rule'], min(n, max(1, int(np.ceil(.75*np.sqrt(n))))))

    def test_invalid_selection_is_not_silent(self):
        for kwargs in (dict(problems=['NO_SUCH_PROBLEM']), dict(solvers=['NO_SUCH_SOLVER']),
                       dict(seeds=[0]), dict(solvers=['zoagp'],seeds=[2])):
            with self.assertRaises(ValueError):
                tasks_for(ROOT, **kwargs)

    def test_constraint_side_order_and_finite_sentinels(self):
        g, _ = convert_cutest_constraints_to_inequalities(np.array([3.,4.]),
                                                         np.array([-1.e20,4.]), np.array([5.,4.]))
        np.testing.assert_array_equal(g, [-2., -1.e20-3., 0., 0.])
        self.assertEqual(nomad.finite_bound_list(np.array([-np.inf,-1e20]), -1), [-1e20,-1e20])

    def test_difference_steps_at_bounds(self):
        x = np.array([1.,0.,0.])
        lower, upper = np.array([-1.,0.,-.01]), np.array([1.,0.,.01])
        self.assertEqual(fd.coordinate_step(x, 0, .1, lower, upper), -.1)
        self.assertEqual(fd.coordinate_step(x, 1, .1, lower, upper), 0.)
        self.assertEqual(fd.coordinate_step(x, 2, .1, lower, upper), 0.)

    def test_shared_base_query_cost(self):
        oracle = ToyConstrainedOracle(4)
        indices = np.array([0,2])
        _, _, grad = fd.finite_difference_lagrangian(oracle, oracle.x0, np.zeros(2), indices, .001, 0)
        self.assertEqual(oracle.query_count, 3)
        np.testing.assert_array_equal(grad[[1,3]], [0.,0.])

    def test_budget_reserves_next_iteration(self):
        oracle = ToyConstrainedOracle()
        recorder = RunRecorder('test','test',1,1e-4)
        oracle.query_count = 7
        self.assertFalse(common.should_continue(oracle,10,3,recorder,0))
        self.assertTrue(common.should_continue(oracle,11,3,recorder,0))

    def test_sgda_zero_penalty_matches_gda(self):
        params = dict(alpha=.03, beta_over_alpha=1., block_rule=2, r0=.01,
                      radius_decay=.5, ybar=100., p=0., gamma=.01, max_time_seconds=0, flush_every=0)
        a = zob_gda.run(ToyConstrainedOracle(), params, 100, 3, 1e-4)
        b = zob_sgda.run(ToyConstrainedOracle(), params, 100, 3, 1e-4)
        keys = ('query','iter','f_norm','violation','best_feasible_f','best_violation','x_norm')
        self.assertEqual([[r[k] for k in keys] for r in a.rows], [[r[k] for k in keys] for r in b.rows])

    def test_sampled_and_terminal_recording(self):
        a, b = RunRecorder('test','test',1,1e-4,50), SampledRunRecorder('test','test',1,1e-4,50)
        for recorder in (a,b):
            for query in range(1,54):
                recorder.record(query,0,1/query,np.array([-1.]),np.array([0.]))
            recorder.mark_status('completed')
        self.assertEqual([r['query'] for r in a.rows], [1,50,53])
        self.assertEqual([r['query'] for r in b.rows], [1,50])
        self.assertEqual(b.best_feasible_f, 1/53)
        self.assertEqual(b.rows[-1]['best_feasible_f'], 1/50)

    def test_nomad_explicit_seed_callback(self):
        task = dict(problem='TOY_NONCONVEX', seed=2, eps_feas=1e-4, record_every=50,
                    query_budget=100, solver_max_time_seconds=1800, scoring_time_seconds=1800,
                    flush_every=0, expected_initial_rng_state='state2')
        oracle = ToyConstrainedOracle(4)
        task['f0'] = oracle.initial_f
        received = {}
        class Point:
            def size(self):
                return 4
            def get_coord(self, i):
                return oracle.x0[i]
            def setBBO(self, value):
                received['bbo'] = value
        def optimize(callback, x0, lower, upper, parameters):
            received['parameters'] = parameters
            self.assertEqual(callback(Point()), 1)
            return {'run_flag':1}
        fake = types.SimpleNamespace(optimize=optimize, getRNGState=lambda:'state2')
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', PyNomad=fake), patch.object(nomad,'version',return_value='4.5.1'):
            result = nomad.run(oracle, task, directory)
            certificate = json.loads((Path(directory)/'certificate.json').read_text())
            self.assertTrue(certificate['seed_state_verified'])
            self.assertFalse(certificate['external_set_seed_called'])
            self.assertEqual(result['actual_queries'], 1)
            self.assertIn('SEED 2', received['parameters'])
            self.assertIn('MAX_TIME 1800', received['parameters'])
            self.assertEqual(len(received['bbo'].split()), 3)


if __name__ == '__main__':
    unittest.main()
