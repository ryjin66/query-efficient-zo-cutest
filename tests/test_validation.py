import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cutest_benchmark.analysis import LABELS, per_run_markdown, read, validate_grid
from cutest_benchmark.recording import BudgetRecorder, captured_recorder_class
from cutest_benchmark.validation import (
    capture_cresc4_queries, check_vector, load_compressed, load_query_points, rescore_cresc4,
)

ROOT = Path(__file__).resolve().parents[1]


class ValidationTests(unittest.TestCase):
    def test_every_retained_run_is_displayed(self):
        records = read(ROOT/'results/run_metrics.csv')
        problems = read(ROOT/'config/problems.csv').problem.tolist()
        validate_grid(records, problems)
        text = per_run_markdown(records, problems)
        self.assertEqual(text, (ROOT/'results/per_run.md').read_text())
        rows = [line for line in text.splitlines()
                if line.startswith('| ') and line.split(' | ')[0][2:] in LABELS.values()]
        self.assertEqual(len(rows), 1530)
        self.assertEqual(text.count('\n## '), 30)

    def test_saved_vectors_match_scored_endpoints(self):
        points = load_compressed(ROOT/'results/incumbents.json.gz')
        records = read(ROOT/'results/run_metrics.csv').set_index(['problem', 'solver', 'seed'])
        self.assertEqual(len(points), 1530)
        seen = set()
        for item in points:
            key = (item['problem'], item['solver'], item['seed'])
            self.assertNotIn(key, seen)
            seen.add(key)
            row = records.loc[key]
            self.assertEqual(item.get('best_x') is not None, bool(row.feasible))
            np.testing.assert_allclose(float(item['best_feasible_f']), row.best_feasible_f,
                                       rtol=0, atol=0)
            np.testing.assert_allclose(float(item['best_violation']), row.best_violation,
                                       rtol=0, atol=0)

    def test_reference_witness_coverage(self):
        references = read(ROOT/'config/references.csv').set_index('problem')
        witnesses = load_compressed(ROOT/'config/reference_points.json.gz')
        finite = references.loc[np.isfinite(references.f_ref)]
        self.assertEqual(set(witnesses), set(finite.index))
        self.assertEqual(len(witnesses), 29)
        self.assertNotIn('CAMSHAPE', witnesses)
        for problem, point in witnesses.items():
            self.assertEqual(point['verified_f'], references.loc[problem, 'f_ref'])
            self.assertLessEqual(point['verified_violation'], 1e-4)

    def test_all_nomad_seeds_have_matching_certificates(self):
        states = json.loads((ROOT/'config/nomad_rng_states.json').read_text())
        info = read(ROOT/'results/run_info.csv')
        nomad = info.loc[info.solver.eq('nomad_default')]
        self.assertEqual(len(nomad), 300)
        self.assertTrue(nomad.nomad_seed_state_verified.all())
        for row in nomad.itertuples():
            self.assertEqual(row.nomad_initial_rng_state, states[str(row.seed)])

    def test_coordinate_capture_keeps_native_feedback(self):
        oracle = types.SimpleNamespace(query_count=0)
        def native(x):
            oracle.query_count += 1
            return float(x[0]), np.array([0.])
        oracle.eval = native
        with tempfile.TemporaryDirectory() as directory:
            capture_cresc4_queries(oracle, directory)
            oracle.eval([99.])
            oracle.query_count = 0
            for value in (2., 3.):
                f, g = oracle.eval([value])
                self.assertEqual(f, value)
                np.testing.assert_array_equal(g, [0.])
            rows = [json.loads(line) for line in (Path(directory)/'query_points.jsonl').read_text().splitlines()]
            self.assertEqual([r['query'] for r in rows], [1, 2])
            self.assertEqual([r['x'] for r in rows], [[2.], [3.]])

    def test_correction_uses_valid_prefix_without_new_queries(self):
        points = [dict(query=i, x=[i], f_norm=-100., g_norm=[0.]) for i in range(1, 4)]
        log = pd.DataFrame(dict(query=[1, 2], elapsed_seconds=[.1, .2],
            best_feasible_f=[-100., -100.], best_violation=[0., 0.],
            f_norm=[-100., -100.], violation=[0., 0.]))
        def verified(problem, point, oracle, precision):
            i = point[0]
            return dict(f_norm=10./i, violation=1. if i == 1 else 0., feasible=i > 1)
        with patch('cutest_benchmark.validation.evaluate', side_effect=verified):
            corrected, checks, prefixes = rescore_cresc4(log, points, None)
            self.assertEqual(corrected.best_feasible_f.tolist(), [math.inf, 5.])
            pd.testing.assert_frame_equal(corrected[['query','elapsed_seconds']], log[['query','elapsed_seconds']])
            self.assertEqual(prefixes[1]['incumbent_query'], 2)
            self.assertFalse(checks.iloc[0].verified_feasible)
            with self.assertRaises(ValueError):
                rescore_cresc4(log, [points[1]], None)

    def test_watchdog_partial_append_keeps_only_complete_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'points.jsonl'
            path.write_text('{"query": 1}\n{"query":')
            self.assertEqual(load_query_points(path), [{'query': 1}])
            path.write_text('{"query":\n{"query": 2}\n')
            with self.assertRaises(json.JSONDecodeError):
                load_query_points(path)

    def test_independent_validation_rejects_false_feasibility(self):
        oracle = types.SimpleNamespace(n=1, bl=np.array([-1.]), bu=np.array([1.]),
                                       eval=lambda x:(1., np.array([0.])))
        with patch('cutest_benchmark.validation.evaluate',
                   return_value=dict(f_norm=1., violation=1., feasible=False)):
            with self.assertRaises(ValueError):
                check_vector(oracle, 'CRESC4', [0.], 'feasible', 1.)

    def test_native_high_precision_difference_is_explicit(self):
        oracle = types.SimpleNamespace(n=1, bl=np.array([-1.]), bu=np.array([1.]),
                                       eval=lambda x:(1., np.array([0.])))
        with patch('cutest_benchmark.validation.evaluate',
                   return_value=dict(f_norm=1.1, violation=0., feasible=True)):
            check = check_vector(oracle, 'CRESC4', [0.], 'feasible', 1.)
            self.assertTrue(check['matches_native'])
            self.assertFalse(check['matches_verified'])
            self.assertEqual(check['objective'], 1.1)

    def test_in_budget_snapshot_survives_later_query(self):
        instances = []
        cls = captured_recorder_class(BudgetRecorder, instances)
        recorder = cls('test', 'test', 1, 1e-4, 1, budget=2, time_cap=1800)
        for q in range(1, 4):
            recorder.record(q, 0, 1./q, np.array([-1.]), np.array([q]))
        self.assertEqual(recorder.scored['query'], 2)
        self.assertEqual(recorder.finite_scored['best_feasible_f'], .5)
        self.assertIs(instances[0], recorder)


if __name__ == '__main__':
    unittest.main()
