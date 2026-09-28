import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.stats import t

from cutest_benchmark.analysis import (eligible_trace, first_hit, mean_ci, normalized_gap,
    read, validate_grid, profile_areas, reconstruct, validate_final_results)

ROOT = Path(__file__).resolve().parents[1]


class AnalysisTests(unittest.TestCase):
    def test_ci_uses_seed_count_standard_error(self):
        values = np.arange(10, dtype=float)
        result = mean_ci(values)
        self.assertAlmostEqual(result['ci_halfwidth'], t.ppf(.975,9)*values.std(ddof=1)/np.sqrt(10))
        self.assertEqual(mean_ci([2.]*10)['ci_halfwidth'], 0.)

    def test_single_run_has_no_seed_interval(self):
        self.assertTrue(math.isnan(mean_ci([22.])['ci_halfwidth']))

    def test_gap_scale_and_improved_reference(self):
        self.assertEqual(normalized_gap(3.,11.,1.), .2)
        self.assertEqual(normalized_gap(-2.,11.,1.), 0.)
        self.assertTrue(math.isnan(normalized_gap(3.,1.,math.inf)))
        self.assertEqual(normalized_gap(math.inf,1.,0.), math.inf)

    def test_query_and_time_limits(self):
        log = pd.DataFrame(dict(query=[1,50,100,150], elapsed_seconds=[.1,1.,1801.,1802.],
                                best_feasible_f=[math.inf,5.,1.,0.], best_violation=[1.,0.,0.,0.]))
        valid = eligible_trace(log,100)
        self.assertEqual(valid['query'].tolist(), [1,50])
        self.assertEqual(first_hit(valid,10.,0.,.1), math.inf)
        self.assertEqual(first_hit(valid,10.,0.,.5), 50)
        self.assertEqual(first_hit(valid,10.,math.inf,.5), math.inf)

    def test_missing_and_duplicated_seeds_fail(self):
        index = read(ROOT/'results/run_metrics.csv')
        problems = read(ROOT/'config/problems.csv').problem.tolist()
        validate_grid(index, problems)
        with self.assertRaises(ValueError):
            validate_grid(index.iloc[1:], problems)
        with self.assertRaises(ValueError):
            validate_grid(pd.concat([index,index.iloc[:1]]), problems)

    def test_unchanged_solver_summary(self):
        summary = read(ROOT/'results/summary.csv')
        expected = {
            'zob_sgda':[22.5,16.9,12.6,12.3], 'zob_gda':[22.3,18.4,14.2,9.7],
            'nomad_default':[25.5,13.9,8.7,4.5],
        }
        for solver, values in expected.items():
            group = summary[summary.solver.eq(solver)].set_index('metric')
            np.testing.assert_allclose(group.loc[['feasible_count','solved_1e_1','solved_1e_2','solved_1e_3'],'mean'], values, rtol=0, atol=1e-12)

    def test_reference_baseline_summary(self):
        summary = read(ROOT/'results/summary.csv')
        expected = {'zoagp':[18.,11.,7.,5.], 'zominmax':[18.2,12.9,10.,8.4],
                    'szo_conex':[16.3,7.2,4.6,2.1]}
        for solver, values in expected.items():
            group = summary[summary.solver.eq(solver)].set_index('metric')
            np.testing.assert_allclose(group.loc[['feasible_count','solved_1e_1','solved_1e_2','solved_1e_3'],'mean'], values, rtol=0, atol=1e-12)

    def test_auc_reconstruction_and_seed_intervals(self):
        hits = read(ROOT/'results/hitting_times.csv')
        curves = read(ROOT/'results/profiles.csv')
        seed, summary = profile_areas(hits, curves, curves.normalized_queries.max())
        pd.testing.assert_frame_equal(seed, read(ROOT/'results/per_seed_profile_areas.csv'), check_exact=True)
        pd.testing.assert_frame_equal(summary, read(ROOT/'results/profile_areas.csv'), check_exact=True)
        for row in summary.itertuples():
            values = seed.loc[seed.solver.eq(row.solver) & seed.tau.eq(row.tau), 'normalized_log_auc']
            self.assertTrue(values.between(0, 1).all())
            if row.solver != 'zoagp':
                self.assertAlmostEqual(row.ci_halfwidth, t.ppf(.975, 9)*values.std(ddof=1)/np.sqrt(10))

    def test_final_table_and_label(self):
        from cutest_benchmark.figures import LABELS
        from cutest_benchmark.reporting import table
        curves = read(ROOT/'results/profiles.csv')
        tex, _ = table(read(ROOT/'results/summary.csv'), read(ROOT/'results/profile_areas.csv'),
                       curves.normalized_queries.max())
        self.assertEqual(tex, (ROOT/'results/summary_table.tex').read_text())
        self.assertNotIn(r'\tau=10^{-2}', tex)
        self.assertIn('Normalized log-AUC', tex)
        self.assertEqual(LABELS['zominmax'], 'ZO-MinMax')

    def test_historical_parameters_are_complete_and_not_active(self):
        old = read(ROOT/'config/historical_problem_parameters.csv')
        self.assertEqual(len(old), 150)
        self.assertFalse(old.duplicated(['problem', 'solver']).any())
        self.assertEqual(set(old.solver), {'zob_gda', 'zob_sgda', 'zoagp', 'zominmax', 'szo_conex'})
        for _, rows in old.groupby('solver'):
            self.assertEqual(rows.problem.nunique(), 30)

    def test_final_summary_loader_needs_no_histories(self):
        records, hits, metadata = reconstruct(ROOT, ROOT/'results')
        self.assertEqual(len(records), 1530)
        self.assertEqual(len(hits), 4590)
        self.assertEqual(len(metadata), 30)
        self.assertFalse((ROOT/'results/traces').exists())

    def test_changed_reference_policy_requires_histories(self):
        with self.assertRaisesRegex(ValueError, 'requires query histories'):
            reconstruct(ROOT, ROOT/'results', 'historical-sensitivity')

    def test_final_summary_rejects_inconsistent_values(self):
        records = read(ROOT/'results/run_metrics.csv')
        hits = read(ROOT/'results/hitting_times.csv')
        metadata = read(ROOT/'config/problems.csv').set_index('problem')
        refs = read(ROOT/'config/references.csv').set_index('problem')
        changed = records.copy()
        changed.loc[0, 'norm_gap'] += 1.
        with self.assertRaises(AssertionError):
            validate_final_results(changed, hits, metadata, refs)
        changed = hits.copy()
        changed.loc[0, 'query_to_target'] = 1e9
        with self.assertRaises(ValueError):
            validate_final_results(records, changed, metadata, refs)
        changed = hits.copy()
        changed.loc[0, 'normalized_queries'] += 1.
        with self.assertRaises(AssertionError):
            validate_final_results(records, changed, metadata, refs)
        with self.assertRaises(ValueError):
            validate_final_results(records, hits.iloc[1:], metadata, refs)


if __name__ == '__main__':
    unittest.main()
