"""Seed-level statistics from final run summaries or new query histories."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ORDER = ('zob_sgda', 'zob_gda', 'nomad_default', 'zoagp', 'zominmax', 'szo_conex')
LABELS = dict(zip(ORDER, ('ZOB-SGDA', 'ZOB-GDA', 'NOMAD', 'ZO-AGP', 'ZO-MinMax', 'SZO-ConEx')))
TARGETS = ((.1, 'solved_1e_1'), (.01, 'solved_1e_2'), (.001, 'solved_1e_3'))
METRICS = ('feasible_count', 'feasible_fraction', *(name for _, name in TARGETS))


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def eligible_trace(log, budget, cap=1800):
    required = ('query', 'elapsed_seconds', 'best_feasible_f', 'best_violation')
    if log.empty or log[list(required)].isna().any().any():
        raise ValueError('Empty trace or missing resource/incumbent value')
    if (log['query'].diff().dropna() < 0).any() or (log.elapsed_seconds.diff().dropna() < 0).any():
        raise ValueError('Nonmonotone resource counter')
    valid = log[log['query'].le(budget) & log.elapsed_seconds.le(cap)]
    if valid.empty:
        raise ValueError('No recorded evaluation within both limits')
    return valid


def normalized_gap(best, f0, reference):
    if not math.isfinite(best):
        return math.inf
    if not math.isfinite(reference):
        return math.nan
    return max(0., (best-reference)/max(1., abs(f0-reference)))


def first_hit(log, f0, reference, tau):
    if not math.isfinite(reference):
        return math.inf
    best = log.best_feasible_f.to_numpy(float)
    hit = np.isfinite(best) & (np.maximum(0., (best-reference)/max(1., abs(f0-reference))) <= tau)
    return float(log.loc[hit, 'query'].min()) if hit.any() else math.inf


def validate_grid(index, problems):
    expected = {(p, s, seed) for p in problems for s in ORDER
                for seed in ([1] if s == 'zoagp' else range(1, 11))}
    actual = list(zip(index.problem, index.solver, index.seed))
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Expected exactly 1530 unique problem/solver/seed records')


def validate_final_results(records, hits, metadata, refs):
    keys = ['problem', 'solver', 'seed']
    validate_grid(records, list(metadata.index))
    if set(hits.tau) != {tau for tau, _ in TARGETS}:
        raise ValueError('Final hitting times must contain all three frozen targets')
    if records.feasible.dtype != bool or any(records[name].dtype != bool for _, name in TARGETS):
        raise ValueError('Final feasibility and target flags must be Boolean')
    dimensions = records.problem.map(metadata.n).to_numpy(int)
    budgets = records.problem.map(metadata.query_budget).to_numpy(int)
    np.testing.assert_array_equal(records.query_budget, budgets)
    if not records.scoring_time_seconds.eq(1800).all():
        raise ValueError('Final scoring time limit differs from the protocol')
    q = records.queries.to_numpy(float)
    if not (np.isfinite(q) & (q >= 1) & (q <= budgets) & (q == np.floor(q))).all():
        raise ValueError('Invalid last recorded query count')
    if not records.solver_seconds.between(0, 1800).all():
        raise ValueError('Final checkpoint is outside the scoring window')
    np.testing.assert_array_equal(records.feasible, np.isfinite(records.best_feasible_f))
    if records.best_violation.isna().any() or records.best_violation.lt(0).any():
        raise ValueError('Invalid final constraint violation')
    if (records.feasible & records.best_violation.gt(1e-4)).any():
        raise ValueError('Final feasible incumbent exceeds the feasibility tolerance')
    gaps = np.array([normalized_gap(row.best_feasible_f, refs.loc[row.problem, 'f0'],
                                   refs.loc[row.problem, 'f_ref']) for row in records.itertuples()])
    np.testing.assert_allclose(records.norm_gap, gaps, rtol=0, atol=0, equal_nan=True)
    order = records.set_index(keys).index
    previous = np.zeros(len(records))
    for tau, metric in TARGETS:
        target = hits.loc[hits.tau.eq(tau)]
        validate_grid(target, list(metadata.index))
        target = target.set_index(keys).loc[order]
        first = target.query_to_target.to_numpy(float)
        finite = np.isfinite(first)
        valid = np.isposinf(first) | (finite & (first >= 1) & (first <= q) & (first == np.floor(first)))
        if not valid.all() or (first < previous).any():
            raise ValueError('Invalid or nonmonotone final hitting times')
        np.testing.assert_array_equal(records[metric], records.feasible & (gaps <= tau))
        np.testing.assert_array_equal(records[metric], finite)
        np.testing.assert_array_equal(records['queries_'+metric], first)
        np.testing.assert_array_equal(target.normalized_queries, first/(dimensions+1))
        previous = first


def reconstruct(root, data, reference_policy='verified'):
    if (data/'trace_index.csv').exists():
        return reconstruct_traces(root, data, reference_policy)
    if reference_policy != 'verified':
        raise ValueError('Changing reference values requires query histories; use --data with a complete run directory')
    versions = json.loads((root/'config/source_versions.json').read_text())
    for relative in ('config/problems.csv', 'config/references.csv'):
        if digest((root/relative).read_bytes()) != versions['configuration_sha256'][relative]:
            raise ValueError('Final hitting times require the frozen dimensions, budgets, and reference values')
    metadata = read(root/'config/problems.csv').set_index('problem')
    refs = read(root/'config/references.csv').set_index('problem')
    records, hits = read(data/'run_metrics.csv'), read(data/'hitting_times.csv')
    validate_final_results(records, hits, metadata, refs)
    return records, hits, metadata


def reconstruct_traces(root, data, reference_policy='verified'):
    metadata = read(root/'config/problems.csv').set_index('problem')
    refs = read(root/'config/references.csv').set_index('problem')
    if reference_policy == 'historical-sensitivity':
        refs['f_ref'] = refs.legacy_f_ref
    index = read(data/'trace_index.csv')
    validate_grid(index, list(metadata.index))
    records, hits = [], []
    for item in index.itertuples(index=False):
        path = (data/item.path).resolve()
        if not path.is_relative_to(data.resolve()):
            raise ValueError('Trace path escapes the results directory')
        content = path.read_bytes()
        if digest(content) != item.file_sha256:
            raise ValueError('File checksum mismatch: '+item.path)
        raw = gzip.decompress(content) if path.suffix == '.gz' else content
        if digest(raw) != item.csv_sha256:
            raise ValueError('CSV checksum mismatch: '+item.path)
        log = read(io.BytesIO(raw))
        for column in ('problem', 'solver', 'seed'):
            if not log[column].eq(getattr(item, column)).all():
                raise ValueError('Trace identity mismatch: '+item.path)
        meta, ref = metadata.loc[item.problem], refs.loc[item.problem]
        valid = eligible_trace(log, int(meta.query_budget))
        best, violation = float(valid.best_feasible_f.min()), float(valid.best_violation.min())
        np.testing.assert_allclose(float(log.iloc[0].f_norm), ref.f0, rtol=1e-9, atol=1e-10)
        feasible = math.isfinite(best)
        if feasible and violation > 1e-4:
            raise ValueError('Inconsistent feasible incumbent')
        gap = normalized_gap(best, ref.f0, ref.f_ref)
        row = dict(problem=item.problem, solver=item.solver, seed=int(item.seed),
                   query_budget=int(meta.query_budget), scoring_time_seconds=1800,
                   best_feasible_f=best, best_violation=violation, feasible=feasible,
                   norm_gap=gap, queries=int(valid['query'].max()),
                   solver_seconds=float(valid.elapsed_seconds.max()), excluded_rows=len(log)-len(valid),
                   status=str(log.iloc[-1].status), final_recorded_f=float(valid.iloc[-1].f_norm),
                   final_recorded_violation=float(valid.iloc[-1].violation))
        for tau, metric in TARGETS:
            q = first_hit(valid, ref.f0, ref.f_ref, tau)
            row[metric] = math.isfinite(q)
            if row[metric] != (feasible and gap <= tau):
                raise ValueError('Endpoint and hitting-time results disagree')
            hits.append(dict(problem=item.problem, solver=item.solver, seed=int(item.seed),
                             tau=tau, query_to_target=q, normalized_queries=q/(int(meta.n)+1)))
        records.append(row)
    records = pd.DataFrame(records)
    if (data/'run_info.csv').exists():
        info = read(data/'run_info.csv').drop(columns='status')
        records = records.merge(info, on=['problem','solver','seed'], validate='one_to_one')
    for tau, metric in TARGETS:
        selected = pd.DataFrame(hits).loc[lambda frame: frame.tau.eq(tau),
            ['problem','solver','seed','query_to_target']].rename(columns={'query_to_target':'queries_'+metric})
        records = records.merge(selected, on=['problem','solver','seed'], validate='one_to_one')
    return records, pd.DataFrame(hits), metadata


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite seed-level observations')
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else math.nan
    se = sd/math.sqrt(len(values))
    half = float(t.ppf(.975, len(values)-1))*se if len(values) > 1 else math.nan
    return dict(n_seeds=len(values), mean=mean, sample_sd=sd, standard_error=se,
                ci_low=mean-half, ci_high=mean+half, ci_halfwidth=half)


def summarize(records, problems):
    validate_grid(records, problems)
    counts, summary = [], []
    for solver in ORDER:
        subset = records[records.solver.eq(solver)]
        seed_rows = []
        for seed, group in subset.groupby('seed', sort=True):
            row = dict(solver=solver, seed=int(seed), problems=len(problems),
                       feasible_count=int(group.feasible.sum()),
                       feasible_fraction=float(group.feasible.mean()))
            row.update({metric: int(group[metric].sum()) for _, metric in TARGETS})
            counts.append(row)
            seed_rows.append(row)
        for metric in METRICS:
            summary.append(dict(solver=solver, metric=metric,
                                **mean_ci([r[metric] for r in seed_rows])))
    return pd.DataFrame(counts), pd.DataFrame(summary)


def problem_rates(records):
    rows = []
    for (solver, problem), group in records.groupby(['solver', 'problem'], sort=False):
        rows.append(dict(solver=solver, problem=problem, seeds=len(group),
            feasible_rate=float(group.feasible.mean()),
            **{name: float(group[name].mean()) for _, name in TARGETS}))
    return pd.DataFrame(rows)


def profiles(hits, metadata):
    maximum = float((metadata.query_budget/(metadata.n+1)).max())
    finite = hits.loc[np.isfinite(hits.normalized_queries), 'normalized_queries']
    grid = np.unique(np.r_[0., 1., maximum, finite])
    if grid[-1] > maximum + 1e-9:
        raise ValueError('Hitting time exceeds the resource budget')
    curves = []
    for solver in ORDER:
        for tau, _ in TARGETS:
            group = hits[hits.solver.eq(solver) & hits.tau.eq(tau)]
            seeds = [1] if solver == 'zoagp' else list(range(1, 11))
            matrix = group.pivot(index='seed', columns='problem', values='normalized_queries')
            times = matrix.loc[seeds, metadata.index].to_numpy(float)
            if np.isnan(times).any():
                raise ValueError('Incomplete hitting-time matrix')
            values = np.array([np.searchsorted(np.sort(row), grid, side='right')/len(metadata) for row in times])
            for j, x in enumerate(grid):
                curves.append(dict(solver=solver, tau=tau, normalized_queries=x, **mean_ci(values[:, j])))
    return pd.DataFrame(curves), maximum


def profile_areas(hits, curves, maximum):
    rows, summaries = [], []
    for (solver, tau), group in hits.groupby(['solver', 'tau']):
        values = []
        for seed, frame in group.groupby('seed'):
            x = frame.normalized_queries.to_numpy(float)
            area = np.where(np.isfinite(x), np.log(maximum/np.clip(x, 1, maximum)), 0).mean()/np.log(maximum)
            rows.append(dict(solver=solver, seed=seed, tau=tau, normalized_log_auc=area))
            values.append(area)
        curve = curves.loc[curves.solver.eq(solver) & curves.tau.eq(tau) & curves.normalized_queries.ge(1)]
        integral = np.sum(curve['mean'].to_numpy()[:-1]
                          * np.diff(np.log(curve.normalized_queries)))/np.log(maximum)
        np.testing.assert_allclose(integral, np.mean(values), rtol=0, atol=1e-12)
        summaries.append(dict(solver=solver, tau=tau, **mean_ci(values)))
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def per_run_markdown(records, problems):
    def number(value):
        return 'NA' if pd.isna(value) else f'{value:.6g}'
    lines = ['# Final Metrics by Problem and Seed', '',
        'All 1,530 retained runs are shown. Values are scored within both resource limits.',
        'The objective is the best feasible normalized value; minimum violation may belong to another point.',
        '`inf` means no finite value or target hit; `NA` means unavailable, including an undefined reference gap.',
        'First-hit query counts and full-precision values are in [run_metrics.csv](run_metrics.csv).', '',
        ' | '.join(f'[{problem}](#{problem.lower()})' for problem in problems), '']
    for problem in problems:
        lines.extend([f'## {problem}', '',
            '| Algorithm | Seed | Feasible | Objective | Min. violation | Gap | Queries | Seconds | Solved 0.1 | Solved 0.001 | Stop |',
            '| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | --- |'])
        for solver in ORDER:
            for row in records.loc[records.problem.eq(problem) & records.solver.eq(solver)].sort_values('seed').itertuples():
                cells = [LABELS[solver], str(row.seed), 'yes' if row.feasible else 'no',
                    number(row.best_feasible_f), number(row.best_violation), number(row.norm_gap),
                    str(row.queries), f'{row.solver_seconds:.3f}',
                    'yes' if row.solved_1e_1 else 'no', 'yes' if row.solved_1e_3 else 'no', row.status]
                lines.append('| '+' | '.join(cells)+' |')
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--data', type=Path)
    parser.add_argument('--out', type=Path, default=Path('reproduced'))
    parser.add_argument('--reference-policy', choices=('verified', 'historical-sensitivity'), default='verified')
    args = parser.parse_args()
    root = args.root.resolve()
    data = (args.data or root/'results').resolve()
    out = args.out.resolve()
    protected = [data, *(root/name for name in ('results', 'config', 'src', 'scripts', 'tests', 'environment'))]
    if out == root or any(out.is_relative_to(path) or path.is_relative_to(out) for path in protected):
        parser.error('Use a separate output directory; retained results are read-only')
    out.mkdir(parents=True, exist_ok=True)
    records, hits, metadata = reconstruct(root, data, args.reference_policy)
    counts, summary = summarize(records, list(metadata.index))
    curves, maximum = profiles(hits, metadata)
    seed_areas, areas = profile_areas(hits, curves, maximum)
    for solver in ORDER:
        for tau, metric in TARGETS:
            endpoint = curves[curves.solver.eq(solver) & curves.tau.eq(tau)].iloc[-1]
            stat = summary[summary.solver.eq(solver) & summary.metric.eq(metric)].iloc[0]
            np.testing.assert_allclose(endpoint[['mean','ci_low','ci_high']].to_numpy(float)*len(metadata),
                                       stat[['mean','ci_low','ci_high']].to_numpy(float), rtol=0, atol=1e-10, equal_nan=True)
    for name, frame in [('run_metrics',records), ('hitting_times',hits), ('per_seed_counts',counts),
                        ('summary',summary), ('profiles',curves), ('per_seed_profile_areas',seed_areas),
                        ('profile_areas',areas), ('per_problem_rates',problem_rates(records))]:
        frame.to_csv(out/(name+'.csv'), index=False)
    from .reporting import table
    tex, tabular = table(summary, areas, maximum, args.reference_policy != 'verified')
    (out/'summary_table.tex').write_text(tex)
    tabular.to_csv(out/'summary_table.csv', index=False)
    (out/'per_run.md').write_text(per_run_markdown(records, list(metadata.index)))
    from .figures import render
    render(curves, out, maximum)
    print(f'Reconstructed {len(records)} runs, {len(counts)} seed-level counts, and 18 profile curves.')
    print(summary.pivot(index='solver', columns='metric', values='mean').to_string())


if __name__ == '__main__':
    main()
