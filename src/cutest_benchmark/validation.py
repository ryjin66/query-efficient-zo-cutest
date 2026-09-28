"""Independent saved-point checks and post-hoc CRESC4 trace correction."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import TARGETS, normalized_gap, read
from .high_precision import evaluate, verify_formula
from .recording import write_json


def load_compressed(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def capture_cresc4_queries(oracle, directory):
    """Record coordinates without changing values returned to the optimizer."""
    original = oracle.eval
    path = Path(directory) / 'query_points.jsonl'

    def recorded(x):
        f, g = original(x)
        row = dict(query=oracle.query_count, x=np.asarray(x).tolist(),
                   f_norm=float(f), g_norm=np.asarray(g).tolist())
        # The uncounted initial check is replaced after NOMAD resets its counter.
        with path.open('w' if oracle.query_count == 1 else 'a') as stream:
            stream.write(json.dumps(row)+'\n')
        return f, g

    oracle.eval = recorded


def load_query_points(path):
    lines = Path(path).read_text().splitlines()
    points = []
    for i, line in enumerate(lines):
        try:
            points.append(json.loads(line))
        except json.JSONDecodeError:
            # A watchdog may interrupt the final append; scored prefixes must still be complete.
            if i != len(lines)-1:
                raise
    return points


def rescore_cresc4(log, points, oracle):
    best, violation, best_x, best_q = math.inf, math.inf, None, None
    checks, prefixes = [], []
    for q, item in enumerate(points, 1):
        if item['query'] != q:
            raise ValueError('Incomplete CRESC4 coordinate sequence')
        low, high = evaluate('CRESC4', item['x'], oracle, 100), evaluate('CRESC4', item['x'], oracle, 200)
        assert low['feasible'] == high['feasible']
        np.testing.assert_allclose([low['f_norm'], low['violation']],
            [high['f_norm'], high['violation']], rtol=1e-13, atol=1e-14, equal_nan=True)
        native_f = float(item['f_norm'])
        native_feasible = math.isfinite(native_f) and max([0., *map(float, item['g_norm'])]) <= 1e-4
        if high['feasible'] and high['f_norm'] < best:
            best, best_x, best_q = high['f_norm'], item['x'], q
        violation = min(violation, high['violation'])
        checks.append(dict(query=q, native_f=native_f, native_feasible=native_feasible,
            verified_f=high['f_norm'], verified_violation=high['violation'], verified_feasible=high['feasible']))
        prefixes.append(dict(best_feasible_f=best, best_violation=violation,
                             best_x=best_x, incumbent_query=best_q))
    if int(log['query'].max()) > len(points):
        raise ValueError('Coordinates do not cover the recorded checkpoints')
    corrected = log.copy()
    for i, row in log.iterrows():
        q = int(row['query'])
        for field in ('best_feasible_f', 'best_violation'):
            corrected.loc[i, field] = prefixes[q-1][field]
        corrected.loc[i, 'f_norm'] = checks[q-1]['verified_f']
        corrected.loc[i, 'violation'] = checks[q-1]['verified_violation']
    pd.testing.assert_frame_equal(log[['query', 'elapsed_seconds']],
                                  corrected[['query', 'elapsed_seconds']], check_exact=True)
    return corrected, pd.DataFrame(checks), prefixes


def correct_recorded_cresc4(directory):
    from .oracle_cutest import CutestOracle
    directory = Path(directory)
    task = json.loads((directory/'task.json').read_text())
    log = read(directory/'trace.csv')
    points = load_query_points(directory/'query_points.jsonl')
    oracle = CutestOracle('CRESC4')
    corrected, checks, prefixes = rescore_cresc4(log, points, oracle)
    (directory/'trace.csv').replace(directory/'uncorrected_trace.csv')
    corrected.to_csv(directory/'trace.csv', index=False)
    checks.to_csv(directory/'query_checks.csv', index=False)
    eligible = log.loc[log['query'].le(task['query_budget']) & log.elapsed_seconds.le(task['scoring_time_seconds'])]
    last = eligible.iloc[-1]
    write_json(directory/'corrected_incumbent.json', dict(problem='CRESC4', solver='nomad_default',
        seed=task['seed'], query=int(last['query']), elapsed_seconds=float(last.elapsed_seconds),
        **prefixes[int(last['query'])-1]))


def check_vector(oracle, problem, point, kind, expected, independent=False):
    x = np.asarray(point, dtype=float)
    if x.shape != (oracle.n,) or not np.isfinite(x).all() or np.any(x < oracle.bl) or np.any(x > oracle.bu):
        raise ValueError('Invalid saved vector or variable bounds')
    native_f, g = oracle.eval(x)
    native_violation = max(0., float(np.max(g))) if len(g) else 0.
    f, violation = native_f, native_violation
    high_precision = (independent or problem == 'CRESC4' or np.max(np.abs(x)) > 1e10
                      or (kind == 'feasible' and abs(f) > 1e10))
    if high_precision:
        low, high = evaluate(problem, x, oracle, 100), evaluate(problem, x, oracle, 200)
        np.testing.assert_allclose([low['f_norm'], low['violation']],
            [high['f_norm'], high['violation']], rtol=1e-13, atol=1e-14, equal_nan=True)
        f, violation = high['f_norm'], high['violation']
    native_value = native_f if kind == 'feasible' else native_violation
    verified_value = f if kind == 'feasible' else violation
    expected = float(expected)
    matches_native = bool(np.isclose(native_value, expected, rtol=1e-9, atol=1e-10))
    matches_verified = bool(np.isclose(verified_value, expected, rtol=1e-9, atol=1e-10))
    if not (matches_verified if independent else matches_native or matches_verified):
        raise ValueError('Saved scalar does not match its vector')
    if kind == 'feasible' and (not math.isfinite(f) or violation > 1e-4):
        raise ValueError('A claimed feasible point fails independent validation')
    if kind == 'violation' and (expected <= 1e-4) != (violation <= 1e-4):
        raise ValueError('Independent minimum-violation feasibility classification changed')
    return dict(objective=f, violation=violation, native_objective=native_f,
        native_violation=native_violation, saved_value=expected,
        high_precision=bool(high_precision), matches_native=matches_native,
        matches_verified=matches_verified, bounds_valid=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path, default=Path('validation'))
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if out == root or out.is_relative_to(root/'results') or out.is_relative_to(root/'config'):
        parser.error('Validation output must be separate from retained data')
    from .oracle_cutest import CutestOracle
    references = load_compressed(root/'config/reference_points.json.gz')
    incumbents = load_compressed(root/'results/incumbents.json.gz')
    metadata = read(root/'config/problems.csv')
    scalar_refs = read(root/'config/references.csv').set_index('problem')
    checks = []
    for problem in metadata.problem:
        oracle = CutestOracle(problem)
        if problem in ('CRESC4', 'MADSEN', 'GIGOMEZ2'):
            verify_formula(oracle, problem)
        if problem in references:
            point = references[problem]
            value = check_vector(oracle, problem, point['x'], 'feasible', point['verified_f'],
                                 point.get('independent_high_precision', False))
            checks.append(dict(problem=problem, role='reference', solver=point['solver'],
                               seed=point['seed'], **value))
        for point in [item for item in incumbents if item['problem'] == problem]:
            for kind, key, field in [('feasible', 'best_x', 'best_feasible_f'),
                                     ('violation', 'best_violation_x', 'best_violation')]:
                if point.get(key) is not None:
                    try:
                        value = check_vector(oracle, problem, point[key], kind, point[field])
                        if kind == 'feasible':
                            ref = scalar_refs.loc[problem]
                            stored_gap = normalized_gap(float(point[field]), ref.f0, ref.f_ref)
                            verified_gap = normalized_gap(value['objective'], ref.f0, ref.f_ref)
                            if any((stored_gap <= tau) != (verified_gap <= tau) for tau, _ in TARGETS):
                                raise ValueError('Independent endpoint target classification changed')
                    except (ValueError, AssertionError) as error:
                        raise ValueError(f'{problem}/{point["solver"]}/{point["seed"]}/{kind}: {error}') from error
                    checks.append(dict(problem=problem, role=kind, solver=point['solver'],
                                       seed=point['seed'], **value))
        print(problem, 'saved-point checks passed', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(checks)
    frame.to_csv(out/'saved_point_checks.csv', index=False)
    write_json(out/'summary.json', dict(status='pass', saved_vectors=len(checks),
        reference_points=len(references), optimizer_reruns=0,
        native_verified_differences=int((~frame.matches_verified).sum())))
    print('Validated', len(checks), 'saved vectors. Query histories are not part of this check.')


if __name__ == '__main__':
    main()
