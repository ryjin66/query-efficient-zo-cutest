"""PyNomad 4.5.1 callback with explicit seeds and sampled incumbents."""
import csv
from importlib.metadata import version
from pathlib import Path

import numpy as np

from ..constraints import violation
from ..fd import project_bounds
from ..recording import write_json
from ..sampled_logging import RunRecorder


def finite_bound_list(values, sign):
    arr = np.asarray(values, dtype=float).reshape(-1)
    replacement = float(sign)*1.0e20
    return [float(v) if np.isfinite(v) else replacement for v in arr]


def run(oracle, task, directory):
    import PyNomad
    if version('PyNomadBBO') != '4.5.1':
        raise ValueError('This protocol requires PyNomadBBO 4.5.1')
    directory = Path(directory)
    x0 = project_bounds(oracle.x0, oracle.bl, oracle.bu)
    f, g = oracle.eval(x0)
    np.testing.assert_allclose(f, task['f0'], rtol=1e-9, atol=1e-10)
    oracle.reset_count()
    raw, journal = directory/'trace.csv', directory/'journal.csv'
    scored, full = directory/'scored_incumbent.json', directory/'incumbent.json'
    recorder = RunRecorder(task['problem'], 'nomad_default', task['seed'], task['eps_feas'],
                           record_every=task['record_every'])
    params = [f'DIMENSION {oracle.n}', 'BB_OUTPUT_TYPE OBJ'+' PB'*oracle.num_ineq,
              f'MAX_BB_EVAL {task["query_budget"]}', 'DISPLAY_DEGREE 0',
              f'MAX_TIME {task["solver_max_time_seconds"]}', f'SEED {task["seed"]}']
    lower, upper = finite_bound_list(oracle.bl, -1), finite_bound_list(oracle.bu, 1)
    saved_best, callback_error = None, None
    certificate = dict(seed=task['seed'], parameters=params, x0=x0.tolist(), lower_bounds=lower,
                       upper_bounds=upper, initial_f=float(f), initial_violation=float(violation(g)),
                       expected_initial_rng_state=task['expected_initial_rng_state'],
                       external_set_seed_called=False)

    def callback(point):
        nonlocal saved_best, callback_error
        # Each optimizer invocation starts in a fresh process; SEED controls its state.
        if oracle.query_count == 0:
            state = PyNomad.getRNGState()
            certificate['observed_initial_rng_state'] = state
            certificate['seed_state_verified'] = state == task['expected_initial_rng_state']
            write_json(directory/'certificate.json', certificate)
            if not certificate['seed_state_verified']:
                callback_error = 'RNG state does not match the requested explicit seed'
                return 0
        x = np.array([point.get_coord(i) for i in range(point.size())], dtype=float)
        f_norm, g_norm = oracle.eval(x)
        previous = len(recorder.rows)
        recorder.record(query=oracle.query_count, iteration=-1, f_norm=f_norm, g_norm=g_norm,
                        x=x, status='nomad_eval')
        if len(recorder.rows) > previous:
            row = recorder.rows[-1]
            with journal.open('a', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if previous == 0:
                    writer.writeheader()
                writer.writerow(row)
            best = (row['best_feasible_f'], row['best_violation'])
            if row['query'] <= task['query_budget'] and row['elapsed_seconds'] <= task['scoring_time_seconds'] and best != saved_best:
                write_json(scored, dict(row, best_x=None if recorder.best_x is None else recorder.best_x.tolist(),
                                       best_violation_x=None if recorder.best_violation_x is None else recorder.best_violation_x.tolist()))
                saved_best = best
        if task['flush_every'] > 0 and oracle.query_count % task['flush_every'] == 0:
            recorder.to_csv(raw)
            recorder.save_incumbent(full)
        point.setBBO(' '.join(f'{v:.17g}' for v in [f_norm, *g_norm]).encode('UTF-8'))
        return 1

    result = PyNomad.optimize(callback, x0.tolist(), lower, upper, params)
    if callback_error:
        raise RuntimeError(callback_error)
    if not recorder.rows:
        raise RuntimeError('NOMAD returned without a saved evaluation')
    recorder.rows[-1]['status'] = f'nomad_done:{result.get("run_flag")}'
    recorder.to_csv(raw)
    recorder.save_incumbent(full)
    if result.get('run_flag') not in (1, 0, -1, -2, -4):
        raise RuntimeError('Unexpected NOMAD termination: '+str(result))
    return dict(status='completed', actual_queries=oracle.query_count, nomad_result=result)
