"""Run frozen problem/solver/seed combinations in isolated processes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
from functools import partial
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .parameters import select_parameters
from .recording import BudgetRecorder, SampledRunRecorder, captured_recorder_class, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def tasks_for(root, problems=None, solvers=None, seeds=None):
    params = json.loads((root/'config/parameters.json').read_text())
    protocols = json.loads((root/'config/protocol.json').read_text())
    states = json.loads((root/'config/nomad_rng_states.json').read_text())
    refs = {r['problem']:r for r in read_rows(root/'config/references.csv')}
    metadata = {r['problem']:r for r in read_rows(root/'config/problems.csv')}
    if problems and not set(problems) <= set(params):
        raise ValueError('Unknown problem name')
    available = set(next(iter(params.values())))
    if solvers and not set(solvers) <= available:
        raise ValueError('Unknown solver name')
    if seeds and not set(seeds) <= set(range(1, 11)):
        raise ValueError('Seeds must be in 1--10')
    tasks = []
    for problem in metadata:
        if problems and problem not in problems:
            continue
        for solver, values in params[problem].items():
            if solvers and solver not in solvers:
                continue
            protocol = protocols[problem][solver]
            expected = select_parameters(solver, int(metadata[problem]['n']), protocol['query_budget'])
            if values != expected:
                raise ValueError('Resolved parameters differ from the uniform rule')
            for seed in protocol['seeds']:
                if seeds and seed not in seeds:
                    continue
                tasks.append(dict(problem=problem, solver=solver, seed=seed, params=values,
                                  n=int(metadata[problem]['n']), m_ineq=int(metadata[problem]['m_ineq']),
                                  f0=float(refs[problem]['f0']), **protocol,
                                  expected_initial_rng_state=states[str(seed)]))
    if not tasks:
        raise ValueError('The requested filters select no runs')
    return tasks


def worker(payload):
    task = json.loads(Path(payload).read_text())
    directory = Path(payload).parent
    from .oracle_cutest import CutestOracle
    oracle = CutestOracle(task['problem'])
    if (oracle.n, oracle.num_ineq) != (task['n'], task['m_ineq']):
        raise ValueError('CUTEst dimensions differ from the frozen problem specification')
    if task['solver'] == 'nomad_default':
        from .solvers.nomad import run
        if task['problem'] == 'CRESC4':
            from .validation import capture_cresc4_queries
            capture_cresc4_queries(oracle, directory)
        result = run(oracle, task, directory)
    elif task['solver'] in ('zoagp', 'zominmax', 'szo_conex'):
        if (task['solver_max_time_seconds'], task['outer_timeout_seconds'],
                task['recording'], task['record_every'], task['eps_feas']) != (1800, 1920, 'terminal', 50, 1e-4):
            raise ValueError('Reference baseline protocol differs from the retained runs')
        solver = importlib.import_module('.solvers.reference.'+task['solver'], package=__package__)
        result, _ = solver.run(oracle, task['params'], task['query_budget'], task['seed'],
            time_cap=task['solver_max_time_seconds'], out=directory, record_every=task['record_every'])
    else:
        from .solvers import common
        instances = []
        if task['recording'] == 'sampled':
            common.RunRecorder = SampledRunRecorder
        elif task['solver'].startswith('zob_'):
            common.RunRecorder = partial(captured_recorder_class(BudgetRecorder, instances),
                budget=task['query_budget'], time_cap=task['solver_max_time_seconds'])
        params = dict(task['params'], max_time_seconds=task['solver_max_time_seconds'],
                      flush_every=task['flush_every'])
        solver = importlib.import_module('.solvers.'+task['solver'], package=__package__)
        error = None
        try:
            recorder = solver.run(oracle, params, task['query_budget'], task['seed'], task['eps_feas'],
                                  out_dir=directory, record_every=task['record_every'])
        except (ArithmeticError, ValueError) as exception:
            if not instances or not instances[0].rows:
                raise
            recorder, error = instances[0], repr(exception)
            recorder.mark_status('solver_exception')
        recorder.to_csv(directory/'trace.csv')
        recorder.save_incumbent(directory/'incumbent.json')
        result = dict(status=recorder.rows[-1]['status'], actual_queries=oracle.query_count,
                      error=error, elapsed_seconds=time.perf_counter()-recorder.start)
    write_json(directory/'worker_result.json', result)


def run_one(task, output):
    directory = output/'traces'/task['solver']/task['problem']/f'seed{task["seed"]}'
    if directory.exists():
        completion = directory/'completion.json'
        if not completion.exists():
            raise ValueError('Incomplete attempt requires review: '+str(directory))
        saved = json.loads(completion.read_text())
        if json.loads((directory/'task.json').read_text()) != task:
            raise ValueError('Refusing resume with changed parameters')
        for relative, expected in saved['output_hashes'].items():
            if sha(directory/relative) != expected:
                raise ValueError('Saved run checksum changed: '+str(directory/relative))
        return saved
    directory.mkdir(parents=True)
    payload = directory/'task.json'
    write_json(payload, task)
    with (directory/'stdout.txt').open('w') as stdout, (directory/'stderr.txt').open('w') as stderr:
        try:
            result = subprocess.run([sys.executable, '-m', 'cutest_benchmark.runner', '--worker', str(payload)],
                                    stdout=stdout, stderr=stderr, timeout=task['outer_timeout_seconds'], check=False)
            if result.returncode != 0 or not (directory/'worker_result.json').exists():
                raise RuntimeError('Worker failed; inspect '+str(directory/'stderr.txt'))
            status = json.loads((directory/'worker_result.json').read_text())['status']
        except subprocess.TimeoutExpired:
            from .logging_utils import run_log_path
            if task['solver'] == 'nomad_default':
                fallback = directory/'journal.csv'
            elif task['solver'] in ('zoagp', 'zominmax', 'szo_conex'):
                fallback = directory/'trace.csv'
            else:
                fallback = run_log_path(directory, task['problem'], task['solver'], task['seed'])
            if not fallback.exists():
                raise RuntimeError('Timeout without recoverable observations: '+str(directory))
            rows = read_rows(fallback)
            if not rows:
                raise RuntimeError('Timeout with an empty trace: '+str(directory))
            with (directory/'trace.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[-1]))
                writer.writeheader()
                writer.writerows(rows)
                writer.writerow(dict(rows[-1], status='hard_timeout'))
            status = 'hard_timeout'
    raw = directory/'trace.csv'
    if not raw.exists() or not read_rows(raw):
        raise RuntimeError('Missing final trace: '+str(directory))
    if task['problem'] == 'CRESC4' and task['solver'] == 'nomad_default':
        from .validation import correct_recorded_cresc4
        correct_recorded_cresc4(directory)
    record = dict(problem=task['problem'], solver=task['solver'], seed=task['seed'], status=status,
                  path=str(raw.relative_to(output)), csv_sha256=sha(raw), file_sha256=sha(raw),
                  output_hashes={p.name:sha(p) for p in directory.iterdir() if p.is_file()})
    write_json(directory/'completion.json', record)
    print(f'{task["problem"]}/{task["solver"]}/{task["seed"]}: {status}', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path, default=Path('runs'))
    parser.add_argument('--problems', nargs='+')
    parser.add_argument('--solvers', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--resource-workers', type=int, default=6)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    root, output = args.root.resolve(), args.out.resolve()
    if output == root or output.is_relative_to(root/'results') or output.is_relative_to(root/'config') or output.is_relative_to(root/'src'):
        parser.error('New runs must not overwrite release files')
    if not 1 <= args.resource_workers <= args.workers:
        parser.error('Require 1 <= resource-workers <= workers')
    tasks = tasks_for(root, args.problems, args.solvers, args.seeds)
    print(f'Selected {len(tasks)} runs. Output: {output}', flush=True)
    if not args.execute:
        print('Add --execute to run the solvers.')
        return
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        if os.environ.get(variable) != '1':
            parser.error('Source scripts/activate.sh to set single-threaded numerical libraries')
    from importlib.metadata import version
    manifest = dict(tasks=tasks, python=platform.python_version(), architecture=platform.machine(),
                    packages={p:version(p) for p in ('numpy','scipy','pandas','pycutest','PyNomadBBO')},
                    source_hashes={str(p.relative_to(root)):sha(p) for parent in ('src','config')
                                   for p in (root/parent).rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = output/'manifest.json'
        if saved.exists() and json.loads(saved.read_text()) != manifest:
            raise ValueError('Resume refused: inputs, selection, source, or environment changed')
        write_json(saved, manifest)
        records = []
        for intensive, workers in ((False,args.workers), (True,args.resource_workers)):
            group = [task for task in tasks if (task['n'] > 30) == intensive]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                records.extend(pool.map(lambda task:run_one(task, output), group))
        fields = ('problem','solver','seed','path','csv_sha256','file_sha256','status')
        with (output/'trace_index.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)
        print('Finished. A complete 1530-run selection can be analyzed with --data '+str(output))


if __name__ == '__main__':
    main()
