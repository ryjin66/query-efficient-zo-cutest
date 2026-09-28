"""Decode problem definitions and check dimensions and initial values."""
import argparse
import csv
from importlib.metadata import version
from pathlib import Path

import numpy as np
from cutest_benchmark.oracle_cutest import CutestOracle
from cutest_benchmark.fd import project_bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    for package, expected in {'numpy':'2.2.6','scipy':'1.15.2','pandas':'2.3.3','pycutest':'1.8.2','PyNomadBBO':'4.5.1'}.items():
        if version(package) != expected:
            raise ValueError(f'{package}: expected {expected}, found {version(package)}')
    with (args.root/'config/references.csv').open() as stream:
        refs = {r['problem']:float(r['f0']) for r in csv.DictReader(stream)}
    with (args.root/'config/problems.csv').open() as stream:
        metadata = list(csv.DictReader(stream))
    for row in metadata:
        oracle = CutestOracle(row['problem'])
        if (oracle.n, oracle.metadata.m_raw, oracle.num_ineq) != tuple(int(row[k]) for k in ('n','m_raw','m_ineq')):
            raise ValueError('Dimension mismatch: '+row['problem'])
        f, _ = oracle.eval(project_bounds(oracle.x0, oracle.bl, oracle.bu))
        np.testing.assert_allclose(f, refs[row['problem']], rtol=1e-9, atol=1e-10)
        print(f'{row["problem"]}: n={oracle.n}, m={oracle.num_ineq}, initial value verified', flush=True)


if __name__ == '__main__':
    main()
