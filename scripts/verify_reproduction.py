"""Compare regenerated statistics, table, and figure with the retained results."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cutest_benchmark.analysis import read


KEYS = {
    'run_metrics': ['problem', 'solver', 'seed'],
    'hitting_times': ['problem', 'solver', 'seed', 'tau'],
    'per_seed_counts': ['solver', 'seed'],
    'summary': ['solver', 'metric'],
    'profiles': ['solver', 'tau', 'normalized_queries'],
    'per_seed_profile_areas': ['solver', 'seed', 'tau'],
    'profile_areas': ['solver', 'tau'],
    'per_problem_rates': ['solver', 'problem'],
    'summary_table': ['solver'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, default=Path('reproduced'))
    args = parser.parse_args()
    retained = args.root/'results'
    for name, keys in KEYS.items():
        left, right = read(retained/(name+'.csv')), read(args.out/(name+'.csv'))
        pd.testing.assert_frame_equal(left.sort_values(keys).reset_index(drop=True),
            right[left.columns].sort_values(keys).reset_index(drop=True), check_exact=True)
    for name in ('summary_table.tex', 'per_run.md'):
        if (retained/name).read_bytes() != (args.out/name).read_bytes():
            raise ValueError('Text differs: '+name)
    png = Path('figures/data_profiles_combined.png')
    np.testing.assert_array_equal(np.asarray(Image.open(retained/png)),
                                  np.asarray(Image.open(args.out/png)))
    print('All nine result tables, LaTeX, per-run page, and PNG pixels match exactly.')


if __name__ == '__main__':
    main()
