"""Publication table from seed-level counts and normalized log-AUCs."""
import numpy as np
import pandas as pd

from .figures import LABELS

ORDER = ('zob_sgda', 'zob_gda', 'nomad_default', 'zoagp', 'zominmax', 'szo_conex')
TARGETS = (.1, .001)


def table(summary, areas, maximum, sensitivity=False):
    data = summary.copy()
    area_rows = areas.loc[areas.tau.isin(TARGETS)].copy()
    area_rows['metric'] = area_rows.tau.map({.1: 'auc_1e_1', .001: 'auc_1e_3'})
    data = pd.concat([data, area_rows], ignore_index=True)
    columns = ('feasible_count', 'feasible_fraction', 'solved_1e_1', 'solved_1e_3', 'auc_1e_1', 'auc_1e_3')
    maxima = data.groupby('metric')['mean'].max()
    rows, csv_rows = [], []
    for solver in ORDER:
        cells = [LABELS[solver]]
        record = {'solver': solver, 'algorithm': LABELS[solver]}
        for metric in columns:
            stat = data.loc[data.solver.eq(solver) & data.metric.eq(metric)].iloc[0]
            digits = 4 if metric.startswith('auc_') else 3 if metric == 'feasible_fraction' else 2
            value = f"{stat['mean']:.{digits}f}"
            if np.isclose(stat['mean'], maxima[metric], rtol=0, atol=1e-12):
                value = r'\mathbf{' + value + '}'
            if metric != 'feasible_fraction' and stat.n_seeds > 1:
                value += rf' \pm {stat.ci_halfwidth:.{digits}f}'
            cells.append('$' + value + '$')
            record[metric] = stat['mean']
            record[metric + '_ci_halfwidth'] = stat.ci_halfwidth
        rows.append(' & '.join(cells) + r' \\')
        if solver == 'zob_gda':
            rows.append(r'\midrule')
        csv_rows.append(record)
    reference_note = (
        'This sensitivity analysis retains the unrecovered historical scalar references '
        'on HALDMADS and HAIFAL; the other references have verified saved points.'
        if sensitivity else
        'All references come from verified saved points. A separate sensitivity analysis '
        'retains unrecovered historical scalar references on HALDMADS and HAIFAL.')
    tex = r"""% Requires \usepackage{booktabs}.
\begin{table}[htbp]
\centering
\caption{Feasibility, optimality, and query efficiency for 30 CUTEst problems}
\label{tab:cutest_combined}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}
\toprule
Algorithm & \shortstack{Mean\\feasible} & \shortstack{Avg.\\feasible rate}
& \multicolumn{2}{c}{Mean problems solved} & \multicolumn{2}{c}{Normalized log-AUC} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
& & & $\tau=10^{-1}$ & $\tau=10^{-3}$ & $\tau=10^{-1}$ & $\tau=10^{-3}$ \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\vspace{0.5em}
\parbox{0.98\textwidth}{\footnotesize
Note: Mean feasible averages the number of problems with a feasible incumbent
(normalized violation $\le10^{-4}$) across seeds. Avg.\ feasible rate is this mean
divided by 30. Mean problems solved additionally requires
$\max\{0,f^{\mathrm{best}}-f^{\mathrm{ref}}\}/\max\{1,|f^0-f^{\mathrm{ref}}|\}\le\tau$.
Normalized log-AUC is the area under each seed's data profile with respect to
$\log q$, divided by $\log B$, over $1\le q\le B$, where
""" + rf'$q=Q/(d_x+1)$ and $B={maximum:,.2f}$.' + r"""
It lies in $[0,1]$; larger values indicate better query efficiency and coverage.
The $\pm$ values are 95\% Student-$t$ confidence half-widths across seeds 1--10,
computed separately from seed-level counts or AUCs.
ZO-AGP has one run per problem and no seed-based confidence interval.
Only observations within the original query budgets and 1,800 seconds are scored.
All algorithms share the same reference pool; problems without a finite reference
remain in the denominator but cannot count as solved.
""" + reference_note + r"""
CRESC4 NOMAD seeds 1 and 10 retain the high-precision post-hoc feasibility correction;
their original optimization feedback was not corrected.
Bold identifies the largest mean, not statistical significance.
}
\end{table}
"""
    return tex, pd.DataFrame(csv_rows)



