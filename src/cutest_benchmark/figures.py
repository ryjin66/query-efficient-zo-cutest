"""Publication data profiles with pointwise seed-level confidence bands."""
import os
import tempfile
from pathlib import Path

import numpy as np

PLOT_ORDER = ('zob_sgda', 'zob_gda', 'zominmax', 'zoagp', 'nomad_default', 'szo_conex')
LABELS = dict(zip(PLOT_ORDER, ('ZOB-SGDA', 'ZOB-GDA', 'ZO-MinMax', 'ZO-AGP', 'NOMAD', 'SZO-ConEx')))
STYLES = {
    'zob_sgda': dict(color='#0072B2', linestyle='-', linewidth=2.6),
    'zob_gda': dict(color='#D55E00', linestyle='-', linewidth=2.6),
    'nomad_default': dict(color='#000000', linestyle='--', linewidth=2.0),
    'zominmax': dict(color='#009E73', linestyle='-.', linewidth=2.0),
    'zoagp': dict(color='#CC79A7', linestyle=':', linewidth=2.2),
    'szo_conex': dict(color='#E69F00', linestyle=(0, (4, 2, 1, 2)), linewidth=2.0),
}


def render(profiles, output, maximum):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'cutest-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif', 'font.size':10, 'axes.labelsize':10,
                         'axes.titlesize':10, 'legend.fontsize':8.5, 'xtick.labelsize':9,
                         'ytick.labelsize':9, 'axes.linewidth':.8, 'pdf.fonttype':42,
                         'ps.fonttype':42, 'savefig.dpi':300})
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05), sharey=True)
    handles, labels = [], []
    for j, tau in enumerate((.1, .001)):
        ax = axes[j]
        for solver in PLOT_ORDER:
            data = profiles[profiles.solver.eq(solver) & profiles.tau.eq(tau)
                            & profiles.normalized_queries.ge(1)].sort_values('normalized_queries')
            x = data.normalized_queries.to_numpy()
            if solver != 'zoagp':
                ax.fill_between(x, np.maximum(0, data.ci_low), np.minimum(1, data.ci_high),
                                step='post', color=STYLES[solver]['color'], alpha=.10, linewidth=0, zorder=1)
            line = ax.step(x, data['mean'], where='post', label=LABELS[solver], zorder=3, **STYLES[solver])[0]
            if j == 0:
                handles.append(line)
                labels.append('ZOAGP' if solver == 'zoagp' else LABELS[solver])
        power = 1 if j == 0 else 3
        ax.set_title(f'({chr(97+j)}) Data profile, '+r'$\tau=$'+rf'$10^{{-{power}}}$')
        ax.set_xlabel(r'Number of queries / $(d_x+1)$')
        ax.set_xscale('log')
        ax.set_xlim(1, maximum)
        ax.set_ylim(-.02, 1.02)
        ax.grid(True, which='major', alpha=.28, linewidth=.65)
        ax.grid(True, which='minor', alpha=.11, linewidth=.45)
    axes[0].set_ylabel('Mean fraction solved')
    legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -.03), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        if legend.get_window_extent(renderer).overlaps(ax.get_tightbbox(renderer)):
            raise ValueError('Legend overlaps the plot labels')
    destination = Path(output)/'figures'
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'png'):
        fig.savefig(destination/('data_profiles_combined.'+suffix), bbox_inches='tight')
    plt.close(fig)
