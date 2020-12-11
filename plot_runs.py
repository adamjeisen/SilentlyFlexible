import matplotlib.pyplot as plt
import pandas as pd

errorbar_kwargs = {'linewidth':1, 'fmt':'o-'}
def plot_perf_over_load(run_stats):
    """
    Plots percent maintained, percent spurious and recall over load
    :param run_stats: Dataframe with run statistics
    :return:
    """
    load_group = run_stats.groupby('load')
    percent_maintained = load_group['percent_maintained']
    percent_spurious = load_group['percent_spurious']
    avg_mu_error = load_group['avg_mu_error']
    f = plt.figure(dpi=200,)
    ax = f.add_subplot(1, 2, 1)
    ax.set_ylim(-.1, 1.1)
    ax.set_xlabel('Load')
    ax.set_ylabel('Percent maintained')
    plt.errorbar(x=percent_maintained.groups.keys(), y=percent_maintained.mean(), yerr=percent_maintained.std(),
                 c='b', **errorbar_kwargs)
    ax = ax.twiny()
    ax.set_ylabel('Percent spurious')
    plt.errorbar(x=percent_spurious.groups.keys(), y=percent_spurious.mean(), yerr=percent_spurious.std(), c='r', **errorbar_kwargs)
    ax = f.add_subplot(1, 2, 2)
    ax.set_ylabel('Recall SD')

    plt.errorbar(x=avg_mu_error.groups.keys(), y=avg_mu_error.mean(), yerr=avg_mu_error.std(), c='r', **errorbar_kwargs)
    plt.tight_layout()
    plt.show()

