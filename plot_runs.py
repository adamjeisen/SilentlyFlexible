import matplotlib.pyplot as plt
import pandas as pd

errorbar_kwargs = {'linewidth':1, 'fmt':'o-', 'markersize': 2, 'capsize': 1}
def _ax_fmt(axs):
    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

plt.rcParams.update({'font.size': 8})


def plot_perf_over_n_rand(run_stats):
    """
    Plots percent maintained, percent spurious and recall over size of random network
    :param run_stats: Dataframe with run statistics
    :return:
    """
    load_group = run_stats.groupby('n_rand')
    percent_maintained = load_group['percent_maintained']
    percent_spurious = load_group['percent_spurious']
    avg_mu_error = load_group['avg_mu_error']
    f = plt.figure(figsize=(2, 4), )
    ax1 = f.add_subplot(2, 1, 1)
    ax1.set_ylim(-.1, 1.1)
    ax1.set_ylabel('Percent maintained/spurious')
    plt.errorbar(x=percent_maintained.groups.keys(), y=percent_maintained.mean(),
                 c='b', label='maintained', **errorbar_kwargs)
    plt.errorbar(x=percent_spurious.groups.keys(), y=percent_spurious.mean(), c='r',
                 label='spurious',  **errorbar_kwargs)
    ax1.legend(bbox_to_anchor=(0.1, 0.6), loc='center left',prop={'size': 6})
    ax2 = f.add_subplot(2, 1, 2,sharex=ax1)
    ax2.set_ylabel('Recall SD')
    ax2.set_xlabel('$N_{rand}$')
    ax2.errorbar(x=list(avg_mu_error.groups.keys()), y=avg_mu_error.mean(), yerr=avg_mu_error.std(), c='r',
                 **errorbar_kwargs)
    _ax_fmt([ax1, ax2])
    plt.tight_layout(pad=0.4)
    plt.show()


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
    f = plt.figure(figsize=(2, 4), )
    ax1 = f.add_subplot(2, 1, 1)
    ax1.set_ylim(-.1, 1.1)
    ax1.set_ylabel('Percent maintained/spurious')
    plt.errorbar(x=percent_maintained.groups.keys(), y=percent_maintained.mean(),
                 c='b', label='maintained', **errorbar_kwargs)
    plt.errorbar(x=percent_spurious.groups.keys(), y=percent_spurious.mean(), c='r',
                 label='spurious',  **errorbar_kwargs)
    ax1.legend(bbox_to_anchor=(0.1, 0.6), loc='center left',prop={'size': 6})
    ax2 = f.add_subplot(2, 1, 2,sharex=ax1)
    ax2.set_ylabel('Recall SD')
    ax2.set_xlabel('Load')
    ax2.errorbar(x=list(avg_mu_error.groups.keys()), y=avg_mu_error.mean(), yerr=avg_mu_error.std(), c='r',
                 **errorbar_kwargs)
    _ax_fmt([ax1, ax2])
    plt.tight_layout(pad=0.4)

    plt.show()

