import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import pearsonr

from analysis import _get_target_sensory_neurons, _get_target_score
from utils import load

def plot_overlaid(run_results, simulation, sens_idx=0, save_path=None, **kwargs):
    f, ax = plt.subplots()
    n_sensory = simulation.N_sensory
    n_sensory_nets = simulation.N_sensory_nets
    x_ff = run_results['x_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    x_ff = x_ff.reshape(x_ff.shape[0], n_sensory_nets, n_sensory)
    u_ff = run_results['u_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    u_ff = u_ff.reshape(u_ff.shape[0], n_sensory_nets, n_sensory)
    target_neurons, non_target_neurons = _get_target_sensory_neurons(run_results, sens_idx=sens_idx)
    target_u_ff = u_ff[1:, sens_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_u_ff = u_ff[1:, sens_idx, non_target_neurons].mean(1)

    target_x_ff = x_ff[1:, sens_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_x_ff = x_ff[1:, sens_idx, non_target_neurons].mean(1)
    spike_times, spike_locations = np.where(run_results['p_sens'][:, sens_idx, :])
    ax.scatter(spike_times/simulation.sens_nets[0].dt, spike_locations, c='black', s=2, label='spikes')
    ax.invert_yaxis()
    time = np.arange(len(target_x_ff))/simulation.sens_nets[0].dt
    ax2 = ax.twinx()
    ax2.plot(time, target_x_ff, c='blue', label='target $x^{FF}$')
    ax2.plot(time, target_u_ff, c='green', label='target $u^{FF}$')
    ax2.plot(time, target_x_ff * target_u_ff, c='magenta', label='target $u^{FF}x^{FF}$')
    ax.axvspan(100, 300, color='r', alpha=0.2)
    ax.axvspan(600, 650, color='r', alpha=0.2)
    plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Sensory Neurons')
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def plot_synaptic(run_results, simulation, sens_idx=0, save_path=None, **kwargs):
    n_sensory = simulation.N_sensory
    n_sensory_nets = simulation.N_sensory_nets
    x_ff = run_results['x_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    x_ff = x_ff.reshape(x_ff.shape[0], n_sensory_nets, n_sensory)
    u_ff = run_results['u_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    u_ff = u_ff.reshape(u_ff.shape[0], n_sensory_nets, n_sensory)
    target_neurons, non_target_neurons = _get_target_sensory_neurons(run_results, sens_idx=sens_idx)
    target_u_ff = u_ff[1:, sens_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_u_ff = u_ff[1:, sens_idx, non_target_neurons].mean(1)

    target_x_ff = x_ff[1:, sens_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_x_ff = x_ff[1:, sens_idx, non_target_neurons].mean(1)

    correl_results = _get_target_score(run_results, simulation, sens_idx=sens_idx)
    correl_u = correl_results['correl_u']
    correl_x = correl_results['correl_x']
    correl_ux = correl_results['correl_ux']
    correl_p = correl_results['correl_p']
    f, axs = plt.subplots(2, 1)
    axs[0].plot(target_x_ff * target_u_ff, c='magenta', label='target $u^{FF}x^{FF}$')
    axs[0].plot(non_target_x_ff * non_target_u_ff, c='violet', label='non-target $u^{FF}x^{FF}$')
    axs[0].plot(target_x_ff, c='blue', label='target $x^{FF}$')
    axs[0].plot(non_target_x_ff, c='lightskyblue', label='non-target $x^{FF}$')
    axs[0].plot(target_u_ff, c='green', label='target $u^{FF}$')
    axs[0].plot(non_target_u_ff, c='limegreen', label='non-target $u^{FF}$')
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_title(f'Sensory Network {sens_idx} Average Synaptic Values - feedforward', fontsize=11)
    axs[1].plot(correl_u, label='correlation with $u$')
    axs[1].plot(correl_x, label='correlation with $x$')
    axs[1].plot(correl_ux, label='correlation with $ux$')
    axs[1].plot(correl_p, label='correlation with $p_{rand}'
                                '$')
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[1].set_title(f'Sensory Network {sens_idx} Target Score Correlations - feedback', fontsize=11)
    axs[1].set_xlabel('Time (ms)')
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def plot_ux_avg(run_results, simulation, sens_idx=0, **kwargs):
    n_sensory = simulation.N_sensory
    n_sensory_nets = simulation.N_sensory_nets
    x_ff = run_results['x_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    x_ff = x_ff.reshape(x_ff.shape[0], n_sensory_nets, n_sensory)
    u_ff = run_results['u_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    u_ff = u_ff.reshape(u_ff.shape[0], n_sensory_nets, n_sensory)
    target_neurons, non_target_neurons = _get_target_sensory_neurons(run_results, sens_idx=sens_idx)
    target_u_ff = u_ff[1:, sens_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_u_ff = u_ff[1:, sens_idx, non_target_neurons].mean(1)

    target_x_ff = x_ff[1:, sens_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_x_ff = x_ff[1:, sens_idx, non_target_neurons].mean(1)

    x_fb = run_results['x_fb']  # shape should be t x n_rand
    u_fb = run_results['u_fb'] # shape should be t x n_rand
    gaussian = np.exp(
        (-1 / (2 * simulation.sigma ** 2)) * (np.arange(len(target_neurons)) - int(len(target_neurons) / 2)) ** 2)
    targeted_score = ((simulation.W_ff > 0).astype(int) -
                      (simulation.W_ff < 0).astype(int))[:, target_neurons] @ gaussian
    correl = np.array([pearsonr(u_fb[i, :]*x_fb[i, :], targeted_score)[0] if not
                        np.array_equal(np.unique(np.diff(u_fb[i, :])), [0]) else 0 for i in range(1, u_fb.shape[0])])


    f, axs = plt.subplots(2, 1)
    axs[0].plot(target_x_ff*target_u_ff, label='target $x^{FF}u^{FF}$')
    axs[0].plot(non_target_x_ff*non_target_u_ff, label='non-target $x^{FF}u^{FF}$')
    axs[0].legend()
    axs[0].set_title('Average $ux$ - feedforward')
    axs[1].plot(correl)
    axs[1].set_title('Target Score Correlation with $ux$ - feedback')
    plt.tight_layout()
    plt.show()

def plot_x_avg(run_results, simulation, sens_idx=0, **kwargs):
    n_sensory = simulation.N_sensory
    n_sensory_nets = simulation.N_sensory_nets
    x_ff = run_results['x_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    x_ff = x_ff.reshape(x_ff.shape[0], n_sensory_nets, n_sensory)
    x_fb = run_results['x_fb'] # shape should be t x n_rand
    target_neurons, non_target_neurons = _get_target_sensory_neurons(run_results, sens_idx=sens_idx)
    target_x_ff = x_ff[1:, sens_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_x_ff = x_ff[1:, sens_idx, non_target_neurons].mean(1)
    gaussian = np.exp(
        (-1 / (2 * simulation.sigma ** 2)) * (np.arange(len(target_neurons)) - int(len(target_neurons) / 2)) ** 2)
    targeted_score = ((simulation.W_ff > 0).astype(int) -
                      (simulation.W_ff < 0).astype(int))[:, target_neurons] @ gaussian
    correl = np.array([pearsonr(x_fb[i, :], targeted_score)[0] if not
                        np.array_equal(np.unique(np.diff(x_fb[i, :])), [0]) else 0 for i in range(1, x_fb.shape[0])])
    f, axs = plt.subplots(2, 1)
    axs[0].plot(target_x_ff, label='target $x^{FF}$')
    axs[0].plot(non_target_x_ff, label='non-target $x^{FF}$')
    axs[0].legend()
    axs[0].set_title('Average $x$ - feedforward')
    axs[1].plot(correl)
    axs[1].set_title('Targeted Score Correlation with $x$ - feedback')
    plt.tight_layout()
    plt.show()


def plot_u_avg(run_results, simulation, sens_idx=0, **kwargs):
    n_sensory = simulation.N_sensory
    n_sensory_nets = simulation.N_sensory_nets
    u_ff = run_results['u_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    u_ff = u_ff.reshape(u_ff.shape[0], n_sensory_nets, n_sensory) # shape should be t x n_rand
    u_fb = run_results['u_fb'] # shape should be t x n_rand
    target_neurons, non_target_neurons = _get_target_sensory_neurons(run_results, sens_idx=sens_idx)
    target_u_ff = u_ff[1:, sens_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_u_ff = u_ff[1:, sens_idx, non_target_neurons].mean(1)
    gaussian = np.exp(
        (-1 / (2 * simulation.sigma ** 2)) * (np.arange(len(target_neurons)) - int(len(target_neurons) / 2)) ** 2)
    targeted_score = ((simulation.W_ff > 0).astype(int) -
                      (simulation.W_ff < 0).astype(int))[:, target_neurons] @ gaussian
    correl = np.array([pearsonr(u_fb[i, :], targeted_score)[0] if not
                        np.array_equal(np.unique(np.diff(u_fb[i, :])), [0]) else 0 for i in range(1, u_fb.shape[0])])
    f, axs = plt.subplots(2, 1)
    axs[0].plot(target_u_ff, label='target $u^{FF}$')
    axs[0].plot(non_target_u_ff, label='non-target $u^{FF}$')
    axs[0].legend()
    axs[0].set_title('Average $u$ - feedforward')
    axs[1].plot(correl)
    axs[1].set_title('Targeted Score Correlation with $u$ - feedback')
    plt.tight_layout()
    plt.show()

# def plot_s_sens_avg(run_results, **kwargs):
#     s_sens = run_results['s_sens']
#     mus = run_results['mus']
#     sigma = run_results['sigma']
#     n_sensory = run_results['n_sensory']
#     for mu_idx, mu in enumerate(mus):
#         target_neurons, non_target_neurons = _get_target_sensory_neurons(run_results, mu_idx)
#         target_activity = s_sens[:, mu_idx, target_neurons].mean(axis=1)
#         plt.plot(target_activity, label='target population')
#
#         non_target_activity = s_sens[:, mu_idx, non_target_neurons].mean(axis=1)
#         plt.plot(non_target_activity)
#     plt.legend()
#     plt.xlabel('Time')
#     plt.ylabel('$s_i$')
#     plt.show()


def plot_r_sens(run_results, **kwargs):
    _plot_sens(key='r_sens', run_results=run_results, title='Firing Rates in Sensory Network', **kwargs)


def plot_s_sens(run_results, **kwargs):
    _plot_sens(key='s_sens', run_results=run_results, title='Synaptic Activity in Sensory Network', **kwargs)


def plot_p_sens(run_results, **kwargs):
    _plot_sens(key='p_sens', run_results=run_results, title='Spiking in Sensory Network', **kwargs)


def plot_r_rand(run_results, **kwargs):
    _plot_rand(key='r_rand', run_results=run_results, title='Firing Rates in Random Network', **kwargs)


def plot_s_rand(run_results, **kwargs):
    _plot_rand(key='s_rand', run_results=run_results, title='Synaptic Activity in Random Network', **kwargs)


def plot_p_rand(run_results, **kwargs):
    _plot_rand(key='p_rand', run_results=run_results, title='Spiking in Random Network', **kwargs)


def plot_s_ext(run_results, **kwargs):
    _plot_sens(key='s_ext', run_results=run_results, title='External Input in Sensory Network', **kwargs)

def plot_s_ext_rand(run_results, **kwargs):
    _plot_rand(key='s_ext_rand', run_results=run_results, title='External Input in Random Network', **kwargs)


def _plot_rand(key, run_results, title='', save_path=None):
    fig = plt.figure()
    if key == 'p_rand':
        spike_times, spike_locations = np.where(run_results[key])
        plt.scatter(spike_times, spike_locations, c='black', s=2)
        plt.gca().invert_yaxis()
    else:
        plt.imshow(run_results[key].T, aspect='auto')
        plt.colorbar()
    plt.xlabel('Time Steps')
    plt.ylabel('Neuron #')
    plt.title(title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close(fig)


def _plot_sens(key, run_results, sens_idx=0, title='', save_path=None):
    fig = plt.figure()
    if key == 'p_sens':
        spike_times, spike_locations = np.where(run_results[key][:, sens_idx, :])
        plt.scatter(spike_times, spike_locations, c='black', s=2)
        plt.gca().invert_yaxis()
    else:
        plt.imshow(run_results[key][:, sens_idx, :].T, aspect='auto')
        plt.colorbar()
    plt.xlabel('Time Steps')
    plt.ylabel('Neuron #')
    if title:
        title += f' {sens_idx}'
    plt.title(title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close(fig)

def save_all_plots(run_stats_df, loaded_only=True):
    plot_dir = os.path.join(os.path.dirname(run_stats_df.path.iloc[0]), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    for i, row in run_stats_df.iterrows():
        save_dir = os.path.join(plot_dir, os.path.basename(row.path))
        os.makedirs(save_dir, exist_ok=True)
        data = load(row.path)
        run_results = data['run_results']
        simulation = data['simulation']

        # sensory network plots
        for sens_idx in range(simulation.N_sensory_nets):
            if (loaded_only and simulation.mus[sens_idx] is not None) or (not loaded_only):
                    plot_r_sens(run_results, sens_idx=sens_idx, save_path=os.path.join(save_dir, f'sens_net_{sens_idx}_r.png'))
                    plot_s_sens(run_results, sens_idx=sens_idx, save_path=os.path.join(save_dir, f'sens_net_{sens_idx}_s.png'))
                    plot_p_sens(run_results, sens_idx=sens_idx, save_path=os.path.join(save_dir, f'sens_net_{sens_idx}_p.png'))
                    plot_s_ext(run_results, sens_idx=sens_idx, save_path=os.path.join(save_dir, f'sens_net_{sens_idx}_s_ext.png'))
            if simulation.mus[sens_idx] is not None:
                plot_synaptic(run_results, simulation, sens_idx=sens_idx, save_path=os.path.join(save_dir, f'sens_net_{sens_idx}_synaptic.png'))

        # random network plots
        plot_r_rand(run_results, save_path=os.path.join(save_dir, f'rand_net_r.png'))
        plot_s_rand(run_results, save_path=os.path.join(save_dir, f'rand_net_s.png'))
        plot_p_rand(run_results, save_path=os.path.join(save_dir, f'rand_net_p.png'))
        plot_s_ext_rand(run_results, save_path=os.path.join(save_dir, f'rand_net_s_ext.png'))


# if __name__ == '__main__':
#     from simulation import Simulation
#
#     sim = Simulation(T=1000, N_sensory_nets=1, N_sensory=512, N_rand=1024, amp_ext=500)
#     sim.reset()
#     run_results = sim.run()
#     plot_s_sens_avg(run_results)
