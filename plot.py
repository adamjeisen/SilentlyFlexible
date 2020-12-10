import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from analysis import _get_target_sensory_neurons, _get_target_score

def plot_overlaid(run_results, simulation, mu_idx=0, **kwargs):
    f, axs = plt.subplots(1, 1, )
    p_sens = run_results['p_sens'][:, mu_idx, :]
    n_sensory = simulation.N_sensory
    n_sensory_nets = simulation.N_sensory_nets
    x_ff = run_results['x_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    x_ff = x_ff.reshape(x_ff.shape[0], n_sensory_nets, n_sensory)
    u_ff = run_results['u_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    u_ff = u_ff.reshape(u_ff.shape[0], n_sensory_nets, n_sensory)
    target_neurons, non_target_neurons = _get_target_sensory_neurons(run_results, mu_idx=mu_idx)
    target_u_ff = u_ff[1:, mu_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_u_ff = u_ff[1:, mu_idx, non_target_neurons].mean(1)

    target_x_ff = x_ff[1:, mu_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_x_ff = x_ff[1:, mu_idx, non_target_neurons].mean(1)
    axs.imshow(p_sens.T, cmap='Greys', origin='left')
    axs.plot(512 * target_x_ff, c='blue', label='target $x^{FF}$')
    axs.plot(512 * target_u_ff, c='green', label='target $u^{FF}$')
    axs.plot(512 * target_x_ff * target_u_ff, c='magenta', label='target $u^{FF}x^{FF}$')
    axs.axvspan(100, 300, color='r', alpha=0.2)
    axs.axvspan(600, 650, color='r', alpha=0.2)
    axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_synaptic(run_results, simulation, mu_idx=0, **kwargs):
    n_sensory = simulation.N_sensory
    n_sensory_nets = simulation.N_sensory_nets
    x_ff = run_results['x_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    x_ff = x_ff.reshape(x_ff.shape[0], n_sensory_nets, n_sensory)
    u_ff = run_results['u_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    u_ff = u_ff.reshape(u_ff.shape[0], n_sensory_nets, n_sensory)
    target_neurons, non_target_neurons = _get_target_sensory_neurons(run_results, mu_idx=mu_idx)
    target_u_ff = u_ff[1:, mu_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_u_ff = u_ff[1:, mu_idx, non_target_neurons].mean(1)

    target_x_ff = x_ff[1:, mu_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_x_ff = x_ff[1:, mu_idx, non_target_neurons].mean(1)

    x_fb = run_results['x_fb']  # shape should be t x n_rand
    u_fb = run_results['u_fb']  # shape should be t x n_rand
    gaussian = np.exp(
        (-1 / (2 * simulation.sigma ** 2)) * (np.arange(len(target_neurons)) - int(len(target_neurons) / 2)) ** 2)
    correl_results = _get_target_score(run_results, simulation, mu_idx=0)
    correl_u = correl_results['correl_u']
    correl_x = correl_results['correl_x']
    correl_ux = correl_results['correl_ux']
    # correl_p = correl_results['correl_p']
    f, axs = plt.subplots(2, 1)
    axs[0].plot(target_x_ff * target_u_ff, c='magenta', label='target $u^{FF}x^{FF}$')
    axs[0].plot(non_target_x_ff * non_target_u_ff, c='violet', label='non-target $u^{FF}x^{FF}$')
    axs[0].plot(target_x_ff, c='blue', label='target $x^{FF}$')
    axs[0].plot(non_target_x_ff, c='lightskyblue', label='non-target $x^{FF}$')
    axs[0].plot(target_u_ff, c='green', label='target $u^{FF}$')
    axs[0].plot(non_target_u_ff, c='limegreen', label='non-target $u^{FF}$')
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[0].set_title('Average Synaptic Values - feedforward')
    axs[1].plot(correl_u, label='correlation with $u$')
    axs[1].plot(correl_x, label='correlation with $x$')
    axs[1].plot(correl_ux, label='correlation with $ux$')
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[1].set_title('Target Score Correlations - feedback')
    plt.tight_layout()
    plt.show()

def plot_ux_avg(run_results, simulation, mu_idx=0, **kwargs):
    n_sensory = simulation.N_sensory
    n_sensory_nets = simulation.N_sensory_nets
    x_ff = run_results['x_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    x_ff = x_ff.reshape(x_ff.shape[0], n_sensory_nets, n_sensory)
    u_ff = run_results['u_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    u_ff = u_ff.reshape(u_ff.shape[0], n_sensory_nets, n_sensory)
    target_neurons, non_target_neurons = _get_target_sensory_neurons(run_results, mu_idx=mu_idx)
    target_u_ff = u_ff[1:, mu_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_u_ff = u_ff[1:, mu_idx, non_target_neurons].mean(1)

    target_x_ff = x_ff[1:, mu_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_x_ff = x_ff[1:, mu_idx, non_target_neurons].mean(1)

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

def plot_x_avg(run_results, simulation, mu_idx=0, **kwargs):
    n_sensory = simulation.N_sensory
    n_sensory_nets = simulation.N_sensory_nets
    x_ff = run_results['x_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    x_ff = x_ff.reshape(x_ff.shape[0], n_sensory_nets, n_sensory)
    x_fb = run_results['x_fb'] # shape should be t x n_rand
    target_neurons, non_target_neurons = _get_target_sensory_neurons(run_results, mu_idx=mu_idx)
    target_x_ff = x_ff[1:, mu_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_x_ff = x_ff[1:, mu_idx, non_target_neurons].mean(1)
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


def plot_u_avg(run_results, simulation, mu_idx=0, **kwargs):
    n_sensory = simulation.N_sensory
    n_sensory_nets = simulation.N_sensory_nets
    u_ff = run_results['u_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    u_ff = u_ff.reshape(u_ff.shape[0], n_sensory_nets, n_sensory) # shape should be t x n_rand
    u_fb = run_results['u_fb'] # shape should be t x n_rand
    target_neurons, non_target_neurons = _get_target_sensory_neurons(run_results, mu_idx=mu_idx)
    target_u_ff = u_ff[1:, mu_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_u_ff = u_ff[1:, mu_idx, non_target_neurons].mean(1)
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


def plot_s_sens_avg(run_results, **kwargs):
    s_sens = run_results['s_sens']
    mus = run_results['mus']
    sigma = run_results['sigma']
    n_sensory = run_results['n_sensory']
    for mu_idx, mu in enumerate(mus):
        target_neurons, non_target_neurons = _get_target_sensory_neurons(run_results, mu_idx)
        target_activity = s_sens[:, mu_idx, target_neurons].mean(axis=1)
        plt.plot(target_activity, label='target population')

        non_target_activity = s_sens[:, mu_idx, non_target_neurons].mean(axis=1)
        plt.plot(non_target_activity)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('$s_i$')
    plt.show()


def plot_r_sens(run_results, **kwargs):
    _plot_sens(key='r_sens', run_results=run_results, **kwargs)


def plot_s_sens(run_results, **kwargs):
    _plot_sens(key='s_sens', run_results=run_results, **kwargs)


def plot_p_sens(run_results, **kwargs):
    _plot_sens(key='p_sens', run_results=run_results, **kwargs)


def plot_r_rand(run_results, **kwargs):
    _plot_rand(key='r_rand', run_results=run_results, **kwargs)


def plot_s_rand(run_results, **kwargs):
    _plot_rand(key='s_rand', run_results=run_results, **kwargs)


def plot_p_rand(run_results, **kwargs):
    _plot_rand(key='p_rand', run_results=run_results, **kwargs)


def plot_s_ext(run_results, **kwargs):
    _plot_sens(key='s_ext', run_results=run_results, **kwargs)


def _plot_rand(key, run_results):
    plt.imshow(run_results[key].T, aspect='auto')
    plt.xlabel('Time Steps')
    plt.ylabel('Neuron #')
    plt.colorbar()
    plt.show()


def _plot_sens(key, run_results, sens_idx=0, title=''):
    plt.imshow(run_results[key][:, sens_idx, :].T, aspect='auto')
    plt.xlabel('Time Steps')
    plt.ylabel('Neuron #')
    plt.title(title)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    from simulation import Simulation

    sim = Simulation(T=1000, N_sensory_nets=1, N_sensory=512, N_rand=1024, amp_ext=500)
    sim.reset()
    run_results = sim.run()
    plot_s_sens_avg(run_results)
