import matplotlib.pyplot as plt
import numpy as np
from analysis import _get_target_sensory_neurons


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

    f, axs = plt.subplots(2, 1)
    axs[0].plot(target_x_ff*target_u_ff, label='target $x^{FF}u^{FF}$')
    axs[0].plot(non_target_x_ff*non_target_u_ff, label='non-target $x^{FF}u^{FF}$')
    axs[0].legend()
    axs[0].set_title('Average $x$ - feedforward')
    plt.tight_layout()
    plt.show()
def plot_x_avg(run_results, simulation, mu_idx=0, **kwargs):
    n_sensory = simulation.N_sensory
    n_sensory_nets = simulation.N_sensory_nets
    x_ff = run_results['x_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    x_ff = x_ff.reshape(x_ff.shape[0], n_sensory_nets, n_sensory)
    # x_fb = run_results['x_fb']
    # x_fb = x_fb.reshape(x_ff.shape[0], n_sensory_nets, n_sensory)  # shape should be t x n_sensory_nets x n_sensory x
    target_neurons, non_target_neurons = _get_target_sensory_neurons(run_results, mu_idx=mu_idx)
    target_x_ff = x_ff[1:, mu_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_x_ff = x_ff[1:, mu_idx, non_target_neurons].mean(1)
    # target_x_fb = x_fb[1:, mu_idx, target_neurons].mean(1)  # averaging across target neurons
    # non_target_x_fb = x_fb[1:, mu_idx, non_target_neurons].mean(1)
    f, axs = plt.subplots(2, 1)
    axs[0].plot(target_x_ff, label='target $x^{FF}$')
    axs[0].plot(non_target_x_ff, label='non-target $x^{FF}$')
    axs[0].legend()
    axs[0].set_title('Average $x$ - feedforward')
    # axs[1].plot(target_x_fb, label='non-target $x^{FB}$')
    # axs[1].plot(non_target_x_fb, label='non-target $x^{FB}$')
    # axs[1].set_title('Average $x$ - feedback')
    # axs[1].legend()
    plt.tight_layout()
    plt.show()


def plot_u_avg(run_results, simulation, mu_idx=0, **kwargs):
    n_sensory = simulation.N_sensory
    n_sensory_nets = simulation.N_sensory_nets
    u_ff = run_results['u_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    u_ff = u_ff.reshape(u_ff.shape[0], n_sensory_nets, n_sensory)
    # u_fb = run_results['u_fb']
    # u_fb = u_fb.reshape(u_ff.shape[0], n_sensory_nets, n_sensory)  # shape should be t x n_sensory_nets x n_sensory x
    target_neurons, non_target_neurons = _get_target_sensory_neurons(run_results, mu_idx=mu_idx)
    target_u_ff = u_ff[1:, mu_idx, target_neurons].mean(1)  # averaging across target neurons
    non_target_u_ff = u_ff[1:, mu_idx, non_target_neurons].mean(1)
    # target_u_fb = u_fb[1:, mu_idx, target_neurons].mean(1)  # averaging across target neurons
    # non_target_u_fb = u_fb[1:, mu_idx, non_target_neurons].mean(1)
    f, axs = plt.subplots(2, 1)
    axs[0].plot(target_u_ff, label='target $u^{FF}$')
    axs[0].plot(non_target_u_ff, label='non-target $u^{FF}$')
    axs[0].legend()
    axs[0].set_title('Average $u$ - feedforward')
    # axs[1].plot(target_u_fb, label='non-target $u^{FB}$')
    # axs[1].plot(non_target_u_fb, label='non-target $u^{FB}$')
    # axs[1].set_title('Average $u$ - feedback')
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
