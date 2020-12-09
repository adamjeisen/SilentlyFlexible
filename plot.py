import matplotlib.pyplot as plt
import numpy as np


def _get_target_neurons(run_results, mu_idx=0):
    sigma = run_results['sigma']
    mu = run_results['mus'][mu_idx]
    n_sensory = run_results['n_sensory']

    target_neurons = np.arange(mu - 3 * sigma, mu + 3 * sigma)
    target_neurons = np.where(target_neurons < 0, n_sensory + target_neurons, target_neurons).astype(int)
    target_neurons = np.where(target_neurons >= n_sensory, target_neurons - n_sensory, target_neurons).astype(int)

    non_target_neurons = np.arange(n_sensory)
    non_target_neurons = [neuron for neuron in non_target_neurons if neuron not in target_neurons]

    return target_neurons, non_target_neurons


def plot_u_avg(run_results, mu_idx=0, **kwargs):
    n_sensory = run_results['n_sensory']
    u_ff = run_results['u_ff']  # shape should be t x n_rand x n_sensory * n_sensory_nets
    u_ff = u_ff.reshape(u_ff.shape[0], u_ff.shape[1], -1, n_sensory)
    u_fb = run_results['u_fb']  # shape should be t x n_sensory_nets x n_sensory x n_rand
    target_neurons, non_target_neurons = _get_target_neurons(run_results, mu_idx=0)
    target_u_ff = u_ff[:, :, mu_idx, target_neurons].mean(2).mean(1) # averaging across target neurons
    non_target_u_ff = u_ff[:, :, mu_idx, non_target_neurons].mean(2).mean(1)
    target_u_fb = u_fb[:, mu_idx, target_neurons, :].mean(2).mean(1)  # averaging across target neurons
    non_target_u_fb = u_fb[:, mu_idx, target_neurons, :].mean(2).mean(1)
    f, axs = plt.subplots(2, 1)
    axs[0].plot(target_u_ff, label='target $u^{FF}$')
    axs[0].plot(non_target_u_ff, label='non-target $u^{FF}$')
    axs[0].legend()
    axs[0].set_title('Average $u$ - feedforward')
    axs[1].plot(target_u_fb, label='non-target $u^{FB}$')
    axs[1].plot(non_target_u_fb, label='non-target $u^{FB}$')
    axs[1].set_title('Average $u$ - feedback')
    plt.tight_layout()
    plt.show()

def plot_s_sens_avg(run_results, **kwargs):
    s_sens = run_results['s_sens']
    mus = run_results['mus']
    sigma = run_results['sigma']
    n_sensory = run_results['n_sensory']
    for mu_idx, mu in enumerate(mus):
        target_neurons, non_target_neurons = _get_target_neurons(run_results, mu_idx)
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
