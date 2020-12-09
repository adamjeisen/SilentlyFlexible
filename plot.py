import matplotlib.pyplot as plt
import numpy as np

def plot_s_sens_avg(run_results, **kwargs):
    s_sens = run_results['s_sens']
    mus = run_results['mus']
    sigma = run_results['sigma']
    n_sensory = run_results['n_sensory']
    # selecting the target neurons
    for i_net, mu in enumerate(mus):
        target_neurons = np.arange(mu - 3 * sigma, mu + 3 * sigma)
        target_neurons = np.where(target_neurons < 0, n_sensory + target_neurons, target_neurons).astype(int)
        target_neurons = np.where(target_neurons >= n_sensory, target_neurons - n_sensory, target_neurons).astype(int)
        target_activity = s_sens[:, i_net, target_neurons].mean(axis=1)
        plt.plot(target_activity, label='target population')

        non_target_neurons = np.arange(n_sensory)
        non_target_neurons = [neuron for neuron in non_target_neurons if neuron not in target_neurons]
        non_target_activity = s_sens[:, i_net, non_target_neurons].mean(axis=1)
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

