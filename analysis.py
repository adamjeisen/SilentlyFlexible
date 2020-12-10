import numpy as np
from scipy.stats import pearsonr


def _get_target_sensory_neurons(run_results, mu_idx=0):
    """
    Returns list of neurons activated by incoming sensory input (and list of non-target neurons)
    """
    sigma = run_results['sigma']
    mu = run_results['mus'][mu_idx]
    n_sensory = run_results['n_sensory']

    target_neurons = np.arange(mu - 3 * sigma, mu + 3 * sigma)
    target_neurons = np.where(target_neurons < 0, n_sensory + target_neurons, target_neurons).astype(int)
    target_neurons = np.where(target_neurons >= n_sensory, target_neurons - n_sensory, target_neurons).astype(int)

    non_target_neurons = np.arange(n_sensory)
    non_target_neurons = [neuron for neuron in non_target_neurons if neuron not in target_neurons]

    return target_neurons, non_target_neurons


def _get_surr_neurons(sigma, mu, n, width=1):
    target_neurons = np.arange(mu - width * sigma, mu + width * sigma)
    target_neurons = np.where(target_neurons < 0, n + target_neurons, target_neurons).astype(int)
    target_neurons = np.where(target_neurons >= n, target_neurons - n, target_neurons).astype(int)
    return target_neurons


def _get_target_random_neurons(run_results, mu_idx=0):
    target_sensory_neurons, non_target_sensory_neurons = _get_target_sensory_neurons(run_results, mu_idx=mu_idx)
    n_sensory = run_results['n_sensory']
    sigma = run_results['sigma']
    n_sensory_nets = run_results['simulation'].N_sensory_nets
    n_rand = run_results['simulation'].N_rand
    mu = run_results['simulation'].mus[mu_idx]
    W_fb = run_results['W_fb']
    W_fb = W_fb.reshape(n_sensory_nets, n_sensory, n_rand)
    target_neurons = _get_surr_neurons(sigma, mu, n_sensory)
    W_fb = W_fb[mu_idx, :, ].squeeze()
    target_random = np.where(W_fb[target_neurons, :] > 0)


def _get_target_score(run_results, simulation, mu_idx=0):
    u_fb = run_results['u_fb']
    x_fb = run_results['x_fb']
    p_rand = run_results['p_rand']
    target_neurons, non_target_neurons = _get_target_sensory_neurons(run_results, mu_idx=mu_idx)
    gaussian = np.exp(
        (-1 / (2 * simulation.sigma ** 2)) * (np.arange(len(target_neurons)) - int(len(target_neurons) / 2)) ** 2)
    targeted_score = ((simulation.W_ff > 0).astype(int) -
                      (simulation.W_ff < 0).astype(int))[:, target_neurons] @ gaussian
    # correl_p = np.array([pearsonr(p_rand[i, :], targeted_score)[0] if not
    #     np.array_equal(np.unique(np.diff(u_fb[i, :])), [0]) else 0 for i in range(1, u_fb.shape[0])])
    correl_u = np.array([pearsonr(u_fb[i, :], targeted_score)[0] if not
        np.array_equal(np.unique(np.diff(u_fb[i, :])), [0]) else 0 for i in range(1, u_fb.shape[0])])
    correl_x = np.array([pearsonr(x_fb[i, :], targeted_score)[0] if not
        np.array_equal(np.unique(np.diff(u_fb[i, :])), [0]) else 0 for i in range(1, u_fb.shape[0])])
    correl_ux = np.array([pearsonr(u_fb[i, :] * x_fb[i, :], targeted_score)[0] if not
        np.array_equal(np.unique(np.diff(u_fb[i, :])), [0]) else 0 for i in range(1, u_fb.shape[0])])
    return dict(
        # correl_p=correl_p,
        correl_u=correl_u,
        correl_x=correl_x,
        correl_ux=correl_ux
    )
