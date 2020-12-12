import os
from analysis import _get_maintained_memory, _get_mle_mus
import numpy as np
from utils import load
import pandas as pd
from plot import save_all_plots


def get_run_stats(directory, gen_plots=False):
    paths = []
    if not isinstance(directory, list):
        directory = [directory]
    for d in directory:
        dir_paths = [os.path.join(d, f) for f in os.listdir(d) if not os.path.isdir(os.path.join(d, f)) and not f.startswith('.') and not f == 'log']
        paths += dir_paths
    run_stats_rows = []
    for path in paths:
        data = load(path)
        run_results = data['run_results']
        simulation = data['simulation']

        # get statistics on how accurate memories were
        mle_mus = _get_mle_mus(run_results, simulation)
        dist_from_mean = np.array([
                        np.abs([(mle_mus[i] - simulation.mus[i]),
                                (mle_mus[i] - (simulation.mus[i] + simulation.N_sensory)),
                                (mle_mus[i] - (simulation.mus[i] - simulation.N_sensory))]).min(axis=0)
                                if simulation.mus[i] is not None else None for i in range(simulation.N_sensory_nets)])
        mu_error = np.array([(dist / (simulation.N_sensory / 2)) * np.pi
                             if dist is not None else None for dist in dist_from_mean])
        # get statistics on how well memories were maintained
        maintained_memory, avg_spikes = _get_maintained_memory(run_results, simulation, mle_mus)
        load_locs = np.array([mu is not None for mu in simulation.mus])
        maintained_memories = sum(maintained_memory[load_locs])
        spurious_memories = sum(maintained_memory[~load_locs])
        percent_maintained = maintained_memories / simulation.load
        if simulation.load < 8:
            percent_spurious = spurious_memories / (simulation.N_sensory_nets - simulation.load)
        else:
            percent_spurious = 0
        avg_mu_error = mu_error[load_locs].mean()
        run_stats_rows.append(dict(
            load=simulation.load,
            n_rand=simulation.N_rand,
            path=path,
            mle_mus=mle_mus,
            mu_error=mu_error,
            maintained_memory=maintained_memory,
            avg_spikes=avg_spikes,
            load_locs=load_locs,
            maintained_memories=maintained_memories,
            spurious_memories=spurious_memories,
            avg_mu_error=avg_mu_error,
            percent_maintained=percent_maintained,
            percent_spurious=percent_spurious
        ))

    run_stats_df = pd.DataFrame(run_stats_rows)

    if gen_plots:
        save_all_plots(run_stats_df)

    return run_stats_df
