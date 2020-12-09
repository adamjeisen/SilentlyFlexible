import pickle as pkl


def save(simulation, run_results, fpath='data/simulation_results'):
    save_dict = dict(
        simulation=simulation,
        run_results=run_results
    )
    with open(fpath, 'wb') as f:
        pkl.dump(save_dict, f)
