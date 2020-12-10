import pickle as pkl
from simulation import Simulation


def save(simulation, run_results, fpath='data/simulation_results'):
    save_dict = dict(
        simulation=simulation,
        run_results=run_results
    )
    with open(fpath, 'wb') as f:
        pkl.dump(save_dict, f)


def load(fpath='data/simulation_results'):
    with open(fpath, 'rb') as f:
        save_dict = pkl.load(f)
    return save_dict
