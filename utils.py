import pickle as pkl
from simulation import Simulation
from datetime import datetime


def get_fpath(simulation):
    tau_f, tau_d, amp_ext, ux_mod, u_init = simulation.rand_net.tau_f, simulation.rand_net.tau_d, simulation.amp_ext, \
                                            simulation.rand_net.ux_mod, simulation.rand_net.u_init
    fpath = f'data/{datetime.now()}_tau_f={tau_f}_tau_d={tau_d}_amp_ext={amp_ext}_ux_mod={ux_mod}_u_init={u_init}'
    return fpath


def save(simulation, run_results, fpath):
    save_dict = dict(
        simulation=simulation,
        run_results=run_results
    )
    print(f'Saving model and run results to {fpath}')
    with open(fpath, 'wb') as f:
        pkl.dump(save_dict, f)


def load(fpath='data/simulation_results'):
    with open(fpath, 'rb') as f:
        save_dict = pkl.load(f)
    return save_dict
