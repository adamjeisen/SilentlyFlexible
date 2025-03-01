import buschman_simulation
import simulation
import os
from utils import *

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    # ===================================================
    # SYNAPTIC
    # ===================================================
    sim = simulation.Simulation(T=1500, N_sensory_nets=8, N_sensory=512, N_rand=1024, amp_ext=1200,
                                amp_ext_nonspecific=5, amp_ext_background=0,
                                amp_ext_nonspecific_rand=5, amp_ext_stim_rand=10, amp_ext_background_rand=0,
                                tau_d=200, tau_f=2000, ux_mod=2, u_init=0.2)
    mus = [None] * sim.N_sensory_nets
    mus[0] = 256
    sim.reset(mus=mus)
    fpath = get_fpath(sim)
    save(simulation=sim, run_results={}, fpath=fpath)
    run_results = sim.run()
    save(simulation=sim, run_results=run_results, fpath=fpath)

    # # ===================================================
    # # Buschman
    # # ===================================================
    # sim = buschman_simulation.Simulation(T=3000, N_sensory_nets=8, N_sensory=512, N_rand=1024, amp_ext=1200)
    # mus = [None]*sim.N_sensory_nets
    # mus[0] = 256
    # mus[1] = 256
    # mus[2] = 256
    # mus[3] = 256
    # mus[4] = 256
    # mus[5] = 256
    # sim.reset(mus=mus)
    # fpath = get_fpath_buschman(sim)
    # save(simulation=sim, run_results={}, fpath=fpath)
    # run_results = sim.run()
    # save(simulation=sim, run_results=run_results, fpath=fpath)