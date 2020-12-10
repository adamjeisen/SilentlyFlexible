from simulation import *
import os
from utils import *

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    sim = Simulation(T=900, N_sensory_nets=8, N_sensory=512, N_rand=1024, amp_ext=900, amp_ext_nonspecific=10,
                     tau_d=100, tau_f=2000, ux_mod=5)
    sim.reset(mus=[256])
    save(simulation=sim, run_results={})
    run_results = sim.run()
    save(simulation=sim, run_results=run_results)