from simulation import *
from utils import *

if __name__ == '__main__':
    sim = Simulation(T=1000, N_sensory_nets=8, N_sensory=512, N_rand=1024, amp_ext=300)
    sim.reset(mus=[256])
    # save(simulation=sim, run_results={})
    run_results = sim.run()
    # save(simulation=sim, run_results=run_results)