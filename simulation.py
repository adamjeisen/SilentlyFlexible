import numpy as np
from synaptic_networks import SensorySynapticNetwork, RandomSynapticNetwork

class Simulation():
    def __init__(self, T=1000, load=1, N_sensory=512, N_rand=1024, N_sensory_nets=2, amp_ext=10, gamma=0.35, alpha=2100,
                 beta=200, **sens_net_kwargs):
        # function arguments
        self.T = T
        self.load = load
        self.N_sensory = N_sensory
        self.N_rand = N_rand
        self.N_sensory_nets = N_sensory_nets
        self.amp_ext = amp_ext
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.sens_net_kwargs = sens_net_kwargs  # rand net has no kwargs other than the base spiking net class ones, so these kwargs should be exclusively for the sensory netowrk

        # relevant variables
        self.sigma = self.N_sensory / 32
        self.s_ext = None
        self.sens_nets = None
        self.rand_net = None
        self.W_ff = None
        self.mus = None
        self.W_fb = None

    def _create_s_ext(self):
        # randomly select center of input to each network
        mus = np.random.choice(self.N_sensory, size=(self.load,))
        # generate Gaussians around the means for each network with std sigma (with wraparound)
        dist_from_mean = np.array([np.abs([(np.arange(self.N_sensory) - mu),
                                           (np.arange(self.N_sensory) - (mu + self.N_sensory)),
                                           (np.arange(self.N_sensory) - (mu - self.N_sensory))]).min(axis=0) for mu in
                                   mus])
        s_ext = np.exp(-1 * (dist_from_mean ** 2) / (2 * self.sigma ** 2))
        s_ext *= self.amp_ext / np.sqrt(2 * np.pi * self.sigma ** 2)
        # zero out anything beyond 3 standard deviations
        s_ext[dist_from_mean > 3 * self.sigma] = 0

        net = 0  # TODO: Make this a randomly selected network?
        s_ext_all = np.zeros((self.N_sensory_nets, self.N_sensory))
        s_ext_all[net] = s_ext
        s_ext_all = s_ext_all.reshape(self.N_sensory * self.N_sensory_nets)
        return dict(
            s_ext=s_ext_all,
            mus=mus
        )

    def _create_weight_matrices(self):
        # initialize which connections are excitatory in the feed forward matrix (sample from Bernoulli)
        W_ff = (np.random.uniform(low=0, high=1, size=(self.N_rand, self.N_sensory_nets * self.N_sensory)) <
                self.gamma).astype(int)
        # set up the feed back matrix
        W_fb = W_ff.T

        # FORWARD: set up excitatory connections to be normalized by number of input connections to neuron i
        W_ff = (self.alpha / (
            np.repeat(W_ff.sum(axis=1), self.N_sensory_nets * self.N_sensory).reshape(W_ff.shape))) * W_ff
        # FORWARD: set up inhibitory connections
        W_ff -= self.alpha / (self.N_sensory_nets * self.N_sensory)

        # # BACK: set up excitatory connections to be normalized by number of input connections to neuron i
        W_fb = (self.beta / (np.repeat(W_fb.sum(axis=1), self.N_rand).reshape(W_fb.shape))) * W_fb
        # BACK: set up inhibitory connections
        W_fb -= self.beta / self.N_rand

        return W_ff, W_fb

    def reset(self):
        # initialize sensory input
        input_dict = self._create_s_ext()
        self.s_ext = input_dict['s_ext']
        self.mus = input_dict['mus']
        # initialize sensory networks

        self.sens_nets = [SensorySynapticNetwork(N=self.N_sensory, **self.sens_net_kwargs) for i in
                          range(self.N_sensory_nets)]
        # initialize random network
        self.rand_net = RandomSynapticNetwork(N=self.N_rand)
        # initialize weight matrices
        self.W_ff, self.W_fb = self._create_weight_matrices()

        # reset sensory networks
        for i, sens_net in enumerate(self.sens_nets):
            sens_net.reset(W_fb=self.W_fb[i * self.N_sensory: (i + 1) * self.N_sensory, :])

        # reset random network
        self.rand_net.reset(W_ff=self.W_ff)

    def forward(self, s_ext, s_rand_prev, s_sens_prev, p_rand_prev):
        """
        Single timestep forward
        s_ext: External inputs (N_sensory_nets * N_sensory, 1)
        s_rand: Synaptic activations of random network at t - 1 (N_rand, 1)
        s_sens: Synaptic activations of sensory networks at t - 1 (N_sensory_nets * N_sensory, 1)
        """


        # reshaping input to sensory networks
        s_ext = s_ext.reshape(self.N_sensory_nets, self.N_sensory)
        s_sens_prev = s_sens_prev.reshape(self.N_sensory_nets, self.N_sensory)

        # initializing empty array to hold current activity of nets
        s_sens = np.zeros((self.N_sensory_nets, self.N_sensory))
        r_sens = np.zeros((self.N_sensory_nets, self.N_sensory))
        p_sens = np.zeros((self.N_sensory_nets, self.N_sensory))

        # initializing empty array to hold synaptic feedforward variables
        u_fb = np.zeros((self.N_sensory_nets, self.N_sensory, self.N_rand))
        x_fb = np.zeros((self.N_sensory_nets, self.N_sensory, self.N_rand))

        # forward step for all sensory networks
        for i, sens_net in enumerate(self.sens_nets):
            sens_step = sens_net.forward(s_ext=s_ext[i], s_rec=s_sens_prev[i], s_rand=s_rand_prev, p_rand=p_rand_prev)
            s_sens[i] = sens_step['s']
            r_sens[i] = sens_step['r']
            p_sens[i] = sens_step['p']
            u_fb[i] = sens_step['u']
            x_fb[i] = sens_step['x']


        # reshaping sensory network activity into a one-dimensional array for random network
        s_sens = s_sens.reshape(self.N_sensory_nets * self.N_sensory, )
        p_sens = p_sens.reshape(self.N_sensory_nets * self.N_sensory, )
        r_sens = r_sens.reshape(self.N_sensory_nets * self.N_sensory, )

        # forward step for random network
        rand_step = self.rand_net.forward(s_sens=s_sens, s_rec=s_rand_prev, p_sens=p_sens)
        s_rand = rand_step['s']
        r_rand = rand_step['r']
        p_rand = rand_step['p']
        u_ff = rand_step['u']
        x_ff = rand_step['x']

        return dict(
            u_ff=u_ff,
            u_fb=u_fb,
            x_ff=x_ff,
            x_fb=x_fb,
            s_sens=s_sens,
            p_sens=p_sens,
            r_sens=r_sens,
            s_rand=s_rand,
            r_rand=r_rand,
            p_rand=p_rand
        )

    def run(self):
        """
        Runs through entire simulation
        """
        # intializing synaptic variables matrices
        u_fb = np.zeros((self.T, self.N_sensory_nets, self.N_sensory, self.N_rand))
        u_ff = np.zeros((self.T, self.N_rand, self.N_sensory * self.N_sensory_nets))
        x_fb = np.zeros((self.T, self.N_sensory_nets, self.N_sensory, self.N_rand))
        x_ff = np.zeros((self.T, self.N_rand, self.N_sensory * self.N_sensory_nets))

        # initializing random network activity
        s_rand_T = np.zeros((self.T, self.N_rand))
        p_rand_T = np.zeros((self.T, self.N_rand))
        r_rand_T = np.zeros((self.T, self.N_rand))

        s_rand_T[0, :] = np.random.uniform(low=0, high=0.01, size=(self.N_rand))

        # initializing sensory networks
        s_sens_T = np.zeros((self.T, self.N_sensory_nets * self.N_sensory))
        p_sens_T = np.zeros((self.T, self.N_sensory_nets * self.N_sensory))
        r_sens_T = np.zeros((self.T, self.N_sensory_nets * self.N_sensory))
        s_sens_T[0, :] = np.random.uniform(low=0, high=0.01, size=(self.N_sensory_nets * self.N_sensory))

        # extend input to be T timesteps and only nonzero for 100 ts
        s_ext_T = np.broadcast_to(self.s_ext, (self.T, self.N_sensory * self.N_sensory_nets)).copy()
        # stimulus is presented for 100 ms
        stim_T = int(10 / 0.1)
        s_ext_T[100 + stim_T:] = 0
        s_ext_T[:100] = 0
        # s_ext_T *= 0

        for t in range(1, self.T):
            if (t + 1) % 100 == 0:
                print(f'step {t} of {self.T}')
            s_sens_prev = s_sens_T[t - 1]
            s_rand_prev = s_rand_T[t - 1]
            p_rand_prev = p_rand_T[t - 1]
            s_ext = s_ext_T[t - 1]
            step = self.forward(s_ext=s_ext, s_rand_prev=s_rand_prev, s_sens_prev=s_sens_prev, p_rand_prev=p_rand_prev)
            s_sens_T[t] = step['s_sens']
            p_sens_T[t] = step['p_sens']
            r_sens_T[t] = step['r_sens']
            s_rand_T[t] = step['s_rand']
            r_rand_T[t] = step['r_rand']
            p_rand_T[t] = step['p_rand']
            u_ff[t] = step['u_ff']
            u_fb[t] = step['u_fb']
            x_ff[t] = step['x_ff']
            x_fb[t] = step['x_fb']

        p_sens_T = p_sens_T.reshape(self.T, self.N_sensory_nets, self.N_sensory)
        s_ext_T = s_ext_T.reshape(self.T, self.N_sensory_nets, self.N_sensory)
        r_sens_T = r_sens_T.reshape(self.T, self.N_sensory_nets, self.N_sensory)
        s_sens_T = s_sens_T.reshape(self.T, self.N_sensory_nets, self.N_sensory)

        return dict(
            u_ff=u_ff,
            u_fb=u_fb,
            x_ff=x_ff,
            x_fb=x_fb,
            n_sensory=self.N_sensory,
            n_rand=self.N_rand,
            mus=self.mus,
            sigma=self.sigma,
            s_ext=s_ext_T,
            s_sens=s_sens_T,
            r_sens=r_sens_T,
            p_sens=p_sens_T,
            s_rand=s_rand_T,
            r_rand=r_rand_T,
            p_rand=p_rand_T
        )


if __name__ == '__main__':
    sim = Simulation(T=800, N_sensory_nets=8, N_sensory=512, N_rand=1024, amp_ext=500)
    sim.reset()
    run_results = sim.run()
    print('done')
