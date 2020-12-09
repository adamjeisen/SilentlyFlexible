import numpy as np
from scipy.linalg import toeplitz
class SpikingNetwork():
    def __init__(self, N=512, tau=10, dt=1):
        self.r = None  # firing rates
        self.W_rec = None  # recurrent weight matrix
        self.W_other = None
        self.s = None  # synaptic activation
        self.dt = dt
        self.N = N
        self.tau = tau

    def phi(self, x):
        return (0.4/self.tau) * (1 + np.tanh(0.4 * x - 3))

    def reset(self, W_rec, W_other):
        if W_rec is None:
            W_rec = np.zeros((self.N, self.N))
        if W_other is None:
            raise ValueError("W_other must be defined at network reset")
        self.W_rec = W_rec  # recurrent weights
        self.W_other = W_other  # weights between random and sensory

    def forward(self, s_ext, s_other, s_rec):
        """
        Forward pass through spiking network
        s_ext: External input (currently only nonzero for sensory networks)
        s_other: Input from other network (should be nonzero for both sensory and random networks)
        s_rec: Recurrent input (currently only nonzero for sensory networks)
        """
        if s_ext is None:  # random network
            s_ext = 0
        if s_other is None:
            raise ValueError('s_other should not be None')
        if s_rec is None:  # random network
            s_rec = np.zeros((self.N,))
        # the tau after
        r = self.phi(self.W_rec @ s_rec + self.W_other @ s_other + s_ext)
        # p = np.random.poisson(r*self.dt, size=(self.N, ))
        p = np.random.rand(self.N, ) < (r * self.dt)
        delta_s = (-s_rec) / self.tau + p
        s = s_rec + self.dt * delta_s
        return dict(
            r=r,
            s=s,
            p=p
        )


class SensorySpikingNetwork(SpikingNetwork):
    def __init__(self, A=2, lamb=0.28, k1=1, k2=0.25, **kwargs):
        super(SensorySpikingNetwork, self).__init__(**kwargs)
        self.A = A
        self.lamb = lamb
        self.k1 = k1
        self.k2 = k2

    def _create_recurrent_weight_matrix(self):
        angle = 2. * np.pi * np.arange(1, self.N + 1) / self.N

        def weight_intrapool(i):
            return self.lamb + self.A * np.exp(self.k1 * (np.cos(i) - 1)) - self.A * np.exp(self.k2 * (np.cos(i) - 1))

        w = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                w[i, j] = weight_intrapool(angle[i] - angle[j])

        w[np.arange(self.N), np.arange(self.N)] = 0
        return w

    def reset(self, W_fb):
        W_rec = self._create_recurrent_weight_matrix()
        super(SensorySpikingNetwork, self).reset(W_rec=W_rec, W_other=W_fb)

    def forward(self, s_ext, s_rand, s_rec):
        """
        Forward pass for sensory network.
        s_ext: External inputs
        s_rand: Inputs from random network
        s_rec: Recurrent input - synaptic activations from previous timestep
        """
        # print(s_ext is None, s_rec is None, s_rand is None)
        return super(SensorySpikingNetwork, self).forward(s_ext=s_ext, s_other=s_rand, s_rec=s_rec)


class RandomNetwork(SpikingNetwork):
    def __init__(self, **kwargs):
        super(RandomNetwork, self).__init__(**kwargs)

    def reset(self, W_ff):
        """
        Intializes recurrent weight matrix.
        W_f: feedforward weights from sensory networks
        """
        super(RandomNetwork, self).reset(W_other=W_ff, W_rec=None)

    def forward(self, s_sens, s_rec):
        """
        Forward pass for random network.
        s_sens: Activity of neurons in sensory network
        s_rand: Activity of neurons in random network in previous timestep
        """
        return super(RandomNetwork, self).forward(s_other=s_sens, s_ext=None, s_rec=s_rec)
