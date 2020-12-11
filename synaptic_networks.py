import numpy as np
from scipy.linalg import toeplitz
class SynapticSpikingNetwork():
    def __init__(self, N=512, tau=10, dt=1, tau_d=200, tau_f=1500, u_init=.2, ux_mod=2, sigmoid_slope=100):

        self.W_rec = None  # recurrent weight matrix
        self.W_other = None # feedforward/back weight matrix

        self.dt = dt
        self.N = N
        self.tau = tau
        self.tau_d = tau_d  # depression time constant
        self.tau_f = tau_f  # facilitation time constant
        self.u = None
        self.x = None
        self.u_init = u_init
        self.ux_mod = ux_mod
        self.sigmoid_slope = sigmoid_slope


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-self.sigmoid_slope * (x - 0.5)))

    def phi(self, x):
        return (0.4/self.tau) * (1 + np.tanh(0.4 * x - 3))

    def reset(self, W_rec, W_other):
        if W_rec is None:
            W_rec = np.zeros((self.N, self.N))
        if W_other is None:
            raise ValueError("W_other must be defined at network reset")
        self.W_rec = W_rec  # recurrent weights
        self.W_other = W_other  # weights between random and sensory
        self.u = np.ones(W_other.shape) * self.u_init
        self.x = np.ones(W_other.shape)


    def forward(self, s_ext, s_other, s_rec, p_other):
        """
        Forward pass through spiking network
        s_ext: External input (currently only nonzero for sensory networks)
        s_other: Input from other network (should be nonzero for both sensory and random networks)
        s_rec: Recurrent input (currently only nonzero for sensory networks)
        p_other: Spikes from presynaptic neurons used to update calcium vals
        """
        if s_ext is None:  # random network
            s_ext = 0
        if s_other is None:
            raise ValueError('s_other should not be None')
        if s_rec is None:  # random network
            s_rec = np.zeros((self.N,))
        facilitation = self.u * self.x
        r = self.phi(self.W_rec @ s_rec + (self.ux_mod * facilitation * self.W_other) @ s_other + s_ext)
        p = np.random.rand(self.N, ) < (r * self.dt)
        delta_s = (-s_rec) / self.tau + p
        s = s_rec + self.dt * delta_s
        self.u, self.x = self._forward_calcium(p_other)
        return dict(
            r=r,
            s=s,
            p=p,
            u=self.u,
            x=self.x,
        )

    def _forward_calcium(self, p):
        """
        p: Presynaptic spike vector
        """
        u_prev = self.u
        x_prev = self.x

        p = p[:, np.newaxis]
        p = p.repeat(self.N, axis=1)
        p = p.T
        delta_u = (self.u_init - u_prev)/self.tau_f + self.u_init*(1 - u_prev)*p
        u = u_prev + self.dt * delta_u

        delta_x = (1 - x_prev)/self.tau_d - u_prev * x_prev * p
        x = x_prev + self.dt * delta_x
        return u, x


class SensorySynapticNetwork(SynapticSpikingNetwork):
    def __init__(self, A=2, lamb=0.28, k1=1, k2=0.25, **kwargs):
        super(SensorySynapticNetwork, self).__init__(**kwargs)
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
        super(SensorySynapticNetwork, self).reset(W_rec=W_rec, W_other=W_fb)

    def forward(self, s_ext, s_rand, s_rec, p_rand):
        """
        Forward pass for sensory network.
        s_ext: External inputs
        s_rand: Inputs from random network
        s_rec: Recurrent input - synaptic activations from previous timestep
        """
        # print(s_ext is None, s_rec is None, s_rand is None)
        return super(SensorySynapticNetwork, self).forward(s_ext=s_ext, s_other=s_rand, s_rec=s_rec, p_other=p_rand)


class RandomSynapticNetwork(SynapticSpikingNetwork):
    def __init__(self, **kwargs):
        super(RandomSynapticNetwork, self).__init__(**kwargs)

    def reset(self, W_ff):
        """
        Intializes recurrent weight matrix.
        W_f: feedforward weights from sensory networks
        """
        super(RandomSynapticNetwork, self).reset(W_other=W_ff, W_rec=None)

    def forward(self, s_sens, s_rec, p_sens, s_ext):
        """
        Forward pass for random network.
        s_sens: Activity of neurons in sensory network
        s_rand: Activity of neurons in random network in previous timestep
        """
        return super(RandomSynapticNetwork, self).forward(s_other=s_sens, s_ext=s_ext, s_rec=s_rec, p_other=p_sens)
