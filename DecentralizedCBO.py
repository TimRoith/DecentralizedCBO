from cbx.dynamics import CBXDynamic
from cbx.dynamics.cbo import cbo_update
import numpy as np

def ring(num_nodes, lam = .5):
    if num_nodes == 1:
        return np.ones((1, 1))
    if num_nodes == 2:
        return np.array([[lam, 1 - lam], [1 - lam, lam]])

    W =  lam * np.eye(num_nodes) + (1 - lam) / 2 * (np.eye(num_nodes, k=1) + np.eye(num_nodes, k=-1))
    W[0, -1] = (1 - lam) / 2
    W[-1, 0] = (1 - lam) / 2
    return W

def identity(num_nodes):
    return np.eye(num_nodes)

def full(num_nodes):
    return np.ones((num_nodes, num_nodes)) / num_nodes

def star(num_nodes):
    W = np.eye(num_nodes) * (num_nodes - 1) / num_nodes
    W[0, :] = 1.0 / num_nodes
    W[:, 0] = 1.0 / num_nodes
    W[0, 0] = 1.0 / num_nodes
    return W

def ring_2hop(num_nodes, lam=0.5):
    I = np.eye(num_nodes)
    near = np.roll(I, 1, axis=1) + np.roll(I, -1, axis=1)
    far  = np.roll(I, 2, axis=1) + np.roll(I, -2, axis=1)
    return lam * I + (1 - lam) / 4 * (near + far)

class DecentralizedCBO(CBXDynamic):
    def __init__(self, f, W = None, lamda_c_drift=0., noise_drift_mode = 'c',
                 compute_z_mode = 'tracking', gamma = 1.0,
                 update_mixing_freq = 1,
                 **kwargs) -> None:
        super().__init__(f, **kwargs)
        self.W = np.eye(self.M) if W is None else W
        self.consensus = np.zeros((self.M, 1,) +  self.d)
        self.z = self.copy(self.consensus)[:, 0, ...]
        self.lamda_c_drift = lamda_c_drift
        self.noise_drift_mode = noise_drift_mode
        self.gamma = gamma
        self.update_mixing_freq = update_mixing_freq

        if compute_z_mode == 'tracking':
            self.compute_z = self.compute_z_tracking
        elif compute_z_mode == 'consensus':
            self.compute_z = self.compute_z_consensus

    def cbo_step(self,):
        self.consensus_old = self.copy(self.consensus)
        self.compute_consensus()
        self.compute_z()

        c_drift    = self.x[self.particle_idx] - self.consensus
        z_drift    = self.x[self.particle_idx] - self.z[:, None,...] + self.lamda_c_drift * c_drift
        self.drift = c_drift if self.noise_drift_mode == 'c' else z_drift
        self.x[self.particle_idx] += cbo_update(
            z_drift, self.lamda, self.dt,
            self.sigma, self.noise()
        )

    def compute_z_tracking(self):
        if self.it % self.update_mixing_freq == 0:
            Wz = self.W @ self.z
        else:
            Wz = self.z
        self.z = Wz + (self.consensus - self.consensus_old)[:, 0, ...]

    def compute_z_consensus(self):
        c      = self.consensus[None, self.active_runs_idx, 0, ...]   # (1, M, d)
        energy = self.f(c)                                             # (1, M)
        self.num_f_eval[self.active_runs_idx] += 1
        z_global, _ = self._compute_consensus(energy, c, self.alpha[0:1, :])
        C_global = z_global[0, 0, :]                                   # (d,)
        self.z = ((1 - self.gamma) * self.consensus[:, 0, :]
                  + self.gamma     * C_global[None, :])                # (M, d)

    inner_step = cbo_step
