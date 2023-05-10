from ..misc import NOT_PARALLEL, DomainCoordinates, COMM, MathDummy
from .. import adjoint_method as am
from .grid import AutoDiffMaterial3D, HybridMaterial3D, AutoDiffMaterial2D, HybridMaterial2D
from . import fdfd
from functools import partial

import numpy as np
import torch

__author__ = "Sean Hooten"
__copyright__ = 'Copyright 2023 Hewlett Packard Enterprise Development LP.'
__license__ = "BSD-3"
__maintainer__ = "Sean Hooten"
__status__ = "development"

### Generator for Topology Adjoint Method Classes ###
# Doing this in lieu of individually instantiated Topology classes (the same methods would be overriden).
# For better accordance with the style of the rest of EMopt, this behavior may change in the future
# Please see additional convenience Topology AM classes below

def _TopologyGenerator(BaseAM):
    # NOTE THIS IS STILL UNDER DEVELOPMENT, POTENTIALLY INCORRECT SOLUTIONS
    assert issubclass(BaseAM, am.AdjointMethod)
    class _Topology(BaseAM):
        def __init__(self, sim, step=None, domain=None, update_mu=False, eps_bounds=None, mu_bounds=None, planar=None, lam=0):
            super().__init__(sim, step=step)
            try:
                Z = self.sim.Z
                dz = self.sim.dz
                self._mode = 3
            except:
                Z = 0
                dz = 1.0
                self._mode = 2

            self._full_domain = DomainCoordinates(0, self.sim.X, 0, self.sim.Y, 0, Z, self.sim.dx, self.sim.dy, dz)
            self._domain = domain if domain is not None else self._full_domain
            self._update_mu = update_mu
            self._planar = planar
            self._lam = lam

            if self._mode == 2:
                self._dom_slices = (self._domain.j, self._domain.k)
                self._GridMat = partial(GridMaterial2D, self.sim.N, self.sim.M)
            else:
                self._dom_slices = (self._domain.i, self._domain.j, self._domain.k)
                self._GridMat = partial(GridMaterial3D, self.sim.Nx, self.sim.Ny, self.sim.Nz)

            self._epsb = eps_bounds
            self._mub = mu_bounds

        def get_params(self):
            self._full_grid_eps = self.sim.eps.get_values_in(self._full_domain, squeeze=True)
            self._grid_eps = self.sim.eps.get_values_in(self._domain, squeeze=True)

            self._full_grid_mu = None
            self._grid_mu = None

            if self._epsb:
                #params = inverse_scaled_sigmoid(self._grid_eps.real, self._epsb[0], self._epsb[1]).ravel()
                params = inverse_scaled_sigmoid(self._grid_eps.real, self._epsb[0], self._epsb[1])
            else:
                #params = np.copy(self._grid_eps.real.ravel())
                params = np.copy(self._grid_eps.real)

            if self._planar is not None:
                params = np.mean(params, axis=self._planar).ravel()
            else:
                params = params.ravel()


            if self._update_mu:
                self._full_grid_mu = self.sim.mu.get_values_in(self._full_domain, squeeze=True)
                self._grid_mu = self.sim.mu.get_values_in(self._domain, squeeze=True)
                if self._mub:
                    #params_mu = inverse_scaled_sigmoid(self._grid_mu.real, self._mub[0], self._mub[1]).ravel()
                    params_mu = inverse_scaled_sigmoid(self._grid_mu.real, self._mub[0], self._mub[1])
                    #params =  np.concatenate([params, params_mu], axis=0)
                else:
                    params_mu = np.copy(self._grid_mu.real)
                    #params =  np.concatenate([params, self._grid_mu.real.ravel()], axis=0)

                if self._planar is not None:
                    params_mu = np.mean(params_mu, axis=self._planar).ravel()
                else:
                    params_mu = params_mu.ravel()

                params = np.concatenate([params, params_mu], axis=0)

            return params

        def update_system(self, params):
            full_grid_eps = self._full_grid_eps
            full_grid_mu = self._full_grid_mu

            if self._update_mu:
                peps = params[:self._grid_eps.size].reshape(self._grid_eps.shape)
                pmu = params[self._grid_eps.size:].reshape(self._grid_mu.shape)

                if self._epsb:
                    full_grid_eps[self._dom_slices] = scaled_sigmoid(peps, self._epsb[0], self._epsb[1])
                else:
                    full_grid_eps[self._dom_slices] = peps

                if self._mub:
                    full_grid_mu[self._dom_slices] = scaled_sigmoid(pmu, self._mub[0], self._mub[1])
                else:
                    full_grid_mu[self._dom_slices] = pmu

                eps = self._GridMat(full_grid_eps)
                mu = self._GridMat(full_grid_mu)
                self.sim.set_materials(eps, mu)
            else:
                peps = params.reshape(self._grid_eps.shape)
                if self._epsb:
                    full_grid_eps[self._dom_slices] = scaled_sigmoid(peps, self._epsb[0], self._epsb[1])
                else:
                    full_grid_eps[self._dom_slices] = peps
                eps = self._GridMat(full_grid_eps)
                self.sim.set_materials(eps, self.sim.mu)

        def calc_gradient(self, sim, params):
            if NOT_PARALLEL:
                if self._update_mu:
                    if self._epsb:
                        delta_eps = self._epsb[1] - self._epsb[0]
                        #sig_eps = params.reshape(self._grid_eps.shape)
                        sig_eps = sigmoid(params[:self._grid_eps.size].reshape(self._grid_eps.shape))
                        sig_eps = (sig_eps, delta_eps)
                    else:
                        sig_eps = None

                    if self._mub:
                        delta_mu = self._mub[1] - self._mub[0]
                        #sig_mu = params[self._grid_eps.size:].reshape(self._grid_mu.shape)
                        sig_mu = sigmoid(params[self._grid_eps.size:].reshape(self._grid_mu.shape))
                        sig_mu = (sig_mu, delta_mu)
                    else:
                        sig_mu = None
                else:
                    sig_mu = None

                    if self._epsb:
                        delta_eps = self._epsb[1] - self._epsb[0]
                        #sig_eps = params.reshape(self._grid_eps.shape)
                        sig_eps = sigmoid(params.reshape(self._grid_eps.shape))
                        sig_eps = (sig_eps, delta_eps)
                    else:
                        sig_eps = None
            else:
                sig_eps = None
                sig_mu = None

            gradient = sim.calc_ydAx_topology(self._domain, self._update_mu, sig_eps=sig_eps, sig_mu=sig_mu, lam=self._lam)
            return gradient
    return _Topology

### CONVENIENCE TOPOLOGY AM CLASSES ###
Topology = _TopologyGenerator(am.AdjointMethod)
TopologyPNF2D = _TopologyGenerator(am.AdjointMethodPNF2D)
TopologyPNF3D = _TopologyGenerator(am.AdjointMethodPNF3D)
TopologyFM2D = _TopologyGenerator(am.AdjointMethodFM2D)
### CONVENIENCE TOPOLOGY AM CLASSES ###

### Generator for AutoDiff Adjoint Method Classes ###
# Doing this in lieu of individually instantiated AutoDiff classes (the same methods would be overriden).
# For better accordance with the style of the rest of EMopt, this behavior may change in the future
# Please see additional convenience AutoDiff AM classes below


def _AutoDiffGenerator(BaseAM):
    # Generator for Autodiff class
    assert issubclass(BaseAM, am.AdjointMethod)
    class _AutoDiff(BaseAM):
        # The implementation is surprisingly elegant.
        # no need for user to create "update_system" class method, only need to define one of: AutogradGridMaterialXD or HybridMaterialXD.
        def __init__(self, sim, domain=None, update_mu=False):
            super().__init__(sim)

            assert isinstance(self.sim.eps, AutoDiffMaterial3D) or isinstance(self.sim.eps, HybridMaterial3D) or \
                   isinstance(self.sim.eps, AutoDiffMaterial2D) or isinstance(self.sim.eps, HybridMaterial2D)

            try:
                Z = self.sim.Z
                dz = self.sim.dz
                self._mode = 3
                self._shifts_e = [[-0.5, 0.0, 0.5], [-0.5, 0.5, 0.0], [0.0, 0.0, 0.0]] # written as [sz,sy,sx], written in order of Ex, Ey, Ez
                self._shifts_m = [[0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [-0.5, 0.5, 0.5]] # written as [sz,sy,sx], written in order of Hx, Hy, Hz
                ds = [self.sim.dz, self.sim.dy, self.sim.dx]
            except:
                Z = 0
                dz = 1.0
                self._mode = 2
                if isinstance(sim, fdfd.FDFD_TE):
                    self._shifts_e = [[0, 0],] # written as [sy,sx] pairs
                    self._shifts_m = [[0, -0.5], [0.5, 0]] # written as [sy,sx] pairs
                elif isinstance(sim, fdfd.FDFD_TM):
                    self._shifts_e = [[0, -0.5], [0.5, 0]] # written as [sy,sx] pairs
                    self._shifts_m = [[0, 0],] # written as [sy,sx] pairs
                ds = [self.sim.dy, self.sim.dx]

            self._shifts_e_ds = [[self._shifts_e[i][j] * ds[j] for j in range(len(ds))] for i in range(len(self._shifts_e))]
            self._shifts_m_ds = [[self._shifts_m[i][j] * ds[j] for j in range(len(ds))] for i in range(len(self._shifts_m))]

            full_domain = DomainCoordinates(0, self.sim.X, 0, self.sim.Y, 0, Z, self.sim.dx, self.sim.dy, dz)
            self._domain = domain if domain is not None else full_domain
            assert isinstance(self._domain, DomainCoordinates)
            self._update_mu = update_mu

            xt = torch.as_tensor(self._domain.x)
            yt = torch.as_tensor(self._domain.y)
            if self._mode == 3:
                zt = torch.as_tensor(self._domain.z)
                self._coords = [zt, yt, xt]
            else:
                self._coords = [yt, xt]

        def update_system(self, params):
            if NOT_PARALLEL:
                self._v = torch.as_tensor(params).requires_grad_()
                eps = []
                for i in range(len(self._shifts_e)):
                    coords = [self._coords[j] + self._shifts_e_ds[i][j] for j in range(len(self._coords))]
                    ep = self.sim.eps._func(self._v, coords)
                    eps.append(ep)
                self._eps_autograd = [eps[i] + 0j for i in range(len(eps))]

                if self._update_mu:
                    mus = []
                    for i in range(len(self._shifts_m)):
                        coords = [self._coords[j] + self._shifts_m_ds[i][j] for j in range(len(self._coords))]
                        mu = self.sim.mu._func(self._v, coords)
                        mus.append(mu)
                    self._mu_autograd = [mus[i] + 0j for i in range(len(mus))]
                else:
                    self._mu_autograd = None

            else:
                self._eps_autograd = MathDummy(); self._mu_autograd = MathDummy()

            v = torch.as_tensor(params)
            self.sim.eps.v = v
            if self._update_mu:
                self.sim.mu.v = v

        def calc_gradient(self, sim, params):
            pseudoloss = sim.calc_ydAx_autograd(self._domain, self._update_mu, self._eps_autograd, self._mu_autograd)
            if NOT_PARALLEL:
                gradient = torch.autograd.grad(pseudoloss, self._v)[0].cpu().numpy()
            else:
                gradient = None
            gradient = COMM.bcast(gradient, root=0)
            return gradient

    return _AutoDiff


### CONVENIENCE AUTODIFF AM CLASSES ###
AutoDiff = _AutoDiffGenerator(am.AdjointMethod)
AutoDiffFM2D = _AutoDiffGenerator(am.AdjointMethodFM2D)
AutoDiffPNF2D = _AutoDiffGenerator(am.AdjointMethodPNF2D)
AutoDiffPNF3D = _AutoDiffGenerator(am.AdjointMethodPNF3D)
### CONVENIENCE AUTODIFF AM CLASSES ###

### CONVENIENCE FUNCTIONS ###
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def scaled_sigmoid(x, lb, ub):
    return lb + (ub-lb)*sigmoid(x)

def inverse_scaled_sigmoid(x, lb, ub):
    arg = (x - lb)/(ub-lb)
    assert np.all(arg>=0.0) and np.all(arg<=1.0)
    return np.log(arg/(1.0-arg))
