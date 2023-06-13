from ..misc import NOT_PARALLEL, DomainCoordinates, COMM, MathDummy
from .. import adjoint_method as am
from .grid import AutoDiffMaterial3D, HybridMaterial3D, AutoDiffMaterial2D, HybridMaterial2D, TopologyMaterial2D, TopologyMaterial3D
from . import fdfd
from functools import partial

import numpy as np
import torch

__author__ = "Sean Hooten"
__copyright__ = 'Copyright 2023 Hewlett Packard Enterprise Development LP.'
__license__ = "BSD-3"
__maintainer__ = "Sean Hooten"
__status__ = "development"


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

### Generator for Topology Adjoint Method Classes ###
# Doing this in lieu of individually instantiated Topology classes (the same methods would be overriden).
# For better accordance with the style of the rest of EMopt, this behavior may change in the future
# Please see additional convenience Topology AM classes below

def _TopologyGenerator(BaseAM):
    # Generator for Toplogy class
    assert issubclass(BaseAM, am.AdjointMethod)
    class _Topology(BaseAM):
        def __init__(self, sim, domain=None, update_mu=False, eps_bounds=None, mu_bounds=None, planar=False, area_penalty=0):
            super().__init__(sim)
            assert isinstance(self.sim.eps, TopologyMaterial2D) or isinstance(self.sim.eps, TopologyMaterial3D)

            try:
                Z = self.sim.Z
                dz = self.sim.dz
            except:
                Z = 0
                dz = 1.0

            full_domain = DomainCoordinates(0, self.sim.X, 0, self.sim.Y, 0, Z, self.sim.dx, self.sim.dy, dz)
            self._domain = domain if domain is not None else full_domain
            assert isinstance(self._domain, DomainCoordinates)
            self._update_mu = update_mu
            self._epsb = eps_bounds
            self._mub = mu_bounds
            self._planar = planar
            self._lam = area_penalty

        def get_params(self):
            self._grid_eps = np.copy(self.sim.eps.grid)
            self._gse = self._grid_eps.shape

            if self._epsb:
                params = inverse_scaled_sigmoid(self._grid_eps.real, self._epsb[0], self._epsb[1])
            else:
                params = np.copy(self._grid_eps.real)

            if self._planar:
                params = np.mean(params, axis=0).ravel()
            else:
                params = params.ravel()
            self._pse = params.size

            if self._update_mu:
                self._grid_mu = np.copy(self.sim.mu.grid)
                self._gsm = self._grid_mu.shape
                if self._mub:
                    params_mu = inverse_scaled_sigmoid(self._grid_mu.real, self._mub[0], self._mub[1])
                else:
                    params_mu = np.copy(self._grid_mu.real)

                if self._planar:
                    params_mu = np.mean(params_mu, axis=0).ravel()
                else:
                    params_mu = params_mu.ravel()

                params = np.concatenate([params, params_mu], axis=0)
                self._psm = params_mu.size

            return params

        def update_system(self, params):
            if self._update_mu:
                peps = params[:self._pse]
                pmu = params[self._pse:]
                #peps = params[:self._grid_eps.size].reshape(self._grid_eps.shape)
                #pmu = params[self._grid_eps.size:].reshape(self._grid_mu.shape)
                if self._epsb:
                    new_grid_eps = scaled_sigmoid(peps, self._epsb[0], self._epsb[1])
                else:
                    new_grid_eps = peps

                if self._mub:
                    new_grid_mu = scaled_sigmoid(pmu, self._mub[0], self._mub[1])
                else:
                    new_grid_mu = pmu

                if self._planar:
                    new_grid_eps = new_grid_eps.reshape(self._gse[1:])
                    new_grid_mu = new_grid_mu.reshape(self._gsm[1:])
                    self.sim.eps.grid = np.broadcast_to(new_grid_eps[np.newaxis, ...], self._gse)
                    self.sim.mu.grid = np.broadcast_to(new_grid_mu[np.newaxis, ...], self._gsm)
                else:
                    self.sim.eps.grid = new_grid_eps.reshape(self._gse)
                    self.sim.mu.grid = new_grid_mu.reshape(self._gsm)
            else:
                peps = params

                if self._epsb:
                    new_grid_eps = scaled_sigmoid(peps, self._epsb[0], self._epsb[1])
                else:
                    new_grid_eps = peps

                if self._planar:
                    new_grid_eps = new_grid_eps.reshape(self._gse[1:])
                    self.sim.eps.grid = np.broadcast_to(new_grid_eps[np.newaxis, ...], self._gse)
                else:
                    self.sim.eps.grid = new_grid_eps.reshape(self._gse)

        def calc_gradient(self, sim, params):
            if NOT_PARALLEL:
                if self._update_mu:
                    if self._epsb:
                        del_eps = self._epsb[1] - self._epsb[0]
                        #sig_eps = sigmoid(params[:self._grid_eps.size].reshape(self._grid_eps.shape))
                        if self._planar:
                            sig_eps = sigmoid(params[:self._pse].reshape(self._gse[1:]))
                            #sig_eps = sigmoid(np.broadcast_to(params[:self._pse].reshape(self._gse[1:])[np.newaxis, ...], self._gse))
                        else:
                            sig_eps = sigmoid(params[:self._pse].reshape(self._gse))
                        #sig_eps = (sig_eps, delta_eps)
                        #sig_eps = delta_eps * sig_eps * (1.0 - sig_eps)
                        self._sig_eps = np.copy(sig_eps)
                        sig_eps = sig_eps * (1.0 - sig_eps)
                    else:
                        del_eps = 1
                        sig_eps = np.array([1])
                        self._sig_eps = sig_eps

                    if self._mub:
                        del_mu = self._mub[1] - self._mub[0]
                        #sig_mu = sigmoid(params[self._grid_eps.size:].reshape(self._grid_mu.shape))
                        if self._planar:
                            #sig_mu = sigmoid(np.broadcast_to(params[self._pse:].reshape(self._gsm[1:])[np.newaxis, ...], self._gsm))
                            sig_mu = sigmoid(params[self._pse:].reshape(self._gsm[1:]))
                        else:
                            sig_mu = sigmoid(params[self._pse:].reshape(self._gsm))
                        #sig_mu = (sig_mu, delta_mu)
                        #sig_mu = delta_mu * sig_mu * (1.0 - sig_mu)
                        sig_mu = sig_mu * (1.0 - sig_mu)
                    else:
                        del_mu = 1
                        #sig_mu = 1
                        sig_mu = np.array([1])
                else:
                    del_mu = 1
                    sig_mu = 1

                    if self._epsb:
                        del_eps = self._epsb[1] - self._epsb[0]
                        if self._planar:
                            #sig_eps = sigmoid(np.broadcast_to(params.reshape(self._gse[1:])[np.newaxis, ...], self._gse))
                            sig_eps = sigmoid(params.reshape(self._gse[1:]))
                        else:
                            sig_eps = sigmoid(params.reshape(self._gse))
                        #sig_eps = (sig_eps, delta_eps)
                        #sig_eps = delta_eps * sig_eps * (1.0 - sig_eps)
                        self._sig_eps = np.copy(sig_eps)
                        sig_eps = sig_eps * (1.0 - sig_eps)
                    else:
                        del_eps = 1
                        #sig_eps = 1
                        sig_eps = np.array([1])
                        self._sig_eps = sig_eps
            else:
                #sig_eps = 1
                #sig_mu = 1
                #del_eps = 1
                #del_mu = 1
                #self._sig_eps = sig_eps
                #sig_eps = MathDummy()
                #sig_mu = MathDummy()
                #del_eps = MathDummy()
                #del_mu = MathDummy()
                #self._sig_eps = 1
                sig_eps = np.array([1])
                sig_mu = np.array([1])
                del_eps = np.array([1])
                del_mu = np.array([1])
                self._sig_eps = 1

            gradient = sim.calc_ydAx_topology(self._domain, self._update_mu, sig_eps=sig_eps, sig_mu=sig_mu, del_eps=del_eps, del_mu=del_mu, planar=self._planar, lam=self._lam)
            return gradient

        def calc_penalty(self, sim, params):
            lam = self._lam
            if lam == 0:
                return 0
            else:
                #return lam * np.mean(self._sig_eps)
                return lam * np.mean(sigmoid(params))

        def calc_grad_p(self, sim, params):
            # this is computed directly in calc_gradient
            return np.zeros_like(params)

    return _Topology


### CONVENIENCE TOPOLOGY AM CLASSES ###
Topology = _TopologyGenerator(am.AdjointMethod)
TopologyFM2D = _TopologyGenerator(am.AdjointMethodFM2D)
TopologyPNF2D = _TopologyGenerator(am.AdjointMethodPNF2D)
TopologyPNF3D = _TopologyGenerator(am.AdjointMethodPNF3D)
### CONVENIENCE TOPOLOGY AM CLASSES ###

### CONVENIENCE FUNCTIONS ###
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def scaled_sigmoid(x, lb, ub):
    return lb + (ub-lb)*sigmoid(x)

def inverse_scaled_sigmoid(x, lb, ub):
    arg = (x - lb)/(ub-lb)
    assert np.all(arg>=0.0) and np.all(arg<=1.0)
    return np.log(arg/(1.0-arg))