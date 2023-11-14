from functools import partial
import numpy as np
import torch

from ..misc import NOT_PARALLEL, DomainCoordinates, COMM, MathDummy
from .. import adjoint_method as am

from .grid import AutoDiffMaterial3D, HybridMaterial3D, AutoDiffMaterial2D, \
                  HybridMaterial2D, TopologyMaterial2D, TopologyMaterial3D
from . import fdfd
from . import fdtd

__author__ = "Sean Hooten"
__copyright__ = 'Copyright 2023 Hewlett Packard Enterprise Development LP.'
__license__ = "BSD-3"
__maintainer__ = "Sean Hooten"
__status__ = "development"

##########################################################################################
# AutoDiff Enhanced Optimization
##########################################################################################
def _AutoDiffGenerator(BaseAM: am.AdjointMethod):
    """Class generator for AutoDiff-enhanced optimization. The generator takes one of the
    base AdjointMethod classes from emopt.adjoint_method and returns a derived class that
    makes the same assumptions as the base class (e.g. power normalization 
    (AdjointMethodPNF2D/PNF3D)).

    Doing this in lieu of individually instantiated AutoDiff classes for each version, since
    the same methods are overriden in the same way for each case. Please see the convenience
    AutoDiff AM classes below, that can be used in the same way as the conventional AM classes,
    e.g., AutoDiff, AutoDiffFM2D, AutoDiffPNF2D, AutoDiffPNF3D.

    Parameters
    ----------
    BaseAM : emopt.adjoint_method.AdjointMethod
        Takes a conventional emopt AdjointMethod class.

    Returns
    -------
    emopt.adjoint_method.AdjointMethod
        Returns an AdjointMethod derived class, with a few methods overriden for use with
        AutoDiff enhanced optimization. These new classes can be instantiated very similarly
        to the conventional AdjointMethod classes. See below.
    """
    assert issubclass(BaseAM, am.AdjointMethod)

    class _AutoDiff(BaseAM):
        """AutoDiff enhanced adjoint method class.

        This class can be used very similarly to the classes in emopt.adjoint_method.
        The main difference is that this class overrides the update_system(...) method,
        the user should **NOT** define update_system(...) themselves. Instead, the simulation
        update functionality has been moved to be performed in one of following classes
        available in emopt.experimental.grid: AutoDiffMaterial2D, AutoDiffMaterial3D,
        HybridMaterial2D, or HybridMaterial3D. There, the user should provide a 
        PyTorch compatible callable that takes in coordinates and structural parameters
        as input, and outputs the simulation material distribution grid as output. The
        Material instance can then be provided directly to one of the new simulation objects
        in emopt.experimental.fdfd or emopt.experimental.fdtd in the usual way. We have
        provided several convenient AutoDiff-compatible shape primitive definitions and
        effective logic operations in emopt.experimental.autodiff_geometry which can
        be used to construct the material function callable.

        Parameters
        ----------
        sim : emopt.experimental.fdfd or emopt.experimental.fdtd
            Simulation object that has calc_ydAx_autograd method defined.
        domain : emopt.misc.DomainCoordinates
            A subdomain of the simulation where the material distribution is updated.
            If None, it will use sim.eps._fd, which will be defined if the user uses
            a emopt.grid.HybridMaterialXD class. Otherwise, uses the full simulation
            domain. Default = None.
        update_mu : bool
            If True, will also consider updates to the permeability in the optimization.
            In which case, sim.mu will need to be defined by one of the AutoDiff compatible
            classes in emopt.experimental.grid. Default = False.

        Attributes
        ----------
        sim : emopt.experimental.fdfd or emopt.experimental.fdtd
            Simulation object that has calc_ydAx_autograd method defined.
        """
        def __init__(self, 
                     sim, 
                     domain: DomainCoordinates = None, 
                     update_mu: bool = False
                     ):
            super().__init__(sim)

            # Check if using compatible simulation and material definitions
            assert isinstance(self.sim, fdfd.FDFD_TE) or isinstance(self.sim, fdfd.FDFD_TM) or \
                   isinstance(self.sim, fdfd.FDFD_3D) or isinstance(self.sim, fdtd.FDTD)

            assert isinstance(self.sim.eps, AutoDiffMaterial3D) or isinstance(self.sim.eps, HybridMaterial3D) or \
                   isinstance(self.sim.eps, AutoDiffMaterial2D) or isinstance(self.sim.eps, HybridMaterial2D)

            # Check if 2D or 3D
            try:
                Z = self.sim.Z
                dz = self.sim.dz
                self._mode = 3
                self._shifts_e = [[-0.5, 0.0, 0.5], [-0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]
                                  # written as [sz,sy,sx], written in order of Ex, Ey, Ez
                self._shifts_m = [[0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [-0.5, 0.5, 0.5]] 
                                  # written as [sz,sy,sx], written in order of Hx, Hy, Hz
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

            # Define staggered grid coordinate shifts
            self._shifts_e_ds = [[self._shifts_e[i][j] * ds[j] for j in range(len(ds))] for i in range(len(self._shifts_e))]
            self._shifts_m_ds = [[self._shifts_m[i][j] * ds[j] for j in range(len(ds))] for i in range(len(self._shifts_m))]

            # Get domain for update
            full_domain = DomainCoordinates(0, self.sim.X, 0, self.sim.Y, 0, Z, self.sim.dx, self.sim.dy, dz)
            if domain is None:
                if isinstance(self.sim.eps, HybridMaterial2D) or isinstance(self.sim.eps, HybridMaterial3D):
                    self._domain = self.sim.eps._fd
                else:
                    self._domain = full_domain
            else:
                self._domain = domain

            assert isinstance(self._domain, DomainCoordinates)

            if update_mu:
                assert isinstance(self.sim.mu, AutoDiffMaterial3D) or isinstance(self.sim.mu, HybridMaterial3D) or \
                       isinstance(self.sim.mu, AutoDiffMaterial2D) or isinstance(self.sim.mu, HybridMaterial2D)
            self._update_mu = update_mu

            # Get coordinates of domain
            xt = torch.as_tensor(self._domain.x)
            yt = torch.as_tensor(self._domain.y)
            if self._mode == 3:
                zt = torch.as_tensor(self._domain.z)
                self._coords = [zt, yt, xt]
            else:
                self._coords = [yt, xt]

        def update_system(self, params: np.ndarray):
            """Update the geometry/material distributions of the system.
            DO NOT OVERRIDE, this is taken care of automatically by making using
            the user's custom function defined in sim.eps / sim.mu.

            Parameters
            ----------
            params : numpy.ndarray
                1D array containing design parameter values (one value per design
                parameter)
            """
            # The way this works is we take the parameter vector, convert it to
            # PyTorch Tensor, enable the computational graph, then pass it through
            # user-defined Material distribution function along with coordinates.
            # The resulting eps/mu are then reduced with the forward/adjoint fields
            # in the calc_gradient method below, and then we simply backpropagate 
            # the sensitivities to the parameter vector. 
            # Here, we have to be somewhat careful about staggered grid coordinates,
            # and furthermore, we just do the full eps / mu forward calculation directly
            # on the head node. This might be improved through MPI, but seems to be
            # quite fast in test cases. Fruthermore, doing things with MPI might
            # prevent us from implementing this on GPU easily.
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

            # On all nodes, we pass the current parameter vector to the material
            # instances. We do not turn on the computational graph, because this
            # will just be used to update the material grids for the simulation
            # objects.
            v = torch.as_tensor(params)
            self.sim.eps.v = v
            if self._update_mu:
                self.sim.mu.v = v

        def calc_gradient(self, sim, params: np.ndarray):
            """Compute the gradient by Automatic Differentiation.
            This assumes that the simulation object has calc_ydAx_autograd
            method defined.

            Parameters
            ----------
            params : numpy.ndarray
                1D array containing design parameter values (one value per design
                parameter)
            """
            # Everything we need was calculated in update_system above.
            pseudoloss = sim.calc_ydAx_autograd(self._domain, 
                                                self._update_mu, 
                                                self._eps_autograd, 
                                                self._mu_autograd)

            # We compute the gradient by reverse-mode AutoDiff on the headnode.
            # Then we convert it to numpy array to be used by the optimizer.
            if NOT_PARALLEL:
                gradient = torch.autograd.grad(pseudoloss, self._v)[0].cpu().numpy()
            else:
                gradient = None
            gradient = COMM.bcast(gradient, root=0)
            return gradient

    return _AutoDiff

#######################################
### CONVENIENCE AUTODIFF AM CLASSES ###
#######################################
# Use these just like the regular base classes,
# but with AutoDiff functionality
AutoDiff = _AutoDiffGenerator(am.AdjointMethod)
AutoDiffFM2D = _AutoDiffGenerator(am.AdjointMethodFM2D)
AutoDiffPNF2D = _AutoDiffGenerator(am.AdjointMethodPNF2D)
AutoDiffPNF3D = _AutoDiffGenerator(am.AdjointMethodPNF3D)
##########################################################################################


##########################################################################################
# Topology Optimization
##########################################################################################
def _TopologyGenerator(BaseAM):
    """Class generator for topology optimization. The generator takes one of the
    base AdjointMethod classes from emopt.adjoint_method and returns a derived class that
    makes the same assumptions as the base class (e.g. power normalization 
    (AdjointMethodPNF2D/PNF3D)).

    Doing this in lieu of individually instantiated Topology classes for each version, since
    the same methods are overriden in the same way for each case. Please see the convenience
    Topology AM classes below, that can be used in the same way as the conventional AM classes,
    e.g., Topology, TopologyFM2D, TopologyPNF2D, TopologyPNF3D.

    Parameters
    ----------
    BaseAM : emopt.adjoint_method.AdjointMethod
        Takes a conventional emopt AdjointMethod class.

    Returns
    -------
    emopt.adjoint_method.AdjointMethod
        Returns an AdjointMethod derived class, with a few methods overriden for use with
        topology optimization. These new classes can be instantiated very similarly
        to the conventional AdjointMethod classes. See below.
    """
    assert issubclass(BaseAM, am.AdjointMethod)

    class _Topology(BaseAM):
        """Topology adjoint method class.

        This class can be used very similarly to the classes in emopt.adjoint_method.
        The main difference is that this class overrides the update_system(...) method,
        the user should **NOT** define update_system(...) themselves. This class will
        automatically update the simulation material distribution. The user simply needs
        to define the material distributions with one of the following classes available
        in emopt.experimental.grid: TopologyMaterial2D, TopologyMaterial3D.
        The Material instance can then be provided directly to one of the new simulation objects
        in emopt.experimental.fdfd or emopt.experimental.fdtd in the usual way. Note that
        we currently do not take into account unique materials at staggered grid coordinates.

        Parameters
        ----------
        sim : emopt.experimental.fdfd or emopt.experimental.fdtd
            Simulation object that has calc_ydAx_topology method defined.
        domain : emopt.misc.DomainCoordinates
            A subdomain of the simulation where the material distribution is updated.
            If None, it will use sim.eps._fd, which will be defined if the user uses
            a emopt.grid.TopologyXD class. Otherwise, uses the full simulation
            domain. Default = None.
        update_mu : bool
            If True, will also consider updates to the permeability in the optimization.
            In which case, sim.mu will need to be defined by one of the Topology compatible
            classes in emopt.experimental.grid. Default = False.
        eps_bounds : tuple
            2-tuple of floats defining lower and upper bounds of eps. If None, will take
            eps to be unbounded. Default = None.
        mu_bounds : tuple
            2-tuple of floats defining lower and upper bounds of mu. If None, will take
            mu to be unbounded. Default = None.
        planar : bool
            If True, will assume that the optimized geometry should be planar extruded about
            domain. This is common for optimizations of, e.g., integrated photonic devices.
            If the simulation is 2D, it will extrude along the y direction (non-trivial features
            in the x direction). If the simulation is 3D, it will extrude along the z direction
            (non trivial features in the xy plane). If False, all pixels in domain will
            be independent degrees-of-freedom. Default = False.
        vol_penalty : float
            Can be used to penalize spurious features in the topology-optimized design. 
            If > 0, it will tend to drive the design towards the lower bound of eps_bounds.
            If < 0, it will tend to drive the design towards the upper bound of eps_bounds.
            Use carefully, it will trade-off with the design objective.

        Methods
        -------
        get_params()
            This should be called by the user after constructing the Topology AM object.
            It gets a parameter vector that can be passed to the emopt optimizer.

        Attributes
        ----------
        sim : emopt.experimental.fdfd or emopt.experimental.fdtd
            Simulation object that has calc_ydAx_topology method defined.
        """
        def __init__(self, 
                     sim, 
                     domain: DomainCoordinates = None, 
                     update_mu: bool = False, 
                     eps_bounds: tuple = None, 
                     mu_bounds: tuple = None, 
                     planar: bool = False, 
                     vol_penalty: float = 0.
                     ):
            super().__init__(sim)

            assert isinstance(self.sim, fdfd.FDFD_TE) or isinstance(self.sim, fdfd.FDFD_TM) or \
                   isinstance(self.sim, fdfd.FDFD_3D) or isinstance(self.sim, fdtd.FDTD)

            assert isinstance(self.sim.eps, TopologyMaterial2D) or isinstance(self.sim.eps, TopologyMaterial3D)

            # Check if 2D or 3D
            try:
                Z = self.sim.Z
                dz = self.sim.dz
            except:
                Z = 0
                dz = 1.0

            # Get domain for update
            self._domain = domain if domain is not None else self.sim.eps._fd
            assert isinstance(self._domain, DomainCoordinates)

            if update_mu:
                assert isinstance(self.sim.mu, TopologyMaterial2D) or isinstance(self.sim.mu, TopologyMaterial3D)
            self._update_mu = update_mu

            self._epsb = eps_bounds
            self._mub = mu_bounds
            self._planar = planar
            self._lam = vol_penalty

        def get_params(self, squish=0.05):
            """Get the current parameter vector, can be passed to the emopt optimizer.

            Parameters
            ----------
            squish : float
                If initial design has material values that are exactly equal to the bounds,
                then errors will result. Squish is a percentage of (upper bound - lower bound)
                used to adjust the material values before inverting the sigmoid, to avoid
                numerical issues of this kind.

            Returns
            -------
            np.ndarray
                Parameter vector (will be large, the size of the designable grid)
            """
            self._grid_eps = np.copy(self.sim.eps.grid)
            self._gse = self._grid_eps.shape

            if self._epsb:
                params = inverse_scaled_sigmoid(self._grid_eps.real, self._epsb[0], self._epsb[1],
                                                squish=squish)
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
                    params_mu = inverse_scaled_sigmoid(self._grid_mu.real, self._mub[0], self._mub[1],
                                                       squish=squish)
                else:
                    params_mu = np.copy(self._grid_mu.real)

                if self._planar:
                    params_mu = np.mean(params_mu, axis=0).ravel()
                else:
                    params_mu = params_mu.ravel()

                params = np.concatenate([params, params_mu], axis=0)
                self._psm = params_mu.size

            return params

        def update_system(self, params: np.ndarray):
            """Update the geometry/material distributions of the system.
            DO NOT OVERRIDE, this is taken care of automatically.

            Parameters
            ----------
            params : numpy.ndarray
                1D array containing design parameter values (one value per design
                parameter)
            """
            if self._update_mu:
                peps = params[:self._pse]
                pmu = params[self._pse:]
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

        def calc_gradient(self, sim, params: np.ndarray):
            """Compute the topology optimization gradient.
            This assumes that the simulation object has calc_ydAx_topology
            method defined.

            Parameters
            ----------
            params : numpy.ndarray
                1D array containing design parameter values (one value per design
                parameter)
            """
            # Currently we just do this on the head node.
            # Note that we do not consider staggered grid coordinates currently
            # We need to handle a whole bunch of cases.
            if NOT_PARALLEL:
                if self._update_mu:
                    if self._epsb:
                        del_eps = self._epsb[1] - self._epsb[0]
                        if self._planar:
                            sig_eps = sigmoid(params[:self._pse].reshape(self._gse[1:]))
                        else:
                            sig_eps = sigmoid(params[:self._pse].reshape(self._gse))
                        self._sig_eps = np.copy(sig_eps)
                        sig_eps = sig_eps * (1.0 - sig_eps)
                    else:
                        del_eps = 1
                        sig_eps = np.array([1])
                        self._sig_eps = sig_eps

                    if self._mub:
                        del_mu = self._mub[1] - self._mub[0]
                        if self._planar:
                            sig_mu = sigmoid(params[self._pse:].reshape(self._gsm[1:]))
                        else:
                            sig_mu = sigmoid(params[self._pse:].reshape(self._gsm))
                        sig_mu = sig_mu * (1.0 - sig_mu)
                    else:
                        del_mu = 1
                        sig_mu = np.array([1])
                else:
                    del_mu = 1
                    sig_mu = 1

                    if self._epsb:
                        del_eps = self._epsb[1] - self._epsb[0]
                        if self._planar:
                            sig_eps = sigmoid(params.reshape(self._gse[1:]))
                        else:
                            sig_eps = sigmoid(params.reshape(self._gse))
                        self._sig_eps = np.copy(sig_eps)
                        sig_eps = sig_eps * (1.0 - sig_eps)
                    else:
                        del_eps = 1
                        sig_eps = np.array([1])
                        self._sig_eps = sig_eps
            else:
                sig_eps = np.array([1])
                sig_mu = np.array([1])
                del_eps = np.array([1])
                del_mu = np.array([1])
                self._sig_eps = 1

            gradient = sim.calc_ydAx_topology(self._domain, 
                                              self._update_mu, 
                                              sig_eps=sig_eps, 
                                              sig_mu=sig_mu, 
                                              del_eps=del_eps, 
                                              del_mu=del_mu, 
                                              planar=self._planar, 
                                              lam=self._lam)
            return gradient

        def calc_penalty(self, sim, params):
            """
            Calculate the penalty term associated with
            vol_penalty. The user should call this via
            super() if they desire to override calc_penalty.
            The penalty value can be accessed with
            self.current_vol_penalty

            Parameters
            ----------
            sim : emopt.experimental.fdfd and emopt.experimental.fdtd
                Simulation object
            params : numpy.ndarray
                1D array containing design parameter values (one value per design
                parameter)
            """
            lam = self._lam
            if lam == 0:
                self.current_vol_penalty = 0
            else:
                self.current_vol_penalty = lam * np.mean(sigmoid(params))
            return self.current_vol_penalty

        def calc_grad_p(self, sim, params):
            # this is computed directly in calc_gradient,
            # no need to update.
            return np.zeros_like(params)

    return _Topology


#######################################
### CONVENIENCE AUTODIFF AM CLASSES ###
#######################################
# Use these just like the regular base classes,
# but with AutoDiff functionality
Topology = _TopologyGenerator(am.AdjointMethod)
TopologyFM2D = _TopologyGenerator(am.AdjointMethodFM2D)
TopologyPNF2D = _TopologyGenerator(am.AdjointMethodPNF2D)
TopologyPNF3D = _TopologyGenerator(am.AdjointMethodPNF3D)
##########################################################################################

#######################################
### CONVENIENCE functions
#######################################
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def scaled_sigmoid(x, lb, ub):
    return lb + (ub-lb)*sigmoid(x)

def inverse_scaled_sigmoid(x, lb, ub, squish=0.05):
    # Give ourselves a bit of breathing room
    # for initialization.
    # This effectively will squish the output
    # variables to something finite.
    delt = ub - lb
    ubb = ub + squish*delt
    lbb = lb - squish*delt
    #arg = (x - lb)/(ub-lb)
    arg = (x - lbb)/(ubb-lbb)
    assert np.all(arg>=0.0) and np.all(arg<=1.0)
    return np.log(arg/(1.0-arg))