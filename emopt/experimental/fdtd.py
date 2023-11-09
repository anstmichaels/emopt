"""
This module defines derived classes of the FDTD solvers in emopt.fdfd for
use with new experimental topology and AutoDiff-enhanced optimization methods
available in emopt.experimental.adjoint_method. It enables: support for
functionally-defined material distributions, and improved calculation of
the adjoint variables method gradient using backpropagation.
Note: currently requires PyTorch for correct functionality.

Examples
--------
See emopt/examples/experimental/ for detailed examples.
"""
from .. import fdtd
from ..misc import NOT_PARALLEL

import numpy as np
import torch

__author__ = "Sean Hooten"
__copyright__ = 'Copyright 2023 Hewlett Packard Enterprise Development LP.'
__license__ = "BSD-3"
__maintainer__ = "Sean Hooten"
__status__ = "development"

class FDTD(fdtd.FDTD):
    """Derived class to simulate Maxwell's equations in 3D.
    Please see emopt.fdtd.FDTD for full documentation (it is used
    the same way, except implements some additional methods for topology /
    AutoDiff-enhanced optimization).
    This class should be used for simulations that use either of the following:
        emopt.experimental.grid classes for simulation material distributions
        emopt.experimental.adjoint_method classes for inverse design
    """
    def __init__(self, *args, **kwargs):
        super(FDTD, self).__init__(*args, **kwargs)

    def calc_ydAx_topology(self,
            domain: DomainCoordinates,
            update_mu: bool,
            sig_eps: np.ndarray = None,
            sig_mu: np.ndarray = None,
            del_eps: float = 1.,
            del_mu: float = 1.,
            planar: bool = False,
            lam: float = 0.) -> np.ndarray:
        """Calculates gradient = -2 * Re(y^T dA/dp * x) for topology optimizations.

        The gradient for bounded topology optimization can be expressed, more
        specifically, as:
        grad = 2 * omega * (sig_eps * del_eps * Im(E o E^adj) - sig_mu * del_mu * Im(H o H^adj))
        where sig_eps = the derivative of sigmoid of variables for permittivity
              sig_mu = the derivative of sigmoid of variables for permeability
              o = the Hadamard (element-wise) product.
              del_eps = the permittivity range (max - min)
              del_mu = the permeability range (max - min)
        This class also allows for planar devices (e.g. for compatibility with
        photolithography). Furthermore, one may use a penalty multiplier lam,
        which penalizes spurious features (ultimately, lam>0 tries to guide designs
        towards features with the lower bound of the permittivity or permeability).

        NOTE: Below is not the most efficent way to be doing things,
              we should be doing multiplication with PETSc vectors for
              parallelization. Currently performing multiplication only on head node
              Something like this would be improved (nonworking code):
                  x = self.x
                  y = self.x_adj
                  product = PETSc.Vec().create()
                  product = product.pointwiseMult(x,y)

        NOTE: Currently we assume that staggered grid coordinates have same material
        value as grid center.

        Parameters
        ----------
        domain : DomainCoordinates
            The designable grid domain.

        update_mu : bool
            Use True if mu is also designable. Default = False.

        sig_eps : np.ndarray
            The derivative of sigmoid of the design parameters for permittivity
            (should take the same shape local designable grid as the field arrays).
            If None, assumes that the problem is unbounded. Default = None.

        sig_mu : np.ndarray
            The derivative of sigmoid of the design parameters for permeability
            (should take the same shape local designable grid as the field arrays).
            If None, assumes that the problem is unbounded. Default = None.
            Note: if update_mu=False, this will be ignored.

        del_eps : float
            The difference between maximum and minimum permittivity for bi-level
            bounded topology optimization.

        del_mu : float
            The difference between maximum and minimum permeability for bi-level
            bounded topology optimization.

        planar : float
            If True, assumes that the local designable region should be constrained
            to extruded shapes (common for integrated photonics). In 3D, this assumes
            that the structure is effectively 2D extruded. Currently assumes that the
            structure has nontrivial features in the x-y directions.
            Default = False.

        lam : float
            A penalty value that can be used to penalize spurious design features.
            If lam > 0, drives the design towards features with lower bound of
            permittivity/permeability. Use carefully, it will trade-off with the
            true design objective. Default = 0.

        Returns
        -------
        np.ndarray
            The gradient of the objective with respect to permittivity/permeability
            pixels.
        """

        Ex = self.get_field('Ex', domain)
        Ex_adj = self.get_adjoint_field('Ex', domain)
        Ey = self.get_field('Ey', domain)
        Ey_adj = self.get_adjoint_field('Ey', domain)
        Ez = self.get_field('Ez', domain)
        Ez_adj = self.get_adjoint_field('Ez', domain)

        if planar:
            grad_eps = sig_eps * (lam/sig_eps.size + \
                    2 * del_eps * np.imag(np.sum(Ex * Ex_adj, axis=0) + \
                                          np.sum(Ey * Ey_adj, axis=0) + \
                                          np.sum(Ez * Ez_adj, axis=0)))
        else:
            grad_eps = sig_eps * (lam/sig_eps.size + \
                    2* del_eps * np.imag(Ex * Ex_adj + Ey * Ey_adj + Ez * Ez_adj))

        if update_mu:
            Hx = self.get_field('Hx', domain)
            Hx_adj = self.get_adjoint_field('Hx', domain)
            Hy = self.get_field('Hy', domain)
            Hy_adj = self.get_adjoint_field('Hy', domain)
            Hz = self.get_field('Hz', domain)
            Hz_adj = self.get_adjoint_field('Hz', domain)

            if planar:
                grad_mu = sig_mu * (lam/sig_mu.size - \
                        2 * del_mu * np.imag(np.sum(Hx * Hx_adj, axis=0) + \
                                             np.sum(Hy * Hy_adj, axis=0) + \
                                             np.sum(Hz * Hz_adj, axis=0)))
            else:
                grad_mu = sig_mu * (lam/sig_mu.size - \
                        2 * del_mu * np.imag(Hx * Hx_adj + Hy * Hy_adj + Hz * Hz_adj))

            grad = np.concatenate([grad_eps.ravel(), grad_mu.ravel()], axis=0)
        else:
            grad = grad_eps.ravel()

        return grad

    def calc_ydAx_autograd(self,
            domain: DomainCoordinates,
            update_mu: bool,
            eps_autograd: list,
            mu_autograd: list) -> torch.tensor:
        """Calculates pseudo_FOM = 2 * sum(Im(eps o E o E^adj) - Im(mu o H o H^adj))
        for reverse-mode AutoDiff enhanced optimizations.

        NOTE: Below is not the most efficent way to be doing things,
              we should be doing multiplication with PETSc vectors for
              parallelization. Currently performing multiplication only on head node
              Something like this would be improved (nonworking code):
                  x = self.x
                  y = self.x_adj
                  product = PETSc.Vec().create()
                  product = product.pointwiseMult(x,y)

        Parameters
        ----------
        domain : DomainCoordinates
            The designable grid domain.

        update_mu : bool
            Use True if mu is also designable. Default = False.

        eps_autograd : list of torch.tensor
            list of permittivity distribution arrays, defined at staggered coordinates in
            x,y,z. In principle, this should be sampled from a emopt.experimental.grid.
            AutoDiffMaterial3D object.

        mu_autograd : list of torch.tensor
            list of permeability distribution arrays, defined at staggered coordinates in
            x,y,z. In principle, this should be sampled from a emopt.experimental.grid.
            AutoDiffMaterial3D object.

        Returns
        -------
        torch.tensor
            pseudo_FOM for use in reverse-mode AutoDiff to compute the gradient.
        """
        Ex = torch.as_tensor(self.get_field('Ex', domain))
        Ex_adj = torch.as_tensor(self.get_adjoint_field('Ex', domain))
        Ey = torch.as_tensor(self.get_field('Ey', domain))
        Ey_adj = torch.as_tensor(self.get_adjoint_field('Ey', domain))
        Ez = torch.as_tensor(self.get_field('Ez', domain))
        Ez_adj = torch.as_tensor(self.get_adjoint_field('Ez', domain))
        pseudoloss = eps_autograd[0] * Ex * Ex_adj + \
                     eps_autograd[1] * Ey * Ey_adj + \
                     eps_autograd[2] * Ez * Ez_adj
        if update_mu:
            Hx = torch.as_tensor(self.get_field('Hx', domain))
            Hx_adj = torch.as_tensor(self.get_adjoint_field('Hx', domain))
            Hy = torch.as_tensor(self.get_field('Hy', domain))
            Hy_adj = torch.as_tensor(self.get_adjoint_field('Hy', domain))
            Hz = torch.as_tensor(self.get_field('Hz', domain))
            Hz_adj = torch.as_tensor(self.get_adjoint_field('Hz', domain))
            pseudoloss = pseudoloss - mu_autograd[0] * Hx * Hx_adj - \
                                      mu_autograd[1] * Hy * Hy_adj - \
                                      mu_autograd[2] * Hz * Hz_adj
        if NOT_PARALLEL:
            return 2 * pseudoloss.imag.sum()
        else:
            return
