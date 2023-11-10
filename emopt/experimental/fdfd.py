"""
This module defines derived classes of the FDFD solvers in emopt.fdfd for
use with new experimental topology and AutoDiff-enhanced optimization methods
available in emopt.experimental.adjoint_method. It enables: support for
functionally-defined material distributions, and improved calculation of
the adjoint variables method gradient using backpropagation.
Note: currently requires PyTorch for correct functionality.

Examples
--------
See emopt/examples/experimental/ for detailed examples.
"""
from .. import fdfd
from ..fdfd import PETSc
from ..misc import RANK, MathDummy, NOT_PARALLEL, DomainCoordinates, warning_message, info_message

import numpy as np
import torch

__author__ = "Sean Hooten"
__copyright__ = 'Copyright 2023 Hewlett Packard Enterprise Development LP.'
__license__ = "BSD-3"
__maintainer__ = "Sean Hooten"
__status__ = "development"

class FDFD_TE(fdfd.FDFD_TE):
    """Derived class to simulate Maxwell's equations in 2D with TE-polarized
    fields. Please see emopt.fdfd.FDFD_TE for full documentation (it is used
    the same way, except implements some additional methods for topology /
    AutoDiff-enhanced optimization).
    This class should be used for simulations that use either of the following:
        emopt.experimental.grid classes for simulation material distributions
        emopt.experimental.adjoint_method classes for inverse design
    """
    def __init__(self, *args, **kwargs):
        super(FDFD_TE, self).__init__(*args, **kwargs)

    def build(self):
        """(Re)Build the system matrix. This has been rewritten for improved
        speed at the expense of more memory usage. Basically instead of calling
        the permittivity/permeability grid objects for each individual
        coordinate, we instead vectorize the operation by calling update
        immediately after initializing the system matrix :math:`A`.
        """
        A = self._A
        M = self._M
        N = self._N
        eps = self._eps
        mu = self._mu
        bc = self._bc

        odx = self._R / self._dx
        ody = self._R / self._dy

        pml_x_Ez, pml_x_Hy = self.__get_pml_x()
        pml_y_Ez, pml_y_Hx = self.__get_pml_y()

        odx_Ez = odx * pml_x_Ez
        odx_Hy = odx * pml_x_Hy
        ody_Ez = ody * pml_y_Ez
        ody_Hx = ody * pml_y_Hx
        Nc = self.Nc

        if(self._eps == None or self._mu == None):
            raise Exception('The material distributions of the system must be \
                            initialized prior to building the system matrix.')

        if(self.verbose and NOT_PARALLEL):
            info_message('Building system matrix...')

        for i in range(self.ib, self.ie):

            ig = int(i/Nc)
            component = int(i - Nc*ig)
            y = int(ig/N)
            x = ig - y*N

            if(component == 0): # Jz row
                # relevant j coordinates
                jEz = ig*Nc
                jHx1 = ig*Nc + 1
                jHx0 = (ig-N)*Nc + 1
                jHy0 = ig*Nc + 2
                jHy1 = (ig+1)*Nc + 2

                # Diagonal element is the permittivity at (x,y)
                A[i,jEz] = 1j

                A[i,jHx1] = -ody_Ez[y,x]
                A[i,jHy0] = -odx_Ez[y,x]

                if(y > 0):
                    A[i,jHx0] = ody_Ez[y,x]

                if(x < N-1):
                    A[i,jHy1] = odx_Ez[y,x]

                #############################
                # enforce boundary conditions
                #############################
                if(y == 0):
                    if(bc[1] == '0'):
                        A[i,jHy0] = 0
                        if(x < N-1): A[i,jHy1] = 0
                    elif(bc[1] == 'E'):
                        A[i,jHx1] = -2*ody_Ez[y,x]
                    elif(bc[1] == 'H'):
                        A[i,jHx1] = 0
                    elif(bc[1] == 'P'):
                        jHx2 = (x + (M-1)*N)*Nc+1
                        A[i,jHx2] = ody_Ez[y,x]

                elif(y == M-1):
                    if(bc[1] == 'M'):
                        A[i,jHx1] = 0

                if(x == N-1):
                    if(bc[0] == '0'):
                        A[i,jHx1] = 0
                        if(y > 0): A[i,jHx0] = 0
                    elif(bc[0] == 'P'):
                        jHy2 = (ig-N-1)*Nc + 2
                        A[i,jHy2] = odx_Ez[y,x]
                elif(x == 0):
                    if(bc[0] == 'M'):
                        A[i,jHy0] = 0

            elif(component == 1): # Mx row
                # relevant j coordinates
                jHx = ig*Nc + 1
                jEz0 = ig*Nc
                jEz1 = (ig+N)*Nc

                j0 = i
                j1 = (y+1)*N + x + M*N

                # diagonal element is permeability at (x,y)
                # if(simple_mu): A[i, j0] = -1j
                A[i,jHx] = -1j

                A[i, jEz0] = -ody_Hx[y,x]

                if(y < M-1):
                    A[i,jEz1] = ody_Hx[y,x]

                #############################
                # enforce boundary conditions
                #############################
                if(y == 0):
                    if(bc[1] == '0'):
                        A[i,jEz0] = 0
                elif(y == M-1):
                    if(bc[1] == 'P'):
                        jEz2 = x*Nc
                        A[i,jEz2] = ody_Hx[y,x]
                    if(bc[1] == 'M'):
                        A[i, jEz0] = 0.0

                if(x == N-1):
                    if(bc[0] == '0'):
                        A[i,jEz0] = 0
                        if(y < M-1): A[i,jEz1] = 0
                elif(x == 0):
                    pass

            else: # My row
                # relevant j coordinates
                jHy = ig*Nc + 2
                jEz0 = (ig-1)*Nc
                jEz1 = ig*Nc

                j0 = i
                j1 = i-1

                # diagonal is permeability at (x,y)
                A[i,jHy] = -1j
                A[i,jEz1] = -odx_Hy[y,x]

                if(x > 0):
                    A[i,jEz0] = odx_Hy[y,x]

                #############################
                # enforce boundary conditions
                #############################
                if(y == 0):
                    if(bc[1] == '0'):
                        A[i,jEz1] = 0
                        if(x > 0): A[i,jEz0] = 0
                elif(y == M-1):
                    pass

                if(x == N-1):
                    if(bc[0] == '0'):
                        A[i,jEz1] = 0
                elif(x == 0):
                    if(bc[0] == 'E'):
                        A[i,jEz1] = 0
                    elif(bc[0] == 'H'):
                        A[i,jEz1] = -2*odx_Hy[y,x]
                    elif(bc[0] == 'P'):
                        jEz2 = (ig + N-1)*Nc
                        A[i,jEz2] = odx_Hy[y,x]

        # communicate off-processor values and setup internal data structures for
        # performing parallel operations
        A.assemblyBegin()
        A.assemblyEnd()

        self.update(bbox=None)

        self._built = True

    def update(self, bbox=None):
        """Update only the material values stored in A.
        See emopt.fdfd.FDFD_TE.update for full documentation.

        This has been rewritten for improved speed/convenience. The update is done
        in Python instead of calling any C code, but seems to be fast. Note this is
        currently memory inefficient, calculates the full grid on each MPI node,
        then downselects what it needs to update :math:`A` locally.
        """
        # implementing the update in python for simplicity
        A = self._A
        M = self._M
        N = self._N
        eps = self._eps
        mu = self._mu

        if(bbox == None):
            x1 = 0
            x2 = self._N
            y1 = 0
            y2 = self._M
        else:
            x1 = int(np.floor(bbox[0]/self._W*N))
            x2 = int(np.ceil(bbox[1]/self._W*N))
            y1 = int(np.floor(bbox[2]/self._H*M))
            y2 = int(np.ceil(bbox[3]/self._H*M))

        A_update = self.A_diag_update

        eps_values = 1j*eps.get_values(x1, x2, y1, y2, sx=0, sy=0).ravel()
        mux_values = -1j*mu.get_values(x1, x2, y1, y2, sx=0, sy=0.5).ravel()
        muy_values = -1j*mu.get_values(x1, x2, y1, y2, sx=-0.5, sy=0).ravel()

        big_array = np.empty(eps_values.shape[0]*3, dtype=np.complex128)
        big_array[0::3] = eps_values
        big_array[1::3] = mux_values
        big_array[2::3] = muy_values

        A_update[:] = big_array[self.ib:self.ie]

        self._workvec.setValues(np.arange(self.ib, self.ie, dtype=np.int32), A_update)
        A.setDiagonal(self._workvec, addv=PETSc.InsertMode.INSERT_VALUES)

        # communicate off-processor values and setup internal data structures for
        # performing parallel operations
        A.assemblyBegin()
        A.assemblyEnd()

    def calc_ydAx_topology(self,
                           domain: DomainCoordinates,
                           update_mu: bool,
                           sig_eps: np.ndarray = None,
                           sig_mu: np.ndarray = None,
                           del_eps: float = 1.,
                           del_mu: float = 1.,
                           planar: bool = False,
                           lam: float = 0.
                           ) -> np.ndarray:
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
            to extruded shapes (common for integrated photonics). In 2D, this assumes
            that the structure is effectively 1D extruded. Currently assumes that the
            structure has nontrivial features in the x direction.
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

        Ez = self.get_field('Ez', domain)
        Ez_adj = self.get_adjoint_field('Ez', domain)

        if planar:
            grad_eps = sig_eps * (lam/sig_eps.size + \
                    2 * del_eps * np.imag(np.sum(Ez * Ez_adj, axis=0)))
        else:
            grad_eps = sig_eps * (lam/sig_eps.size + \
                    2 * del_eps * np.imag(Ez * Ez_adj))

        if update_mu:
            Hx = self.get_field('Hx', domain)
            Hy = self.get_field('Hy', domain)
            Hx_adj = self.get_adjoint_field('Hx', domain)
            Hy_adj = self.get_adjoint_field('Hy', domain)

            if planar:
                grad_mu = sig_mu * (lam/sig_mu.size - \
                        2 * del_mu * (np.imag(np.sum(Hx * Hx_adj, axis=0) + \
                                              np.sum(Hy * Hy_adj, axis=0))))
            else:
                grad_mu =  sig_mu * (lam/sig_mu.size - \
                        2 * del_mu * (np.imag(Hx * Hx_adj + Hy * Hy_adj)))

            grad = np.concatenate([grad_eps.ravel(), grad_mu.ravel()], axis=0)
        else:
            grad = grad_eps.ravel()

        return grad

    def calc_ydAx_autograd(self,
                           domain: DomainCoordinates,
                           update_mu: bool,
                           eps_autograd: list,
                           mu_autograd: list
                           ) -> torch.Tensor:
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

        eps_autograd : list of torch.Tensor
            list of The permittivity distribution arrays, in this case defined at grid
            center for Ez component. Index [0] corresponds to Ez component. In principle,
            this should be sampled from a emopt.experimental.grid.AutoDiffMaterial2D object.

        mu_autograd : list of torch.Tensor
            list of permeability distribution arrays, defined at staggered coordinates
            in x and y. Index [0] corresponds to Hy component, index [1] corresponds
            to Hx component. In principle, these should be sampled from a
            emopt.experimental.grid.AutoDiffMaterial2D object.

        Returns
        -------
        torch.Tensor
            pseudo_FOM for use in reverse-mode AutoDiff to compute the gradient.
        """
        Ez = torch.as_tensor(self.get_field('Ez', domain))
        Ez_adj = torch.as_tensor(self.get_adjoint_field('Ez', domain))
        pseudoloss = eps_autograd[0] * Ez * Ez_adj
        if update_mu:
            Hx = torch.as_tensor(self.get_field('Hx', domain))
            Hx_adj = torch.as_tensor(self.get_adjoint_field('Hx', domain))
            Hy = torch.as_tensor(self.get_field('Hy', domain))
            Hy_adj = torch.as_tensor(self.get_adjoint_field('Hy', domain))
            pseudoloss = pseudoloss - mu_autograd[1] * Hx * Hx_adj - mu_autograd[0] * Hy * Hy_adj
        return 2 * pseudoloss.imag.sum()

class FDFD_TM(fdfd.FDFD_TM):
    """Derived class to simulate Maxwell's equations in 2D with TM-polarized
    fields. Please see emopt.fdfd.FDFD_TM for full documentation (it is used
    the same way, except implements some additional methods for topology /
    AutoDiff-enhanced optimization).
    This class should be used for simulations that use either of the following:
        emopt.experimental.grid classes for simulation material distributions
        emopt.experimental.adjoint_method classes for inverse design
    """
    def __init__(self, *args, **kwargs):
        super(FDFD_TM, self).__init__(*args, **kwargs)

    def build(self):
        FDFD_TE.build(self)

    def update(self, bbox=None):
        FDFD_TE.update(self, bbox=bbox)

    def calc_ydAx_topology(self,
                           domain: DomainCoordinates,
                           update_mu: bool,
                           sig_eps: np.ndarray = None,
                           sig_mu: np.ndarray = None,
                           del_eps: float = 1.,
                           del_mu: float = 1.,
                           planar: bool = False,
                           lam: float = 0.
                           ) -> np.ndarray:
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
            to extruded shapes (common for integrated photonics). In 2D, this assumes
            that the structure is effectively 1D extruded. Currently assumes that the
            structure has nontrivial features in the x direction.
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

        if planar:
            grad_eps = sig_eps * (lam/sig_eps.size + \
                    2 * del_eps * np.imag(np.sum(Ex * Ex_adj, axis=0) + \
                                          np.sum(Ey * Ey_adj, axis=0)))
        else:
            grad_eps = sig_eps * (lam/sig_eps.size + \
                    2 * del_eps * np.imag(Ex * Ex_adj + Ey * Ey_adj))

        if update_mu:
            Hz = self.get_field('Hz', domain)
            Hz_adj = self.get_adjoint_field('Hz', domain)

            if planar:
                grad_mu = sig_mu * (lam/sig_mu.size - \
                        2 * del_mu * np.imag(np.sum(Hz * Hz_adj, axis=0)))
            else:
                grad_mu = sig_mu * (lam/sig_mu.size - \
                        2 * del_mu * np.imag(Hz * Hz_adj))

            grad = np.concatenate([grad_eps.ravel(), grad_mu.ravel()], axis=0)
        else:
            grad = grad_eps.ravel()

        return grad

    def calc_ydAx_autograd(self,
                           domain: DomainCoordinates,
                           update_mu: bool,
                           eps_autograd: list,
                           mu_autograd: list
                           ) -> torch.Tensor:
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

        eps_autograd : list of torch.Tensor
            list of permittivity distribution arrays, defined at staggered coordinates
            in x and y. Index [0] corresponds to Ey component, index[1] corresponds
            to Ex component. In principle, this should be sampled from a
            emopt.experimental.grid.AutoDiffMaterial2D object.

        mu_autograd : list of torch.Tensor
            list of permeability distribution arrays, in this case defined at grid
            center for Hz component. Index [0] corresponds to Hz component. In principle,
            this should be sampled from a emopt.experimental.grid.AutoDiffMaterial2D object.

        Returns
        -------
        torch.Tensor
            pseudo_FOM for use in reverse-mode AutoDiff to compute the gradient.
        """
        Ex = torch.as_tensor(self.get_field('Ex', domain))
        Ex_adj = torch.as_tensor(self.get_adjoint_field('Ex', domain))
        Ey = torch.as_tensor(self.get_field('Ey', domain))
        Ey_adj = torch.as_tensor(self.get_adjoint_field('Ey', domain))
        pseudoloss = eps_autograd[1] * Ex * Ex_adj + eps_autograd[0] * Ey * Ey_adj
        if update_mu:
            Hz = torch.as_tensor(self.get_field('Hz', domain))
            Hz_adj = torch.as_tensor(self.get_adjoint_field('Hz', domain))
            pseudoloss = pseudoloss - mu_autograd[0] * Hz * Hz_adj
        return 2*pseudoloss.imag.sum()

class FDFD_3D(fdfd.FDFD_3D):
    """Derived class to simulate Maxwell's equations in 3D. Please see
    emopt.fdfd.FDFD_3D for full documentation (it is used the same way, except
    implements some additional methods for topology / AutoDiff-enhanced optimization).
    This class should be used for simulations that use either of the following:
        emopt.experimental.grid classes for simulation material distributions
        emopt.experimental.adjoint_method classes for inverse design

    NOTE: Currently untested. Recommended to try emopt.experimental.fdtd.FDTD
    """
    def __init__(self, *args, **kwargs):
        super(FDFD_3D, self).__init__(*args, **kwargs)

    def calc_ydAx_topology(self,
                           domain: DomainCoordinates,
                           update_mu: bool,
                           sig_eps: np.ndarray = None,
                           sig_mu: np.ndarray = None,
                           del_eps: float = 1.,
                           del_mu: float = 1.,
                           planar: bool = False,
                           lam: float = 0.
                           ) -> np.ndarray:
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
                           mu_autograd: list
                           ) -> torch.Tensor:
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

        eps_autograd : list of torch.Tensor
            list of permittivity distribution arrays, defined at staggered coordinates in
            x,y,z. In principle, this should be sampled from a emopt.experimental.grid.
            AutoDiffMaterial3D object.

        mu_autograd : list of torch.Tensor
            list of permeability distribution arrays, defined at staggered coordinates in
            x,y,z. In principle, this should be sampled from a emopt.experimental.grid.
            AutoDiffMaterial3D object.

        Returns
        -------
        torch.Tensor
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
