"""Solve for the modes of electromagnetic waveguides in 2D and 3D.

Waveguide modes can be computed by setting up a generalized eigenvalue problem
corresponding to the source-free Maxwell's equations assuming a solution to
:math:`\mathbf{E}` and :math:`\mathbf{H}` which is proportional to :math:`e^{i
k_z z}`, i.e.

.. math::
    \\nabla \\times e^{i k_z z} \mathbf{E} - i \\mu_r \\omega
    e^{i k_z z}\\mathbf{H} = 0

    \\nabla \\times e^{i k_z z} \mathbf{H} + i \\epsilon_r \\omega
    e^{i k_z z}\\mathbf{E} = 0

where we have used the non-dimensionalized Maxwell's equations. These equations
can be written in the form

.. math::
    A x = n_z B x

where :math:`A` contains the discretized curls and material values, :math:`B` is
singular matrix containing only 1s and 0s, and :math:`n_z` is the effective
index of the mode whose field components are contained in :math:`x`.  Although
formulating the problem like this results in a sparse matrix with ~2x the
number of values compared to other formulations discussed in the literature[1],
it has the great advantage that the equations remain very simple which
simplifies the code. This formulation also makes it almost trivial to implement
anisotropic materials (tensors) in the future, if desired.

In addition to solving for the fields of a waveguide's modes, we can also
compute the current sources which excite only that mode. This can be used in
conjunction with :class:`emopt.fdfd.FDFD` to simulated waveguide structures
which are particularly interesting for applications in silicon photonics, etc.

References
----------
[1] A. B. Fallahkhair, K. S. Li and T. E. Murphy, "Vector Finite Difference
Modesolver for Anisotropic Dielectric Waveguides", J. Lightwave Technol. 26(11),
1423-1431, (2008).
"""
# Initialize petsc first
import sys, slepc4py
slepc4py.init(sys.argv)

from defs import FieldComponent
from misc import info_message, warning_message, error_message, \
NOT_PARALLEL, run_on_master, MathDummy

import grid

from abc import ABCMeta, abstractmethod
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
import numpy as np

__author__ = "Andrew Michaels"
__license__ = "GPL License, Version 3.0"
__version__ = "0.2"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

class ModeSolver(object):
    """A generic interface for electromagnetic mode solvers.

    At a minimum, a mode solver must provide functions for solving for the
    modes of a structure, retrieving the fields of a desired mode, retrieving
    the effective index of a desired mode, and calculating the current sources
    which excite that mode.

    Attributes
    ----------
    wavelength : float
        The wavelength of the solved modes.
    neff : list of floats
        The list of solved effective indices
    n0 : float
        The effective index near which modes are found
    neigs : int
        The number of modes to solve for.

    Methods
    -------
    build(self)
        Build the system of equations and prepare the mode solver for the solution
        process.
    solve(self)
        Solve for the modes of the structure.
    get_field(self, i, component)
        Get the desired field component of the i'th mode.
    get_field_interp(self, i, component)
        Get the desired interpolated field component of the i'th mode
    get_source(self, i, ds1, ds2, ds3=0.0)
        Get the source current distribution for the i'th mode.
    """
    __metaclass__ = ABCMeta

    def __init__(self, wavelength, n0=1.0, neigs=1):
        self._neff = []
        self.n0 = n0
        self.neigs = neigs
        self.wavelength = wavelength

    @property
    def neff(self):
        return self._neff

    @neff.setter
    def neff(self, value):
        warning_message('neff cannot be set by the user.', \
                        module='emopt.modes')

    @abstractmethod
    def build(self):
        """Build the system of equations and prepare the mode solver for the
        solution process.
        """
        pass

    @abstractmethod
    def solve(self):
        """Solve for the fields of the desired modes.
        """
        pass

    @abstractmethod
    def get_field(self, i, component):
        """Get the raw field of the i'th mode.

        This function should only be called after :func:`solve`.

        Parameters
        ----------
        i : int
            The number of the desired mode
        component : str
            The desired field component.

        Returns
        -------
        numpy.ndarray
            (Master node only) The desired field component.
        """
        pass

    @abstractmethod
    def get_field_interp(self, i, component):
        """Get the interpolated field of the i'th mode.

        This function should only be called after :func:`solve`. In general,
        this field should be prefered over :func:`get_field`.

        Parameters
        ----------
        i : int
            The number of the desired mode
        component : str
            The desired field component.

        Returns
        -------
        numpy.ndarray
            (Master node only) The desired interpolated field component.
        """
        pass

    @abstractmethod
    def get_source(self, i, ds1, ds2, ds3=0.0):
        """Calculate the current source distribution which will excite the
        desired mode.

        The current source distribution can be computed by assuming the
        computed mode fields are proportional to :math:`e^{i k_z z}` and
        eminate from a 'virtual' plane (hence the fields are zero on one side
        of plane, and have the desired z-dependence on the other side).  This
        assumed field can be plugged into the source-containing Maxwell's
        equations to solve for :math`J` and :math:`M`.

        Parameters
        ----------
        i : int
            The index of the desired mode.
        ds1 : float
            The grid spacing in the first spatial dimension.
        ds2 : float
            The grid spacing in the second spatial dimension.
        ds3 : float
            The grid spacing in the third spatial dimension
        """
        pass

class ModeTE(ModeSolver):
    """Solve for the TE polarized modes of a 1D slice of a 2D structure.

    The TE polarization consists of a non-zeros :math:`E_z`, :math:`H_x`,
    :math:`H_y`. The mode is assumed to propagate in the x direction and the
    mode field is a function of the y-position, i.e. the fields are

    .. math::
        E_z(x,y) = E_{mz}(y) e^{i k_x x}

        H_x(x,y) = H_{mx}(y) e^{i k_x x}

        H_y(x,y) = H_{my}(y) e^{i k_x x}

    where :math:`E_{mz}`, :math:`H_{mx}`, and :math:`H_{my}` are the mode
    fields.

    Parameters
    ----------
    wavelength : float
        The wavelength of the modes.
    ds : float
        The grid spacing in the mode field (y) direction.
    eps : numpy.ndarray
        The array containing the slice of permittivity for which the modes are
        calculated.
    mu : numpy.ndarray
        The array containing the slice of permeabilities for which the modes are
        calculated.
    n0 : float (optional)
        The 'guess' for the effective index around which the modes are
        computed. In general, this value should be larger than the index of the
        mode you are looking for. (default = 1.0)
    neigs : int (optional)
        The number of modes to compute. (default = 1)
    backwards : bool
        Defines whether or not the mode propagates in the forward +x direction
        (False) or the backwards -x direction (True). (default = False)

    Attributes
    ----------
    wavelength : float
        The wavelength of the solved modes.
    neff : list of floats
        The list of solved effective indices
    n0 : float
        The effective index near which modes are found
    neigs : int
        The number of modes to solve for.
    dir : int
        Direction of mode propagation (1 = forward, -1 = backward)
    bc : str
        The boundary conditions used. The possible boundary conditions are:
            0 -- Perfect electric conductor (top and bottom)
            M -- Perfect magnetic conductor (top and bottom)
            E -- Electric field symmetry (bottom) and PEC (top)
            H -- Magnetic field symmetry (bottom) and PEC (top)
            P -- Periodicity (top and bottom)
            EM -- Electric field symmetry (bottom) PMC (top)
            HM -- Magnetic field symmetry (bottom) PMC (top)

    Methods
    -------
    build(self)
        Build the system of equations and prepare the mode solver for the solution
        process.
    solve(self)
        Solve for the modes of the structure.
    get_field(self, i, component)
        Get the desired raw field component of the i'th mode.
    get_field_interp(self, i, component)
        Get the desired interpolated field component of the i'th mode
    get_mode_number(self, i):
        Estimate the number X of the given TE_X mode.
    find_mode_index(self, X):
        Find the index of a TE_X mode with the desired X.
    get_source(self, i, ds1, ds2, ds3=0.0)
        Get the source current distribution for the i'th mode.
    """

    def __init__(self, wavelength, eps, mu, domain, n0=1.0, neigs=1, \
                 backwards=False):
        super(ModeTE, self).__init__(wavelength, n0, neigs)

        # Generated fields/source will be reshaped to match input eps
        self.domain = domain

        # figure out how 2D domain is oriented
        n = None
        if(domain.Nx == 1):
            n = 'x'
            self._N = domain.Ny
            self.ds = domain.dy
        elif(domain.Ny == 1):
            n = 'y'
            self._N = domain.Nx
            self.ds = domain.dx
        else:
            raise AttributeError('2D domains are not supported for 1D mode ' \
                                 'calculations!')
        self.ndir = n
        self._fshape = [domain.Ny, domain.Nx]
        self.eps = eps.get_values_in(domain, squeeze=True)
        self.mu = mu.get_values_in(domain, squeeze=True)

        if(backwards):
            self._dir = -1.0
        else:
            self._dir = 1.0

        self._bc = '0'

        # non-dimensionalization for spatial variables
        self.R = self.wavelength/(2*np.pi)

        # Solve problem of the form Ax = lBx
        # define A and B matrices here
        # factor of 3 due to 3 field components
        self._A = PETSc.Mat()
        self._A.create(PETSc.COMM_WORLD)
        self._A.setSizes([3*self._N, 3*self._N])
        self._A.setType('aij')
        self._A.setUp()

        self._B = PETSc.Mat()
        self._B.create(PETSc.COMM_WORLD)
        self._B.setSizes([3*self._N, 3*self._N])
        self._B.setType('aij')
        self._B.setUp()

        # setup the solver
        self._solver = SLEPc.EPS()
        self._solver.create()

        # we need to set up the spectral transformation so that it doesnt try
        # to invert 
        st = self._solver.getST()
        st.setType('sinvert')

        # Let's use MUMPS for any system solving since it is fast
        ksp = st.getKSP()
        #ksp.setType('gmres')
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverPackage('mumps')

        # setup vectors for the solution
        self._x = []
        self._neff = np.zeros(neigs, dtype=np.complex128)
        vr, wr = self._A.getVecs()
        self._x.append(vr)

        for i in range(neigs-1):
            self._x.append(self._x[0].copy())

        self._fields = [np.array([]) for i in range(self.neigs)]
        self._Ez = [np.zeros(self._N, dtype=np.complex128) for i in \
                    range(self.neigs)]
        self._Hx = [np.zeros(self._N, dtype=np.complex128) for i in \
                    range(self.neigs)]
        self._Hy = [np.zeros(self._N, dtype=np.complex128) for i in \
                    range(self.neigs)]

        ib, ie = self._A.getOwnershipRange()
        self.ib = ib
        self.ie = ie

    @property
    def dir(self):
        return self._dir

    @dir.setter
    def dir(self, new_dir):
        if(np.abs(new_dir) != 1):
            error_message('Direction must be 1 or -1 (forward or backwards).')

        self._dir = new_dir

    @property
    def bc(self):
        return self._bc

    @bc.setter
    def bc(self, bc):
        if(bc not in ['0', 'M', 'E', 'H', 'P', 'EM', 'HM']):
            error_message("Boundary condition type '%s' not found. Use "
                          "0, M, E, H, P, EM, HM." % (bc), "emopt.modes")

        self._bc = bc

    def build(self):
        """Build the system of equations and prepare the mode solver for the solution
        process.

        In order to solve for the eigen modes, we must first assemble the
        relevant matrices :math:`A` and :math:`B` for the generalized
        eigenvalue problem given by :math:`A x = n_x B x` where :math:`n_x` is
        the eigenvalue and :math:`x` is the vector containing the eigen modes.

        Notes
        -----
        This function is run on all nodes.
        """
        ds = self.ds/self.R # non-dimensionalize

        A = self._A
        B = self._B
        mu = self.mu
        eps = self.eps
        N = self._N

        for I in xrange(self.ib, self.ie):

            # (stuff) = n_x B H_y
            if(I < N):
                i = I
                y = I

                j0 = I
                j1 = I+N

                A[i,j0] = 1j*eps[j0]

                A[i,j1] = -1.0/ds
                if(j0 > 0):
                    A[i,j1-1] = 1.0/ds

                #############################
                # enforce boundary conditions
                #############################
                if(y == 0):
                    if(self._bc == 'E'):
                        A[i,j1] = -2.0/ds
                    elif(self._bc == 'H'):
                        A[i,j1] = 0.0
                    elif(self._bc == 'P'):
                        j2 = j1 + N-1
                        A[i,j2] = 1.0/ds
                elif(y == N-1):
                    if(self._bc == 'M' or self._bc == 'EM' or self._bc == 'HM'):
                        A[i,j1] = 0

            # (stuff) = n_x B H_x
            elif(I < 2*N):
                i = I
                y = I-N

                j0 = I
                j1 = I-N

                # the second or should be an and, but that results in a
                # singular matrix -- is this a code error or physics error?
                A[i,j0] = -1j*mu[j1]

                A[i,j1] = -1.0/ds

                if(j1 < N-1):
                    A[i,j1+1] = 1.0/ds

                #############################
                # enforce boundary conditions
                #############################
                if(y == 0):
                    if(self._bc == '0'):
                        A[i,j1] = 0
                elif(y == N-1):
                    if(self._bc == 'P'):
                        j2 = 0
                        A[i,j2] = 1.0/ds


            # (stuff) = n_x B E_z
            else:
                i = I
                y = I-2*N
                j0 = I

                A[i,j0] = 1j*mu[i-2*N]

                #############################
                # enforce boundary conditions
                #############################
                if(y == 0):
                    if(self._bc == '0'):
                        A[i,j0] = 0
                elif(y == N-1):
                    pass

        # Define B. It contains ones on the first and last third of the
        # diagonals
        for i in xrange(self.ib, self.ie):
            if(i < N):
                B[i,i+2*N] = -1j*self._dir # _dir=1 corresponds to exp(-ikx)
            elif(i < 2*N):
                B[i,i] = 0
            else:
                B[i,i-2*N] = -1j*self._dir

        self._A.assemble()
        self._B.assemble()

    def solve(self):
        """Solve for the modes of the structure.

        In addition to solving for the modes, this function saves the results
        to the master node so that they can be easily retrieved for
        visualization, etc.

        Notes
        -----
        This function is run on all nodes.
        """
        self._solver.setOperators(self._A, self._B)
        self._solver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        self._solver.setDimensions(self.neigs, PETSc.DECIDE)
        self._solver.setTarget(self.n0)
        self._solver.setFromOptions()

        self._solver.solve()
        nconv = self._solver.getConverged()

        if(nconv < self.neigs):
            warning_message('%d eigenmodes were requested, however only %d ' \
                            'eigenmodes were found.' % (self.neigs, nconv), \
                            module='emopt.modes')

        # nconv can be bigger than the desired number of eigen values
        if(nconv > self.neigs):
            neigs = self.neigs
        else:
            neigs = nconv

        for i in range(neigs):
            self.neff[i] = self._solver.getEigenvalue(i)
            self._solver.getEigenvector(i, self._x[i])

            # Save the full result on the master node so it can be accessed in the
            # future
            scatter, x_full = PETSc.Scatter.toZero(self._x[i])
            scatter.scatter(self._x[i], x_full, False, PETSc.Scatter.Mode.FORWARD)

            if(NOT_PARALLEL):
                self._fields[i] = x_full.getArray()
                field = self._fields[i]

                N = self._N

                self._Ez[i] = field[0:N]
                self._Hx[i] = field[N:2*N]
                self._Hy[i] = field[2*N:3*N]

                # unfortunate hacks for PMC :(
                if(self._bc == 'M'):
                    self._Hx[i][-1] = 0

    @run_on_master
    def get_field(self, i, component):
        """Get the desired raw field component of the i'th mode.

        Use this function with care: Ez/Hy and Hx are specified at different
        points in space (separated by half of a grid cell).  In general
        :func:`.ModeTE.get_field_interp` should be prefered.

        In general, you may wish to solve for more than one mode.  In order to
        get the desired mode, you must specify its index.  If you do not know
        the index but you do know the desired mode number, then
        :func:`.ModeTE.find_mode_index` may be used to determine the index of
        the desired mode.

        Notes
        -----
        This function only returns a non-None result on the master node. On all
        other nodes, None is returned.

        See Also
        --------
        :func:`.ModeTE.get_field_interp`

        :func:`.ModeTE.find_mode_index`

        Parameters
        ----------
        i : int
            The index of the desired mode
        component : str
            The desired field component (Ez, Hx, or Hy)

        Returns
        -------
        numpy.ndarray or None
            (Master node only) an array containing the desired component of the
            mode field.
        """
        # since we artificially extended the structure by one element during
        # initialization, we need to be careful to return fields of the
        # expected size, hence the [1:]
        if(component == 'Ez'):
            return np.reshape(self._Ez[i], self._fshape)
        elif(component == 'Hx'):
            return np.reshape(self._Hx[i], self._fshape)
        elif(component == 'Hy'):
            return np.reshape(self._Hy[i], self._fshape)
        else:
            raise ValueError('Unrecongnized field componenet "%s". The allowed'
                             'field components are Ez, Hx, Hy.' % (component))

    @run_on_master
    def get_field_interp(self, i, component):
        """Get the desired interpolated field component of the i'th mode.

        In general, this function should be preferred over
        :func:`.ModeTE.get_field`.

        In general, you may wish to solve for more than one mode.  In order to
        get the desired mode, you must specify its index.  If you do not know
        the index but you do know the desired mode number, then
        :func:`.ModeTE.find_mode_index` may be used to determine the index of
        the desired mode.

        Notes
        -----
        This function only returns a non-None result on the master node. On all
        other nodes, None is returned.

        See Also
        --------
        :func:`.ModeTE.get_field`

        :func:`.ModeTE.find_mode_index`

        Parameters
        ----------
        i : int
            The index of the desired mode
        component : str
            The desired field component (Ez, Hx, or Hy)

        Returns
        -------
        numpy.ndarray or None
            (Master node only) an array containing the desired component of the
            interpolated mode field.
        """
        # since we artificially extended the structure by one element during
        # initialization, we need to be careful to return fields of the
        # expected size, hence the [1:]
        if(component == 'Ez'):
            return np.reshape(self._Ez[i], self._fshape)
        elif(component == 'Hy'):
            return np.reshape(self._Hy[i], self._fshape)
        elif(component == 'Hx'):
            Hxi = np.pad(self._Hx[i], 1, 'constant', constant_values=0)
            Hxi[1:] += Hxi[0:-1]
            return np.reshape(Hxi[1:-1] / 2.0, self._fshape)
        else:
            raise ValueError('Unrecongnized field componenet "%s". The allowed'
                             'field components are Ez, Hx, Hy.' % (component))

    @run_on_master
    def get_mode_number(self, i):
        """Estimate the number X of the given TE_X mode.

        Often times, we will look for a specific TE_X mode where X is the
        number of the mode. Because of the way that the eigenvalue problem is
        solved, it is not known a priori which mode is found during the
        solution process.  In order to get around this, we can estimate which X
        a given solved mode corresponds to by looking at the number of phase
        steps in the electric field. To avoid weird phase errors that might
        appear due to the approximate numerical solution process, we use an
        thresholded amplitude-weighted phase process.

        Notes
        -----
        This function makes not guarantees that the X determined is meaningful.
        In particular, the solver may find non-physical modes whose number of
        phase crossings is equal to the desired TE_X mode.  In general, it is a
        good idea to visualize the mode to verify that it is infact the desired
        mode.

        Parameters
        ----------
        i : int
            The index of the mode to analyze.

        Returns
        -------
        int
            The number X of the specified TE_X mode.
        """
        Ez = self._Ez[i]
        if(self._bc == 'E' or self._bc == 'EM'):
            Ez = np.concatenate([Ez[::-1], Ez])
        if( self._bc == 'H' or self._bc == 'HM'):
            Ez = np.concatenate([-1*Ez[::-1], Ez])
        if(self._bc == 'P'):
            warning_message('get_mode_number may not work as expected for ' \
                            'periodic boundary conditions.', 'emopt.modes')

        dphase = 0.5
        thresh_frac = 0.05

        phase = np.angle(Ez)
        wphase = (phase - np.mean(phase))*np.abs(Ez)
        pthresh = np.max(np.abs(wphase))*thresh_frac

        wphase[wphase > pthresh] = 1.0
        wphase[wphase < -pthresh] = -1.0
        wphase[np.abs(wphase) < pthresh] = 0.0

        phase_crossings = np.sum(np.abs(np.diff(wphase)) > dphase)

        return int(phase_crossings/2 - 1)

    @run_on_master
    def find_mode_index(self, X):
        """Find the index of a TE_X mode with the desired X.

        This function makes no guarantees that the mode found is in fact a TE_X
        mode and not some other non-physical mode.  It is important to verify
        the result by checking its effective index or by visualizing it.

        Parameters
        ----------
        X : int
            The number of the desired mode.

        Returns
        -------
        int
            The index of the mode with the desired number.
        """
        for i in range(self.neigs):
            if(self.get_mode_number(i) == X):
                return i

        warning_message('Desired mode number was not found.', 'emopt.modes')
        return 0

    def get_source(self, i, dx, dy, dz=0.0):
        """Get the source current distribution for the i'th mode.

        Notes
        -----
        For this calculation to work out, we assume that all field components
        are zero to the left of the center of the Yee cell (i.e. the positions
        of the Ez values). To the right of the center of the Yee cell, we
        assume the field components have an exp(ikx) dependence.

        dy should be equal to ds.

        This class assumes all modes propagate in the x direction. In order to
        propagate a mode in the y direction, x and y (dx and dy) can be
        permuted.

        TODO
        ----
        Implement in parallelized manner.

        Parameters
        ----------
        i : int
            Index of the mode for which the corresponding current sources are
            desired.
        dx : float
            The grid spacing in the x direction.
        dy : float
            The grid spacing in the y direction.
        dz : float
            Unused in :class:`.ModeTE`

        Returns
        -------
        tuple of numpy.ndarray
            (On ALL nodes) The tuple (Jz, Mx, My) containing arrays of the
            source distributions.  In 2D, these source distributions are N x 1
            arrays.
        """
        N = self._N

        if(NOT_PARALLEL):
            Jz = np.zeros(N, dtype=np.complex128)
            Mx = np.zeros(N, dtype=np.complex128)
            My = np.zeros(N, dtype=np.complex128)

            # need to include boundary values
            Ez = np.pad(self._Ez[i], 1, 'constant', constant_values=0)
            Hx = np.pad(self._Hx[i], 1, 'constant', constant_values=0)
            Hy = np.pad(self._Hy[i], 1, 'constant', constant_values=0)

            # account for symmetry and periodic boundary conditions
            if(self._bc[0] == 'E'):
                Ez[0] = Ez[1]
                Hx[0] = -Hx[1]
                Hy[0] = Hy[1]
            elif(self._bc[0] == 'H'):
                Ez[0] = -Ez[1]
                Hx[0] = Hx[1]
                Hy[0] = -Hy[1]
            elif(self._bc[0] == 'P'):
                Ez[0] = Ez[-2]
                Ez[-1] = Ez[1]
                Hx[0] = Hx[-2]
                Hx[-1] = Hx[1]
                Hy[0] = Hy[-2]
                Hy[-1] = Hy[1]

            neff = self.neff[i]
            dx = dx/self.R # non-dimensionalize
            dy = dy/self.R # non-dimensionalize

            dHxdy = np.diff(Hx) / dy
            dHxdy = dHxdy[:-1]
            dHydx = Hy[1:-1]*np.exp(self._dir*1j*neff*dx/2.0) / dx
            dEzdy = np.diff(Ez)[1:] / dy
            dEzdx = Ez[1:-1] / dx

            Jz = 1j*(self.eps*Ez[1:-1]) + dHydx - dHxdy
            Mx = dEzdy - 1j*(self.mu*Hx[1:-1])
            My = -dEzdx

        else:
            Jz = None
            Mx = None
            My = None

        comm = MPI.COMM_WORLD
        Jz = comm.bcast(Jz, root=0)
        Mx = comm.bcast(Mx, root=0)
        My = comm.bcast(My, root=0)

        Jz = np.reshape(Jz, self._fshape)
        Mx = np.reshape(Mx, self._fshape)
        My = np.reshape(My, self._fshape)
        return (Jz, Mx, My)

class ModeTM(ModeTE):
    """Solve for the TM polarized modes of a 1D slice of a 2D structure.

    The TM polarization consists of a non-zeros :math:`H_z`, :math:`E_x`,
    :math:`E_y`. The mode is assumed to propagate in the x direction and the
    mode field is a function of the y-position, i.e. the fields are

    .. math::
        H_z(x,y) = H_{mz}(y) e^{i k_x x}

        E_x(x,y) = E_{mx}(y) e^{i k_x x}

        E_y(x,y) = E_{my}(y) e^{i k_x x}

    where :math:`H_{mz}`, :math:`E_{mx}`, and :math:`E_{my}` are the mode
    fields.

    Parameters
    ----------
    wavelength : float
        The wavelength of the modes.
    ds : float
        The grid spacing in the mode field (y) direction.
    eps : numpy.ndarray
        The array containing the slice of permittivity for which the modes are
        calculated.
    mu : numpy.ndarray
        The array containing the slice of permeabilities for which the modes are
        calculated.
    n0 : float (optional)
        The 'guess' for the effective index around which the modes are
        computed. In general, this value should be larger than the index of the
        mode you are looking for. (default = 1.0)
    neigs : int (optional)
        The number of modes to compute. (default = 1)
    backwards : bool
        Defines whether or not the mode propagates in the forward +x direction
        (False) or the backwards -x direction (True). (default = False)

    Attributes
    ----------
    wavelength : float
        The wavelength of the solved modes.
    neff : list of floats
        The list of solved effective indices
    n0 : float
        The effective index near which modes are found
    neigs : int
        The number of modes to solve for.

    Methods
    -------
    build(self)
        Build the system of equations and prepare the mode solver for the solution
        process.
    solve(self)
        Solve for the modes of the structure.
    get_field(self, i, component)
        Get the desired raw field component of the i'th mode.
    get_field_interp(self, i, component)
        Get the desired interpolated field component of the i'th mode
    get_mode_number(self, i):
        Estimate the number X of the given TE_X mode.
    find_mode_index(self, X):
        Find the index of a TE_X mode with the desired X.
    get_source(self, i, ds1, ds2, ds3=0.0)
        Get the source current distribution for the i'th mode.
    """

    def __init__(self, wavelength, eps, mu, domain, n0=1.0, neigs=1, \
                 backwards=False):

        # A TM mode source is the same as a TE mode source except with the
        # permittivity and permeability smapped and the E and H and J and M
        # components swapped around.
        super(ModeTM, self).__init__(wavelength, mu, eps, domain, n0, neigs, \
                                      backwards)

        self.bc = 'M' # really PEC since we use TE mode solver

    @property
    def bc(self):
        # since we use the TE solver, we internally swap E for H and 0 for M.
        # We need to unmix this up so the user isnt confused.
        bc = self._bc + ''
        if(len(bc) == 2):
            if(bc[0] == 'E'): return 'H'
            else: return 'E'
        else:
            if(bc[0] == 'E'): return 'HM'
            elif(bc[0] == 'H'): return 'EM'
            elif(bc[0] == '0'): return 'M'
            elif(bc[0] == 'M'): return '0'
            else: return bc

    @bc.setter
    def bc(self, bc):
        if(bc not in ['0', 'M', 'E', 'H', 'P', 'EM', 'HM']):
            error_message("Boundary condition type '%s' not found. Use "
                          "0, M, E, H, P, EM, HM." % (bc), "emopt.modes")

        # we need to swap Es and Hs and 0s and Ms since we use the TE solver to
        # find the TM fields
        if(len(bc) == 2):
            if(bc[0] == 'E'): self._bc = 'H'
            else: self._bc = 'E'
        else:
            if(bc[0] == 'E'): self._bc = 'HM'
            elif(bc[0] == 'H'): self._bc = 'EM'
            elif(bc[0] == '0'): self._bc = 'M'
            elif(bc[0] == 'M'): self._bc = '0'
            else: self._bc = bc

    @run_on_master
    def get_field(self, i, component):
        """Get the desired raw field component of the i'th mode.

        Use this function with care: Hz/Ey and Ex are specified at different
        points in space (separated by half of a grid cell).  In general
        :func:`.ModeTM.get_field_interp` should be prefered.

        In general, you may wish to solve for more than one mode.  In order to
        get the desired mode, you must specify its index.  If you do not know
        the index but you do know the desired mode number, then
        :func:`.ModeTM.find_mode_index` may be used to determine the index of
        the desired mode.

        Notes
        -----
        This function only returns a non-None result on the master node. On all
        other nodes, None is returned.

        See Also
        --------
        :func:`.ModeTM.get_field_interp`

        :func:`.ModeTM.find_mode_index`

        Parameters
        ----------
        i : int
            The index of the desired mode
        component : str
            The desired field component (Hz, Ex, or Ey)

        Returns
        -------
        numpy.ndarray or None
            (Master node only) an array containing the desired component of the
            mode field.
        """
        te_comp = ''
        if(component == 'Hz'): te_comp = 'Ez'
        elif(component == 'Ex'): te_comp = 'Hx'
        elif(component == 'Ey'): te_comp = 'Hy'
        else: te_comp = 'invalid'

        field = super(ModeTM, self).get_field(i, te_comp)

        if(component == 'Hz'):
            return field*-1
        else:
            return field

    @run_on_master
    def get_field_interp(self, i, component):
        """Get the desired interpolated field component of the i'th mode.

        In general, this function should be preferred over
        :func:`.ModeTM.get_field`.

        In general, you may wish to solve for more than one mode.  In order to
        get the desired mode, you must specify its index.  If you do not know
        the index but you do know the desired mode number, then
        :func:`.ModeTM.find_mode_index` may be used to determine the index of
        the desired mode.

        Notes
        -----
        This function only returns a non-None result on the master node. On all
        other nodes, None is returned.

        See Also
        --------
        :func:`.ModeTM.get_field`

        :func:`.ModeTM.find_mode_index`

        Parameters
        ----------
        i : int
            The index of the desired mode
        component : str
            The desired field component (Hz, Ex, or Ey)

        Returns
        -------
        numpy.ndarray or None
            (Master node only) an array containing the desired component of the
            interpolated mode field.
        """
        te_comp = ''
        if(component == 'Hz'): te_comp = 'Ez'
        elif(component == 'Ex'): te_comp = 'Hx'
        elif(component == 'Ey'): te_comp = 'Hy'
        else: te_comp = 'invalid'

        field = super(ModeTM, self).get_field_interp(i, te_comp)

        if(component == 'Hz'):
            return field*-1
        else:
            return field

    def get_source(self, dx, dy, dz=0.0):
        """Get the source current distribution for the i'th mode.

        Notes
        -----
        For this calculation to work out, we assume that all field components
        are zero to the left of the center of the Yee cell (i.e. the positions
        of the Hz values). To the right of the center of the Yee cell, we
        assume the field components have an exp(ikx) dependence.

        dy should be equal to ds.

        This class assumes all modes propagate in the x direction. In order to
        propagate a mode in the y direction, x and y (dx and dy) can be
        permuted.

        TODO
        ----
        Implement in parallelized manner.

        Parameters
        ----------
        i : int
            Index of the mode for which the corresponding current sources are
            desired.
        dx : float
            The grid spacing in the x direction.
        dy : float
            The grid spacing in the y direction.
        dz : float
            Unused in :class:`.ModeTE`

        Returns
        -------
        tuple of numpy.ndarray
            (On ALL nodes) The tuple (Mz, Jx, Jy) containing arrays of the
            source distributions.  In 2D, these source distributions are N x 1
            arrays.
        """
        src = super(ModeTM, self).get_source(dx, dy, dz)

        # In order to make use of the TE subclass, we need to flip the sign of
        # the Jx and Jy sources
        return (src[0], -1*src[1], -1*src[2])

class ModeFullVector(ModeSolver):
    """Solve for the modes for a 2D slice of a 3D structure.

    Parameters
    ----------
    wavelength : float
        The wavelength of the modes.
    ds : float
        The grid spacing in the mode field (y) direction.
    eps : numpy.ndarray
        The array containing the slice of permittivity for which the modes are
        calculated.
    mu : numpy.ndarray
        The array containing the slice of permeabilities for which the modes are
        calculated.
    n0 : float (optional)
        The 'guess' for the effective index around which the modes are
        computed. In general, this value should be larger than the index of the
        mode you are looking for. (default = 1.0)
    neigs : int (optional)
        The number of modes to compute. (default = 1)
    backwards : bool
        Defines whether or not the mode propagates in the forward +x direction
        (False) or the backwards -x direction (True). (default = False)

    Attributes
    ----------
    wavelength : float
        The wavelength of the solved modes.
    neff : list of floats
        The list of solved effective indices
    n0 : float
        The effective index near which modes are found
    neigs : int
        The number of modes to solve for.

    Methods
    -------
    build(self)
        Build the system of equations and prepare the mode solver for the solution
        process.
    solve(self)
        Solve for the modes of the structure.
    get_field(self, i, component)
        Get the desired raw field component of the i'th mode.
    get_field_interp(self, i, component)
        Get the desired interpolated field component of the i'th mode
    get_mode_number(self, i):
        Estimate the number X of the given TE_X mode.
    find_mode_index(self, X):
        Find the index of a TE_X mode with the desired X.
    get_source(self, i, ds1, ds2, ds3=0.0)
        Get the source current distribution for the i'th mode.
    """

    def __init__(self, wavelength, eps, mu, domain, n0=1.0, neigs=1, \
                 backwards=False, verbose=True):
        super(ModeFullVector, self).__init__(wavelength, n0, neigs)

        self.eps = eps
        self.mu = mu
        self.domain = domain

        # figure out how 2D domain is oriented
        n = None
        if(domain.Nx == 1):
            n = 'x'
            self._M = domain.Nz
            self._N = domain.Ny
            self.dx = domain.dy
            self.dy = domain.dz
        elif(domain.Ny == 1):
            n = 'y'
            self._M = domain.Nz
            self._N = domain.Nx
            self.dx = domain.dx
            self.dy = domain.dz
        elif(domain.Nz == 1):
            n = 'z'
            self._M = domain.Ny
            self._N = domain.Nx
            self.dx = domain.dx
            self.dy = domain.dy
        else:
            raise AttributeError('3D domains are not supported for 2D mode ' \
                                 'calculations!')
        self.ndir = n

        self.verbose = verbose

        # this minus sign needs attention... It doesnt really make sense,
        # currently
        if(backwards):
            self._dir = -1.0
        else:
            self._dir = 1.0

        # non-dimensionalization for spatial variables
        self.R = self.wavelength/(2*np.pi)

        # Solve problem of the form Ax = nBx
        # define A and B matrices here
        # 6 fields
        Nfields = 6
        self._A = PETSc.Mat()
        self._A.create(PETSc.COMM_WORLD)
        self._A.setSizes([Nfields*self._M*self._N, Nfields*self._M*self._N])
        self._A.setType('aij')
        self._A.setUp()

        self._B = PETSc.Mat()
        self._B.create(PETSc.COMM_WORLD)
        self._B.setSizes([Nfields*self._M*self._N, Nfields*self._M*self._N])
        self._B.setType('aij')
        self._B.setUp()

        # setup the solver
        self._solver = SLEPc.EPS()
        self._solver.create()

        # we need to set up the spectral transformation so that it doesnt try
        # to invert B
        st = self._solver.getST()
        st.setType('sinvert')

        # Let's use MUMPS for any system solving since it is fast
        ksp = st.getKSP()
        #ksp.setType('gmres')
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverPackage('mumps')

        # setup vectors for the solution
        self._x = []
        self._neff = np.zeros(neigs, dtype=np.complex128)
        vr, wr = self._A.getVecs()
        self._x.append(vr)

        for i in range(neigs-1):
            self._x.append(self._x[0].copy())

        ib, ie = self._A.getOwnershipRange()
        self.ib = ib
        self.ie = ie
        #indset = self._A.getOwnershipIS()

        #self._ISEx = indset[0].createBlock(self._M*self._N, [0])
        #self._ISEy = indset[0].createBlock(self._M*self._N, [1])
        #self._ISEz = indset[0].createBlock(self._M*self._N, [2])
        #self._ISHx = indset[0].createBlock(self._M*self._N, [3])
        #self._ISHy = indset[0].createBlock(self._M*self._N, [4])
        #self._ISHz = indset[0].createBlock(self._M*self._N, [5])

        # Boundary conditions default to PEC
        self._bc = ['0', '0']

    @property
    def bc(self):
        return self._bc

    @bc.setter
    def bc(self, bc):
        if(bc[0] not in ['0', 'M', 'E', 'H', 'P', 'EM', 'HM']):
            error_message("Boundary condition type '%s' not found. Use "
                          "0, M, E, H, P, EM, HM." % (bc[0]), "emopt.modes")
        if(bc[1] not in ['0', 'M', 'E', 'H', 'P', 'EM', 'HM']):
            error_message("Boundary condition type '%s' not found. Use "
                          "0, M, E, H, P, EM, HM." % (bc[1]), "emopt.modes")

        self._bc = bc

    def build(self):
        """Build the system of equations and prepare the mode solver for the solution
        process.

        In order to solve for the eigen modes, we must first assemble the
        relevant matrices :math:`A` and :math:`B` for the generalized
        eigenvalue problem given by :math:`A x = n_x B x` where :math:`n_x` is
        the eigenvalue and :math:`x` is the vector containing the eigen modes.

        Notes
        -----
        This function is run on all nodes.
        """
        if(self.verbose and NOT_PARALLEL):
            info_message('Building mode solver system matrix...')

        dx = self.dx/self.R # non-dimensionalize
        dy = self.dy/self.R # non-dimensionalize

        odx = 1.0/dx
        ody = 1.0/dy

        A = self._A
        B = self._B
        eps = self.eps
        mu = self.mu
        M = self._M
        N = self._N

        bc = self._bc

        # handle coordinate permutations for 3D slices. This is necessary since
        # the solver assumes the mode propagates in the z direction however
        # slices pointing in other directions may be provided when used in
        # conjunction with the 3D solver.
        get_eps = None
        get_mu = None
        i0 = self.domain.i1
        j0 = self.domain.j1
        k0 = self.domain.k1
        if(isinstance(eps, grid.Material2D)):
            get_eps_x = lambda x,y : eps.get_value(k0+x+0.5, j0+y)
            get_eps_y = lambda x,y : eps.get_value(k0+x, j0+y+0.5)
            get_eps_z = lambda x,y : eps.get_value(k0+x, j0+y)
            get_mu_x = lambda x,y : mu.get_value(k0+x, j0+y+0.5)
            get_mu_y = lambda x,y : mu.get_value(k0+x+0.5, j0+y)
            get_mu_z = lambda x,y : mu.get_value(k0+x+0.5, j0+y+0.5)
        elif(isinstance(eps, grid.Material3D)):
            if(self.ndir == 'x'):
                get_eps_x = lambda x,y : eps.get_value(k0,j0+x+0.5,i0+y-0.5)
                get_eps_y = lambda x,y : eps.get_value(k0,j0+x,i0+y)
                get_eps_z = lambda x,y : eps.get_value(k0,j0+x,i0+y-0.5)
                get_mu_x = lambda x,y : mu.get_value(k0,j0+x,i0+y)
                get_mu_y = lambda x,y : mu.get_value(k0,j0+x+0.5,i0+y-0.5)
                get_mu_z = lambda x,y : mu.get_value(k0,j0+x+0.5,i0+y)
            elif(self.ndir == 'y'):
                get_eps_x = lambda x,y : eps.get_value(k0+y,j0,i0+x)
                get_eps_y = lambda x,y : eps.get_value(k0+y+0.5,j0,i0+x-0.5)
                get_eps_z = lambda x,y : eps.get_value(k0+y,j0,i0+x-0.5)
                get_mu_x = lambda x,y : mu.get_value(k0+y+0.5,j0,i0+x-0.5)
                get_mu_y = lambda x,y : mu.get_value(k0+y,j0,i0+x)
                get_mu_z = lambda x,y : mu.get_value(k0+y+0.5,j0,i0+x)
            elif(self.ndir == 'z'):
                get_eps_x = lambda x,y : eps.get_value(k0+x,j0+y,i0)
                get_eps_y = lambda x,y : eps.get_value(k0+x,j0+y,i0)
                get_eps_z = lambda x,y : eps.get_value(k0+x,j0+y,i0)
                get_mu_x = lambda x,y : mu.get_value(k0+x,j0+y,i0)
                get_mu_y = lambda x,y : mu.get_value(k0+x,j0+y,i0)
                get_mu_z = lambda x,y : mu.get_value(k0+x,j0+y,i0)

        for I in xrange(self.ib, self.ie):
            A[I,I] = 0.0
            B[I,I] = 0.0

            # (stuff) = n_z B H_y
            if(I < N*M):
                y = int((I-0*M*N)/N)
                x = (I-0*M*N) - y * N

                JHz1 = 5*M*N + y*N + x
                JHz0 = 5*M*N + (y-1)*N + x
                JEx = y*N + x
                JHy = 4*M*N + y*N + x

                # derivative of Hz
                if(y > 0): A[I, JHz0] = -ody
                A[I, JHz1] = ody

                # Ex
                A[I, JEx] = 1j*get_eps_x(x,y)

                # Setup the LHS B matrix
                B[I,JHy] = 1j*self._dir

                #############################
                # enforce boundary conditions
                #############################
                if(x == 0):
                    pass
                elif(x == N-1):
                    pass

                if(y == 0):
                    if(bc[1] == '0'): A[I, JEx] = 0
                    elif(bc[1] == 'E'): A[I, JHz1] = 2*ody
                    elif(bc[1] == 'H'):
                        A[I, JHz1] = 0
                        A[I, JEx] = 0
                elif(y == M-1):
                    pass


            # (stuff) = n_z B H_x
            elif(I < 2*N*M):
                y = int((I-1*M*N)/N)
                x = (I-1*M*N) - y * N

                JHz1 = 5*N*M + y*N + x
                JHz0 = 5*N*M + y*N + x - 1

                JEy = M*N + y*N + x
                JHx = 3*M*N + y*N + x

                # derivative of Hz
                if(x > 0): A[I, JHz0] = odx
                A[I, JHz1] = -odx

                # Ey
                A[I, JEy] = 1j*get_eps_y(x,y)

                # Setup the LHS B matrix
                B[I,JHx] = -1j*self._dir

                #############################
                # enforce boundary conditions
                #############################
                if(x == 0):
                    if(bc[0] == '0'): A[I, JEy] = 0
                    elif(bc[0] == 'E'): A[I, JHz1] = -2*odx
                    elif(bc[0] == 'H'): A[I, JHz1] = 0
                elif(x == N-1):
                    pass

                if(y == 0):
                    pass
                elif(y == M-1):
                    pass

            # (stuff) = n_z B E_z
            elif(I < 3*N*M):
                y = int((I-2*M*N)/N)
                x = (I-2*M*N) - y * N

                JEy0 = M*N + y*N + x
                JEy1 = M*N + y*N + x + 1

                JEx0 = y*N + x
                JEx1 = (y+1)*N + x

                JHz = 5*M*N + y*N + x
                JEz = 2*M*N + y*N + x

                # derivative of Ey
                A[I, JEy0] = -odx
                if(x < N-1): A[I,JEy1] = odx

                # derivative of Ex
                A[I, JEx0] = ody
                if(y < M-1): A[I, JEx1] = -ody

                # Hz at x,y
                A[I, JHz] = -1j*get_mu_z(x,y)


                #############################
                # enforce boundary conditions
                #############################
                if(x == 0):
                    if(bc[0] == '0'): A[I, JEy0] = 0
                elif(x == N-1):
                    pass

                if(y == 0):
                    if(bc[1] == '0'): A[I, JEx0] = 0
                    if(bc[1] == 'H'): A[I, JEx0] = 0
                elif(y == M-1):
                    pass

            # (stuff) = n_z B E_y
            elif(I < 4*N*M):
                y = int((I-3*M*N)/N)
                x = (I-3*M*N) - y * N

                JEz0 = 2*M*N + y*N + x
                JEz1 = 2*M*N + (y+1)*N + x

                JHx = 3*M*N + y*N + x
                JEy = M*N + y*N+x

                # derivative of Ez
                A[I,JEz0] = -ody
                if(y < M-1): A[I,JEz1] = ody

                # Hx at x,y
                A[I,JHx] = -1j*get_mu_x(x,y)

                # Setup the LHS B matrix
                B[I,JEy] = 1j*self._dir

                #############################
                # enforce boundary conditions
                #############################
                if(x == 0):
                    if(bc[0] == '0'): A[I, JHx] = 0
                elif(x == N-1):
                    pass

                if(y == 0):
                    if(bc[1] == '0'): A[I, JEz0] = 0
                    if(bc[1] == 'H'): A[I, JEz0] = 0
                elif(y == M-1):
                    pass

            # (stuff) = n_z B E_x
            elif(I < 5*N*M):
                y = int((I-4*M*N)/N)
                x = (I-4*M*N) - y * N

                JEz0 = 2*M*N + y*N + x
                JEz1 = 2*M*N + y*N + x + 1

                JHy = 4*M*N + y*N + x
                JEx = y*N + x

                # derivative of Ez
                A[I,JEz0] = odx
                if(x < N-1): A[I,JEz1] = -odx

                # Hy at x,y
                A[I,JHy] = -1j*get_mu_y(x,y)

                # Setup the LHS B matrix
                B[I,JEx] = -1j*self._dir

                #############################
                # enforce boundary conditions
                #############################
                if(x == 0):
                    if(bc[0] == '0'): A[I, JEz0] = 0
                elif(x == N-1):
                    pass

                if(y == 0):
                    if(bc[1] == '0'): A[I, JHy] = 0
                    if(bc[1] == 'H'): A[I, JHy] = 0
                elif(y == M-1):
                    pass

            # (stuff) = Hz (zero)
            elif(I < 6*M*N):
                y = int((I-5*M*N)/N)
                x = (I-5*M*N) - y * N

                JHy0 = 4*M*N + y*N + x - 1
                JHy1 = 4*M*N + y*N + x

                JHx0 = 3*M*N + (y-1)*N + x
                JHx1 = 3*M*N + y*N + x

                JEz = 2*M*N + y*N + x
                JEz1 = 2*M*N + y*N + x + 1
                JEz2 = 2*M*N + (y+1)*N + x
                JEz3 = 2*M*N + (y+1)*N + x + 1

                # derivative of Hy
                if(x > 0): A[I, JHy0] = -odx
                A[I,JHy1] = odx

                # derivative of Hx
                if(y > 0): A[I, JHx0] = ody
                A[I, JHx1] = -ody

                # Ez
                A[I, JEz] = 1j*get_eps_z(x,y)

                #############################
                # enforce boundary conditions
                #############################
                if(x == 0):
                    if(bc[0] == '0'):
                        A[I, JHx1] = 0
                        if(y > 0): A[I, JHx0] = 0
                    elif(bc[0] == 'E'): A[I,JHy1] = 2*odx
                    elif(bc[0] == 'H'): A[I,JHy1] = 0
                elif(x == N-1):
                    pass

                if(y == 0):
                    if(bc[1] == '0'):
                        A[I, JHy1] = 0
                        if(x > 0): A[I, JHy0] = 0
                    elif(bc[1] == 'E'): A[I, JHx1] = -2*ody
                    elif(bc[1] == 'H'):
                        A[I, JHx1] = 0
                        A[I, JHy1] = 0
                        if(x > 0): A[I, JHy0] = 0
                elif(y == M-1):
                    pass

        self._A.assemble()
        self._B.assemble()

    def solve(self):
        """Solve for the modes of the structure.

        Notes
        -----
        This function is run on all nodes.
        """
        if(self.verbose and NOT_PARALLEL):
            info_message('Solving...')

        # Setup the solve options. We are solving a generalized non-hermitian
        # eigenvalue problem (GNHEP)
        self._solver.setOperators(self._A, self._B)
        self._solver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        self._solver.setDimensions(self.neigs, PETSc.DECIDE)
        self._solver.setTarget(self.n0) # "guess" for effective index
        self._solver.setFromOptions()

        # Solve Ax=nBx using SLEPc.
        # Internally, we use a direct solve (MUMPS) to handle the heavy
        # lifting.
        self._solver.solve()
        nconv = self._solver.getConverged()

        if(nconv < self.neigs):
            warning_message('%d eigenmodes were requested, however only %d ' \
                            'eigenmodes were found.' % (self.neigs, nconv), \
                            module='emopt.modes')

        # nconv can be bigger than the desired number of eigen values
        if(nconv > self.neigs):
            neigs = self.neigs
        else:
            neigs = nconv

        for i in range(neigs):
            self.neff[i] = self._solver.getEigenvalue(i)
            self._solver.getEigenvector(i, self._x[i])

    def __permute_field_component(self, component):
        ## Permute the field components to account for planes with non-z normal
        # directions.
        if(self.ndir == 'x'):
            if(component == FieldComponent.Ex): component = FieldComponent.Ez
            elif(component == FieldComponent.Ey): component = FieldComponent.Ex
            elif(component == FieldComponent.Ez): component = FieldComponent.Ey
            elif(component == FieldComponent.Hx): component = FieldComponent.Hz
            elif(component == FieldComponent.Hy): component = FieldComponent.Hx
            elif(component == FieldComponent.Hz): component = FieldComponent.Hy
        elif(self.ndir == 'y'):
            if(component == FieldComponent.Ex): component = FieldComponent.Ey
            elif(component == FieldComponent.Ey): component = FieldComponent.Ez
            elif(component == FieldComponent.Ez): component = FieldComponent.Ex
            elif(component == FieldComponent.Hx): component = FieldComponent.Hy
            elif(component == FieldComponent.Hy): component = FieldComponent.Hz
            elif(component == FieldComponent.Hz): component = FieldComponent.Hx
        elif(self.ndir == 'z'):
            pass # No need to permute!

        return component

    def get_field(self, i, component, permute=True):
        """Get the desired raw field component of the i'th mode.

        Notes
        -----
        This function only returns a non-None result on the master node. On all
        other nodes, None is returned.

        See Also
        --------
        :func:`.ModeFullVector.get_field_interp`

        :func:`.ModeFullVector.find_mode_index`

        Parameters
        ----------
        i : int
            The index of the desired mode
        component : str
            The desired field component (Ex, Ey, Ez, Hx, Hy, Hz)
        permute : bool
            Permute the field components according to the normal direction of
            the supplied domain. Because all computation is internally
            performed assuming the mode propagates in the z direction, the
            field components need to be permuted in order to produce the
            desired result. (default = True)

        Returns
        -------
        numpy.ndarray or None
            (Master node only) an array containing the desired component of the
            mode field.
        """
        M = self._M
        N = self._N

        # Permute components
        if(permute):
            component = self.__permute_field_component(component)

        # fields are split up among processes --> we only request data from
        # processors which store the desired field component. We do this by
        # finding out the initial and final indices of the field vector in each
        # process which stores the desired field component.
        if(component == FieldComponent.Ex):
            if(self.ib >= M*N):
                I0 = 0
                I1 = 0
            else:
                I0 = 0
                if(self.ie >= M*N):
                    I1 = M*N-self.ib
                else:
                    I1 = self.ie-self.ib
        elif(component == FieldComponent.Ey):
            if(self.ib >= 2*M*N or self.ie < M*N):
                I0 = 0
                I1 = 0
            else:
                if(self.ib < M*N):
                    I0 = M*N - self.ib
                else:
                    I0 = 0
                if(self.ie >= 2*M*N):
                    I1 = 2*M*N-self.ib
                else:
                    I1 = self.ie-self.ib
        elif(component == FieldComponent.Ez):
            if(self.ib >= 3*M*N or self.ie < 2*M*N):
                I0 = 0
                I1 = 0
            else:
                if(self.ib < 2*M*N):
                    I0 = 2*M*N - self.ib
                else:
                    I0 = 0
                if(self.ie >= 3*M*N):
                    I1 = 3*M*N-self.ib
                else:
                    I1 = self.ie-self.ib
        elif(component == FieldComponent.Hx):
            if(self.ib >= 4*M*N or self.ie < 3*M*N):
                I0 = 0
                I1 = 0
            else:
                if(self.ib < 3*M*N):
                    I0 = 3*M*N - self.ib
                else:
                    I0 = 0
                if(self.ie >= 4*M*N):
                    I1 = 4*M*N-self.ib
                else:
                    I1 = self.ie-self.ib
        elif(component == FieldComponent.Hy):
            if(self.ib >= 5*M*N or self.ie < 4*M*N):
                I0 = 0
                I1 = 0
            else:
                if(self.ib < 4*M*N):
                    I0 = 4*M*N - self.ib
                else:
                    I0 = 0
                if(self.ie >= 5*M*N):
                    I1 = 5*M*N-self.ib
                else:
                    I1 = self.ie-self.ib
        elif(component == FieldComponent.Hz):
            if(self.ie < 4*M*N):
                I0 = 0
                I1 = 0
            else:
                if(self.ib < 5*M*N):
                    I0 = 5*M*N - self.ib
                else:
                    I0 = 0

                I1 = self.ie-self.ib
        else:
            raise ValueError('Unrecongnized field componenet "%s". The allowed'
                             'field components are Ex, Ey, Ez, Hx, Hy, Hz.' % (component))

        comm = MPI.COMM_WORLD
        x = self._x[i].getArray()[I0:I1]
        x_full = comm.gather(x, root=0)

        #scatter, x_full = PETSc.Scatter.toZero(x)
        #scatter.scatter(x, x_full, False, PETSc.Scatter.Mode.FORWARD)

        if(NOT_PARALLEL):
            x_assembled = np.concatenate(x_full)
            field = np.reshape(x_assembled, (M,N))
            return field
        else:
            return MathDummy()

    def get_field_interp(self, i, component):
        """Get the desired interpolated field component of the i'th mode.

        In general, this function should be preferred over
        :func:`.ModeFullVector.get_field`.

        In general, you may wish to solve for more than one mode.  In order to
        get the desired mode, you must specify its index.  If you do not know
        the index but you do know the desired mode number, then
        :func:`.ModeFullVector.find_mode_index` may be used to determine the index of
        the desired mode.

        Notes
        -----
        The fields are solved for on a grid made up of compressed 2D Yee cells.
        The fields are thus interpolated at the center of this Yee cell (which
        happens to coincide with the position of Hz)

        This function only returns a non-None result on the master node. On all
        other nodes, None is returned.

        See Also
        --------
        :func:`.ModeFullVector.get_field`

        :func:`.ModeFullVector.find_mode_index`

        Parameters
        ----------
        i : int
            The index of the desired mode
        component : str
            The desired field component (Ex, Ey, Ez, Hx, Hy, Hz)

        Returns
        -------
        numpy.ndarray or None
            (Master node only) an array containing the desired component of the
            interpolated mode field.
        """
        f_raw = self.get_field(i, component)

        # zero padding is equivalent to including boundary values outside of
        # the metal boundaries. This is needed to compute the interpolated
        # values.
        f_raw = np.pad(f_raw, 1, 'constant', constant_values=0)

        # Permute components
        component = self.__permute_field_component(component)

        bc = self._bc

        # interpolate and return the fields on the rank 0 nodes. Consult the
        # All field components are interpolated onto the Ez grid.
        if(NOT_PARALLEL):
            if(component == FieldComponent.Ex): # Ex --> average along x
                Ex = np.copy(f_raw)
                if(bc[0] == 'E'): f_raw[:,0] = -1*f_raw[:,1]
                elif(bc[0] == 'H'): f_raw[:,0] = f_raw[:,1]

                Ex[:,1:] += f_raw[:,0:-1]
                return Ex[1:-1, 1:-1]/2.0

            elif(component == FieldComponent.Ey): # Ey --> average along y
                Ey = np.copy(f_raw)
                if(bc[1] == 'E'): f_raw[0,:] = -1*f_raw[1,:]
                elif(bc[1] == 'H'): f_raw[0,:] = f_raw[1,:]

                Ey[1:,:] += f_raw[0:-1,:]
                return Ey[1:-1, 1:-1]/2.0

            elif(component == FieldComponent.Ez): # Ez --> dont average
                return f_raw[1:-1, 1:-1]

            elif(component == FieldComponent.Hx): # Hx --> average along y
                Hx = np.copy(f_raw)
                if(bc[1] == 'E'): f_raw[0,:] = -1*f_raw[1,:]
                elif(bc[1] == 'H'): f_raw[0,:] = f_raw[1,:]

                Hx[1:,:] += f_raw[0:-1,:]
                return Hx[1:-1, 1:-1]/2.0

            elif(component == FieldComponent.Hy): # Hy --> average along x
                Hy = np.copy(f_raw)
                if(bc[0] == 'E'): f_raw[:,0] = -1*f_raw[:,1]
                elif(bc[0] == 'H'): f_raw[:,0] = f_raw[:,1]

                Hy[:,1:] += f_raw[:,0:-1]
                return Hy[1:-1, 1:-1]/2.0

            elif(component == FieldComponent.Hz): # Hz --> average along x & y
                Hz = np.copy(f_raw)
                if(bc[1] == 'E'): f_raw[0,:] = -1*f_raw[1,:]
                elif(bc[1] == 'H'): f_raw[0,:] = f_raw[1,:]
                if(bc[0] == 'E'): f_raw[:,0] = -1*f_raw[:,1]
                elif(bc[0] == 'H'): f_raw[:,0] = f_raw[:,1]

                Hz[:,1:] += f_raw[:,0:-1]
                Hz[1:,:] += f_raw[0:-1,:]
                Hz[1:, 1:] += f_raw[0:-1,0:-1]
                return Hz[1:-1, 1:-1]/4.0
        else:
            return MathDummy()

    def component_energy(self, i):
        """Get the fraction of energy stored in each field component.

        Parameters
        ----------
        i : int
            The index of the mode to analyze

        Returns
        -------
        [float, float, float, float, float, float]
            The list of energy fractions corresponding to Ex, Ey, Ez, Hx, Hy,
            Hz
        """
        eps = self.eps.get_values_in(self.domain)
        mu = self.mu.get_values_in(self.domain)

        Ex = self.get_field(i, FieldComponent.Ex)
        WEx = np.sum(eps.real*np.abs(Ex)**2)
        del Ex

        Ey = self.get_field(i, FieldComponent.Ey)
        WEy = np.sum(eps.real*np.abs(Ey)**2)
        del Ey

        Ez = self.get_field(i, FieldComponent.Ez)
        WEz = np.sum(eps.real*np.abs(Ez)**2)
        del Ez

        Hx = self.get_field(i, FieldComponent.Hx)
        WHx = np.sum(mu.real*np.abs(Hx)**2)
        del Hx

        Hy = self.get_field(i, FieldComponent.Hy)
        WHy = np.sum(mu.real*np.abs(Hy)**2)
        del Hy

        Hz = self.get_field(i, FieldComponent.Hz)
        WHz = np.sum(mu.real*np.abs(Hz)**2)
        del Hz

        Wtot = WEx+WEy+WEz+WHx+WHy+WHz
        return [WEx/Wtot, WEy/Wtot, WEz/Wtot,
                WHx/Wtot, WHy/Wtot, WHz/Wtot]

    def get_mode_number(self, i):
        """Determine the mode number.

        This is based on the number of phase crossings.

        Todo
        ----
        Implement this function...

        Parameters
        ----------
        i : int
            The index of the mode to analyze.

        Returns
        -------
        int, int
            The numbers X and Y of the mode.
        """
        pass

    def find_mode_index(self, P, Q):
        """Find the index of the index of the mode with X horizontal phase
        transitions and Y vertical phase transitions.

        Parameters
        ----------
        P : int
            The number of horizontal phase transitions
        Q : int
            The number of vertical phase transitions

        Returns
        -------
        int
            The index of the mode with the desired number.
        """
        for i in range(self.neigs):
            p, q = self.get_mode_number(i)
            if(p == P and q == Q):
                return i

        warning_message('Desired mode number was not found.', 'emopt.modes')
        return 0

    def get_source(self, i, dx, dy, dz):
        """Get the source current distribution for the i'th mode.

        Notes
        -----
        1) Source calculations are only supported for Material3D structures.
        2) The sources are computed assuming the mode is propagating in the z
        direction. In order to support modes propagating in different
        directions, the spatial coordinates are permuted. Fortunately, the
        underlying dislocated grids are invariant under this transformation.
        This is only necessary for calculating the material cross-sections.

        TODO
        ----
        Implement in parallelized manner.

        Parameters
        ----------
        i : int
            Index of the mode for which the corresponding current sources are
            desired.
        dx : float
            Grid spacing along x direction.
        dy : float
            Grid spacing along y direction
        dz : float
            Grid spacing along z direction

        Returns
        -------
        tuple of numpy.ndarray
            (On master node) The tuple (Jx, Jy, Jz, Mx, My, Mz) containing arrays of the
            source distributions.
        """
        if(not isinstance(self.eps, grid.Material3D)):
            raise Exception("Source calculation only possible for slices of " \
                            "Material3D's..")

        neff = self.neff[i]
        bc = self._bc

        # setup arrays for storing the calculated current sources
        # These are only computed and stored on the rank 0 process 
        if(NOT_PARALLEL):
            Jx = np.zeros([self._M, self._N], dtype=np.complex128)
            Jy = np.zeros([self._M, self._N], dtype=np.complex128)
            Jz = np.zeros([self._M, self._N], dtype=np.complex128)
            Mx = np.zeros([self._M, self._N], dtype=np.complex128)
            My = np.zeros([self._M, self._N], dtype=np.complex128)
            Mz = np.zeros([self._M, self._N], dtype=np.complex128)

            eps = np.zeros([self._M, self._N], dtype=np.complex128)
            mu = np.zeros([self._M, self._N], dtype=np.complex128)
        else:
            Jx = None; Jy = None; Jz = None
            Mx = None; My = None; Mz = None

        # Handle coordinate permutations. This allows us to support launching
        # modes in any cartesian direction. At the end, we will once again
        # permute the calculated current density components to match the users
        # supplied coordinate system
        get_mu = None
        if(self.ndir == 'x'):
            get_mu = lambda sx,sy : self.mu.get_values_in(self.domain, sy=sx, sz=sy, squeeze=True)
            mydx = dy
            mydy = dz
            mydz = dz
        elif(self.ndir == 'y'):
            get_mu = lambda sx,sy : self.mu.get_values_in(self.domain, sx=sx, sz=sy, squeeze=True)
            mydx = dz
            mydy = dx
            mydz = dy
        elif(self.ndir == 'z'):
            get_mu = lambda sx,sy : self.mu.get_values_in(self.domain, sx=sx, sy=sy, squeeze=True)
            mydx = dx
            mydy = dy
            mydz = dz

        # phase factor
        dx = mydx/self.R
        dy = mydy/self.R
        dz = mydz/self.R
        ekz = np.exp(1j*self._dir*neff*dz/2)

        # Get field data
        Ex = self.get_field(i, FieldComponent.Ex, permute=False)
        Ey = self.get_field(i, FieldComponent.Ey, permute=False)
        Ez = self.get_field(i, FieldComponent.Ez, permute=False)
        Hx = self.get_field(i, FieldComponent.Hx, permute=False)
        Hy = self.get_field(i, FieldComponent.Hy, permute=False)
        Hz = self.get_field(i, FieldComponent.Hz, permute=False)

        # normalize power to ~1.0 -- not really necessary
        S = 0.5*mydx*mydy*np.sum(Ex*np.conj(Hy)-Ey*np.conj(Hx))
        Ex = Ex/np.sqrt(S); Ey = Ey/np.sqrt(S); Ez = Ez/np.sqrt(S)
        Hx = Hx/np.sqrt(S); Hy = Hy/np.sqrt(S); Hz = Hz/np.sqrt(S)

        if(NOT_PARALLEL):
            ## Calculate contribution of Ex
            Ex = np.pad(Ex, 1, 'constant', constant_values=0)
            My += Ex[1:-1, 1:-1]*ekz/dz

            ## Calculate contribution of Ey
            Ey = np.pad(Ey, 1, 'constant', constant_values=0)
            Mx += -Ey[1:-1, 1:-1]*ekz/dz

            ## Calculate contribution of Ez
            Ez = np.pad(Ez, 1, 'constant', constant_values=0)
            eps = self.eps.get_values_in(self.domain, squeeze=True)
            Jz += 1j*eps*Ez[1:-1, 1:-1]
            Mx += (Ez[2:, 1:-1] - Ez[1:-1, 1:-1])/dy
            My += -1*(Ez[1:-1, 2:] - Ez[1:-1, 1:-1])/dx

            ## Calculate contribution of Hx
            Hx = np.pad(Hx, 1, 'constant', constant_values=0)

            # Handle boundary conditions
            if(bc[1] == 'E'): Hx[0,:] = -1*Hx[1,:]
            elif(bc[1] == 'H'): Hx[0,:] = Hx[1,:]

            mu = get_mu(0.0, 0.5)
            Jy += Hx[1:-1, 1:-1]/dz
            Jz += -1*(Hx[1:-1, 1:-1] - Hx[0:-2, 1:-1])/dy
            Mx += -1j*mu*Hx[1:-1, 1:-1]

            ## Calculate contribution of Hy
            Hy = np.pad(Hy, 1, 'constant', constant_values=0)

            # Handle boundary conditions
            if(bc[0] == 'E'): Hy[:,0] = -1*Hy[:,1]
            elif(bc[0] == 'H'): Hy[:,0] = Hy[:,1]

            mu = get_mu(0.5, 0.0)
            Jx += -Hy[1:-1, 1:-1]/dz
            Jz += (Hy[1:-1, 1:-1] - Hy[1:-1, 0:-2])/dx
            My += -1j*mu*Hy[1:-1, 1:-1]

            ## Calculate contribution of Hz
            # There isn't one

            # reshape the output to make things seemless
            Jx = np.reshape(Jx, self.domain.shape)
            Jy = np.reshape(Jy, self.domain.shape)
            Jz = np.reshape(Jz, self.domain.shape)
            Mx = np.reshape(Mx, self.domain.shape)
            My = np.reshape(My, self.domain.shape)
            Mz = np.reshape(Mz, self.domain.shape)

        # permute (if necessary) and return the results
        if(self.ndir == 'x'):
            return Jz, Jx, Jy, Mz, Mx, My
        elif(self.ndir == 'y'):
            return Jy, Jz, Jx, My, Mz, Mx
        elif(self.ndir == 'z'):
            return Jx, Jy, Jz, Mx, My, Mz

# Done modes.py!
