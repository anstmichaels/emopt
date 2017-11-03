"""Solve for the modes of electromagnetic waveguides in 2D and 3D.

Waveguide modes can be computed by setting up a generalized eigenvalue problem
corresponding to the source-free Maxwell's equations assuming a solution to
:math:`\mathbf{E}` and :math:`\mathbf{H}` which is proportional to :math:`e^{i
k_z z}`, i.e.

.. math::
    \\nabla \\times e^{i k_z z} \mathbf{E} + i \\mu_r \\nabla \\times
    e^{i k_z z}\\mathbf{H} = 0

    \\nabla \\times e^{i k_z z} \mathbf{H} - i \\epsilon_r \\nabla \\times
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

from misc import info_message, warning_message, error_message, RANK, \
NOT_PARALLEL, run_on_master

from grid import row_wise_A_update

from math import pi
from abc import ABCMeta, abstractmethod
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
import numpy as np

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

class Mode_TE(ModeSolver):
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

    def __init__(self, wavelength, ds, eps, mu, n0=1.0, neigs=1, \
                 backwards=False):
        super(Mode_TE, self).__init__(wavelength, n0, neigs)

        # We extend the size of the inputs by one element on both sides in order to
        # accomodate taking derivatives which will be necessary for finding
        # sources.  Any returned quantities will be the same length as the
        # input eps/mu
        N = len(eps)
        self._N = N+2

        self.eps = np.concatenate((eps[0:1], eps, eps[-1:]))
        self.mu = np.concatenate((mu[0:1], mu, mu[-1:]))

        self.ds = ds

        if(backwards):
            self._dir = -1.0
        else:
            self._dir = 1.0

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

            # (stuff) = n_x B E_z
            if(I < N):
                i = I
                j0 = I+2*N

                A[i,j0] = -1*mu[i]

            # (stuff) = n_x B H_x
            elif(I < 2*N):
                i = I
                j0 = I
                j1 = I-N

                A[i,j0] = -1j*mu[I-N]
                A[i,j1] = -1.0/ds

                if(j1 < N-1):
                    A[i,j1+1] = 1.0/ds

            # (stuff) = n_x B H_y
            else:
                i = I
                j0 = I-2*N
                j1 = I-N

                A[i,j0] = -1.0*eps[j0]
                A[i,j1] = -1j/ds

                if(j1 > N):
                    A[i,j1-1] = 1j/ds

        # Define B. It contains ones on the first and last third of the
        # diagonals
        for i in xrange(self.ib, self.ie):
            if(i < N):
                B[i,i] = self._dir
            elif(i < 2*N):
                B[i,i] = 0
            else:
                B[i,i] = self._dir

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

    @run_on_master
    def get_field(self, i, component):
        """Get the desired raw field component of the i'th mode.

        Use this function with care: Ez/Hy and Hx are specified at different
        points in space (separated by half of a grid cell).  In general
        :func:`.Mode_TE.get_field_interp` should be prefered.

        In general, you may wish to solve for more than one mode.  In order to
        get the desired mode, you must specify its index.  If you do not know
        the index but you do know the desired mode number, then
        :func:`.Mode_TE.find_mode_index` may be used to determine the index of
        the desired mode.

        Notes
        -----
        This function only returns a non-None result on the master node. On all
        other nodes, None is returned.

        See Also
        --------
        :func:`.Mode_TE.get_field_interp`

        :func:`.Mode_TE.find_mode_index`

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
            return self._Ez[i][1:-1]
        elif(component == 'Hx'):
            return self._Hx[i][1:-1]
        elif(component == 'Hy'):
            return self._Hy[i][1:-1]
        else:
            raise ValueError('Unrecongnized field componenet "%s". The allowed'
                             'field components are Ez, Hx, Hy.' % (component))

    @run_on_master
    def get_field_interp(self, i, component):
        """Get the desired interpolated field component of the i'th mode.

        In general, this function should be preferred over
        :func:`.Mode_TE.get_field`.

        In general, you may wish to solve for more than one mode.  In order to
        get the desired mode, you must specify its index.  If you do not know
        the index but you do know the desired mode number, then
        :func:`.Mode_TE.find_mode_index` may be used to determine the index of
        the desired mode.

        Notes
        -----
        This function only returns a non-None result on the master node. On all
        other nodes, None is returned.

        See Also
        --------
        :func:`.Mode_TE.get_field`

        :func:`.Mode_TE.find_mode_index`

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
            return self._Ez[i][1:-1]
        elif(component == 'Hy'):
            return self._Hy[i][1:-1]
        elif(component == 'Hx'):
            Hxi = np.copy(self._Hx[i])
            Hxi[1:] += Hxi[0:self._N-1]
            return Hxi[1:-1] / 2.0
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
            Unused in :class:`.Mode_TE`

        Returns
        -------
        tuple of numpy.ndarray
            (On ALL nodes) The tuple (Jz, Mx, My) containing arrays of the
            source distributions.  In 2D, these source distributions are N x 1
            arrays.
        """
        N = self._N

        if(NOT_PARALLEL):
            Jz = np.zeros(N-2, dtype=np.complex128)
            Mx = np.zeros(N-2, dtype=np.complex128)
            My = np.zeros(N-2, dtype=np.complex128)

            Ez = self._Ez[i]
            Hx = self._Hx[i]
            Hy = self._Hy[i]
            neff = self.neff[i]
            dx = dx/self.R # non-dimensionalize
            dy = dy/self.R # non-dimensionalize

            dHxdy = np.diff(Hx) / dy
            dHxdy = dHxdy[:-1]
            dHydx = Hy[1:-1]*np.exp(self._dir*1j*neff*dx/2.0) / dy
            dEzdy = np.diff(Ez)[1:] / dy
            dEzdx = Ez[1:-1] / dy

            Jz = 1j*(self.eps*Ez)[1:-1] + dHydx - dHxdy
            Mx = dEzdy - 1j*(self.mu*Hx)[1:-1]
            My = -dEzdx

        else:
            Jz = None
            Mx = None
            My = None

        comm = MPI.COMM_WORLD
        Jz = comm.bcast(Jz, root=0)
        Mx = comm.bcast(Mx, root=0)
        My = comm.bcast(My, root=0)
        return (Jz, Mx, My)

class Mode_TM(Mode_TE):
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

    def __init__(self, wavelength, ds, eps, mu, n0=1.0, neigs=1, \
                 backwards=False):

        # A TM mode source is the same as a TE mode source except with the
        # permittivity and permeability smapped and the E and H and J and M
        # components swapped around.
        super(Mode_TM, self).__init__(wavelength, ds, mu, eps, n0, neigs, \
                                      backwards)
    @run_on_master
    def get_field(self, i, component):
        """Get the desired raw field component of the i'th mode.

        Use this function with care: Hz/Ey and Ex are specified at different
        points in space (separated by half of a grid cell).  In general
        :func:`.Mode_TM.get_field_interp` should be prefered.

        In general, you may wish to solve for more than one mode.  In order to
        get the desired mode, you must specify its index.  If you do not know
        the index but you do know the desired mode number, then
        :func:`.Mode_TM.find_mode_index` may be used to determine the index of
        the desired mode.

        Notes
        -----
        This function only returns a non-None result on the master node. On all
        other nodes, None is returned.

        See Also
        --------
        :func:`.Mode_TM.get_field_interp`

        :func:`.Mode_TM.find_mode_index`

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

        field = super(Mode_TM, self).get_field(i, te_comp)

        if(component == 'Hz'):
            return field*-1
        else:
            return field

    @run_on_master
    def get_field_interp(self, i, component):
        """Get the desired interpolated field component of the i'th mode.

        In general, this function should be preferred over
        :func:`.Mode_TM.get_field`.

        In general, you may wish to solve for more than one mode.  In order to
        get the desired mode, you must specify its index.  If you do not know
        the index but you do know the desired mode number, then
        :func:`.Mode_TM.find_mode_index` may be used to determine the index of
        the desired mode.

        Notes
        -----
        This function only returns a non-None result on the master node. On all
        other nodes, None is returned.

        See Also
        --------
        :func:`.Mode_TM.get_field`

        :func:`.Mode_TM.find_mode_index`

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

        field = super(Mode_TM, self).get_field_interp(i, te_comp)

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
            Unused in :class:`.Mode_TE`

        Returns
        -------
        tuple of numpy.ndarray
            (On ALL nodes) The tuple (Mz, Jx, Jy) containing arrays of the
            source distributions.  In 2D, these source distributions are N x 1
            arrays.
        """
        src = super(Mode_TM, self).get_source(dx, dy, dz)

        # In order to make use of the TE subclass, we need to flip the sign of
        # the Jx and Jy sources
        return (src[0], -1*src[1], -1*src[2])
