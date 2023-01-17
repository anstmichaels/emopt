"""
This module defines a standard interface for defining finite difference
frequency domain (FDFD) solvers.  The FDFD method formulates the frequency
domain Maxwell's equations explicitly in the form :math:`A x = b`.  The matrix
:math:`A` is built up by discretizing the materials of the system on a
rectangular grid.  The curl operators in Maxwell's equations are then
approximated on this grid using centered finite differences.  In order to
accomodate these centered differences, a set of dislocated grids (i.e. an array
of Yee cells) is used.  The fields are solved for on this same grid.

In order to run an FDFD simulation, the materials and sources of the system
must be defined.  Once these are defined, the system matrix (:math:`A`) can be
constructed and a simulation performed.  In addition to methods for obtaining
the resulting fields and calculating the source power, the FDFD interface also
defines a function which exposes the diagonal elements of :math:`A`.  This is
useful for performing sensitivity analysis/calculating gradients which is a
core goal of EMOpt.

In addition to the FDFD interface, a variety of specific implementations are
defined.  In particular:

    FDFD_TE : A 2D FDFD solver with TE-polarized fields.  The system is
    infinitely extruded in z and waves can propagate alogn x and y.  The
    allowed sources of the system are :math:`J_z`, :math:`M_x`, :math:`M_y` and
    the non-zeros field components are :math:`E_z`, :math:`H_x`, and
    :math:`H_y`.

    FDFD_TM : A 2D FDFD solver with TM-polarized fields.  The system is
    infinitely extruded in z and waves can propagate alogn x and y.  The
    allowed sources of the system are :math:`M_z`, :math:`J_x`, :math:`J_y` and
    the non-zeros field components are :math:`H_z`, :math:`E_x`, and
    :math:`E_y`.

    FDFD_3D : A full vector 3D FDFD solver. All current source and field
    components are used.  Material distributions in this solver are currently
    restricted to planar structures.

It is important to note that the terminology used here for 'TE' and 'TM' is
consistent with the more classical electromagnetic definition and not with the
silicon photonics definition.

Examples
--------
See emopt/examples/ for detailed examples.

TODO
----
1. Reimplement get_field and get_field_interp in parallel and add support for
Domains in the form FDFD.get_field(self, component, domain=None). The parallel
support will be needed when 3D solvers are implemented regardless, so it is a
good idea to implement the functionality in the already working 2D solvers.
Note: this will require modification of the AdjointMethod class as well; to
make this as simple as possible, the get_field function should return a
correctly-sized zeroed numpy array on the non-master nodes.

2. Reimplement the set_source function with Domain support.  This will be
necessary in 3D when the total grid size is very large.
"""
from __future__ import absolute_import

# Initialize petsc first
from builtins import zip
from builtins import range
import petsc4py
import sys
from future.utils import with_metaclass
petsc4py.init(sys.argv)

from .misc import info_message, warning_message, error_message, RANK, \
NOT_PARALLEL, run_on_master, MathDummy, DomainCoordinates, COMM
from .defs import FieldComponent, SourceComponent
from . import modes
from .simulation import MaxwellSolver

from .grid import row_wise_A_update

import numpy as np
from math import pi
from abc import ABCMeta, abstractmethod
from petsc4py import PETSc
import array
from mpi4py import MPI

__author__ = "Andrew Michaels"
__license__ = "BSD-3"
__version__ = "2019.5.6"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

class FDFD(with_metaclass(ABCMeta, MaxwellSolver)):
    """Finite difference frequency domain solver.

    This class provides the generalized interface for a finite diffrence
    frequency domain forward and adjoint solver.  It may be extended to
    implement 2D semi-vector and 3D full-vector Maxwell solvers.

    Notes
    -----
    Finite difference frequency domain solvers must express the discretized
    Maxwell's equations in the form :math:`A x = b`.  In EMOpt, solving this
    system of equations is handled using the petsc4py library, although this is
    not strictly necessary.

    The majority of methods in this class are abstract and need to be
    implemented by a child class. The exception to this is the function
    :func:`get_A_diag` which should not be necessary to override.

    In general, there are two different ways to access the results of an FDFD
    simulation. The first way is through direct calls to the :func:`get_field`
    and :func:`get_field_interp` functions. The second is by specifying field
    domains before running a forward simulation. If field domains are provided,
    the fields in these domains will be saved immediately after running the
    simulation. This is useful in situations where the same fields will be
    retrieved multiple times between simulations and in cases where the fields
    need to be accessed from the master node without involving other
    higher-rank nodes.

    Methods
    -------
    solve_forward(self)
        Simulate Maxwell's equations
    solve_adjoint(self)
        Simulate the transposed Maxwell's equations
    get_field(self, component)
        Get the desired component of the raw (uninterpolated) field
    get_field_interp(self, component)
        Get the desired component of the interpolated field
    get_adjoint_field(self, component)
        Get the desired component of the raw (uninterpolated) adjoint field
    get_adjoint_field_interp(self, component)
        Get the desired component of the interpolated adjoint field
    build(self)
        Build the system matrix :math:`A`.
    update(self)
        Update the system matrix :math:`A`
    set_sources(self, src)
        Set the discretized current sources used in the forward simulation.
    set_adjoint_sources(self, src)
        Set the discretized current sources used in the adjoint simulation.
    get_source_power(self)
        Get the total power generated by the sources.
    get_A_diag(self, vdiag=None)
        Retrieve the diagonal elements of :math:`A`

    Attributes
    ----------
    nunks
        The number of unknowns solved for in the simulation.
    field_domains
        The list of DomainCoordinates in which fields are recorded immediately
        following a forward solve.
    saved_fields
        The list of fields saved in in the stored DomainCoordinates
    source_power
        The source power injected into the system.
    """

    def __init__(self, ndims):
        super(FDFD, self).__init__(ndims)
        self._A = PETSc.Mat()
        self._A.create(PETSc.COMM_WORLD)

        # number of unknowns (field componenets * grid points)
        self._nunks = 0

    @property
    def nunks(self):
        return self._nunks

    def get_A_diag(self, vdiag=None):
        """Get the diagonal entries of the system matrix A.

        TODO
        ----
        Implement this using a sub-class-dependent in-place copy to speed
        things up?

        Parameters
        ----------
        vdiag : petsc4py.PETSc.Vec
            Vector with dimensions Mx1 where M is equal to the number of
            diagonal entries in A.

        Returns
        -------
        **(Master node only)** the diagonal entries of A.
        """
        if(vdiag == None):
            vdiag = PETSc.Vec()
        self._A.getDiagonal(vdiag)

        #scatter, vdiag_full = PETSc.Scatter.toZero(vdiag)
        #scatter.scatter(vdiag, vdiag_full, False, PETSc.Scatter.Mode.FORWARD)

        #if(NOT_PARALLEL):
        #    return vdiag_full[...]
        return vdiag

    @abstractmethod
    def calc_ydAx(self, Adiag0):
        """Calculate y^T * (A1-A0) * x.

        Parameters
        ----------
        Adiag0 : PETSc.Vec
            The diagonal of the FDFD matrix.

        Returns
        -------
        complex
            The product y^T * (A1-A0) * x
        """
        pass

    @run_on_master
    def spy_A(self):
        """Visualize A.

        This is only run on the master node. If you want to visualize the whole
        A matrix, the code must be run on a single processor. In general, dont
        use this unless you know what you are doing.
        """
        ai, aj, av = self._A.getValuesCSR()

        from scipy.sparse import csr_matrix
        import matplotlib.pyplot as plt

        A_csr = csr_matrix((av, aj, ai))

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.spy(A_csr, markersize=5)
        plt.show()


class FDFD_TE(FDFD):
    """Simulate Maxwell's equations in 2D with TE-polarized fields.

    Notes
    -----
    1. Units used for length parameters do not matter as long as they are
    consistent with one another

    2. The width and the height of the system will be modified in order to
    ensure that they are an integer multiple of dx and dy. It is important that
    this modified width and height be used in any future calculations.

    Parameters
    ----------
    X : float
        Approximate width of the simulation region. If this is not an integer
        multiple of dx, this will be increased slightly
    Y : float
        Approximate height of the simulation region. If this is not an integer
        multiple of dy, this will be increased slightly
    dx : float
        Grid spacing in the x direction.
    dy : float
        Grid spacing in the y direction.
    wavelength : float
        Vacuum wavelength of EM fields.
    solver : str
        The type of solver to use. The possible options are 'direct' (direct LU
        solver), 'iterative' (unpreconditioned iterative solver), 'iterative_lu'
        (iterative solver with LU preconditioner, or 'auto'. (default = 'auto')
    ksp_solver : str
        The type of Krylov subspace solver to use. See the petsc4py documentation for
        the possible types. Note: this flag is ignored if 'direct' or 'auto' is
        chosen for the solver parameter. (default = 'gmres')
    verbose : boolean (Optional)
        Indicates whether or not progress info should be printed on the master
        node. (default=True)

    Attributes
    ----------
    dx : float
        The grid spacing in the x direction
    dy : float
        The grid spacing in the y direction
    W : float
        The width of the simulation region (including PMLs)
    H : float
        The height of the simulation region (including PMLs)
    X : float
        The width of the simulation region (including PMLs)
    Y : float
        The height of the simulation region (including PMLs)
    M : int
        The number of grid cells in the y direction
    N : int
        The number of grid cells in the x direction
    wavelength : float
        The wavelength used for simulation. [arbitrary length unit]
    w_pml_left : int
        The number of grid cells which make up the left PML region
    w_pml_right : int
        The number of grid cells which make up the right PML region
    w_pml_top : int
        The number of grid cells which make up the top PML region
    w_pml_bottom : int
        The number of grid cells which make up the bottom PML region
    field_domains : list of DomainCoordinates
        The list of DomainCoordinates in which fields are recorded immediately
        following a forward solve.
    saved_fields : list of numpy.ndarray
        The list of (Ez, Hx, Hy) fields saved in in the stored
        DomainCoordinates
    source_power : float
        The source power injected into the system.
    w_pml : list of float
        List of PML widths in real spatial units. This variable can be
        reinitialized in order to change the PML widths
    real_materials : bool
        If True, assume that epsilon and mu are real-valued. This can speed up
        some calculations. (defaults to False)
    """

    def __init__(self, X, Y, dx, dy, wavelength, solver='auto',
                 ksp_solver='gmres'):
        super(FDFD_TE, self).__init__(2)

        # Temporary
        W = X
        H = Y

        self._dx = dx
        self._dy = dy
        self._wlen = wavelength

        # scalaing factor used in non-dimensionalizing spatial units
        self._R = wavelength/(2*pi)

        # pml widths for left, right, top, bottom
        self._w_pml = [wavelength/2 for i in range(4)]
        Npx = wavelength/2/dx
        Npy = wavelength/2/dy
        self._w_pml_left = int(Npx)
        self._w_pml_right = int(Npx)
        self._w_pml_top = int(Npy)
        self._w_pml_bottom = int(Npy)

        # Boundary conditions. Default type is PEC on all sim boundaries
        # Note: This will result in PMC boundaries for the 2D TM simulations.
        # This should be a non-issue as long as PMLs are used
        self._bc = ['0', '0']

        # PML parameters -- these can be changed
        self.pml_sigma = 3.0
        self.pml_power = 3.0

        # dx and dy are the only dimension rigorously enforced
        self._M = int(np.ceil(H/dy) + 1)
        self._N = int(np.ceil(W/dx) + 1)

        # The width and height are as close to the desired W and H as possible
        # given the desired grid spacing
        self._W = (self._N - 1) * dx
        self._H = (self._M - 1) * dy


        self._eps = None
        self._mu = None
        self.real_materials = False

        # factor of 3 due to 3 field components
        self.Nc = 3
        self._nunks = self.Nc*self._M*self._N
        self._A.setSizes([self._nunks, self._nunks])
        self._A.setType('aij')
        self._A.setUp()

        self._workvec = PETSc.Vec().create()
        self._workvec.setSizes(self._nunks)
        self._workvec.setUp()

        #obtain solution and RHS vectors
        x, b = self._A.getVecs()
        x.set(0)
        b.set(1)
        self.x = x
        self.b = b

        self.x_adj = x.copy()
        self.b_adj = b.copy()

        self._Adiag1 = x.copy() # used for retrieving Adiag in calc_ydAx

        self.ib, self.ie = self._A.getOwnershipRange()
        self.A_diag_update = np.zeros(self.ie-self.ib, dtype=np.complex128)

        # iterative or direct
        # create an iterative linear solver
        self.ksp_iter = PETSc.KSP()
        self.ksp_iter.create(PETSc.COMM_WORLD)

        self.ksp_iter.setType(ksp_solver)
        self.ksp_iter.setInitialGuessNonzero(True)
        if(ksp_solver == 'gmres'):
            self.ksp_iter.setGMRESRestart(1000)
        pc = self.ksp_iter.getPC()
        if(solver == 'iterative_lu'):
            pc.setType('lu')

            # setFactorSolverPackage was renamed recently
            try:
                pc.setFactorSolverPackage('mumps')
            except AttributeError as ae:
                pc.setFactorSolverType('mumps')
            pc.setReusePreconditioner(True)
        else:
            pc.setType('none')

        # create a direct linear solver
        self.ksp_dir = PETSc.KSP()
        self.ksp_dir.create(PETSc.COMM_WORLD)

        self.ksp_dir.setType('preonly')
        pc = self.ksp_dir.getPC()
        pc.setType('lu')

        try:
            pc.setFactorSolverPackage('mumps')
        except AttributeError as ae:
            pc.setFactorSolverType('mumps')

        self._solver_type = solver

        self.Ez = np.array([])
        self.Hx = np.array([])
        self.Hy = np.array([])

        self.Ez_adj = np.array([])
        self.Hx_adj = np.array([])
        self.Hy_adj = np.array([])

        self.verbose = True
        self._built = False

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def W(self):
        """
        The width of the systme. This width will only exactly match the width
        passed during initialization if it is equal to an integer multiple of
        grid cells.
        """
        warning_message('The W property is deprecated. Use X instead.',
                        'FDFD_TE')
        return self._W

    @property
    def H(self):
        """
        The height of the systme. This height will only exactly match the
        height passed during initialization if it is equal to an integer
        multiple of grid cells.
        """
        warning_message('The H property is deprecated. Use Y instead.',
                        'FDFD_TE')
        return self._H

    @property
    def X(self):
        """This is needed for MaxwellSolver implementation.

        TODO
        ----
        Replace self._W with self._X
        """
        return self._W

    @property
    def Y(self):
        """Also needed for MaxwellSolver implementation.

        TODO
        ----
        Replace self._H with self._Y
        """
        return self._H

    @property
    def Wreal(self):
        """Width of the simulation, excluding PMLs.
        """
        warning_message('The Wreal property is deprecated. Use Xreal instead.',
                       'FDFD_TE')
        return self._W - self._w_pml_left*self._dx - self._w_pml_right*self._dx

    @property
    def Hreal(self):
        """Height of the simulation, excluding PMLs.
        """
        warning_message('The Hreal property is deprecated. Use Yreal instead.',
                       'FDFD_TE')
        return self._H - self._w_pml_top*self._dy - self._w_pml_bottom*self._dy

    @property
    def Xreal(self):
        """Width of the simulation, excluding PMLs.
        """
        return self._W - self._w_pml_left*self._dx - self._w_pml_right*self._dx

    @property
    def Yreal(self):
        """Height of the simulation, excluding PMLs.
        """
        return self._H - self._w_pml_top*self._dy - self._w_pml_bottom*self._dy

    @property
    def M(self):
        return self._M

    @property
    def N(self):
        return self._N

    @property
    def wavelength(self):
        return self._wlen

    @property
    def w_pml_left(self):
        return self._w_pml_left

    @property
    def w_pml_right(self):
        return self._w_pml_right

    @property
    def w_pml_top(self):
        return self._w_pml_top

    @property
    def w_pml_bottom(self):
        return self._w_pml_bottom

    @property
    def solver_type(self):
        return self._solver_type

    @solver_type.setter
    def solver_type(self, val):
        self._solver_type = val

    @wavelength.setter
    def wavelength(self, val):
        if(val > 0):
            self._wlen = val
            self._R = val/(2*pi)
            self._built = False
        else:
            raise ValueError('Wavelength must be a positive number!')

    @property
    def eps(self):
        return self._eps

    @property
    def mu(self):
        return self._mu

    @property
    def w_pml(self):
        return self._w_pml

    @w_pml.setter
    def w_pml(self, val):
        self._w_pml = val

        dx = self._dx
        dy = self._dy
        self._w_pml_left = int(val[0]/dx)
        self._w_pml_right = int(val[1]/dx)
        self._w_pml_top = int(val[2]/dy)
        self._w_pml_bottom = int(val[3]/dy)

        self._built = False

    @property
    def bc(self):
        return ''.join(self._bc)

    @bc.setter
    def bc(self, val):
        self._bc = list(val)
        self._built = False

        if(val[0] in 'EH' and self._w_pml_left != 0):
            warning_message('Symmetry imposed on right boundary with finite width PML.',
                            'emopt.fdfd')

        if(val[1] in 'EH' and self._w_pml_bottom != 0):
            warning_message('Symmetry imposed on bottom boundary with finite width PML.',
                            'emopt.fdfd')

        if(val[0] in 'PB' and (self._w_pml_left != 0 or self._w_pml_right !=0)):
            warning_message('Periodicity imposed along x direction with finite width PML.',
                            'emopt.fdfd')

        if(val[1] in 'PB' and (self._w_pml_top != 0 or self._w_pml_bottom !=0)):
            warning_message('Periodicity imposed along y direction with finite width PML.',
                            'emopt.fdfd')

        for v in val:
            if(v not in '0MEHPB'):
                error_message('Boundary condition type %s unknown. Use 0, M, E, H, '
                              'P, or B.' % (v))

    def set_materials(self, eps, mu):
        """Set the material distributions of the system to be simulated.

        Parameters
        ----------
        eps : emopt.grid.Material
            The spatially-dependent permittivity of the system
        mu : emopt.grid.Material
            The spatially-dependent permeability of the system
        """
        self._eps = eps
        self._mu = mu

    def set_sources(self, src, src_domain=None, mindex=0):
        """Set the sources of the system used in the forward solve.

        The sources can be defined either using three numpy arrays or using a
        mode source. When using a mode source, the corresponding current
        density arrays will be automatically generated and extracted.

        Notes
        -----
        Like the underlying fields, the current sources are represented on a
        set of shifted grids.  In particular, :math:`J_z`'s are all located at
        the center of a grid cell, the :math:`M_x`'s are shifted in the
        positive y direction by half a grid cell, and the :math:`M_y`'s are
        shifted in the negative x direction by half of a grid cell.  It is
        important to take this into account when defining the current sources.

        Todo
        ----
        1. Implement a more user-friendly version of these sources (so that you do
        not need to deal with the Yee cell implementation).

        2. Implement this in a better parallelized way

        Parameters
        ----------
        src : tuple of numpy.ndarray or modes.ModeTE
            (Option 1) The current sources in the form (Jz, Mx, My).  Each array in the
            tuple should be a 2D numpy.ndarry with dimensions MxN.
            (Option 2), a mode source which has been built and solved.
        src_domain : emopt.misc.DomainCoordinates (optional)
            Specifies the location of the provided current source distribution
            or mode source. If None, it is assumed that source arrays have been
            provided and those source arrays span the whole simulation region
            (i.e. have size MxN)
        mindx : int (optional)
            Specifies the index of the mode source (only used if a mode source
            is passed in as src)
        """
        if(isinstance(src, modes.ModeTE)):
            Jz = np.zeros((self._M, self._N), dtype=np.complex128)
            Mx = np.zeros((self._M, self._N), dtype=np.complex128)
            My = np.zeros((self._M, self._N), dtype=np.complex128)

            msrc = src.get_source(mindex, self._dx, self._dy)

            Jz[src_domain.j, src_domain.k] = msrc[0]
            Mx[src_domain.j, src_domain.k] = msrc[1]
            My[src_domain.j, src_domain.k] = msrc[2]

            self.Jz = Jz
            self.Mx = Mx
            self.My = My

        elif(src_domain is not None):
            Jz = np.zeros((self._M, self._N), dtype=np.complex128)
            Mx = np.zeros((self._M, self._N), dtype=np.complex128)
            My = np.zeros((self._M, self._N), dtype=np.complex128)

            Jz[src_domain.j, src_domain.k] = src[0]
            Mx[src_domain.j, src_domain.k] = src[1]
            My[src_domain.j, src_domain.k] = src[2]

            self.Jz = Jz
            self.Mx = Mx
            self.My = My
        else:
            self.Jz = src[0]
            self.Mx = src[1]
            self.My = src[2]

        src_arr = np.zeros(self.Nc*self._M*self._N, dtype=np.complex128)
        src_arr[0::self.Nc] = self.Jz.ravel()
        src_arr[1::self.Nc] = self.Mx.ravel()
        src_arr[2::self.Nc] = self.My.ravel()

        self.b.setArray(src_arr[self.ib:self.ie])

    def set_adjoint_sources(self, src):
        """Set the sources of the system used in the adjoint solve.

        The details of these sources are identical to the forward solution
        current sources.

        Parameters
        ----------
        src : tuple of numpy.ndarray
            The current sources in the form (Jz_adj, Mx_adj, My_adj).  Each array
            in the tiple should be a 2D numpy.ndarry with dimensions MxN.
        """
        self.Jz_adj = src[0]
        self.Mx_adj = src[1]
        self.My_adj = src[2]

        src_arr = np.zeros(self.Nc*self._M*self._N, dtype=np.complex128)
        src_arr[0::self.Nc] = self.Jz_adj.ravel()
        src_arr[1::self.Nc] = self.Mx_adj.ravel()
        src_arr[2::self.Nc] = self.My_adj.ravel()

        self.b_adj.setArray(src_arr[self.ib:self.ie])

    def __get_pml_x(self):
        ## Generate the PML values for the left and right boundaries.
        # this process is vectorized in order to speed things up.  The result
        # is a 2D numpy.ndarray which can be quickly accessed when the system
        # needs to be built
        M = self._M
        N = self._N

        x = np.arange(0,N)
        y = np.arange(0,M)
        X,Y = np.meshgrid(x,y)

        pml_x_Ez = np.ones([M,N], dtype=np.complex128)
        pml_x_Hy = np.ones([M,N], dtype=np.complex128)


        # define the left PML
        w_pml = self._w_pml_left
        x = X[:, 0:w_pml]
        pml_x_Ez[:, 0:w_pml] = 1.0 / (1.0 + 1j*self.pml_sigma *
                                     ((w_pml - x)*1.0/w_pml)**self.pml_power)
        pml_x_Hy[:, 0:w_pml] = 1.0 / (1.0 + 1j*self.pml_sigma *
                                     ((w_pml - x + 0.5)*1.0/w_pml)**self.pml_power)

        # define the right PML
        w_pml = self._w_pml_right
        x = X[:, N-w_pml:]
        pml_x_Ez[:, N-w_pml:] = 1.0 / (1.0 + 1j*self.pml_sigma *
                                      ((x-N+w_pml)*1.0/w_pml)**self.pml_power)
        pml_x_Hy[:, N-w_pml:] = 1.0 / (1.0 + 1j*self.pml_sigma *
                                      ((x-N+w_pml-0.5)*1.0/w_pml)**self.pml_power)

        return pml_x_Ez, pml_x_Hy

    def __get_pml_y(self):
        ## Generate the PML values for the top and bottom boundaries.
        # this process is vectorized in order to speed things up.  The result
        # is a 2D numpy.ndarray which can be quickly accessed when the system
        # needs to be built
        M = self._M
        N = self._N

        x = np.arange(0,N)
        y = np.arange(0,M)
        X,Y = np.meshgrid(x,y)

        pml_y_Ez = np.ones([M,N], dtype=np.complex128)
        pml_y_Hx = np.ones([M,N], dtype=np.complex128)

        # PML for bottom
        w_pml = self._w_pml_bottom
        y = Y[0:w_pml, :]
        pml_y_Ez[0:w_pml, :] = 1.0 / (1.0 + 1j*self.pml_sigma *
                                     ((w_pml - y)*1.0/w_pml)**self.pml_power)
        pml_y_Hx[0:w_pml, :] = 1.0 / (1.0 + 1j*self.pml_sigma *
                                     ((w_pml - y - 0.5)*1.0/w_pml)**self.pml_power)
        # PML for top
        w_pml = self._w_pml_top
        y = Y[M-w_pml:, :]
        pml_y_Ez[M-w_pml:, :] = 1.0 / (1.0 + 1j*self.pml_sigma *
                                      ((y-M+w_pml)*1.0/w_pml)**self.pml_power)
        pml_y_Hx[M-w_pml:, :] = 1.0 / (1.0 + 1j*self.pml_sigma *
                                      ((y-M+w_pml+0.5)*1.0/w_pml)**self.pml_power)

        return pml_y_Ez, pml_y_Hx

    def build(self):
        """(Re)Build the system matrix.

        Maxwell's equations are solved by compiling them into a linear system
        of the form :math:`Ax=b`. Here, we build up the structure of A which contains
        the curl operators, mateiral parameters, and boundary conditions.

        This function must be called at least once after the material
        distributions of the system have been defined (through a call to
        :func:`set_materials`).

        Notes
        -----
        1. In the current Yee cell configuration, the :math:`E_z`'s are
        positioned at the center of the cell, the :math:`H_x`'s are shifted in
        the positive y direction by half of a grid cell, and the :math:`H_y`'s
        are shifted in the negative x direction by half a grid cell.

        2. This function can be rather slow for larger problems.  In general,
        callin this function more than once should be avoided.  Instead, the
        :func:`update` function should be sufficient when only the material
        distributions of the system have been modified

        3. The current PML implementation is not likely ideal. Nonetheless, it
        seems to work ok.

        Raises
        ------
        Exception
            If the material distributions of the simulation have not been set
            prior to calling this function.
        """
        # store local versions of class variables for a bit of speedup
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

                #j1 = i+1
                #j2 = (y-1)*N + x

                # Diagonal element is the permittivity at (x,y)
                A[i,jEz] = 1j * eps.get_value(x,y)

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
                A[i,jHx] = -1j * mu.get_value(x,y+0.5)

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
                A[i,jHy] = -1j * mu.get_value(x-0.5,y)
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

        self._built = True

    def update(self, bbox=None):
        """Update only the material values stored in A.

        In many situations, it is desirable to be able to modify the physical
        structure of the simulation. Doing so only requires updating the
        material data in A, which corresponds to the diagonal elements. This is
        a much faster.

        In many cases, only a small part of the full underlying grid needs to
        be updated.  In order to handle this, an optional bounding box can be
        passed to limit the extent of the update.

        To further improve performance, the bulk of this function has been
        implemented directly in C++.  Refer to Grid_ctypes.hpp for details.

        Parameters
        ----------
        bbox : tuple of ints
            bounding box defining the block of x and y indices to update in the
            format (x1, x2, y1, y2). If None, then the whole grid is updated.
            (default = None)
        """
        A = self._A
        M = self._M
        N = self._N
        eps = self._eps
        mu = self._mu

        if(not self._built):
            warning_message('The system should be built prior to being \
                            updated. Expect unexpect behavior.', \
                            module='emopt.fdfd')

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

        self.A_diag_update = row_wise_A_update(eps, mu, self.ib, self.ie, M, N,\
                                               x1, x2, y1, y2, \
                                               self.A_diag_update)

        A_update = self.A_diag_update
        self._workvec.setValues(np.arange(self.ib, self.ie, dtype=np.int32), A_update)
        A.setDiagonal(self._workvec, addv=PETSc.InsertMode.INSERT_VALUES)

        # communicate off-processor values and setup internal data structures for
        # performing parallel operations
        A.assemblyBegin()
        A.assemblyEnd()

    def solve_forward(self):
        """Solve the forward simulation.

        This is equivalent to a solution of Maxwell's equations, which is
        achieved by either iteratively or directly solving :math:`Ax=b`.

        After the completing the forward solve, the full solution contained in
        x is gathered on the master node so that the fields can be accessed.

        """
        if(not self._built):
            error_message('The system matrix has not be been built. Call' \
                          ' self.build before running a simulation')

        if(self.verbose and NOT_PARALLEL):
            info_message('Running forward solver...')

        # setup and solve Ax=b using petsc4py
        # unless otherwise specified, MUMPS (direct solver) is used.
        # Alternatively, the bicgstab iterative solver may be used.
        if(self._solver_type == 'iterative' or self._solver_type == 'iterative_lu'):
            ksp = self.ksp_iter
            ksp.setOperators(self._A, self._A)
            ksp.setFromOptions()

        elif(self._solver_type == 'direct' or self._solver_type == 'auto'):
            ksp = self.ksp_dir
            ksp.setOperators(self._A, self._A)
            ksp.setFromOptions()

        ksp.solve(self.b, self.x)

        if(RANK == 0):
            convergence = ksp.getConvergedReason()
            if(convergence < 0):
                error_message('Forward solution did not converge with error '
                              'code %d.' % (convergence))

        # Save the full result on the master node so it can be accessed in the
        # future
        scatter, x_full = PETSc.Scatter.toZero(self.x)
        scatter.scatter(self.x, x_full, False, PETSc.Scatter.Mode.FORWARD)

        if(NOT_PARALLEL):
            fields = x_full[...]
            self.fields = fields

            MN = self._M*self._N

            Nc = self.Nc
            self.Ez = np.reshape(fields[0::Nc], [self._M, self._N])
            self.Hx = np.reshape(fields[1::Nc], [self._M, self._N])
            self.Hy = np.reshape(fields[2::Nc], [self._M, self._N])

        # store the source power
        self._source_power = self.get_source_power()

        self.update_saved_fields()

    def update_saved_fields(self):
        # collect data on the field domains
        del self._saved_fields
        self._saved_fields = []
        for d in self._field_domains:
            Ez = self.get_field_interp('Ez', d)
            Hx = self.get_field_interp('Hx', d)
            Hy = self.get_field_interp('Hy', d)
            self._saved_fields.append((Ez, Hx, Hy))

    def solve_adjoint(self):
        """Solve the adjoint simulation.

        The adjoint simulation a transposed version of the forward simulation
        of the form :math:`A^T u = c`.  The adjoint simulation is very useful for
        calculating sensitivies of the structure using the adjoint method.
        """
        if(self.verbose and NOT_PARALLEL):
            info_message('Running adjoint solver...')

        # setup and solve A^Tu=c using petsc4py
        # unless otherwise specified, MUMPS (direct solver) is used.
        # Alternatively, the bicgstab iterative solver may be used.
        if(self._solver_type == 'iterative' or self._solver_type == 'iterative_lu'):
            ksp = self.ksp_iter
            ksp.setOperators(self._A, self._A)
            ksp.setFromOptions()

        elif(self._solver_type == 'direct' or self._solver_type == 'auto'):
            ksp = self.ksp_dir
            ksp.setOperators(self._A, self._A)
            ksp.setFromOptions()

        ksp.solveTranspose(self.b_adj, self.x_adj)

        if(NOT_PARALLEL):
            convergence = ksp.getConvergedReason()
            if(convergence < 0):
                error_message('Adjoint solution did not converge.')

        # Save the full result on the master node so it can be accessed in the
        # future
        scatter, x_adj_full = PETSc.Scatter.toZero(self.x_adj)
        scatter.scatter(self.x_adj, x_adj_full, False, PETSc.Scatter.Mode.FORWARD)

        if(NOT_PARALLEL):
            fields = x_adj_full[...]
            self.fields_adj = fields

            MN = self._M*self._N

            Nc = self.Nc
            self.Ez_adj = np.reshape(fields[0::Nc], [self._M, self._N])
            self.Hx_adj = np.reshape(fields[1::Nc], [self._M, self._N])
            self.Hy_adj = np.reshape(fields[2::Nc], [self._M, self._N])

    def get_field(self, component, domain=None):
        """Get the desired (forward) field component.

        This function returns the RAW field which is defined on the dislocated
        grids.  In most situations, the :func:`get_field_interp` function
        should be prefered.

        Notes
        -----
        This function only returns a non-empty field on the master node.

        Parameters
        ----------
        component : str
            The desired field component to be retrieved (either 'Ez', 'Hx', or
            'Hy')
        domain : emopt.misc.DomainCoordinates
            The domain in which the field is retrieved


        Raises
        ------
        ValueError
            If the supplied component is not 'Ez', 'Hx', or 'Hy'

        Returns
        -------
        numpy.ndarray
            (Master node only) A 2D numpy.ndarray containing the desired field
            component
        """
        if(RANK != 0):
            return MathDummy()

        if(domain is not None):
            j = domain.j
            k = domain.k

        if(component == 'Ez'):
            if(domain is not None):
                return self.Ez[j,k]
            else:
                return np.copy(self.Ez)
        elif(component == 'Hx'):
            if(domain is not None):
                return self.Hx[j,k]
            else:
                return np.copy(self.Hx)
        elif(component == 'Hy'):
            if(domain is not None):
                return self.Hy[j,k]
            else:
                return np.copy(self.Hy)
        else:
            raise ValueError('Unrecongnized field componenet "%s". The allowed'
                             'field components are Ez, Hx, Hy.' % (component))

    def get_field_interp(self, component, domain=None):
        """Get the desired (forward) interpolated field.

        When solving Maxwell's equations on a rectangular grid, we actually set
        up a set of dislocated grids in which the electric and magnetic field
        components are computed at slightly different positions in space.  As
        as result, it is often convenient or even necessary to interpolate the
        fields such that they are all known at the same points in space.  This
        is particularly important when computing power-related quantities.

        For the sake of simplicity, a simple linear interpolation scheme is
        used to compute Hx and Hy at the position of Ez.  Ez is the same as the
        uninterpolated Ez field.

        Parameters
        ----------
        component : str
            The desired field component to be retrieved (either 'Ez', 'Hx', or
            'Hy')
        domain : emopt.misc.DomainCoordinates
            The domain in which the field is retrieved

        Raises
        ------
        ValueError
            If the supplied component is not 'Ez', 'Hx', or 'Hy'

        Returns
        -------
        numpy.ndarray
            (Master node only) A 2D numpy.ndarray containing the desired field
            component
        """
        bc = self._bc
        if(RANK != 0):
            return MathDummy()

        if(domain is not None):
            j = domain.j
            k = domain.k

        if(component == 'Ez'):
            if(domain is not None):
                return self.Ez[j, k]
            else:
                return np.copy(self.Ez)
        elif(component == 'Hx'):
            Hx = np.pad(self.Hx, 1, 'constant', constant_values=0)
            if(bc[1] == 'E'):
                Hx[0,:] = -1*Hx[1,:]
            elif(bc[1] == 'H'):
                Hx[0,:] = Hx[1,:]
            elif(bc[1] == 'P'):
                Hx[0,:] = Hx[-2,:]
                Hx[-1,:] = Hx[1,:]

            Hx0 = np.copy(Hx)
            Hx0[1:, :] += Hx[0:-1,:]
            Hx0 = Hx0[1:-1, 1:-1]

            if(domain is not None):
                return Hx0[j, k]/2.0
            else:
                return Hx0 / 2.0
        elif(component == 'Hy'):
            Hy = np.pad(self.Hy, 1, 'constant', constant_values=0)
            if(bc[0] == 'E'):
                Hy[:,0] = -1*Hy[:,1]
            elif(bc[0] == 'H'):
                Hy[:,0] = Hy[:,1]
            elif(bc[0] == 'P'):
                Hy[:,0] = Hy[:,-2]
                Hy[:,-1] = Hy[:,1]

            Hy0 = np.copy(Hy)
            Hy0[:, 0:-1] += Hy[:, 1:]
            Hy0 = Hy0[1:-1, 1:-1]

            if(domain is not None):
                return Hy0[j, k]/2.0
            else:
                return Hy0 / 2.0
        else:
            raise ValueError('Unrecongnized field componenet "%s". The allowed'
                             'field components are Ez, Hx, Hy.' % (component))

    def get_adjoint_field(self, component, domain=None):
        """Get the desired raw adjoint field component.

        Notes
        -----
        this function only returns a non-empty field on the master node.

        Parameters
        ----------
        component : str
            The desired field component to be retrieved (either 'Ez', 'Hx', or
            'Hy')
        domain : emopt.misc.DomainCoordinates
            The domain in which the field is retrieved

        Raises
        ------
        ValueError
            If the supplied component is not 'Ez', 'Hx', or 'Hy'

        Returns
        -------
        numpy.ndarray
            (Master node only) A numpy.ndarray containing the desired field component
        """
        if(RANK != 0):
            return MathDummy()

        if(domain is not None):
            j = domain.j
            k = domain.k

        if(component == 'Ez'):
            if(domain is not None):
                return self.Ez_adj[j, k]
            else:
                return self.Ez_adj
        elif(component == 'Hx'):
            if(domain is not None):
                return self.Hx_adj[j, k]
            else:
                return self.Hx_adj
        elif(component == 'Hy'):
            if(domain is not None):
                return self.Hy_adj[j, k]
            else:
                return self.Hy_adj
        else:
            raise ValueError('Unrecongnized field componenet "%s". The allowed'
                             'field components are Ez, Hx, Hy.' % (component))

    def get_source_power(self):
        """Get the source power.

        In general, FDFD_TE.source_power should be used to access the source
        power instead. The source power is computed following every forward
        solve, so calling this function explicitly is generally unnecessary.

        Notes
        -----
        1. The source power is computed using the interpolated fields.

        Returns
        -------
        float
            Electromagnetic power generated by source.
        """
        # easier access to PML widths
        w_pml_l = self._w_pml_left
        w_pml_r = self._w_pml_right
        w_pml_t = self._w_pml_top
        w_pml_b = self._w_pml_bottom

        M = self._M
        N = self._N
        dx = self._dx
        dy = self._dy

        Ez = self.get_field_interp('Ez')
        Hx = self.get_field_interp('Hx')
        Hy = self.get_field_interp('Hy')

        if(RANK != 0):
            return MathDummy()

        # calculate the Poynting vectors around the boundaries of the sytem
        # (excluding the PML regions)
        Sx = -0.5*(Ez*np.conj(Hy)).real
        Sy = 0.5*(Ez*np.conj(Hx)).real

        S1 = -Sy[w_pml_b, w_pml_l:N-1-w_pml_r]
        S2 = Sy[M-1-w_pml_t, w_pml_l:N-1-w_pml_r]
        S3 = -Sx[w_pml_b:M-1-w_pml_t, w_pml_l]
        S4 = Sx[w_pml_b:M-1-w_pml_t, N-1-w_pml_r]

        # total power flowing our of boundaries
        P_S = np.sum(S1 + S2) * dx + np.sum(S3 + S4) * dy

        x_all = np.arange(w_pml_l, N-w_pml_r)
        y_all = np.arange(w_pml_b, M-w_pml_t)
        y_all = y_all.reshape(y_all.shape[0], 1).astype(np.int)

        if(not self.real_materials):
            eps = self._eps.get_values(0, N, 0, M)
            mu = self._mu.get_values(0, N, 0, M)
        else:
            eps = np.zeros(Ez.shape, dtype=np.complex128)
            mu = np.zeros(Ez.shape, dtype=np.complex128)

        # power dissipated due to material absorption
        Ez2 = Ez[y_all, x_all]*np.conj(Ez[y_all, x_all])
        Hx2 = Hx[y_all, x_all]*np.conj(Hx[y_all, x_all])
        Hy2 = Hy[y_all, x_all]*np.conj(Hy[y_all, x_all])

        P_loss = 0.25 * dx * dy * np.sum(eps[y_all, x_all].imag*Ez2 + \
                                         mu[y_all, x_all].imag*(Hx2 + Hy2))

        return P_S + P_loss.real

    def calc_ydAx(self, Adiag0):
        """Calculate y^T * (A1-A0) * x.

        Parameters
        ----------
        Adiag0 : PETSc.Vec
            The diagonal of the FDFD matrix.

        Returns
        -------
        complex
            The product y^T * (A1-A0) * x
        """
        x = self.x
        y = self.x_adj
        Adiag1 = self._Adiag1
        self.get_A_diag(Adiag1)

        product = y * (Adiag1-Adiag0) * x
        return np.sum(product[...])

class FDFD_TM(FDFD_TE):
    """Simulate Maxwell's equations in 2D with TM-polarized fields.

    Notes
    -----
    1. Units used for length parameters do not matter as long as they are
    consistent with one another

    2. The width and the height of the system will be modified in order to
    ensure that they are an integer multiple of dx and dy. It is important that
    this modified width and height be used in any future calculations.

    Parameters
    ----------
    X : float
        Approximate width of the simulation region. If this is not an integer
        multiple of dx, this will be increased slightly
    Y : float
        Approximate height of the simulation region. If this is not an integer
        multiple of dy, this will be increased slightly
    dx : float
        Grid spacing in the x direction.
    dy : float
        Grid spacing in the y direction.
    wavelength : float
        Vacuum wavelength of EM fields.
    solver : str
        The type of solver to use. The possible options are 'direct' (direct LU
        solver), 'iterative' (unpreconditioned iterative solver), 'iterative_lu'
        (iterative solver with LU preconditioner, or 'auto'. (default = 'auto')
    ksp_solver : str
        The type of Krylov subspace solver to use. See the petsc4py documentation for
        the possible types. Note: this flag is ignored if 'direct' or 'auto' is
        chosen for the solver parameter. (default = 'gmres')
    verbose : boolean (Optional)
        Indicates whether or not progress info should be printed on the master
        node. (default=True)

    Attributes
    ----------
    dx : float
        The grid spacing in the x direction
    dy : float
        The grid spacing in the y direction
    W : float
        The width of the simulation region (including PMLs)
    H : float
        The height of the simulation region (including PMLs)
    X : float
        The width of the simulation region (including PMLs)
    Y : float
        The height of the simulation region (including PMLs)
    M : int
        The number of grid cells in the y direction
    N : int
        The number of grid cells in the x direction
    wavelength : float
        The wavelength used for simulation. [arbitrary length unit]
    w_pml_left : int
        The number of grid cells which make up the left PML region
    w_pml_right : int
        The number of grid cells which make up the right PML region
    w_pml_top : int
        The number of grid cells which make up the top PML region
    w_pml_bottom : int
        The number of grid cells which make up the bottom PML region
    field_domains : list of DomainCoordinates
        The list of DomainCoordinates in which fields are recorded immediately
        following a forward solve.
    saved_fields : list of numpy.ndarray
        The list of (Ez, Hx, Hy) fields saved in in the stored
        DomainCoordinates
    source_power : float
        The source power injected into the system.
    w_pml : list of float
        List of PML widths in real spatial units. This variable can be
        reinitialized in order to change the PML widths
    """

    def __init__(self, X, Y, dx, dy, wavelength, solver='auto',
                 ksp_solver='gmres'):
        super(FDFD_TM, self).__init__(X, Y, dx, dy, wavelength, solver=solver,
              ksp_solver=ksp_solver)

        self.bc = 'MM'

    @property
    def eps(self):
        return self._eps_actual

    @property
    def mu(self):
        return self._mu_actual

    @property
    def bc(self):
        val = self._bc[:]
        for i in range(2):
            if(val[i] == 'E'): val[i] = 'H'
            elif(val[i] == 'H'): val[i] = 'E'
            if(val[i] == '0'): val[i] = 'M'
            elif(val[i] == 'M'): val[i] = '0'

        return ''.join(val)

    @bc.setter
    def bc(self, val):
        self._bc = list(val)
        self._built = False

        # since TM solver uses TE build() function, we need to swap E and H
        # boundary conditions as well as 0 and M
        for i in range(2):
            if(self._bc[i] == 'E'): self._bc[i] = 'H'
            elif(self._bc[i] == 'H'): self._bc[i] = 'E'
            if(self._bc[i] == '0'): self._bc[i] = 'M'
            elif(self._bc[i] == 'M'): self._bc[i] = '0'

        if(val[0] in 'EH' and self._w_pml_left != 0):
            warning_message('Symmetry imposed on right boundary with finite width PML.',
                            'emopt.fdfd')

        if(val[1] in 'EH' and self._w_pml_bottom != 0):
            warning_message('Symmetry imposed on bottom boundary with finite width PML.',
                            'emopt.fdfd')

        if(val[0] in 'PB' and (self._w_pml_left != 0 or self._w_pml_right !=0)):
            warning_message('Periodicity imposed along x direction with finite width PML.',
                            'emopt.fdfd')

        if(val[1] in 'PB' and (self._w_pml_top != 0 or self._w_pml_bottom !=0)):
            warning_message('Periodicity imposed along y direction with finite width PML.',
                            'emopt.fdfd')

        for v in val:
            if(v not in '0MEHPB'):
                error_message('Boundary condition type %s unknown. Use 0, M, E, H, '
                              'P, or B.' % (v))


    def set_materials(self, eps, mu):
        """Set the material distributions of the system to be simulated.

        Parameters
        ----------
        eps : emopt.grid.Material
            The spatially-dependent permittivity of the system
        mu : emopt.grid.Material
            The spatially-dependent permeability of the system
        """
        # swap mu and eps to go from TE to TM
        super(FDFD_TM, self).set_materials(mu, eps)

        self._eps_actual = eps
        self._mu_actual = mu

    def set_sources(self, src, src_domain=None, mindex=0):
        """Set the sources of the system used in the forward solve.

        The sources can be defined either using three numpy arrays or using a
        mode source. When using a mode source, the corresponding current
        density arrays will be automatically generated and extracted.

        Notes
        -----
        Like the underlying fields, the current sources are represented on a
        set of shifted grids.  In particular, :math:`M_z`'s are all located at
        the center of a grid cell, the :math:`J_x`'s are shifted in the
        positive y direction by half a grid cell, and the :math:`J_y`'s are
        shifted in the negative x direction by half of a grid cell.  It is
        important to take this into account when defining the current sources.

        Todo
        ----
        1. Implement a more user-friendly version of these sources (so that you do
        not need to deal with the Yee cell implementation).

        2. Implement this in a better parallelized way

        Parameters
        ----------
        src : tuple of numpy.ndarray or modes.ModeTM
            (Option 1) The current sources in the form (Mz, Jx, Jy).  Each array in the
            tuple should be a 2D numpy.ndarry with dimensions MxN.
            (Option 2), a mode source which has been built and solved.
        src_domain : emopt.misc.DomainCoordinates (optional)
            Specifies the location of the provided current source distribution
            or mode source. If None, it is assumed that source arrays have been
            provided and those source arrays span the whole simulation region
            (i.e. have size MxN)
        mindx : int (optional)
            Specifies the index of the mode source (only used if a mode source
            is passed in as src)
        """
        if(isinstance(src, modes.ModeTM)):
            msrc = src.get_source(mindex, self._dx, self._dy)

            Mz = msrc[0]
            Jx = msrc[1]
            Jy = msrc[2]
        else:
            Mz = src[0]
            Jx = src[1]
            Jy = src[2]
        # In order to properly make use of the TE subclass, we need to flip the
        # sign of Jx and Jy
        super(FDFD_TM, self).set_sources((Mz, -1*Jx, -1*Jy),
                                         src_domain, mindex)

    def set_adjoint_sources(self, src):
        """Set the sources of the system used in the adjoint solve.

        The details of these sources are identical to the forward solution
        current sources.

        Parameters
        ----------
        src : tuple of numpy.ndarray
            The current sources in the form (Mz_adj, Jx_adj, Jy_adj).  Each array 
            in the tiple should be a 2D numpy.ndarry with dimensions MxN.
        """
        # In order to properly make use of the TE subclass, we need to flip the
        # sign of Jx_adj and Jy_adj
        super(FDFD_TM, self).set_adjoint_sources((src[0], -1*src[1], -1*src[2]))

    def build(self):
        """(Re)Build the system matrix.

        Maxwell's equations are solved by compiling them into a linear system
        of the form :math:`Ax=b`. Here, we build up the structure of A which contains
        the curl operators, mateiral parameters, and boundary conditions.

        This function must be called at least once after the material
        distributions of the system have been defined (through a call to
        :func:`set_materials`).

        Notes
        -----
        1. In the current Yee cell configuration, the :math:`H_z`'s are
        positioned at the center of the cell, the :math:`E_x`'s are shifted in
        the positive y direction by half of a grid cell, and the :math:`E_y`'s
        are shifted in the negative x direction by half a grid cell.

        2. This function can be rather slow for larger problems.  In general,
        callin this function more than once should be avoided.  Instead, the
        :func:`update` function should be sufficient when only the material
        distributions of the system have been modified

        3. The current PML implementation is not likely ideal. Nonetheless, it
        seems to work ok.

        Raises
        ------
        Exception
            If the material distributions of the simulation have not been set
            prior to calling this function.
        """
        super(FDFD_TM, self).build()



    def get_field(self, component, domain=None):
        """Get the desired (forward) field component.

        This function returns the RAW field which is defined on the dislocated
        grids.  In most situations, the :func:`get_field_interp` function
        should be prefered.

        Notes
        -----
        This function only returns a non-empty field on the master node.

        Parameters
        ----------
        component : str
            The desired field component to be retrieved (either 'Hz', 'Ex', or
            'Ey')
        domain : emopt.misc.DomainCoordinates
            The domain in which the field is retrieved

        Raises
        ------
        ValueError
            If the supplied component is not 'Hz', 'Ex', or 'Ey'

        Returns
        -------
        numpy.ndarray
            (Master node only) A 2D numpy.ndarray containing the desired field
            component
        """
        te_comp = ''
        if(component == 'Hz'): te_comp = 'Ez'
        elif(component == 'Ex'): te_comp = 'Hx'
        elif(component == 'Ey'): te_comp = 'Hy'
        else: te_comp = component

        field = super(FDFD_TM, self).get_field(te_comp, domain)

        if(component == 'Hz'):
            return -1 * field
        else:
            return field

    def get_field_interp(self, component, domain=None):
        """Get the desired (forward) interpolated field.

        When solving Maxwell's equations on a rectangular grid, we actually set
        up a set of dislocated grids in which the electric and magnetic field
        components are computed at slightly different positions in space.  As
        as result, it is often convenient or even necessary to interpolate the
        fields such that they are all known at the same points in space.  This
        is particularly important when computing power-related quantities.

        For the sake of simplicity, a simple linear interpolation scheme is
        used to compute Ex and Ey at the position of Hz.  Hz is the same as the
        uninterpolated Hz field.

        Parameters
        ----------
        component : str
            The desired field component to be retrieved (either 'Hz', 'Ex', or
            'Ey')
        domain : emopt.misc.DomainCoordinates
            The domain in which the field is retrieved

        Raises
        ------
        ValueError
            If the supplied component is not 'Hz', 'Ex', or 'Ey'

        Returns
        -------
        numpy.ndarray
            (Master node only) A 2D numpy.ndarray containing the desired field
            component
        """
        te_comp = ''
        if(component == 'Hz'): te_comp = 'Ez'
        elif(component == 'Ex'): te_comp = 'Hx'
        elif(component == 'Ey'): te_comp = 'Hy'
        else: te_comp = component

        field = super(FDFD_TM, self).get_field_interp(te_comp, domain)

        if(component == 'Hz'):
            return -1 * field
        else:
            return field


    def get_adjoint_field(self, component, domain=None):
        """Get the desired raw adjoint field component.

        Notes
        -----
        this function only returns a non-empty field on the master node.

        Parameters
        ----------
        component : str
            The desired field component to be retrieved (either 'Hz', 'Ex', or
            'Ey')
        domain : emopt.misc.DomainCoordinates
            The domain in which the field is retrieved

        Raises
        ------
        ValueError
            If the supplied component is not 'Hz', 'Ex', or 'Ey'

        Returns
        -------
        numpy.ndarray
            (Master node only) A numpy.ndarray containing the desired field component
        """
        te_comp = ''
        if(component == 'Hz'): te_comp = 'Ez'
        elif(component == 'Ex'): te_comp = 'Hx'
        elif(component == 'Ey'): te_comp = 'Hy'
        else: te_comp = component

        field = super(FDFD_TM, self).get_adjoint_field(te_comp, domain)

        if(component == 'Hz'):
            return -1 * field
        else:
            return field

    def update_saved_fields(self):
        # collect data on the field domains
        del self._saved_fields
        self._saved_fields = []
        for d in self._field_domains:
            Hz = self.get_field_interp('Hz', d)
            Ex = self.get_field_interp('Ex', d)
            Ey = self.get_field_interp('Ey', d)
            self._saved_fields.append((Hz, Ex, Ey))

    def get_source_power(self):
        """Get the source power.

        Notes
        -----
        1. The source power is computed using the interpolated fields.

        Returns
        -------
        float
            Electromagnetic power generated by source.
        """
        # easier access to PML widths
        w_pml_l = self._w_pml_left
        w_pml_r = self._w_pml_right
        w_pml_t = self._w_pml_top
        w_pml_b = self._w_pml_bottom

        M = self._M
        N = self._N
        dx = self._dx
        dy = self._dy

        Hz = self.get_field_interp('Hz')
        Ex = self.get_field_interp('Ex')
        Ey = self.get_field_interp('Ey')

        if(RANK != 0):
            return MathDummy()

        # calculate the Poynting vectors around the boundaries of the sytem
        # (excluding the PML regions)
        Sx = 0.5*(Ey*np.conj(Hz)).real
        Sy = -0.5*(Ex*np.conj(Hz)).real

        S1 = -Sy[w_pml_b, w_pml_l:N-1-w_pml_r]
        S2 = Sy[M-1-w_pml_t, w_pml_l:N-1-w_pml_r]
        S3 = -Sx[w_pml_b:M-1-w_pml_t, w_pml_l]
        S4 = Sx[w_pml_b:M-1-w_pml_t, N-1-w_pml_r]

        # total power flowing our of boundaries
        P_S = np.sum(S1 + S2) * dx + np.sum(S3 + S4) * dy

        x_all = np.arange(w_pml_l, N-w_pml_r)
        y_all = np.arange(w_pml_b, M-w_pml_t)
        y_all = y_all.reshape(y_all.shape[0], 1).astype(np.int)

        if(not self.real_materials):
            eps = self._eps_actual.get_values(0, N, 0, M)
            mu = self._mu_actual.get_values(0, N, 0, M)
        else:
            eps = np.zeros(Hz.shape, dtype=np.complex128)
            mu = np.zeros(Hz.shape, dtype=np.complex128)

        # power dissipated due to material absorption
        Hz2 = Hz[y_all, x_all]*np.conj(Hz[y_all, x_all])
        Ex2 = Ex[y_all, x_all]*np.conj(Ex[y_all, x_all])
        Ey2 = Ey[y_all, x_all]*np.conj(Ey[y_all, x_all])

        P_loss = 0.25 * dx * dy * np.sum(eps[y_all, x_all].imag*(Ex2 + Ey2) + \
                                         mu[y_all, x_all].imag*Hz2).real
        return P_S + P_loss

    def get_A_diag(self, vdiag=None):
        """Get the diagonal entries of the system matrix A.

        Parameters
        ----------
        vdiag : petsc4py.PETSc.Vec
            Vector with dimensions Mx1 where M is equal to the number of
            diagonal entries in A.

        Returns
        -------
        **(Master node only)** the diagonal entries of A.
        """
        # We need to override this function since the TE matrix diagonals do
        # not match the TM matrix diagonals (even when swapping eps and mu).
        # This is because the signs on epsilon and mu in Maxwell's equations
        # are flipped when moving from TE to TM.  In most cases, it is easiest
        # to handle this change by swapping Ez with -Hz, Mx with -Jx, and My
        # with -Jy in the TE equations, which can be achieved by simply
        # overriding the corresponding setter and getter functions.  In
        # reality, a better way to handle reusing the TE equations is to swap E
        # and H, J and M, eps with -mu, and mu with -eps.  This way of doing
        # things, however, is harder to achieve programmatically if we want to
        # reuse as much of the TE code as possible.  When using the FDFD object
        # with an AdjointMethod, it turns out that simply swapping field and
        # source components is insufficient and knowledge of the A's diagonals
        # is needed, hence this overriden function.
        if(vdiag == None):
            vdiag = PETSc.Vec()
        self._A.getDiagonal(vdiag)

        vdiag *= -1
        return vdiag

class FDFD_3D(FDFD):
    """Simulate Maxwell's equations in 3D.

    Notes
    -----
    1. Units used for length parameters do not matter as long as they are
    consistent with one another

    2. The dimensions of the system will be modified such that they are
    consistent with the provided grid spacing. You should always reinitialize
    your scripts X, Y, Z variables with sim.X, sim.Y, and sim.Z in order to
    make sure everything is consistent.

    3. The 3D solver internally uses an iterative Krylov subspace method with a
    multigrid preconditioner. This is quite a bit more complicated than the
    direct solver used in 2D so careful attention should be paid when setting
    up the 3D solver. Additional documentation on this will come in the future.

    4. The solver supports both real and complex material values, however the
    source power calculation currently only accounts for real-valued materials.
    This will be updated in the future.

    Parameters
    ----------
    W : float
        Approximate width of the simulation region. If this is not an integer
        multiple of dx, this will be increased slightly
    H : float
        Approximate height of the simulation region. If this is not an integer
        multiple of dy, this will be increased slightly
    dx : float
        Grid spacing in the x direction.
    dy : float
        Grid spacing in the y direction.
    wavelength : float
        Vacuum wavelength of EM fields.
    mglevels : int
        The number of levels to use in the multigrid prevonditioner. If running
        very coarse resolutions, this value should be decreased (to 2). If
        running a very high resolution simulation, this value should be
        increased (to 4 or more depending on resolution).
    rtol : float
        The relative tolerance used to terminate the internal iterative solver.
        Smaller number = higher accuracy solution.
    low_memory : boolean
        If True, reduce the memory footprint of the solver at the cost of
        performance. This is achieved by using an iterative method for the
        coarse solver instead of a direct LU factorization. (default = False)

    Attributes
    ----------
    dx : float
        The grid spacing in the x direction
    dy : float
        The grid spacing in the y direction
    W : float
        The width of the simulation region (including PMLs)
    H : float
        The height of the simulation region (including PMLs)
    M : int
        The number of grid cells in the y direction
    N : int
        The number of grid cells in the x direction
    wavelength : float
        The wavelength used for simulation. [arbitrary length unit]
    w_pml_left : int
        The number of grid cells which make up the left PML region
    w_pml_right : int
        The number of grid cells which make up the right PML region
    w_pml_top : int
        The number of grid cells which make up the top PML region
    w_pml_bottom : int
        The number of grid cells which make up the bottom PML region
    pml_sigma : float
        The graded parameter of the PML. Increase this to get a "stronger" PML
    pml_power : float
        The power of the polynomial used to define the spatial-dependence of
        pml_sigma.
    field_domains : list of DomainCoordinates
        The list of DomainCoordinates in which fields are recorded immediately
        following a forward solve.
    saved_fields : list of numpy.ndarray
        The list of (Ez, Hx, Hy) fields saved in in the stored
        DomainCoordinates
    source_power : float
        The source power injected into the system.
    w_pml : list of float
        List of PML widths in real spatial units. This variable can be
        reinitialized in order to change the PML widths. The list is ordered
        as follows: [xmin, xmax, ymin, ymax, zmin, zmax]
    real_mats : boolean
        If True, assume all permittivity and permeability values are real
        valued (in power calculations).

    """

    def __init__(self, X, Y, Z, dx, dy, dz, wavelength, mglevels=None,
                 rtol=1e-6, low_memory=False, verbose=True):
        super(FDFD_3D, self).__init__(3)

        self._dx = dx
        self._dy = dy
        self._dz = dz
        self._wlen = wavelength

        # scalaing factor used in non-dimensionalizing spatial units
        self._R = wavelength/(2*pi)

        # pml widths
        self._w_pml = [15*dx, 15*dx, 15*dy, 15*dy, 15*dz, 15*dz]
        self._w_pml_xmin = 15
        self._w_pml_xmax= 15
        self._w_pml_ymin = 15
        self._w_pml_ymax = 15
        self._w_pml_zmin = 15
        self._w_pml_zmax = 15

        # Boundary conditions. Default type is PEC on all sim boundaries
        # Note: This will result in PMC boundaries for the 2D TM simulations.
        # This should be a non-issue as long as PMLs are used
        self._bc = ['0', '0', '0']

        # PML parameters -- these can be changed
        self.pml_sigma = 2.0
        self.pml_power = 1.5

        # dx and dy are the only dimension rigorously enforced
        Nx = int(np.ceil(X/dx) + 1); self._Nx = Nx
        Ny = int(np.ceil(Y/dy) + 1); self._Ny = Ny
        Nz = int(np.ceil(Z/dz) + 1); self._Nz = Nz
        self._bsize = 6
        self._nunks = self._bsize*Nx*Ny*Nz
        self._nunks2h = self._bsize*int(Nz/2)*int(Ny/2)*int(Nx/2)

        # The dimensions of the simulation region snap to the nearest grid cell
        self._X = (self._Nx - 1) * dx
        self._Y = (self._Ny - 1) * dy
        self._Z = (self._Nz - 1) * dz

        self._eps = None
        self._mu = None
        self.real_mats = False

        # Get PETSc options database
        optDB = PETSc.Options()

        # The size of the system matrix A is given by the number of unknowns (6
        # field components * Nx*Ny*Nz)
        self._A.setSizes([self._nunks, self._nunks])
        self._A.setType('aij')
        self._A.setUp()

        # We now prepare system matrices for coarser levels
        self._As = []
        self._Rst = []

        if(mglevels == None):
            min_ds = 0.0775 # this yields good values in general
            mglevels = 0
            ds = np.min([dx, dy, dz]); ds /= wavelength

            while(min_ds >= ds):
                mglevels += 1
                min_ds /= 2

            if(mglevels == 0): mglevels = 1

        self._mglevels = mglevels

        for l in range(0,mglevels-1):
            divl = 2*(mglevels-l-1)
            divlm1 = 2*(mglevels-l)
            if(divl == 0): divl = 1

            nunksl = self._bsize * int(Nx/divl) * int(Ny/divl) * int(Nz/divl)
            nunkslm1 = self._bsize * int(Nx/divlm1) * int(Ny/divlm1) * int(Nz/divlm1)

            Al = PETSc.Mat()
            Al.create(PETSc.COMM_WORLD)
            Al.setSizes([nunksl, nunksl])
            Al.setType('aij')
            Al.setUp()

            self._As.append(Al)

        for l in range(1, mglevels):
            divl = 2*(mglevels-l-1)
            divlm1 = 2*(mglevels-l)
            if(divl == 0): divl = 1

            nunksl = self._bsize * int(Nx/divl) * int(Ny/divl) * int(Nz/divl)
            nunkslm1 = self._bsize * int(Nx/divlm1) * int(Ny/divlm1) * int(Nz/divlm1)

            Rst = PETSc.Mat()
            Rst.create(PETSc.COMM_WORLD)
            Rst.setSizes([nunkslm1, nunksl])
            Rst.setType('aij')
            Rst.setPreallocationNNZ([8,8])
            Rst.setUp()

            self._Rst.append(Rst)


        self._As.append(self._A)
        self._AsT = [None for i in range(len(self._As))]

        # Build the resitriction and interpolation matrices
        for l in range(1,mglevels):
            self.buildRst(l)

        #obtain solution and RHS vectors
        x, b = self._A.getVecs()
        x.set(0)
        b.set(0)
        self.x = x
        self.b = b

        self.x_adj = x.copy()
        self.b_adj = b.copy()

        self._Adiag1 = x.copy() # used in calc_ydAx to avoid reallocation

        self.ib, self.ie = self._A.getOwnershipRange()
        self.A_diag_update = np.zeros(self.ie-self.ib, dtype=np.complex128)

        # create an iterative linear solver which uses a multigrid
        # preconditioner and a matrix preconditioner in each smoother. The
        # matrix is the Hermitian of the A matrix in each level (yielding a
        # positive semi definite system which is compatible with multigrid)
        # NOTE: at the coarsest level, we use LU factorization (via MUMPS). In
        # order to save the symbolic factorization to speed up repeated calls,
        # we need two separate solvers: one for the forward solve and one for
        # the adjoint solve. In reality this shouldnt be necessary, but we dont
        # have direct access to the internals of the multigrid code.

        #FORWARD solver
        self.ksp_iter_fwd = PETSc.KSP()
        self.ksp_iter_fwd.create(PETSc.COMM_WORLD)

        self.ksp_iter_fwd.setType('fgmres')
        self.ksp_iter_fwd.setGMRESRestart(20)
        self.ksp_iter_fwd.setTolerances(rtol=rtol)
        #optDB['-ksp_gcr_restart'] = 10
        self.ksp_iter_fwd.setFromOptions()

        #ADJOINT solver
        self.ksp_iter_adj = PETSc.KSP()
        self.ksp_iter_adj.create(PETSc.COMM_WORLD)

        self.ksp_iter_adj.setType('fgmres')
        self.ksp_iter_adj.setGMRESRestart(20)
        self.ksp_iter_adj.setTolerances(rtol=rtol)
        #optDB['-ksp_gcr_restart'] = 10
        self.ksp_iter_adj.setFromOptions()

        # Setup multigrid preconditioner
        ## Basic setup
        pc_fwd = self.ksp_iter_fwd.getPC()
        pc_adj = self.ksp_iter_adj.getPC()
        pc_fwd.setType('mg')
        pc_adj.setType('mg')
        optDB['-pc_mg_levels'] = mglevels
        pc_fwd.setFromOptions()
        pc_adj.setFromOptions()

        pc_fwd.setMGType(PETSc.PC.MGType.MULTIPLICATIVE) # Multiplicative
        pc_adj.setMGType(PETSc.PC.MGType.MULTIPLICATIVE) # Multiplicative
        pc_fwd.setMGCycleType(PETSc.PC.MGCycleType.W) # W cycle
        pc_adj.setMGCycleType(PETSc.PC.MGCycleType.W) # W cycle

        ## Setup coarse solver
        if(not low_memory):
            ksp_crs = pc_fwd.getMGCoarseSolve()
            ksp_crs.setType('preonly')
            pc_crs = ksp_crs.getPC()
            pc_crs.setType('lu')

            try:
                pc_crs.setFactorSolverPackage('mumps')
            except AttributeError as ae:
                pc_crs.setFactorSolverType('mumps')

            pc_crs.setFromOptions()
            self._ksp_crs_fwd = ksp_crs

            ksp_crs = pc_adj.getMGCoarseSolve()
            ksp_crs.setType('preonly')
            pc_crs = ksp_crs.getPC()
            pc_crs.setType('lu')

            try:
                pc_crs.setFactorSolverPackage('mumps')
            except AttributeError as ae:
                pc_crs.setFactorSolverType('mumps')

            pc_crs.setFromOptions()
            self._ksp_crs_adj = ksp_crs
        else:
            ksp_crs = pc_fwd.getMGCoarseSolve()
            ksp_crs.setType('bcgsl')
            ksp_crs.setTolerances(rtol=1e-2)
            pc_crs = ksp_crs.getPC()
            pc_crs.setType('mat')
            self._ksp_crs_fwd = ksp_crs

            ksp_crs = pc_adj.getMGCoarseSolve()
            ksp_crs.setType('bcgsl')
            ksp_crs.setTolerances(rtol=1e-2)
            pc_crs = ksp_crs.getPC()
            pc_crs.setType('mat')
            self._ksp_crs_adj = ksp_crs


        ## Setup Down smoothers
        for l in range(1,mglevels):
            ksp_smooth = pc_fwd.getMGSmootherDown(l)
            ksp_smooth.setType('gmres')
            ksp_smooth.setGMRESRestart(10)
            ksp_smooth.setTolerances(max_it=8)
            pc_smooth = ksp_smooth.getPC()
            pc_smooth.setType('mat')
            pc_smooth.setFromOptions()

            ksp_smooth = pc_adj.getMGSmootherDown(l)
            ksp_smooth.setType('gmres')
            ksp_smooth.setGMRESRestart(10)
            ksp_smooth.setTolerances(max_it=8)
            pc_smooth = ksp_smooth.getPC()
            pc_smooth.setType('mat')
            pc_smooth.setFromOptions()

        ## Setup Up Smoothers
        for l in range(1,mglevels):
            ksp_smooth = pc_fwd.getMGSmootherUp(l)
            ksp_smooth.setType('gmres')
            ksp_smooth.setTolerances(max_it=4)
            pc_smooth = ksp_smooth.getPC()
            pc_smooth.setType('mat')
            pc_smooth.setFromOptions()

            ksp_smooth = pc_adj.getMGSmootherUp(l)
            ksp_smooth.setType('gmres')
            ksp_smooth.setTolerances(max_it=4)
            pc_smooth = ksp_smooth.getPC()
            pc_smooth.setType('mat')
            pc_smooth.setFromOptions()

        ## Set restriction and interpolation
        for l in range(1,mglevels):
            pc_fwd.setMGRestriction(l, self._Rst[l-1])
            pc_fwd.setMGInterpolation(l, self._Rst[l-1])

            pc_adj.setMGRestriction(l, self._Rst[l-1])
            pc_adj.setMGInterpolation(l, self._Rst[l-1])

        self.verbose = verbose
        self._built = False

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def Z(self):
        return self._Z

    @property
    def Nx(self):
        return self._Nx

    @property
    def Ny(self):
        return self._Ny

    @property
    def Nz(self):
        return self._Nz

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def dz(self):
        return self._dz

    @property
    def bc(self):
        return ''.join(self._bc)

    @bc.setter
    def bc(self, newbc):
        self._bc = list(newbc)

    @property
    def w_pml(self):
        return self._w_pml

    @w_pml.setter
    def w_pml(self, ws):
        self._w_pml = list(ws)
        self._w_pml_xmin = int(ws[0] / self._dx)
        self._w_pml_xmax= int(ws[1] / self._dx)
        self._w_pml_ymin = int(ws[2] / self._dy)
        self._w_pml_ymax = int(ws[3] / self._dy)
        self._w_pml_zmin = int(ws[4] / self._dz)
        self._w_pml_zmax = int(ws[5] / self._dz)

    def __pml_x(self, k):
        ## Generate the PML values for the left and right boundaries.
        Nx = self._Nx
        pwr = self.pml_power
        sigma = self.pml_sigma

        w_xmin = self._w_pml_xmin
        w_xmax = self._w_pml_xmax
        if(k <= w_xmin and w_xmin > 0):
            v0 = ((w_xmin - k)*1.0/w_xmin)**pwr
            return 1.0 / (1.0 + 1j*sigma*v0)
        elif(k >= Nx-1-w_xmax and w_xmax > 0):
            v1 = ((k - (Nx-1-w_xmax))*1.0/w_xmax)**pwr
            return 1.0 / (1.0 + 1j*sigma*v1)
        else:
            return 1.0

    def __pml_y(self, j):
        ## Generate the PML values for the top and bottom boundaries.
        Ny = self._Ny
        pwr = self.pml_power
        sigma = self.pml_sigma

        w_ymin = self._w_pml_ymin
        w_ymax = self._w_pml_ymax
        if(j <= w_ymin and w_ymin > 0):
            v0 = ((w_ymin - j)*1.0/w_ymin)**pwr
            return 1.0 / (1.0 + 1j*sigma * v0)
        elif(j >= Ny-1-w_ymax and w_ymax > 0):
            v1 = ((j - (Ny-1-w_ymax))*1.0/w_ymax)**pwr
            return 1.0 / (1.0 + 1j*sigma * v1)
        else:
            return 1.0

    def __pml_z(self, i):
        ## Generate the PML values for the top and bottom boundaries.
        Nz = self._Nz
        pwr = self.pml_power
        sigma = self.pml_sigma

        w_zmin = self._w_pml_zmin
        w_zmax = self._w_pml_zmax


        if(i <= w_zmin and w_zmin > 0):
            v0 = ((w_zmin - i)*1.0/w_zmin)**pwr
            return 1.0 / (1.0 + 1j*sigma * v0)
        elif(i >= Nz-1-w_zmax and w_zmax > 0):
            v1 = ((i - (Nz-1-w_zmax))*1.0/w_zmax)**pwr
            return 1.0 / (1.0 + 1j*sigma * v1)
        else:
            return 1.0

    def buildA(self, l):
        ## Build the system matrix for the lth layer (0 == finest grid)
        A = self._As[l]

        ib, ie = A.getOwnershipRange()

        # for l > 0, we will generate A for a coarser version of the problem
        # Coarsening occurs in factors of 2. We must modify Nx, Ny, Nz
        # accordingly
        Nx = self._Nx
        Ny = self._Ny
        Nz = self._Nz

        divl = 2*(self._mglevels-l-1)
        if(divl == 0): divl = 1

        Nx = int(Nx/divl)
        Ny = int(Ny/divl)
        Nz = int(Nz/divl)

        NxNy = Nx*Ny
        Ngrid = Nx*Ny*Nz

        eps = self._eps
        mu = self._mu
        bc = self._bc

        # similarly, dx, dy, dz must be modified
        dx = self._X / (Nx-1)
        dy = self._Y / (Ny-1)
        dz = self._Z / (Nz-1)

        odx = self._R / dx
        ody = self._R / dy
        odz = self._R / dz

        if(self._eps == None or self._mu == None):
            raise Exception('The material distributions of the system must be \
                            initialized prior to building the system matrix.')

        ig = 0
        component = 0
        x = 0; y = 0; z = 0
        Nc = 6 # 6 field components
        for i in range(ib, ie):

            ig = int(i/Nc)
            component = int(i - 6*ig)
            z = int(ig/NxNy)
            y = int((ig-z*NxNy)/Nx)
            x = ig - z*NxNy - y*Nx

            xh = x*divl + divl/2
            yh = y*divl + divl/2
            zh = z*divl + divl/2

            if(component == 0): # Jx row
                jEx = i
                jHz0 = (ig - Nx)*Nc + 5
                jHz1 = ig*Nc + 5
                jHy0 = (ig - NxNy)*Nc + 4
                jHy1 = ig*Nc + 4

                # Ex
                A[i, jEx] = 1j*eps.get_value(xh+0.5*divl,yh,zh-0.5*divl) # todo: partial grid steps

                # Hz
                pml_y = self.__pml_y(yh)
                A[i, jHz1] = ody * pml_y
                if(y > 0): A[i, jHz0] = -1*ody * pml_y

                # Hy
                pml_z = self.__pml_z(zh - 0.5*divl)
                A[i, jHy1] = -1 * odz * pml_z
                if(z > 0): A[i, jHy0] = odz * pml_z

                #############################
                # enforce boundary conditions
                #############################
                if(z == 0):
                    if(bc[2] == '0'):
                        A[i, jHz1] = 0.0
                        if(y > 0): A[i, jHz0] = 0.0
                    elif(bc[2] == 'E'):
                        A[i, jHy1] = -2 * odz
                    elif(bc[2] == 'H'):
                        A[i, jHy1] = 0
                elif(z == Nz-1):
                    pass

                if(y == 0):
                    if(bc[1] == '0'):
                        A[i, jHy1] = 0
                        if(z > 0): A[i, jHy0] = 0
                    elif(bc[1] == 'E'):
                        A[i, jHz1] = 2 * ody
                    elif(bc[1] == 'H'):
                        A[i, jHz1] = 0
                elif(y == Ny-1):
                    pass

                if(x == 0):
                    pass
                elif(x == Nx-1):
                    pass

            elif(component == 1): #Jy row
                jEy = i
                jHx0 = (ig - NxNy)*Nc + 3
                jHx1 = ig*Nc + 3
                jHz0 = (ig - 1)*Nc + 5
                jHz1 = ig*Nc + 5

                # Ey
                A[i, jEy] = 1j * eps.get_value(xh,yh+0.5*divl,zh-0.5*divl)

                # Hx
                pml_z = self.__pml_z(zh - 0.5*divl)
                A[i, jHx1] = odz * pml_z
                if(z > 0): A[i,jHx0] = -1*odz * pml_z

                # Hz
                pml_x = self.__pml_x(xh)
                A[i, jHz1] = -1*odx * pml_x
                if(x > 0): A[i, jHz0] = odx * pml_x

                #############################
                # enforce boundary conditions
                #############################
                if(z == 0):
                    if(bc[2] == '0'):
                        A[i, jHz1] = 0.0
                        if(x > 0): A[i, jHz0] = 0.0
                    elif(bc[2] == 'E'):
                        A[i, jHx1] = 2 * odz
                    elif(bc[2] == 'H'):
                        A[i, jHx1] = 0
                elif(z == Nz-1):
                    pass

                if(y == 0):
                    pass
                elif(y == Ny-1):
                    pass

                if(x == 0):
                    if(bc[0] == '0'):
                        A[i, jHx1] = 0.0
                        if(z > 0): A[i, jHx0] = 0.0
                    elif(bc[0] == 'E'):
                        A[i, jHz1] = -2*odx
                    elif(bc[0] == 'H'):
                        A[i, jHz1] = 0
                elif(x == Nx-1):
                    pass

            elif(component == 2): # Jz row
                jEz = i
                jHy0 = (ig-1)*Nc + 4
                jHy1 = ig*Nc + 4
                jHx0 = (ig-Nx)*Nc + 3
                jHx1 = ig*Nc + 3

                # Ez
                A[i, jEz] = 1j * eps.get_value(xh,yh,zh)

                # Hy
                pml_x = self.__pml_x(xh)
                A[i, jHy1] = odx * pml_x
                if(x > 0): A[i, jHy0] = -1*odx * pml_x

                # Hx
                pml_y = self.__pml_y(yh)
                A[i, jHx1] = -1*ody * pml_y
                if(y > 0): A[i, jHx0] = ody * pml_y

                #############################
                # enforce boundary conditions
                #############################
                if(z == 0):
                    pass
                elif(z == Nz-1):
                    pass

                if(y == 0):
                    if(bc[1] == '0'):
                        A[i, jHy1] = 0.0
                        if(x > 0): A[i, jHy0] = 0.0
                    elif(bc[1] == 'E'):
                        A[i, jHx1] = -2*ody
                    elif(bc[1] == 'H'):
                        A[i, jHx1] = 0
                elif(y == Ny-1):
                    pass

                if(x == 0):
                    if(bc[0] == '0'):
                        A[i, jHx1] = 0.0
                        if(y > 0): A[i, jHx0] = 0.0
                    elif(bc[0] == 'E'):
                        A[i, jHy1] = 2*odx
                    elif(bc[0] == 'H'):
                        A[i, jHy1] = 0
                elif(x == Nx-1):
                    pass

            elif(component == 3): # Mx row
                jHx = i
                jEz0 = ig*Nc + 2
                jEz1 = (ig+Nx)*Nc + 2
                jEy0 = ig*Nc + 1
                jEy1 = (ig+NxNy)*Nc + 1

                # Hx
                A[i, jHx] = -1j * mu.get_value(xh,yh+0.5*divl,zh)

                # Ez
                pml_y = self.__pml_y(yh + 0.5*divl)
                if(y < Ny-1): A[i, jEz1] = ody * pml_y
                A[i, jEz0] = -1*ody * pml_y

                # Ey
                pml_z = self.__pml_z(zh)
                if(z < Nz-1): A[i, jEy1] = -1*odz * pml_z
                A[i, jEy0] = odz * pml_z

                #############################
                # enforce boundary conditions
                #############################
                if(z == 0):
                    pass
                elif(z == Nz-1):
                    A[i, jEz0] = 0.0
                    if(y < Ny-1): A[i, jEz1] = 0.0

                if(y == 0):
                    pass
                elif(y == Ny-1):
                    A[i, jEy0] = 0.0
                    if(z < Nz-1): A[i, jEy1] = 0.0

                if(x == 0):
                    pass
                elif(x == Nx-1):
                    pass

            elif(component == 4): # My row
                jHy = i
                jEx0 = ig*Nc
                jEx1 = (ig+NxNy)*Nc
                jEz0 = ig*Nc + 2
                jEz1 = (ig+1)*Nc + 2

                # Hy
                A[i, jHy] = -1j*mu.get_value(xh+0.5*divl,yh,zh)

                # Ex
                pml_z = self.__pml_z(zh)
                if(z < Nz-1): A[i, jEx1] = odz * pml_z
                A[i, jEx0] = -1*odz * pml_z

                # Ez
                pml_x = self.__pml_x(xh + 0.5*divl)
                if(x < Nx-1): A[i, jEz1] = -1*odx * pml_x
                A[i, jEz0] = odx * pml_x

                #############################
                # enforce boundary conditions
                #############################
                if(z == 0):
                    pass
                elif(z == Nz-1):
                    A[i, jEz0] = 0.0
                    if(x < Nx-1): A[i, jEz1] = 0.0

                if(y == 0):
                    pass
                elif(y == Ny-1):
                    pass

                if(x == 0):
                    pass
                elif(x == Nx-1):
                    A[i, jEx0] = 0.0
                    if(z < Nz-1): A[i, jEx1] = 0.0

            else: # Mz row
                jHz = i
                jEy0 = ig*Nc + 1
                jEy1 = (ig+1)*Nc + 1
                jEx0 = ig*Nc
                jEx1 = (ig+Nx)*Nc

                # Hz
                A[i, jHz] = -1j * mu.get_value(xh+0.5*divl,yh+0.5*divl,zh-0.5*divl)

                # Ey
                pml_x = self.__pml_x(xh + 0.5*divl)
                if(x < Nx-1): A[i, jEy1] = odx * pml_x
                A[i, jEy0] = -1*odx * pml_x

                # Ex
                pml_y = self.__pml_y(yh + 0.5*divl)
                if(y < Ny-1): A[i, jEx1] = -1*ody * pml_y
                A[i, jEx0] = ody * pml_y

                #############################
                # enforce boundary conditions
                #############################
                if(z == 0):
                    pass
                elif(z == Nz-1):
                    pass

                if(y == 0):
                    pass
                elif(y == Ny-1):
                    A[i, jEy0] = 0.0
                    if(x < Nx-1): A[i, jEy1] = 0.0

                if(x == 0):
                    pass
                elif(x == Nx-1):
                    A[i, jEx0] = 0.0
                    if(y < Ny-1): A[i, jEx1] = 0.0

        # communicate off-processor values and setup internal data structures for
        # performing parallel operations
        A.assemblyBegin()
        A.assemblyEnd()

        self._built = True

    def build(self):
        """(Re)Build the system matrix.

        Maxwell's equations are solved by compiling them into a linear system
        of the form :math:`Ax=b`. Here, we build up the structure of A which contains
        the curl operators, mateiral parameters, and boundary conditions.

        This function must be called at least once after the material
        distributions of the system have been defined (through a call to
        :func:`set_materials`).

        Raises
        ------
        Exception
            If the material distributions of the simulation have not been set
            prior to calling this function.
        """
        if(self.verbose and NOT_PARALLEL):
            info_message('Building system matrices...')

        for l in range(0,self._mglevels):
            self.buildA(l)

            self._AsT[l] = self._As[l].duplicate(copy=True)
            self._As[l].transpose(self._AsT[l])
            self._AsT[l].conjugate()

    def buildRst(self, l):
        ## Build the lth restriction matrix. This matrix "restricts" the grid
        ## from l to l+1, producing a coarser grid with half the resolution
        R = self._Rst[l-1]
        bsize = self._bsize

        ib, ie = R.getOwnershipRange()

        # determine how much the grid will be scaled down
        divl = 2*(self._mglevels-l-1)
        divlm1 = 2*(self._mglevels-l)
        if(divl == 0): divl = 1

        Nx = int(self._Nx/divl)
        Ny = int(self._Ny/divl)
        Nz = int(self._Nz/divl)
        NxNy = Nx*Ny

        Nxl = int(self._Nx/divlm1)
        Nyl = int(self._Ny/divlm1)
        Nzl = int(self._Nz/divlm1)
        NxlNyl = Nxl*Nyl

        alpha = 1.0/7.0

        for i in range(ib,ie):
            ig = int(i/bsize)
            comp = int(i - bsize*ig)

            z = int(ig/NxlNyl)
            y = int((ig-z*NxlNyl)/Nxl)
            x = ig - z*NxlNyl - y*Nxl

            j = 2*(z*NxNy+y*Nx+x)*bsize

            R[i, j+comp] = alpha

            if(2*x<Nx-1):
                R[i, j+comp+bsize] = 0.125
            if(2*y<Ny-1):
                R[i, j+comp+Nx*bsize] = 0.125
            if(2*x<Nx-1 and 2*y<Ny-1):
                R[i, j+comp+bsize+Nx*bsize] = 0.125
            if(2*z<Nz-1):
                R[i, j+comp+NxNy*bsize] = 0.125
            if(2*z<Nz-1 and 2*x<Nx-1):
                R[i, j+comp+NxNy*bsize+bsize] = 0.125
            if(2*z<Nz-1 and 2*y<Ny-1):
                R[i, j+comp+NxNy*bsize+Nx*bsize] = 0.125
            if(2*z<Nz-1 and 2*y<Ny-1 and 2*x<Nx-1):
                R[i, j+comp+NxNy*bsize+Nx*bsize+bsize] = 0.125

        R.assemblyBegin()
        R.assemblyEnd()

    def update(self, bbox=None):
        """(Partially) Update the system matrix.

        Updating the system matrix involves recomputing the material
        distribution of the system.

        If a bounding box is provided, only that portion of the grid will be
        updates.

        Notes
        -----
        This only updates the high resolution grid.
        """
        A = self._As[self._mglevels-1]

        ib, ie = A.getOwnershipRange()

        # for l > 0, we will generate A for a coarser version of the problem
        # Coarsening occurs in factors of 2. We must modify Nx, Ny, Nz
        # accordingly
        Nx = self._Nx
        Ny = self._Ny
        Nz = self._Nz

        NxNy = Nx*Ny
        Ngrid = Nx*Ny*Nz

        eps = self._eps
        mu = self._mu
        bc = self._bc

        # similarly, dx, dy, dz must be modified
        dx = self._dx
        dy = self._dy
        dz = self._dz

        if(self._eps == None or self._mu == None):
            raise Exception('The material distributions of the system must be \
                            initialized prior to building the system matrix.')

        if(bbox == None):
            k1 = 0; k2 = self._Nx
            j1 = 0; j2 = self._Ny
            i1 = 0; i2 = self._Nz
        else:
            k1 = int(bbox[0]/self._X*Nx); k2 = int(bbox[1]/self._X*Nx)
            j1 = int(bbox[2]/self._Y*Ny); j2 = int(bbox[3]/self._Y*Ny)
            i1 = int(bbox[4]/self._Z*Nz); i2 = int(bbox[5]/self._Z*Nz)

        ib = self.ib
        ie = self.ie
        ig = 0
        component = 0
        x = 0; y = 0; z = 0
        Nc = 6 # 6 field components

        ig = int(ib/Nc); zmin = int(ig/NxNy)
        ig = int(ie/Nc); zmax = int(ig/NxNy)

        if(zmin > i1): i1 = zmin
        if(zmax < i2): i2 = zmax+1

        for z in range(zmin, zmax):
            for y in range(j1, j2):
                for x in range(k1, k2):

                    jEx = Nc*(z*NxNy+y*Nx+x) + 0
                    if(jEx >= ib and jEx <ie):
                        A[jEx, jEx] = 1j*eps.get_value(x+0.5, y, z-0.5)

                    jEy = Nc*(z*NxNy+y*Nx+x) + 1
                    if(jEy >= ib and jEy <ie):
                        A[jEy, jEy] = 1j * eps.get_value(x, y+0.5, z-0.5)

                    jEz = Nc*(z*NxNy+y*Nx+x) + 2
                    if(jEz >= ib and jEz <ie):
                        A[jEz, jEz] = 1j * eps.get_value(x, y, z)

                    jHx = Nc*(z*NxNy+y*Nx+x) + 3
                    if(jHx >= ib and jHx <ie):
                        A[jHx, jHx] = -1j * mu.get_value(x, y+0.5, z)

                    jHy = Nc*(z*NxNy+y*Nx+x) + 4
                    if(jHy >= ib and jHy <ie):
                        A[jHy, jHy] = -1j*mu.get_value(x+0.5, y, z)

                    jHz = Nc*(z*NxNy+y*Nx+x) + 5
                    if(jHz >= ib and jHz <ie):
                        A[jHz, jHz] = -1j * mu.get_value(x+0.5, y+0.5, z-0.5)

        # communicate off-processor values and setup internal data structures for
        # performing parallel operations
        A.assemblyBegin()
        A.assemblyEnd()

    def update_multigrid(self, l):
        """Update the material distributions of the restricted system matrices

        Notes
        -----
        l : int
            The multigrid level to update.
        """
        A = self._As[l]

        ib, ie = A.getOwnershipRange()

        # for l > 0, we will generate A for a coarser version of the problem
        # Coarsening occurs in factors of 2. We must modify Nx, Ny, Nz
        # accordingly
        Nx = self._Nx
        Ny = self._Ny
        Nz = self._Nz

        divl = 2*(self._mglevels-l-1)
        if(divl == 0): divl = 1

        Nx = int(Nx/divl)
        Ny = int(Ny/divl)
        Nz = int(Nz/divl)

        NxNy = Nx*Ny
        Ngrid = Nx*Ny*Nz

        eps = self._eps
        mu = self._mu
        bc = self._bc

        # similarly, dx, dy, dz must be modified
        dx = self._X / (Nx-1)
        dy = self._Y / (Ny-1)
        dz = self._Z / (Nz-1)

        if(self._eps == None or self._mu == None):
            raise Exception('The material distributions of the system must be \
                            initialized prior to building the system matrix.')

        ig = 0
        component = 0
        x = 0; y = 0; z = 0
        Nc = 6 # 6 field components
        for i in range(ib, ie):

            ig = int(i/Nc)
            component = int(i - 6*ig)
            z = int(ig/NxNy)
            y = int((ig-z*NxNy)/Nx)
            x = ig - z*NxNy - y*Nx

            xh = x*divl + divl/2
            yh = y*divl + divl/2
            zh = z*divl + divl/2

            if(component == 0): # Jx row
                jEx = i
                A[i, jEx] = 1j*eps.get_value(xh+0.5*divl,yh,zh-0.5*divl)

            elif(component == 1): #Jy row
                jEy = i
                A[i, jEy] = 1j * eps.get_value(xh,yh+0.5*divl,zh-0.5*divl)

            elif(component == 2): # Jz row
                jEz = i
                A[i, jEz] = 1j * eps.get_value(xh,yh,zh)

            elif(component == 3): # Mx row
                jHx = i
                A[i, jHx] = -1j * mu.get_value(xh,yh+0.5*divl,zh)

            elif(component == 4): # My row
                jHy = i
                A[i, jHy] = -1j*mu.get_value(xh+0.5*divl,yh,zh)

            else: # Mz row
                jHz = i
                A[i, jHz] = -1j * mu.get_value(xh+0.5*divl,yh+0.5*divl,zh-0.5*divl)


        # communicate off-processor values and setup internal data structures for
        # performing parallel operations
        A.assemblyBegin()
        A.assemblyEnd()

    def solve_forward(self):
        """Solve the forward simulation.

        This is equivalent to a solution of Maxwell's equations, which is
        achieved by either iteratively or directly solving :math:`Ax=b`.

        After the completing the forward solve, the full solution contained in
        x is gathered on the master node so that the fields can be accessed.

        """
        if(not self._built):
            error_message('The system matrix has not be been built. Call' \
                          ' self.build before running a simulation')

        if(self.verbose and NOT_PARALLEL):
            info_message('Updating preconditioner...')

        # Update multigrid matrices
        for l in range(0,self._mglevels-1):
            self.update_multigrid(l)

        # Update the transposed matrices
        for l in range(0, self._mglevels):
            A = self._As[l]
            AT = self._AsT[l]
            ib, ie = A.getOwnershipRange()
            for i in range(ib, ie):
                AT[i,i] = np.conj(A[i,i])
            AT.assemblyBegin()
            AT.assemblyEnd()

        if(self.verbose and NOT_PARALLEL):
            info_message('Running forward solver...')

        # setup and solve Ax=b using petsc4py
        # unless otherwise specified, MUMPS (direct solver) is used.
        # Alternatively, the bicgstab iterative solver may be used.
        ksp = self.ksp_iter_fwd
        ksp.setInitialGuessNonzero(False)
        ksp.setOperators(self._A, self._A)
        ksp.setFromOptions()

        ksp_crs = self._ksp_crs_fwd
        ksp_crs.setInitialGuessNonzero(False)
        ksp_crs.setOperators(self._As[0], self._As[0])
        ksp_crs.setFromOptions()

        for l in range(1,self._mglevels):
            ksp_smooth = ksp.getPC().getMGSmootherDown(l)
            ksp_smooth.setOperators(self._As[l], self._AsT[l])
            ksp_smooth.setFromOptions()

        for l in range(1,self._mglevels):
            ksp_smooth = ksp.getPC().getMGSmootherUp(l)
            ksp_smooth.setOperators(self._As[l], self._AsT[l])
            ksp_smooth.setFromOptions()

        ksp.solve(self.b, self.x)

        if(RANK == 0):
            convergence = ksp.getConvergedReason()
            if(convergence < 0):
                error_message('Forward solution did not converge with error '
                              'code %d.' % (convergence))

        Psrc = self.get_source_power()
        if(NOT_PARALLEL): self._source_power = Psrc
        else: self._source_power = MathDummy()
        self.update_saved_fields()

    def solve_adjoint(self):
        """Solve the adjoint simulation.

        This is equivalent to a solution of the transpose of Maxwell's
        equations, which is achieved by either iteratively or directly
        solving :math:`Ax=b`.
        """
        if(not self._built):
            error_message('The system matrix has not be been built. Call' \
                          ' self.build before running a simulation')

        if(self.verbose and NOT_PARALLEL):
            info_message('Running adjoint solver...')

        # setup and solve Ax=b using petsc4py
        # unless otherwise specified, MUMPS (direct solver) is used.
        # Alternatively, the bicgstab iterative solver may be used.
        ksp = self.ksp_iter_adj
        ksp.setInitialGuessNonzero(False)
        ksp.setOperators(self._AsT[self._mglevels-1], self._AsT[-1])
        ksp.setFromOptions()

        ksp_crs = self._ksp_crs_adj
        ksp_crs.setInitialGuessNonzero(False)
        ksp_crs.setOperators(self._AsT[0], self._AsT[0])
        ksp_crs.setFromOptions()

        for l in range(1,self._mglevels):
            ksp_smooth = ksp.getPC().getMGSmootherDown(l)
            ksp_smooth.setOperators(self._AsT[l], self._As[l])
            ksp_smooth.setFromOptions()

        for l in range(1,self._mglevels):
            ksp_smooth = ksp.getPC().getMGSmootherUp(l)
            ksp_smooth.setOperators(self._AsT[l], self._As[l])
            ksp_smooth.setFromOptions()

        self.b_adj.conjugate()
        ksp.solve(self.b_adj, self.x_adj)
        self.b_adj.conjugate()

        if(RANK == 0):
            convergence = ksp.getConvergedReason()
            if(convergence < 0):
                error_message('Forward solution did not converge with error '
                              'code %d.' % (convergence))


    def get_field(self, component, domain=None, squeeze=False):
        """Get the uninterpolated field.

        Notes
        -----
        Currently aggregation from the different nodes is done is a slow way.
        It is best to avoid making many repetative calls to this function.

        Parameters
        ----------
        component : str
            The desired field component to retrieve (Ex, Ey, Ez, Hx, Hy, Hz)
        domain : emopt.misc.DomainCoordinates (optional)
            On field data from the specified domain is retrieved. If no domain
            is specified, the whole simulation domain will be used.
            (default=None)

        Returns
        -------
        numpy.ndarray
            The desired fields contained within the specified domain.
        """
        if(domain == None):
            domain = DomainCoordinates(0, self._X, 0, self._Y, 0, self._Z,
                                       self._dx, self._dy, self._dz)

        # get domain bounds for quicker access
        k1 = domain.k1; k2 = domain.k2
        j1 = domain.j1; j2 = domain.j2
        i1 = domain.i1; i2 = domain.i2

        # grid properties. Needed in a sec
        Nc = 6
        Nx = self._Nx
        NxNy = self._Nx * self._Ny
        solution_vec = self.x.getArray()
        values = []

        # In order to get the fields only in the desired domain, we need to
        # extract the corresponding field values from the different processes.
        # For many 3D problems, it would be terribly inefficient (in terms of
        # memory) to assemble all of the parts of x together and then slice out
        # the desired rectangular piece. Instead, we will have each process
        # extract out only the desired values and then send these values to the
        # rank 0 process. To do this efficiently, we will limit our range of
        # indices of x that we consider.
        ib = self.ib
        ie = self.ie

        # based on ie, we can set a limit on z
        ig = int(ie/Nc)
        zmax = int(ig/NxNy)

        ig = int(ib/Nc)
        zmin = int(ig/NxNy)

        if(i1 < zmin): i1 = zmin-1
        if(i2 > zmax): i2 = zmax+1

        c = 0
        if(component == FieldComponent.Ex): c = 0
        elif(component == FieldComponent.Ey): c = 1
        elif(component == FieldComponent.Ez): c = 2
        elif(component == FieldComponent.Hx): c = 3
        elif(component == FieldComponent.Hy): c = 4
        elif(component == FieldComponent.Hz): c = 5

        for z in range(i1, i2):
            for y in range(j1, j2):
                for x in range(k1, k2):
                    index = Nc*(z*NxNy+y*Nx+x)+c

                    if(index >= ib and index < ie):
                        values.append(solution_vec[index-ib])


        # share the data between processors
        comm = MPI.COMM_WORLD
        field = comm.gather(values, root=0)

        # combine and reshape the data
        if(NOT_PARALLEL):
            field = np.concatenate(field)
            field = np.reshape(field, domain.shape)
            if(squeeze): return np.squeeze(field)
            else: return field
        else:
            return MathDummy()


    def get_field_interp(self, component, domain=None, squeeze=False):
        """Get the desired field component.

        Internally, fields are solved on a staggered grid. In most cases, it is
        desirable to know all of the field components at the same sets of
        positions. This requires that we interpolate the fields onto a single
        grid. In emopt, we interpolate all field components onto the Ez grid.

        Parameters
        ----------
        component : str
            The desired field component.
        domain : misc.DomainCoordinates (optional)
            The domain from which the field is retrieved. (default = None)

        Returns
        -------
        numpy.ndarray
            The interpolated field
        """
        # Ez does not need to be interpolated
        if(component == FieldComponent.Ez):
            if(squeeze): return np.squeeze(self.get_field(component, domain))
            else: return self.get_field(component, domain)
        else:
            # if no domain was provided
            if(domain == None):
                domain_interp = DomainCoordinates(0, self._X, 0, self._Y, 0, self._Z,
                                                  self._dx, self._dy, self._dz)
                domain = domain_interp

                k1 = domain_interp.k1; k2 = domain_interp.k2
                j1 = domain_interp.j1; j2 = domain_interp.j2
                i1 = domain_interp.i1; i2 = domain_interp.i2

            # in order to properly handle interpolation at the boundaries, we
            # need to expand the domain
            else:
                k1 = domain.k1; k2 = domain.k2
                j1 = domain.j1; j2 = domain.j2
                i1 = domain.i1; i2 = domain.i2

                if(k1 > 0): k1 -= 1
                if(k2 < self._Nx-1): k2 += 1
                if(j1 > 0): j1 -= 1
                if(j2 < self._Ny-1): j2 += 1
                if(i1 > 0): i1 -= 1
                if(i2 < self._Nz-1): i2 += 1

                domain_interp = DomainCoordinates(k1*self._dx, k2*self._dx,
                                                  j1*self._dy, j2*self._dy,
                                                  i1*self._dz, i2*self._dz,
                                                  self._dx, self._dy, self._dz)

                k1 = domain_interp.k1; k2 = domain_interp.k2
                j1 = domain_interp.j1; j2 = domain_interp.j2
                i1 = domain_interp.i1; i2 = domain_interp.i2

            fraw = self.get_field(component, domain_interp)

            if(RANK != 0):
                return MathDummy()

            fraw = np.pad(fraw, 1, 'constant', constant_values=0)

            # set boundary values equal to original boundary values. These may
            # be changed depending on boundary conditions in a second
            #fraw[0, :, :] = fraw[1, :, :]; fraw[-1, :, :] = fraw[-2, :, :]
            #fraw[:, 0, :] = fraw[:, 1, :]; fraw[:, -1, :] = fraw[:, -2, :]
            #fraw[:, :, 0] = fraw[:, :, 1]; fraw[:, :, -1] = fraw[:, :, -2]

            # after interpolation, we will need to crop the field so that it
            # matches the supplied domain
            crop_field = lambda f : f[1+domain.i1-i1:-1-(i2-domain.i2), \
                                      1+domain.j1-j1:-1-(j2-domain.j2), \
                                      1+domain.k1-k1:-1-(k2-domain.k2)]

            field = None
            bc = self._bc
            if(component == FieldComponent.Ex):
                # Handle special boundary conditions
                if(k1 == 0 and bc[0] == 'E'):
                    fraw[:, :, 0] = -1*fraw[:,:,1]
                elif(k1 == 0 and bc[0] == 'H'):
                    fraw[:, :, 0] = fraw[:, :, 1]

                Ex = np.copy(fraw)
                Ex [1:-1, :, 1:-1] += fraw[1:-1, :, 0:-2]
                Ex [1:-1, :, 1:-1] += fraw[2:, :, 1:-1]
                Ex [1:-1, :, 1:-1] += fraw[2:, :, 0:-2]
                Ex = Ex/4.0
                field = crop_field(Ex)

            elif(component == FieldComponent.Ey):
                # handle special boundary conditions
                if(j1 == 0 and bc[1] == 'E'):
                    fraw[:, 0, :] = -1*fraw[:, 1, :]
                elif(j1 == 0 and bc[1] == 'H'):
                    fraw[:, 0, :] = fraw[:, 1, :]

                Ey = np.copy(fraw)
                Ey[1:-1, 1:-1, :] += fraw[1:-1, 0:-2, :]
                Ey[1:-1, 1:-1, :] += fraw[2:, 1:-1, :]
                Ey[1:-1, 1:-1, :] += fraw[2:, 0:-2, :]
                Ey = Ey/4.0
                field = crop_field(Ey)

            elif(component == FieldComponent.Hx):
                # handle special boundary conditions
                if(j1 == 0 and bc[1] == 'E'):
                    fraw[:, 0, :] = -1*fraw[:, 1, :]
                elif(j1 == 0 and bc[1] == 'H'):
                    fraw[:, 0, :] = fraw[:, 1, :]

                Hx = np.copy(fraw)
                Hx[:, 1:, :] += fraw[:, 0:-1, :]
                Hx = Hx/2.0
                field = crop_field(Hx)

            elif(component == FieldComponent.Hy):
                # Handle special boundary conditions
                if(k1 == 0 and bc[0] == 'E'):
                    fraw[:, :, 0] = -1*fraw[:,:,1]
                elif(k1 == 0 and bc[0] == 'H'):
                    fraw[:, :, 0] = fraw[:, :, 1]

                Hy = np.copy(fraw)
                Hy[:, :, 1:] += fraw[:, :, 0:-1]
                Hy = Hy/2.0
                field = crop_field(Hy)

            elif(component == FieldComponent.Hz):
                # Handle special boundary conditions
                if(k1 == 0 and bc[0] == 'E'):
                    fraw[:, :, 0] = -1*fraw[:,:,1]
                elif(k1 == 0 and bc[0] == 'H'):
                    fraw[:, :, 0] = fraw[:, :, 1]

                # handle special boundary conditions
                if(j1 == 0 and bc[1] == 'E'):
                    fraw[:, 0, :] = -1*fraw[:, 1, :]
                elif(j1 == 0 and bc[1] == 'H'):
                    fraw[:, 0, :] = fraw[:, 1, :]

                Hz = np.copy(fraw)
                Hz[0:-1, 1:, 1:] += fraw[0:-1, 1:, 0:-1]
                Hz[0:-1, 1:, 1:] += fraw[0:-1, 0:-1, 1:]
                Hz[0:-1, 1:, 1:] += fraw[0:-1, 0:-1, 0:-1]
                Hz[0:-1, 1:, 1:] += fraw[1:, 1:, 1:]
                Hz[0:-1, 1:, 1:] += fraw[1:, 1:, 0:-1]
                Hz[0:-1, 1:, 1:] += fraw[1:, 0:-1, 1:]
                Hz[0:-1, 1:, 1:] += fraw[1:, 0:-1, 0:-1]
                Hz = Hz/8.0
                field = crop_field(Hz)
            else:
                pass

            if(squeeze): return np.squeeze(field)
            else: return field


    def get_adjoint_field(self, component, domain=None, squeeze=False):
        """Get the adjoint field.

        Notes
        -----
        1. This function is primarily intended as a diagnostics tool. If
        implementing and adjoint method, consult the emopt.adjoint_method
        documentation.

        2. Currently aggregation from the different nodes is done is a slow way.
        It is best to avoid making many repetative calls to this function.

        Parameters
        ----------
        component : str
            The desired adjoint field component to retrieve (Ex, Ey, Ez, Hx,
            Hy, Hz). Note that the adjoint field is somewhat fictitious, so
            "field component" refers to the part of the solution vector from
            which to collect the fields.
        domain : emopt.misc.DomainCoordinates (optional)
            On field data from the specified domain is retrieved. If no domain
            is specified, the whole simulation domain will be used.
            (default=None)

        Returns
        -------
        numpy.ndarray
            The desired fields contained within the specified domain.
        """
        if(domain == None):
            domain = DomainCoordinates(0, self._X, 0, self._Y, 0, self._Z,
                                       self._dx, self._dy, self._dz)

        # get domain bounds for quicker access
        k1 = domain.k1; k2 = domain.k2
        j1 = domain.j1; j2 = domain.j2
        i1 = domain.i1; i2 = domain.i2

        # grid properties. Needed in a sec
        Nc = 6
        Nx = self._Nx
        NxNy = self._Nx * self._Ny
        solution_vec = self.x_adj.getArray()
        values = []

        # In order to get the fields only in the desired domain, we need to
        # extract the corresponding field values from the different processes.
        # For many 3D problems, it would be terribly inefficient (in terms of
        # memory) to assemble all of the parts of x together and then slice out
        # the desired rectangular piece. Instead, we will have each process
        # extract out only the desired values and then send these values to the
        # rank 0 process. To do this efficiently, we will limit our range of
        # indices of x that we consider.
        ib = self.ib
        ie = self.ie

        # based on ie, we can set a limit on z
        ig = int(ib/Nc)
        zb = int(ig/NxNy)

        ig = int(ie/Nc)
        ze = int(ig/NxNy)

        if(i1 < zb): i1 = zb
        if(i2 > ze): i2 = ze+1

        c = 0
        if(component == FieldComponent.Ex): c = 0
        elif(component == FieldComponent.Ey): c = 1
        elif(component == FieldComponent.Ez): c = 2
        elif(component == FieldComponent.Hx): c = 3
        elif(component == FieldComponent.Hy): c = 4
        elif(component == FieldComponent.Hz): c = 5

        for z in range(i1, i2):
            for y in range(j1, j2):
                for x in range(k1, k2):
                    index = Nc*(z*NxNy+y*Nx+x)+c

                    if(index >= ib and index < ie):
                        values.append(solution_vec[index-ib])


        # share the data between processors
        comm = MPI.COMM_WORLD
        field = comm.gather(values, root=0)

        # combine and reshape the data
        if(NOT_PARALLEL):
            field = np.concatenate(field)
            field = np.reshape(field, domain.shape)
            if(squeeze): return np.squeeze(field)
            else: return field
        else:
            return MathDummy()


    def set_materials(self, eps, mu):
        """Set the material distributions of the system to be simulated.

        Parameters
        ----------
        eps : emopt.grid.Material
            The spatially-dependent permittivity of the system
        mu : emopt.grid.Material
            The spatially-dependent permeability of the system
        """
        self._eps = eps
        self._mu = mu

    def set_sources(self, src, domain, mode_num=0):
        """Set the sources of the system used in the forward solve.

        Notes
        -----
        1. Like the underlying fields, the current sources are represented on a
        set of shifted grids. If manually setting the source, this should be
        taken into account.

        2. If using a mode source to set the sources, the mode source must be
        built and run prior to calling this function.

        3. This function can be called repeatedly to modify the source
        distribution. Calling this function does not zero the current
        distribution.

        Parameters
        ----------
        src : modes.ModeFullVector or tuple of numpy.ndarray
            Either a ModeFullVector object or a set of arrays containing
            current source distributions.
        """
        Nc = 6
        Nx = self._Nx
        NxNy = self._Nx * self._Ny
        ib = self.ib
        ie = self.ie
        src_arr = self.b.getArray()

        if(type(src) == tuple or type(src) == list):
            # the source vector is distributed across the different processes.
            # We need to extract out only the locally stored values from the
            # provided arrays.
            k1 = domain.k1; k2 = domain.k2
            j1 = domain.j1; j2 = domain.j2
            i1 = domain.i1; i2 = domain.i2

            # restrict the range of zs to only those owned locally by this
            # processor
            ig = int(ie/Nc)
            zmax = int(ig/NxNy)+1

            ig = int(ib/Nc)
            zmin = int(ig/NxNy)-1

            if(i1 > zmin): zmin = i1
            if(i2 < zmax): zmax = i2

            Jx = src[0]; Jy = src[1]; Jz = src[2]
            Mx = src[3]; My = src[4]; Mz = src[5]

            for i in range(zmin, zmax):
                for j in range(j1,j2):
                    for k in range(k1,k2):
                        index = Nc*(i*NxNy+j*Nx+k) + 0
                        if(index >= ib and index < ie):
                            src_arr[index - ib] = Jx[i-i1, j-j1, k-k1]

                        index = Nc*(i*NxNy+j*Nx+k)+1
                        if(index >= ib and index < ie):
                            src_arr[index - ib] = Jy[i-i1, j-j1, k-k1]

                        index = Nc*(i*NxNy+j*Nx+k)+2
                        if(index >= ib and index < ie):
                            src_arr[index - ib] = Jz[i-i1, j-j1, k-k1]

                        index = Nc*(i*NxNy+j*Nx+k)+3
                        if(index >= ib and index < ie):
                            src_arr[index - ib] = Mx[i-i1, j-j1, k-k1]

                        index = Nc*(i*NxNy+j*Nx+k)+4
                        if(index >= ib and index < ie):
                            src_arr[index - ib] = My[i-i1, j-j1, k-k1]

                        index = Nc*(i*NxNy+j*Nx+k)+5
                        if(index >= ib and index < ie):
                            src_arr[index - ib] = Mz[i-i1, j-j1, k-k1]

            self.b.setArray(src_arr)

        elif(type(src) == modes.ModeFullVector):
            Jx, Jy, Jz, Mx, My, Mz = src.get_source(mode_num,
                                                    self._dx,
                                                    self._dy,
                                                    self._dz)

            # for now, we are just going to send all current sources to every
            # processor. This is inefficient but very simple. Fortunately, the
            # mode source should have a small memory footprint.
            Jx = COMM.bcast(Jx, root=0)
            Jy = COMM.bcast(Jy, root=0)
            Jz = COMM.bcast(Jz, root=0)
            Mx = COMM.bcast(Mx, root=0)
            My = COMM.bcast(My, root=0)
            Mz = COMM.bcast(Mz, root=0)

            # recursively call set_source with the distributed arrays. this is
            # gross, I know...
            self.set_sources([Jx, Jy, Jz, Mx, My, Mz], domain)


    def set_adjoint_sources(self, dFdxs):
        """Set the adjoint sources.

        This function exists to maintain compatibility with the adjoint method
        class which expects a function which accepts a single argument in order
        to set the adjoint source distribution.

        This functionality is complicated in 3D as we have to be a bit more
        careful with how we share data between processors.

        Notes
        -----
        1. This function clears the current adjoint source vector when called.

        Parameters
        ----------
        dFdxs : tuple of lists
            Tuple containing a list of numpy.ndarray as the first element and a
            list of correpsonding DomainCoordinates as the second argument.
        """
        self.b_adj.set(0)
        for src, domain in zip(dFdxs[0], dFdxs[1]):
            self.update_adjoint_sources(src, domain)

    def update_adjoint_sources(self, src, domain):
        """Update the adjoint source.

        Parameters
        ----------
        src : numpy.ndarray
            The source distribution
        domain : misc.DomainCoordinates
            The region in the grid corresponding to src.
        """
        Nc = 6
        Nx = self._Nx
        Ny = self._Ny
        Nz = self._Nz
        NxNy = self._Nx * self._Ny
        ib = self.ib
        ie = self.ie
        src_arr = self.b_adj.getArray()

        # the source vector is distributed across the different processes.
        # We need to extract out only the locally stored values from the
        # provided arrays.
        k1 = domain.k1; k2 = domain.k2
        j1 = domain.j1; j2 = domain.j2
        i1 = domain.i1; i2 = domain.i2

        # restrict the range of zs to only those owned locally by this
        # processor
        ig = int(ie/Nc)
        zmax = int(ig/NxNy)+1

        ig = int(ib/Nc)
        zmin = int(ig/NxNy)-1

        if(i1 > zmin): zmin = i1
        if(i2 < zmax): zmax = i2

        # we need to be careful that any part of the supplied source/domain
        # which extends outside of the simulation region is ignored
        xmin = k1; xmax = k2
        ymin = j1; ymax = j2
        if(xmin < 0): xmin = 0
        if(ymin < 0): ymin = 0
        if(zmin < 0): zmin = 0

        if(xmax > Nx-1): xmax = Nx-1
        if(ymax > Ny-1): ymax = Ny-1
        if(zmax > Nz-1): zmax = Nz-1

        #print xmin, xmin-k1, ymin, ymin-j1, zmin, zmin-i1

        Jx = src[0]
        Jy = src[1]
        Jz = src[2]
        Mx = src[3]
        My = src[4]
        Mz = src[5]

        # these are most likely only know on rank 0--communicate data
        Jx = COMM.bcast(Jx, root=0)
        Jy = COMM.bcast(Jy, root=0)
        Jz = COMM.bcast(Jz, root=0)
        Mx = COMM.bcast(Mx, root=0)
        My = COMM.bcast(My, root=0)
        Mz = COMM.bcast(Mz, root=0)

        for i in range(zmin,zmax):
            for j in range(ymin,ymax):
                for k in range(xmin,xmax):
                    index = Nc*(i*NxNy+j*Nx+k)+0
                    if(index >= ib and index < ie):
                        src_arr[index - ib] += Jx[i-i1, j-j1, k-k1]

                    index = Nc*(i*NxNy+j*Nx+k)+1
                    if(index >= ib and index < ie):
                        src_arr[index - ib] += Jy[i-i1, j-j1, k-k1]

                    index = Nc*(i*NxNy+j*Nx+k)+2
                    if(index >= ib and index < ie):
                        src_arr[index - ib] += Jz[i-i1, j-j1, k-k1]

                    index = Nc*(i*NxNy+j*Nx+k)+3
                    if(index >= ib and index < ie):
                        src_arr[index - ib] += Mx[i-i1, j-j1, k-k1]

                    index = Nc*(i*NxNy+j*Nx+k)+4
                    if(index >= ib and index < ie):
                        src_arr[index - ib] += My[i-i1, j-j1, k-k1]

                    index = Nc*(i*NxNy+j*Nx+k)+5
                    if(index >= ib and index < ie):
                        src_arr[index - ib] += Mz[i-i1, j-j1, k-k1]

        self.b_adj.setArray(src_arr)

    def update_saved_fields(self):
        """Update the saved fields based on the domains contained in
        FDFD_3D.field_domains.

        Notes
        -----
        1. This is primarily for internal use only.
        2. The fields are NOT squeezed
        """
        del self._saved_fields
        self._saved_fields = []

        for d in self._field_domains:
            Ex = self.get_field_interp(FieldComponent.Ex, domain=d)
            Ey = self.get_field_interp(FieldComponent.Ey, domain=d)
            Ez = self.get_field_interp(FieldComponent.Ez, domain=d)
            Hx = self.get_field_interp(FieldComponent.Hx, domain=d)
            Hy = self.get_field_interp(FieldComponent.Hy, domain=d)
            Hz = self.get_field_interp(FieldComponent.Hz, domain=d)

            self._saved_fields.append((Ex, Ey, Ez, Hx, Hy, Hz))

    def get_source_power(self):
        """Get source power.

        The source power is the total electromagnetic power radiated by the
        electric and magnetic current sources.

        Returns
        -------
        float
            The source power.
        """
        Psrc = 0.0

        # define pml boundary domains
        dx = self._dx; dy = self._dy; dz = self._dz
        if(self._w_pml[0] > 0): xmin = self._w_pml[0] + dx
        else: xmin = 0.0

        if(self._w_pml[1] > 0): xmax = self._X - self._w_pml[1] - dx
        else: xmax = self._X

        if(self._w_pml[2] > 0): ymin = self._w_pml[2] + dy
        else: ymin = 0.0

        if(self._w_pml[3] > 0): ymax = self._Y - self._w_pml[3] - dy
        else: ymax = self._Y

        if(self._w_pml[4] > 0): zmin = self._w_pml[4] + dz
        else: zmin = 0.0

        if(self._w_pml[5] > 0): zmax = self._Z - self._w_pml[5] - dz
        else: zmax = self._Z

        x1 = DomainCoordinates(xmin, xmin, ymin, ymax, zmin, zmax, dx, dy, dz)
        x2 = DomainCoordinates(xmax, xmax, ymin, ymax, zmin, zmax, dx, dy, dz)
        y1 = DomainCoordinates(xmin, xmax, ymin, ymin, zmin, zmax, dx, dy, dz)
        y2 = DomainCoordinates(xmin, xmax, ymax, ymax, zmin, zmax, dx, dy, dz)
        z1 = DomainCoordinates(xmin, xmax, ymin, ymax, zmin, zmin, dx, dy, dz)
        z2 = DomainCoordinates(xmin, xmax, ymin, ymax, zmax, zmax, dx, dy, dz)

        # calculate power transmitter through xmin boundary
        Ey = self.get_field_interp('Ey', x1)
        Ez = self.get_field_interp('Ez', x1)
        Hy = self.get_field_interp('Hy', x1)
        Hz = self.get_field_interp('Hz', x1)

        if(NOT_PARALLEL and self._bc[0] != 'E' and self._bc[0] != 'H'):
            Px = -0.5*dy*dz*np.sum(np.real(Ey*np.conj(Hz)-Ez*np.conj(Hy)))
            #print Px
            Psrc += Px
        del Ey; del Ez; del Hy; del Hz

        # calculate power transmitter through xmax boundary
        Ey = self.get_field_interp('Ey', x2)
        Ez = self.get_field_interp('Ez', x2)
        Hy = self.get_field_interp('Hy', x2)
        Hz = self.get_field_interp('Hz', x2)

        if(NOT_PARALLEL):
            Px = 0.5*dy*dz*np.sum(np.real(Ey*np.conj(Hz)-Ez*np.conj(Hy)))
            #print Px
            Psrc += Px
        del Ey; del Ez; del Hy; del Hz

        # calculate power transmitter through ymin boundary
        Ex = self.get_field_interp('Ex', y1)
        Ez = self.get_field_interp('Ez', y1)
        Hx = self.get_field_interp('Hx', y1)
        Hz = self.get_field_interp('Hz', y1)

        if(NOT_PARALLEL and self._bc[1] != 'E' and self._bc[1] != 'H'):
            Py = 0.5*dy*dz*np.sum(np.real(Ex*np.conj(Hz)-Ez*np.conj(Hx)))
            #print Py
            Psrc += Py
        del Ex; del Ez; del Hx; del Hz

        # calculate power transmitter through ymax boundary
        Ex = self.get_field_interp('Ex', y2)
        Ez = self.get_field_interp('Ez', y2)
        Hx = self.get_field_interp('Hx', y2)
        Hz = self.get_field_interp('Hz', y2)

        if(NOT_PARALLEL):
            Py = -0.5*dy*dz*np.sum(np.real(Ex*np.conj(Hz)-Ez*np.conj(Hx)))
            #print Py
            Psrc += Py
        del Ex; del Ez; del Hx; del Hz

        # calculate power transmitter through zmin boundary
        Ex = self.get_field_interp('Ex', z1)
        Ey = self.get_field_interp('Ey', z1)
        Hx = self.get_field_interp('Hx', z1)
        Hy = self.get_field_interp('Hy', z1)

        if(NOT_PARALLEL and self._bc[2] != 'E' and self._bc[2] != 'H'):
            Pz = -0.5*dy*dz*np.sum(np.real(Ex*np.conj(Hy)-Ey*np.conj(Hx)))
            #print Pz
            Psrc += Pz
        del Ex; del Ey; del Hx; del Hy

        # calculate power transmitter through zmin boundary
        Ex = self.get_field_interp('Ex', z2)
        Ey = self.get_field_interp('Ey', z2)
        Hx = self.get_field_interp('Hx', z2)
        Hy = self.get_field_interp('Hy', z2)

        if(NOT_PARALLEL):
            Pz = 0.5*dy*dz*np.sum(np.real(Ex*np.conj(Hy)-Ey*np.conj(Hx)))
            #print Pz
            Psrc += Pz
        del Ex; del Ey; del Hx; del Hy

        return Psrc

    def calc_ydAx(self, Adiag0):
        """Calculate y^T * (A1-A0) * x.

        Parameters
        ----------
        Adiag0 : PETSc.Vec
            The diagonal of the FDFD matrix.

        Returns
        -------
        complex
            The product y^T * (A1-A0) * x
        """
        x = self.x
        y = self.x_adj
        Adiag1 = self._Adiag1
        self.get_A_diag(Adiag1)

        product = np.conj(y) * (Adiag1-Adiag0) * x
        return np.sum(product[...])
