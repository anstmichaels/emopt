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


# Initialize petsc first
import petsc4py
import sys
petsc4py.init(sys.argv)

from misc import info_message, warning_message, error_message, RANK, \
NOT_PARALLEL, run_on_master, MathDummy

from grid import row_wise_A_update

import numpy as np
from math import pi
from abc import ABCMeta, abstractmethod
from petsc4py import PETSc

__author__ = "Andrew Michaels"
__license__ = "Apache License, Version 2.0"
__version__ = "0.2"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

class FDFD(object):
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
    __metaclass__ = ABCMeta

    def __init__(self):
        self._A = PETSc.Mat()
        self._A.create(PETSc.COMM_WORLD)

        # number of unknowns (field componenets * grid points)
        self._nunks = 0

        self._field_domains = []
        self._saved_fields = []
        self._source_power = []

    @property
    def nunks(self):
        return self._nunks

    @property
    def field_domains(self):
        return self._field_domains

    @field_domains.setter
    def field_domains(self, domains):
        self._field_domains = domains

    @property
    def saved_fields(self):
        return self._saved_fields

    @saved_fields.setter
    def saved_fields(self, newf):
        warning_message('Saved fields cannot be modified externally.',
                        'emopt.fdfd')

    @property
    def source_power(self):
        return self._source_power

    @source_power.setter
    def source_power(self, newp):
        warning_message('The source power cannot be modified.', 'emopt.fdfd')

    @abstractmethod
    def solve_forward(self):
        pass

    @abstractmethod
    def solve_adjoint(self):
        pass

    @abstractmethod
    def get_field(self, component, domain=None):
        pass

    @abstractmethod
    def get_field_interp(self, component, domain=None):
        pass

    @abstractmethod
    def get_adjoint_field(self, component, domain=None):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def update(self, x1, x2, y1, y2):
        pass

    @abstractmethod
    def set_sources(self, src):
        pass

    @abstractmethod
    def set_adjoint_sources(self, src):
        pass

    @abstractmethod
    def update_saved_fields(self):
        pass

    @abstractmethod
    def get_source_power(self, src):
        """
        Notes
        -----
        This should exclude any influence due to non-physical boundary
        conditions like PMLs (if possible)
        """
        pass

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

    2. In the photonics community "TE" is called "TM" and vice versa.  It is
    silly.

    3. The width and the height of the system will be modified in order to
    ensure that they are an integer multiple of dx and dy. It is important that
    this modified width and height be used in any future calculations.

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
    w_pml : list of floats
        pml widths in [left, right, top, bottom] format
    use_iterative : boolean (optional)
        If True, use the BICGSTAB iterative solver. If false, use the MUMPS
        direct solve (defulat = False)
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
    M : int
        The number of grid cells in the y direction
    N : int
        The number of grid cells in the x direction
    wavelength : float
        The wavelength used for simulation. [arbitrary length unit]
    w_pml_left : int
        The number of grid cells which make up the left PML region
    w_pml_right
        The number of grid cells which make up the right PML region
    w_pml_top
        The number of grid cells which make up the top PML region
    w_pml_bottom
        The number of grid cells which make up the bottom PML region
    field_domains
        The list of DomainCoordinates in which fields are recorded immediately
        following a forward solve.
    saved_fields
        The list of (Ez, Hx, Hy) fields saved in in the stored
        DomainCoordinates
    source_power
        The source power injected into the system.

    """

    def __init__(self, W, H, dx, dy, wavelength, w_pml, use_iterative=False, verbose=True):
        super(FDFD_TE, self).__init__()

        self._dx = dx
        self._dy = dy
        self._wlen = wavelength

        # scalaing factor used in non-dimensionalizing spatial units
        self._R = wavelength/(2*pi)

        # pml widths for left, right, top, bottom snap to nearest number of grid
        # cells
        self._w_pml_left = int(w_pml[0]/dx)
        self._w_pml_right = int(w_pml[1]/dx)
        self._w_pml_top = int(w_pml[2]/dy)
        self._w_pml_bottom = int(w_pml[3]/dy)

        # PML parameters -- these can be changed
        self.pml_sigma = 2.0 * wavelength
        self.pml_power = 3.0

        # dx and dy are the only dimension rigorously enforced
        self._M = int(np.ceil(H/dy) + 1)
        self._N = int(np.ceil(W/dx) + 1)
        self._nunks = 3*self._M*self._N

        # The width and height are as close to the desired W and H as possible
        # given the desired grid spacing
        self._W = (self._N - 1) * dx
        self._H = (self._M - 1) * dy

        self._eps = None
        self._mu = None

        # factor of 3 due to 3 field components
        self._A.setSizes([3*self._M*self._N, 3*self._M*self._N])
        self._A.setType('aij')
        self._A.setUp()

        #obtain solution and RHS vectors
        x, b = self._A.getVecs()
        x.set(0)
        b.set(1)
        self.x = x
        self.b = b

        self.x_adj = x.copy()
        self.b_adj = b.copy()

        self.ib, self.ie = self._A.getOwnershipRange()
        self.A_diag_update = np.zeros(self.ie-self.ib, dtype=np.complex128)



        # use conjugate gradients or bicgstab
        if(use_iterative):
            # create a linear solver
            self.ksp_iter = PETSc.KSP()
            self.ksp_iter.create(PETSc.COMM_WORLD)

            self.ksp_iter.setType('gmres')
            pc = self.ksp_iter.getPC()
            pc.setType('ilu')
        else:
            # create a linear solver
            self.ksp = PETSc.KSP()
            self.ksp.create(PETSc.COMM_WORLD)

            self.ksp.setType('preonly')
            pc = self.ksp.getPC()
            pc.setType('lu')
            pc.setFactorSolverPackage('mumps')

        self._use_iterative = use_iterative

        self.Ez = np.array([])
        self.Hx = np.array([])
        self.Hy = np.array([])

        self.Ez_adj = np.array([])
        self.Hx_adj = np.array([])
        self.Hy_adj = np.array([])

        self.verbose = verbose
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
        return self._W

    @property
    def H(self):
        """
        The height of the systme. This height will only exactly match the
        height passed during initialization if it is equal to an integer 
        multiple of grid cells.
        """
        return self._H

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

    @wavelength.setter
    def wavelength(self, val):
        if(val > 0):
            self._wlen = val
        else:
            raise ValueError('Wavelength must be a positive number!')

    @property
    def eps(self):
        return self._eps

    @property
    def mu(self):
        return self._mu

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

    def set_sources(self, src):
        """Set the sources of the system used in the forward solve.

        Currently, sources are defined using 3 numpy.ndarrays.  The elements of
        the array correspond to spatially-dependent electric or magnetic current
        sources.  In the future, more structured source elements may be
        implemented.

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
        src : tuple of numpy.ndarray
            The current sources in the form (Jz, Mx, My).  Each array in the
            tiple should be a 2D numpy.ndarry with dimensions MxN.
        """
        self.Jz = src[0]
        self.Mx = src[1]
        self.My = src[2]

        src_arr = np.concatenate([self.Jz.ravel(), self.Mx.ravel(),
                                  self.My.ravel()])

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

        src_arr = np.concatenate([self.Jz_adj.ravel(), self.Mx_adj.ravel(),
                                  self.My_adj.ravel()])

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

        pml_x = np.ones([M,N], dtype=np.complex128)


        # define the left PML
        w_pml = self._w_pml_left
        x = X[:, 0:w_pml]
        pml_x[:, 0:w_pml] = 1.0 / (1.0 + 1j*self.pml_sigma *
                                              ((w_pml - x)*1.0/w_pml)**self.pml_power)

        # define the right PML
        w_pml = self._w_pml_right
        x = X[:, N-w_pml:]
        pml_x[:, N-w_pml:] = 1.0 / (1.0 + 1j*self.pml_sigma *
                                              ((x-N+w_pml)*1.0/w_pml)**self.pml_power)

        return pml_x

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

        pml_y = np.ones([M,N], dtype=np.complex128)

        # PML for bottom
        w_pml = self._w_pml_bottom
        y = Y[0:w_pml, :]
        pml_y[0:w_pml, :] = 1.0 / (1.0 + 1j*self.pml_sigma *
                                              ((w_pml - y)*1.0/w_pml)**self.pml_power)
        # PML for top
        w_pml = self._w_pml_top
        y = Y[M-w_pml:, :]
        pml_y[M-w_pml:, :] = 1.0 / (1.0 + 1j*self.pml_sigma *
                                              ((y-M+w_pml)*1.0/w_pml)**self.pml_power)

        return pml_y

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

        odx = self._R / self._dx
        ody = self._R / self._dy

        pml_x = self.__get_pml_x()
        pml_y = self.__get_pml_y()

        odx = odx * pml_x
        ody = ody * pml_y

        if(self._eps == None or self._mu == None):
            raise Exception('The material distributions of the system must be \
                            initialized prior to building the system matrix.')

        if(self.verbose and NOT_PARALLEL):
            info_message('Building system matrix...')

        for i in xrange(self.ib, self.ie):
            if(i < M*N ): # Jz row
                y = int(i/N)
                x = i - y * N

                # relevant j coordinates
                j0 = i
                j1 = i+1
                j2 = (y-1)*N + x

                # Diagonal element is the permittivity at (x,y)
                A[i,j0] = 1j * eps.get_value(x,y)

                A[i,j0+M*N] = -ody[y,x]
                A[i,j0+2*M*N] = -odx[y,x]

                if(y > 0):
                    A[i,j2 + M*N] = ody[y,x]

                if(x < N-1):
                    A[i,j1 + 2*M*N] = odx[y,x]

            elif(i < 2 * M*N ): # Mx row
                y = int((i-M*N)/N)
                x = (i-M*N) - y * N

                # relevant j coordinates
                j0 = i
                j1 = (y+1)*N + x + M*N

                # diagonal element is permeability at (x,y)
                # if(simple_mu): A[i, j0] = -1j
                A[i,j0] = -1j * mu.get_value(x,y+0.5)

                A[i, j0-M*N] = -ody[y,x]

                if(y < M-1):
                    A[i,j1-M*N] = ody[y,x]

            else: # My row
                y = int((i-2*M*N)/N)
                x = (i-2*M*N) - y * N

                # relevant j coordinates
                j0 = i
                j1 = i-1

                # diagonal is permeability at (x,y)
                A[i,j0] = -1j * mu.get_value(x-0.5,y)
                A[i,j0-2*M*N] = -odx[y,x]

                if(x > 0):
                    A[i,j1-2*M*N] = odx[y,x]

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
            x2 = self.N
            y1 = 0
            y2 = self.M
        else:
            x1 = bbox[0]
            x2 = bbox[1]
            y1 = bbox[2]
            y2 = bbox[3]

        self.A_diag_update = row_wise_A_update(eps, mu, self.ib, self.ie, M, N,\
                                               x1, x2, y1, y2, \
                                               self.A_diag_update)

        A_update = self.A_diag_update
        for i in xrange(self.ib, self.ie):
            A[i,i] = A_update[i-self.ib]

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
        self.ksp.setOperators(self._A, self._A)
        self.ksp.setFromOptions()
        if(self._use_iterative):
            self.ksp.setInitialGuessNonzero(True)
        self.ksp.solve(self.b, self.x)

        if(RANK == 0):
            convergence = self.ksp.getConvergedReason()
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

            self.Ez = np.reshape(fields[0:MN], [self._M, self._N])
            self.Hx = np.reshape(fields[MN:2*MN], [self._M, self._N])
            self.Hy = np.reshape(fields[2*MN:3*MN], [self._M, self._N])

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

        # solve A^T u = c
        self.ksp.setOperators(self._A)
        self.ksp.setFromOptions()
        if(self._use_iterative):
            self.ksp.setInitialGuessNonzero(True)
        self.ksp.solveTranspose(self.b_adj, self.x_adj)

        if(NOT_PARALLEL):
            convergence = self.ksp.getConvergedReason()
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

            self.Ez_adj = np.reshape(fields[0:MN], [self._M, self._N])
            self.Hx_adj = np.reshape(fields[MN:2*MN], [self._M, self._N])
            self.Hy_adj = np.reshape(fields[2*MN:3*MN], [self._M, self._N])

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
        if(RANK is not 0):
            return MathDummy()

        if(domain is not None):
            j = domain.j
            k = domain.k

        if(component is 'Ez'):
            if(domain is not None):
                return self.Ez[j,k]
            else:
                return np.copy(self.Ez)
        elif(component is 'Hx'):
            if(domain is not None):
                return self.Hx[j,k]
            else:
                return np.copy(self.Hx)
        elif(component is 'Hy'):
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
        if(RANK is not 0):
            return MathDummy()

        if(domain is not None):
            j = domain.j
            k = domain.k

        if(component is 'Ez'):
            if(domain is not None):
                return self.Ez[j, k]
            else:
                return np.copy(self.Ez)
        elif(component is 'Hx'):
            Hx0 = np.copy(self.Hx)
            Hx0[1:, :] += self.Hx[0:self._M-1,:]

            if(domain is not None):
                return Hx0[j, k]/2.0
            else:
                return Hx0 / 2.0
        elif(component is 'Hy'):
            Hy0 = np.copy(self.Hy)
            Hy0[:, 0:self._N-1] += self.Hy[:, 1:]

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
        if(RANK is not 0):
            return MathDummy()

        if(domain is not None):
            j = domain.j
            k = domain.k

        if(component is 'Ez'):
            if(domain is not None):
                return self.Ez_adj[j, k]
            else:
                return self.Ez_adj
        elif(component is 'Hx'):
            if(domain is not None):
                return self.Hx_adj[j, k]
            else:
                return self.Hx_adj
        elif(component is 'Hy'):
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
        1. According to Poynting's theorem, instead of integrating the fields
        along the boundary AND the interior, we could integrate E dot J +
        M dot H.  This would be slightly faster. HOWEVER, it looks like the
        presence of the PMLs makes these two values slightly different as they
        introduce non-physical loss.  Ultimately we only care about the losses
        associated with materials not in the PMLs and thus we use the integral
        of the pointing vector and field energy to compute the total source
        power.

        2. The source power is computed using the interpolated fields.

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

        if(RANK is not 0):
            return MathDummy()

        # calculate the Poynting vectors around the boundaries of the sytem
        # (excluding the PML regions)
        Sx = -0.5*(Ez*np.conj(Hy)).real
        Sy = 0.5*(Ez*np.conj(Hx)).real

        S1 = -Sy[w_pml_b, w_pml_l:N-w_pml_r]
        S2 = Sy[M-w_pml_t, w_pml_l:N-w_pml_r]
        S3 = -Sx[w_pml_b:M-w_pml_t, w_pml_l]
        S4 = Sx[w_pml_b:M-w_pml_t, N-w_pml_r]

        # total power flowing our of boundaries
        P_S = np.sum(S1 + S2) * dx + np.sum(S3 + S4) * dy

        x_all = np.arange(w_pml_l, N-w_pml_r)
        y_all = np.arange(w_pml_b, M-w_pml_t)
        y_all = y_all.reshape(y_all.shape[0], 1).astype(np.int)
        eps = self._eps.get_values(0, M, 0, N)
        mu = self._mu.get_values(0, M, 0, N)

        # power dissipated due to material absorption
        Ez2 = Ez[y_all, x_all]*np.conj(Ez[y_all, x_all])
        Hx2 = Hx[y_all, x_all]*np.conj(Hx[y_all, x_all])
        Hy2 = Hy[y_all, x_all]*np.conj(Hy[y_all, x_all])

        P_loss = 0.25 * dx * dy * np.sum(eps[y_all, x_all].imag*Ez2 + \
                                         mu[y_all, x_all].imag*(Hx2 + Hy2))

        return P_S + P_loss.real

class FDFD_TM(FDFD_TE):
    """Simulate Maxwell's equations in 2D with TM-polarized fields."""

    def __init__(self, W, H, dx, dy, wavelength, w_pml, use_iterative=False, verbose=True):
        super(FDFD_TM, self).__init__(W, H, dx, dy, wavelength, w_pml, use_iterative=False, verbose=True)

    @property
    def eps(self):
        return self._eps_actual

    @property
    def mu(self):
        return self._mu_actual


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

    def set_sources(self, src):
        """Set the sources of the system used in the forward solve.

        Currently, sources are defined using 3 numpy.ndarrays.  The elements of
        the array correspond to spatially-dependent electric or magnetic current
        sources.  In the future, more structured source elements may be
        implemented.

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
        Implement a more user-friendly version of these sources (so that you do
        not need to deal with the Yee cell implementation).

        Parameters
        ----------
        src : tuple of numpy.ndarray
            The current sources in the form (Mz, Jx, Jy).  Each array in the
            tiple should be a 2D numpy.ndarry with dimensions MxN.
        """
        # In order to properly make use of the TE subclass, we need to flip the
        # sign of Jx and Jy
        super(FDFD_TM, self).set_sources((src[0], -1*src[1], -1*src[2]))

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
        1. According to Poynting's theorem, instead of integrating the fields
        along the boundary AND the interior, we could integrate E dot J +
        M dot H.  This would be slightly faster. HOWEVER, it looks like the
        presence of the PMLs makes these two values slightly different as they
        introduce non-physical loss.  Ultimately we only care about the losses
        associated with materials not in the PMLs and thus we use the integral
        of the pointing vector and field energy to compute the total source
        power.

        2. The source power is computed using the interpolated fields.

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

        if(RANK is not 0):
            return MathDummy()

        # calculate the Poynting vectors around the boundaries of the sytem
        # (excluding the PML regions)
        Sx = 0.5*(Ey*np.conj(Hz)).real
        Sy = -0.5*(Ex*np.conj(Hz)).real

        S1 = -Sy[w_pml_b, w_pml_l:N-w_pml_r]
        S2 = Sy[M-w_pml_t, w_pml_l:N-w_pml_r]
        S3 = -Sx[w_pml_b:M-w_pml_t, w_pml_l]
        S4 = Sx[w_pml_b:M-w_pml_t, N-w_pml_r]

        # total power flowing our of boundaries
        P_S = np.sum(S1 + S2) * dx + np.sum(S3 + S4) * dy

        x_all = np.arange(w_pml_l, N-w_pml_r)
        y_all = np.arange(w_pml_b, M-w_pml_t)
        y_all = y_all.reshape(y_all.shape[0], 1).astype(np.int)
        eps = self._eps_actual.get_values(0, M, 0, N)
        mu = self._mu_actual.get_values(0, M, 0, N)

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
