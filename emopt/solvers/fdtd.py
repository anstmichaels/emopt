"""Implements a simple 3D CW-FDTD (continuous wave finite difference frequenct domain)
solver for simulating Maxwell's equations.

The FDTD method provides a simple, efficient, highly parallelizable, and scalable
solution for solving Maxwell's equations. Typical implementations of FDTD, however,
can be tricky to use with highly dispersive materials and are tough to use in
conjunction with the adjoint method for inverse electromagnetic design.

In order to overcome these issues, this implementation uses a ramped-CW source which
allows us to solve for the frequency domain fields at the exact frequency we care
about without doing any form of interpolation. While this eliminates one of FDTD's
advantages (obtaining broadband info with a single simulation), it is easier to
implement and plays nice with the adjoint method which is easily derivable for
frequency-domain solvers. Furthermore, contrary to explicit frequency domain solvers,
CW-FDTD is highly parallelizable which enables it to run quite a bit faster.
Furthermore, compared to frequency domain solvers like FDFD, CW-FDTD consumes
considerably less RAM, making it useful for optimizing much larger devices at much
higher resolutions.

"""
from .simulation import MaxwellSolver
from ..defs import FieldComponent
from ..misc import DomainCoordinates, RANK, MathDummy, NOT_PARALLEL, COMM, \
info_message, warning_message, N_PROC, run_on_master
from ._fdtd_ctypes import libFDTD
from .modes import Mode2D
import petsc4py
import sys
petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np
from math import pi
import sys

__author__ = "Andrew Michaels"
__license__ = "GPL License, Version 3.0"

class SourceArray(object):
    """A container for source arrays and its associated parameters.

    Parameters
    ----------
    Jx : numpy.ndarray
        The x-component of the electric current density
    Jy : numpy.ndarray
        The x-component of the electric current density
    Jz : numpy.ndarray
        The x-component of the electric current density
    Mx : numpy.ndarray
        The x-component of the magnetic current density
    My : numpy.ndarray
        The x-component of the magnetic current density
    Mz : numpy.ndarray
        The x-component of the magnetic current density
    i0 : int
        The LOCAL starting z index of the source arrays
    j0 : int
        The LOCAL starting y index of the source arrays
    k0 : int
        The LOCAL starting k index of the source arrays
    I : int
        The z width of the source array
    J : int
        The y width of the source array
    K : int
        The z width of the source array
    """
    def __init__(self, Jx, Jy, Jz, Mx, My, Mz, i0, j0, k0, I, J, K):
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.Mx = Mx
        self.My = My
        self.Mz = Mz
        self.i0 = i0
        self.j0 = j0
        self.k0 = k0
        self.I  = I
        self.J  = J
        self.K  = K

class Maxwell3D(MaxwellSolver):
    """A 3D continuous-wave finite difference time domain (CW-FDTD) solver for Maxwell's
    equations.

    This class implements a continuous-wave finite difference time domain
    solver which solves for the freqeuncy-domain fields using the finite
    difference time domain method. Unlike the typical pulsed FDTD method, this
    implementation uses a ramped CW source which ensures that the simulation
    error exhibits convergent behavior (which is useful for optimization).
    Furthermore, this CW eliminates the need to perform discrete fourier
    transforms and allows us to compute the fields at the exact
    frequency/wavelength we desire (without relying on any interpolation which
    complicates optimizations).

    Compared to :class:`fdfd.FDFD_3D`, FDTD will generally scale to
    larger/higher resolution problems much better. For such problems, the
    memory consumption can be an order of magnitude or more lower and it
    parallelizes significantly better. For very small problems, however,
    :class:`fdfd.FDFD_3D` may be faster.

    Notes
    -----
    1. Complex materials are not yet supported, however they will be in the
    future!

    2. Because of how power is computed, you should not do any calculations
    within 1 grid cells of the PMLs

    3. Currently, the adjoint simulation can have some difficulty converging
    when used for a power-normalized figure of merit. To get around this, you
    can either increase the relative tolerance, or better yet set a maximum
    number of time steps using Nmax.

    4. Power is currently computed using a transmission box which is placed 1
    grid cell within the PML regions and does not account for material
    absorption.

    Parameters
    ----------
    X : float
        The width of the simulation in the x direction
    Y : float
        The width of the simulation in the y direction
    Z : float
        The width of the simulation in the z direction
    dx : float
        The grid spacing in the x direction
    dy : float
        The grid spacing in the y direction
    dz : float
        The grid spacing in the z direction
    wavelength : float
        The wavelength of the source (and fields)
    rtol : float (optional)
        The relative tolerance used to terminate the simulation.
        (default = 1e-6)
    nconv : int (optional)
        The number of field points to check for convergence. (default = 100)
    min_rindex : float (optional)
        The minimum refractive index of the simulation. Setting this parameter
        correctly can speed up your simulations by quite a bit, however you
        must be careful that its value does not exceed the minimum refractive
        index in your simulation. (default = 1.0)
    complex_eps : boolean (optional)
        Tells the solver if the permittivity is complex-valued. Setting this to
        False can speed up the solver when run on fewer cores (default = False)
    dist_method : str
        Method for distributing the underlying field arrays. Options are: 'auto', 'x', 'y',
        'z'. 'auto' will automatically determine the optimal distribution of the arrays
        between the processors. 'x', 'y', and 'z' will force the array to be distributed as
        slices stacked along the x, y, or z axis respectively.

    Attributes
    ----------
    X : float
        The width of the simulation in the x direction
    Y : float
        The width of the simulation in the y direction
    Z : float
        The width of the simulation in the z direction
    dx : float
        The grid spacing in the x direction
    dy : float
        The grid spacing in the y direction
    dz : float
        The grid spacing in the z direction
    Nx : int
        The number of grid cells along the x direction
    Ny : int
        The number of grid cells along the y direction
    Nz : int
        The number of grid cells along the z direction
    courant_num : float
        The courant number (fraction of maximum time step needed for
        stability). This must be <= 1. By default it is 0.95. Don't change this
        unless you simulations appear to diverge for some reason.
    wavelength : float
        The wavelength of the source (and fields)
    eps : emopt.grid.Material3D
        The permittivity distribution.
    mu : emopt.grid.Material3D
        The permeability distribution.
    rtol : float (optional)
        The relative tolerance used to terminate the simulation.
        (default = 1e-6)
    src_min_value : float
        The starting value of the ramped source. The source is ramped using a
        smooth envelope function which never truely goes to zero. This
        attribute lets us set the "zero" value which will effect the
        convergence time and minimum error. (default = 1e-5)
    src_ramp_time : int
        The number of time steps over which the source is ramped. This will
        affect the convergence time and final error. Note: faster ramps lead to
        higher error (in the current implementation) and do not gurantee that
        the simulation will converge faster. (default = 15*Nlambda)
    Nmax : int
        Max number of time steps
    Nlambda : float
        Number of spatial steps per wavelength (in minimum index material)
    Ncycle : float
        Number of time steps per period of oscillation of the fields.
    bc : str
        The the boundary conditions (PEC, field symmetries, etc)
    w_pml : list of floats
        The list of pml widths (in real coordinates) in the format [xmin, xmax,
        ymin, ymax, zmin, zmax]. Use this to change the PML widths.
    w_pml_xmin : int
        The number of grid cells making up with PML at the minimum x boundary.
    w_pml_xmax : int
        The number of grid cells making up with PML at the maximum x boundary.
    w_pml_ymin : int
        The number of grid cells making up with PML at the minimum y boundary.
    w_pml_ymax : int
        The number of grid cells making up with PML at the maximum y boundary.
    w_pml_zmin : int
        The number of grid cells making up with PML at the minimum z boundary.
    w_pml_zmax : int
        The number of grid cells making up with PML at the maximum z boundary.
    X_real : float
        The width of the simulation excluding PMLs.
    Y_real : float
        The height of the simulation excluding PMLs.
    """

    def __init__(self, X, Y, Z, dx, dy, dz, wavelength, rtol=1e-6, nconv=None,
                 min_rindex=1.0, complex_eps=False, dist_method='auto'):
        super().__init__(3)

        if(nconv is None):
            nconv = N_PROC*10
        elif(nconv < N_PROC*2):
            warning_message('Number of convergence test points (nconv) is ' \
                            'likely too low. If the simulation does not ' \
                            'converge, increase nconv to > 2 * # processors',
                            'emopt.fdtd')

        self._dx = dx
        self._dy = dy
        self._dz = dz

        Nx = int(np.ceil(X/dx)+1); self._Nx = Nx
        Ny = int(np.ceil(Y/dy)+1); self._Ny = Ny
        Nz = int(np.ceil(Z/dz)+1); self._Nz = Nz

        self._X = dx * (Nx-1)
        self._Y = dy * (Ny-1)
        self._Z = dz * (Nz-1)

        self._wavelength = wavelength
        self._R = wavelength/(2*pi)

        ## Courant number < 1
        self._Sc = 0.95
        self._min_rindex = min_rindex
        dt = self._Sc * np.min([dx, dy, dz])/self._R / np.sqrt(3) * min_rindex
        self._dt = dt


        # stencil_type=1 => box
        # stencil_width=1 => 1 element ghosted region
        # boundary_type=1 => ghosted simulation boundary (padded everywhere)
        if(dist_method == 'auto'):
            da = PETSc.DA().create(sizes=[Nx, Ny, Nz], dof=1, stencil_type=0,
                                   stencil_width=1, boundary_type=1)
        elif(dist_method == 'x'):
            da = PETSc.DA().create(sizes=[Nx, Ny, Nz], proc_sizes=[1,1,N_PROC],
                                   dof=1, stencil_type=0,
                                   stencil_width=1, boundary_type=1)
        elif(dist_method == 'y'):
            da = PETSc.DA().create(sizes=[Nx, Ny, Nz], proc_sizes=[1,N_PROC,1],
                                   dof=1, stencil_type=0,
                                   stencil_width=1, boundary_type=1)
        elif(dist_method == 'z'):
            da = PETSc.DA().create(sizes=[Nx, Ny, Nz], proc_sizes=[N_PROC,1,1],
                                   dof=1, stencil_type=0,
                                   stencil_width=1, boundary_type=1)

        self._da = da
        ## Setup the distributed array. Currently, we need 2 for each field
        ## component (1 for the field, and 1 for the averaged material) and 1
        ## for each current density component
        self._vglobal = da.createGlobalVec() # global for data sharing

        pos, lens = da.getCorners()
        k0, j0, i0 = pos
        K, J, I = lens

        # field arrays
        self._Ex = np.zeros(((K+2)*(J+2)*(I+2),), dtype=np.double)
        self._Ey = np.zeros(((K+2)*(J+2)*(I+2),), dtype=np.double)
        self._Ez = np.zeros(((K+2)*(J+2)*(I+2),), dtype=np.double)
        self._Hx = np.zeros(((K+2)*(J+2)*(I+2),), dtype=np.double)
        self._Hy = np.zeros(((K+2)*(J+2)*(I+2),), dtype=np.double)
        self._Hz = np.zeros(((K+2)*(J+2)*(I+2),), dtype=np.double)

        # material arrays -- global since we dont need to pass values around
        self._eps_x = da.createGlobalVec()
        self._eps_y = da.createGlobalVec()
        self._eps_z = da.createGlobalVec()
        self._mu_x = da.createGlobalVec()
        self._mu_y = da.createGlobalVec()
        self._mu_z = da.createGlobalVec()

        # Frequency-domain field arrays for forward simulation
        # Two sets of fields for two snapshots in time. The frequency-domain
        # fields are stored in the t0 field set
        self._Ex_fwd_t0 = da.createGlobalVec()
        self._Ey_fwd_t0 = da.createGlobalVec()
        self._Ez_fwd_t0 = da.createGlobalVec()
        self._Hx_fwd_t0 = da.createGlobalVec()
        self._Hy_fwd_t0 = da.createGlobalVec()
        self._Hz_fwd_t0 = da.createGlobalVec()

        self._Ex_fwd_t1 = da.createGlobalVec()
        self._Ey_fwd_t1 = da.createGlobalVec()
        self._Ez_fwd_t1 = da.createGlobalVec()
        self._Hx_fwd_t1 = da.createGlobalVec()
        self._Hy_fwd_t1 = da.createGlobalVec()
        self._Hz_fwd_t1 = da.createGlobalVec()

        # Frequency-domain field arrays for ajdoint simulation
        self._Ex_adj_t0 = da.createGlobalVec()
        self._Ey_adj_t0 = da.createGlobalVec()
        self._Ez_adj_t0 = da.createGlobalVec()
        self._Hx_adj_t0 = da.createGlobalVec()
        self._Hy_adj_t0 = da.createGlobalVec()
        self._Hz_adj_t0 = da.createGlobalVec()

        self._Ex_adj_t1 = da.createGlobalVec()
        self._Ey_adj_t1 = da.createGlobalVec()
        self._Ez_adj_t1 = da.createGlobalVec()
        self._Hx_adj_t1 = da.createGlobalVec()
        self._Hy_adj_t1 = da.createGlobalVec()
        self._Hz_adj_t1 = da.createGlobalVec()


        # setup the C library which takes care of the E and H updates
        # Nothing complicated here -- just passing all of the sim parameters
        # and work vectors over to the c library
        self._libfdtd = libFDTD.FDTD_new()
        libFDTD.FDTD_set_wavelength(self._libfdtd, wavelength)
        libFDTD.FDTD_set_physical_dims(self._libfdtd, X, Y, Z, dx, dy, dz)
        libFDTD.FDTD_set_grid_dims(self._libfdtd, Nx, Ny, Nz)
        libFDTD.FDTD_set_local_grid(self._libfdtd, k0, j0, i0, K, J, I)
        libFDTD.FDTD_set_dt(self._libfdtd, dt)
        libFDTD.FDTD_set_field_arrays(self._libfdtd,
                self._Ex, self._Ey, self._Ez,
                self._Hx, self._Hy, self._Hz)
        libFDTD.FDTD_set_mat_arrays(self._libfdtd,
                self._eps_x.getArray(), self._eps_y.getArray(), self._eps_z.getArray(),
                self._mu_x.getArray(), self._mu_y.getArray(), self._mu_z.getArray())

        # set whether or not materials are complex valued
        libFDTD.FDTD_set_complex_eps(self._libfdtd, complex_eps)

        ## Setup default PML properties
        w_pml = 15
        self._w_pml = [w_pml*dx, w_pml*dx, \
                      w_pml*dy, w_pml*dy, \
                      w_pml*dz, w_pml*dz]
        self._w_pml_xmin = w_pml
        self._w_pml_xmax = w_pml
        self._w_pml_ymin = w_pml
        self._w_pml_ymax = w_pml
        self._w_pml_zmin = w_pml
        self._w_pml_zmax = w_pml

        self._pml_sigma = 3.0
        self._pml_alpha = 0.0
        self._pml_kappa = 1.0
        self._pml_pow = 3.0

        libFDTD.FDTD_set_pml_widths(self._libfdtd, w_pml, w_pml,
                                                   w_pml, w_pml,
                                                   w_pml, w_pml)
        libFDTD.FDTD_set_pml_properties(self._libfdtd, self._pml_sigma,
                                        self._pml_alpha, self._pml_kappa,
                                        self._pml_pow)
        libFDTD.FDTD_build_pml(self._libfdtd)

        ## Setup the source properties
        Nlambda = wavelength / np.min([dx, dy, dz]) / self._min_rindex
        Ncycle = Nlambda * np.sqrt(3)
        self._Nlambda = Nlambda # spatial steps per wavelength
        self._Ncycle = Ncycle #  time steps per period of oscillation

        self._src_T    = Ncycle * 20.0 * self._dt
        self._src_min  = 1e-4
        self.Nmax = Nlambda*500
        libFDTD.FDTD_set_source_properties(self._libfdtd, self._src_T,
                                           self._src_min)

        self._rtol = rtol
        self.verbose = 2

        ## determine the points used to check for convergence on this process
        # these are roughly evenly spaced points in the local-stored field
        # vectors. We exclude PMLs from these checks. Consider moving this to
        # C++ if this ever gets slow (possible for larger problems?)
        n_pts = int(nconv / N_PROC)
        self._conv_pts = np.arange(0, I*J*K, int(I*J*K/n_pts))
        self._nconv = len(self._conv_pts)

        self._sources = []
        self._adj_sources = []

        ## Define a natural vector for field retrieval
        # We could do this much more efficiently, but except for huge problems,
        # it isn't really necessary and this way the code is simpler
        self._vn = da.createNaturalVec()

        # default boundary conditions = PEC
        self._bc = ['0', '0', '0']
        libFDTD.FDTD_set_bc(self._libfdtd, ''.join(self._bc).encode('ascii'))

        # make room for eps and mu
        self._eps = None
        self._mu = None

        # defome a GhostComm object which we will use to share edge values of
        # the fields between processors
        self._gc = GhostComm(k0, j0, i0, K, J, I, Nx, Ny, Nz)

        # keep track of number of simulations
        self.forward_count = 0
        self.adjoint_count = 0

    @property
    def wavelength(self):
        self._wavelength = self._wavelength

    @wavelength.setter
    def wavelength(self, wlen):
        if(wlen <= 0):
            raise ValueError('Wavelength must be >= 0.')
        else:
            self._wavelength = wlen
            self._R = wlen/2/pi
            libFDTD.FDTD_set_wavelength(self._libfdtd, wlen)

            ds = np.min([self._dx, self._dy, self._dz])
            dt = self._Sc * ds/self._R / np.sqrt(3) * self._min_rindex
            self._dt = dt
            libFDTD.FDTD_set_dt(self._libfdtd, dt)

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
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def dz(self):
        return self._dz

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
    def eps(self):
        return self._eps

    @property
    def mu(self):
        return self._mu

    @property
    def courant_num(self):
        return self._Sc

    @courant_num.setter
    def courant_num(self, Sc):
        if(Sc > 1.0):
            warning_message('A courant number greater than 1 will lead to ' \
                            'instability.', 'emopt.fdtd')
        self._Sc = Sc
        ds = np.min([self._dx, self._dy, self._dz])
        dt = self._Sc * ds/self._R / np.sqrt(3) * self._min_rindex
        self._dt = dt
        libFDTD.FDTD_set_dt(self._libfdtd, dt)

    @property
    def rtol(self):
        return self._rtol

    @rtol.setter
    def rtol(self, rtol):
        if(rtol <= 0):
            raise ValueError('Relative tolerance must be greater than zero.')
        self._rtol = rtol

    @property
    def src_min_value(self):
        return self._src_min

    @src_min_value.setter
    def src_min_value(self, new_min):
        if(new_min <= 0):
            raise ValueError('Minimum source value must be greater than zero.')

        self._src_min = new_min
        libFDTD.FDTD_set_source_properties(self._libfdtd, self._src_T,
                                           self._src_min)

    @property
    def Nlambda(self):
        return self._Nlambda

    @property
    def Ncycle(self):
        return self._Ncycle

    @property
    def src_ramp_time(self):
        return self._src_T

    @src_ramp_time.setter
    def src_ramp_time(self, new_T):
        if(new_T <= 0):
            raise ValueError('Source ramp time must be greater than zero.')
        elif(new_T <= self._Nlambda*5):
            warning_message('Source ramp time is likely too short. '\
                            'Simulations may not converge.',
                            'emopt.fdtd')
            self._src_T = new_T
            libFDTD.FDTD_set_source_properties(self._libfdtd, self._src_T,
                                               self._src_min)
        else:
            self._src_T = new_T
            libFDTD.FDTD_set_source_properties(self._libfdtd, self._src_T,
                                               self._src_min)
    @property
    def bc(self):
        return ''.join(self._bc)

    @bc.setter
    def bc(self, newbc):
        if(len(newbc) != 3):
            raise ValueError('Incorrect number of boundary conditions specified!')

        for bc in newbc:
            if(not bc in '0EHP'):
                raise ValueError('Unrecognized boundary condition: %s' % (bc))
            if(bc == 'P'):
                raise NotImplementedError('Periodic boundary conditions not yet ' \
                                          'implemented')

        self._bc = list(newbc)
        libFDTD.FDTD_set_bc(self._libfdtd, ''.join(self._bc).encode('ascii'))

    @property
    def w_pml(self):
        return self._w_pml

    @w_pml.setter
    def w_pml(self, w_pml):
        if(len(w_pml) != 6):
            raise ValueError('Incorrect number of pml widths specified. You ' \
                             'must specify 6 widths (xmin, xmax, ymin, ymax, '\
                             'zmin, zmax).')

        dx = self._dx; dy = self._dy; dz = self._dz

        self._w_pml_xmin = int(w_pml[0]/dx)
        self._w_pml_xmax = int(w_pml[1]/dx)
        self._w_pml_ymin = int(w_pml[2]/dy)
        self._w_pml_ymax = int(w_pml[3]/dy)
        self._w_pml_zmin = int(w_pml[4]/dy)
        self._w_pml_zmax = int(w_pml[5]/dy)

        self._w_pml = [self._w_pml_xmin*dx, self._w_pml_xmax*dx, \
                       self._w_pml_ymin*dy, self._w_pml_ymax*dy, \
                       self._w_pml_zmin*dz, self._w_pml_zmax*dz]

        # rebuild the PMLs
        libFDTD.FDTD_set_pml_widths(self._libfdtd, self._w_pml_xmin,
                                                   self._w_pml_xmax,
                                                   self._w_pml_ymin,
                                                   self._w_pml_ymax,
                                                   self._w_pml_zmin,
                                                   self._w_pml_zmax)
        libFDTD.FDTD_build_pml(self._libfdtd)

    @property
    def w_pml_xmin(self):
        return self._w_pml_xmin

    @property
    def w_pml_xmax(self):
        return self._w_pml_xmax

    @property
    def w_pml_ymin(self):
        return self._w_pml_ymin

    @property
    def w_pml_ymax(self):
        return self._w_pml_ymax

    @property
    def w_pml_zmin(self):
        return self._w_pml_zmin

    @property
    def w_pml_zmax(self):
        return self._w_pml_zmax

    @property
    def X_real(self):
        return self._X-self._w_pml[0]-self._w_pml[1]

    @property
    def Y_real(self):
        return self._Y-self._w_pml[2]-self._w_pml[3]

    @property
    def Z_real(self):
        return self._Z-self._w_pml[4]-self._w_pml[5]

    def set_materials(self, eps, mu):
        self._eps = eps
        self._mu = mu

    def __get_local_domain_overlap(self, domain):
        ## Get the local and global ijk bounds which correspond to the overlap
        # between the supplied domain and the grid chunk owned by this
        # processor
        #
        # Returns four tuples or four None
        #   None if no overlap
        #   (global start indices), (local start indices), 
        #        (domain start indices), (widths of overlap)
        pos, lens = self._da.getCorners()
        k0, j0, i0 = pos
        K, J, I = lens

        # determine the local portion of the array that is relevant
        # First: don't do anything if this process does not contain the
        # provided source arrays
        if(i0 >= domain.i2):       return None, None, None, None
        elif(i0 + I <= domain.i1): return None, None, None, None
        elif(j0 >= domain.j2):     return None, None, None, None
        elif(j0 + J <= domain.j1): return None, None, None, None
        elif(k0 >= domain.k2):     return None, None, None, None
        elif(k0 + K <= domain.k1): return None, None, None, None

        imin = 0; jmin = 0; kmin = 0
        imax = 0; jmax = 0; kmax = 0
        if(i0 + I > domain.i2): imax = domain.i2
        else: imax = i0 + I

        if(j0 + J > domain.j2): jmax = domain.j2
        else: jmax = j0 + J

        if(k0 + K > domain.k2): kmax = domain.k2
        else: kmax = k0 + K

        if(i0 > domain.i1): imin = i0
        else: imin = domain.i1

        if(j0 > domain.j1): jmin = j0
        else: jmin = domain.j1

        if(k0 > domain.k1): kmin = k0
        else: kmin = domain.k1

        id1 = imin - domain.i1
        jd1 = jmin - domain.j1
        kd1 = kmin - domain.k1

        return (imin, jmin, kmin), \
               (imin - i0, jmin - j0, kmin - k0), \
               (id1, jd1, kd1), \
               (imax - imin, jmax - jmin, kmax - kmin)

    def __set_sources(self, src, domain, adjoint=False):
        ## Set the source arrays. The process is identical for forward and
        # adjoint, so this function manages both using a flag. The process
        # consists of finding which portion of the supplied arrays overlaps
        # with the space owned by this processor, copying that piece, and then
        # submitting the cropped arrays to libFDTD which will convert the
        # complex-valued source to an amplitude and phase.
        Jx, Jy, Jz, Mx, My, Mz = src

        g_inds, l_inds, d_inds, sizes = self.__get_local_domain_overlap(domain)
        if(g_inds == None): return # no overlap between source and this chunk

        id1 = d_inds[0]; id2 = d_inds[0] + sizes[0]
        jd1 = d_inds[1]; jd2 = d_inds[1] + sizes[1]
        kd1 = d_inds[2]; kd2 = d_inds[2] + sizes[2]

        # get the pieces which are relevant to this processor
        Jxs = np.copy(Jx[id1:id2, jd1:jd2, kd1:kd2]).ravel()
        Jys = np.copy(Jy[id1:id2, jd1:jd2, kd1:kd2]).ravel()
        Jzs = np.copy(Jz[id1:id2, jd1:jd2, kd1:kd2]).ravel()
        Mxs = np.copy(Mx[id1:id2, jd1:jd2, kd1:kd2]).ravel()
        Mys = np.copy(My[id1:id2, jd1:jd2, kd1:kd2]).ravel()
        Mzs = np.copy(Mz[id1:id2, jd1:jd2, kd1:kd2]).ravel()

        src = SourceArray(Jxs, Jys, Jzs,
                          Mxs, Mys, Mzs,
                          l_inds[0], l_inds[1], l_inds[2],
                          sizes[0], sizes[1], sizes[2])

        if(adjoint): self._adj_sources.append(src)
        else: self._sources.append(src)

        libFDTD.FDTD_add_source(self._libfdtd,
                                src.Jx, src.Jy, src.Jz,
                                src.Mx, src.My, src.Mz,
                                src.i0, src.j0, src.k0,
                                src.I, src.J, src.K,
                                True)

    def set_sources(self, src, domain, mindex=0):
        """Set a simulation source.

        Simulation sources can be set either using a set of 6 arrays (Jx, Jy,
        Jz, Mx, My, Mz) or a :class:`modes.Mode2D` object. In either
        case, a domain must be provided which tells the simulation where to put
        those sources.

        This function operates in an additive manner meaning that multiple
        calls will add multiple sources to the simulation. To replace the
        existing sources, simply call :def:`clear_sources` first.

        Parameters
        ----------
        src : tuple or modes.Mode2D
            The source arrays or mode object containing source data
        domain : misc.DomainCoordinates
            The domain which specifies where the source is located
        mindex : int (optional)
            The mode source index. This is only relevant if using a
            Mode2D object to set the sources. (default = 0)
        """
        if(type(src) == Mode2D):
            Jxs, Jys, Jzs, Mxs, Mys, Mzs = src.get_source(mindex, self._dx,
                                                                  self._dy,
                                                                  self._dz)

            Jxs = COMM.bcast(Jxs, root=0)
            Jys = COMM.bcast(Jys, root=0)
            Jzs = COMM.bcast(Jzs, root=0)
            Mxs = COMM.bcast(Mxs, root=0)
            Mys = COMM.bcast(Mys, root=0)
            Mzs = COMM.bcast(Mzs, root=0)

            src_arrays = (Jxs, Jys, Jzs, Mxs, Mys, Mzs)
        else:
            src_arrays = src

        self.__set_sources(src_arrays, domain, adjoint=False)
        COMM.Barrier()

    def set_adjoint_sources(self, src):
        """Set the adjoint sources.

        This function works a bit differently than set_sources in order to
        maintain compatibility with the adjoint_method class. It takes a single
        argument which is a list of lists containing multiple sets of source
        arrays and corresponding domains.

        TODO
        ----
        Clean up EMopt interfaces so that this doesn't feel so hacky.

        Paramters
        ---------
        src : list
            List containing two lists: 1 with sets of source arrays and on with
            DomainCoordinates which correspond to those source array sets.
        """
        # clear the old sources
        self.clear_adjoint_sources()

        # add the new sources
        for dFdx, domain in zip(src[0], src[1]):
            self.add_adjoint_sources(dFdx, domain)

    def add_adjoint_sources(self, src, domain):
        """Add an adjoint source.

        Add an adjoint source using a set of 6 current source arrays.

        Parameters
        ----------
        src : list or tuple of numpy.ndarray
            The list/tuple of source arrays in the following order: (Jx, Jy, Jz,
            Mx, My, Mz)
        domain : misc.DomainCoordinates
            The domain which specifies the location of the source arrays
        """
        self.__set_sources(src, domain, adjoint=True)

    def clear_sources(self):
        """Clear simulation sources."""
        # Cleanup old source arrays -- not strictly necessary but it's nice to
        # guarantee that the space is freed up
        for src in self._sources: del src
        self._sources = []

    def clear_adjoint_sources(self):
        """Clear the adjoint simulation sources."""
        for src in self._adj_sources: del src
        self._adj_sources = []

    def build(self):
        """Assemble the strcutre.

        This involves computing the permittiviy and permeability distribution
        for the simulation.
        """
        if(self.verbose > 0 and NOT_PARALLEL):
            info_message('Building FDTD system...')

        eps = self._eps
        mu = self._mu

        pos, lens = self._da.getCorners()
        k0, j0, i0 = pos
        K, J, I = lens

        eps.get_values(k0, k0+K, j0, j0+J, i0, i0+I,
                       sx=0.5, sy=0.0, sz=-0.5,
                       arr=self._eps_x.getArray())

        eps.get_values(k0, k0+K, j0, j0+J, i0, i0+I,
                       sx=0.0, sy=0.5, sz=-0.5,
                       arr=self._eps_y.getArray())

        eps.get_values(k0, k0+K, j0, j0+J, i0, i0+I,
                       sx=0.0, sy=0.0, sz=0.0,
                       arr=self._eps_z.getArray())

        mu.get_values(k0, k0+K, j0, j0+J, i0, i0+I,
                      sx=0.0, sy=0.5, sz=0.0,
                      arr=self._mu_x.getArray())

        mu.get_values(k0, k0+K, j0, j0+J, i0, i0+I,
                       sx=0.5, sy=0.0, sz=0.0,
                       arr=self._mu_y.getArray())

        mu.get_values(k0, k0+K, j0, j0+J, i0, i0+I,
                       sx=0.5, sy=0.5, sz=-0.5,
                       arr=self._mu_z.getArray())

    def update(self, bbox=None):
        """Update the permittivity and permeability distribution.

        A small portion of the whole simulation region can be updated by
        supplying a bounding box which encompasses the desired region to be
        updated.

        Notes
        -----
        The bounding box accepted by this class is currently different from the
        FDFD solvers in that it is specified using real-space coordinates not
        indices. In the future, the other solvers will implement this same
        functionality.

        TODO
        ----
        Implement 3 point frequency domain calculation. This should be
        more accurate and produce more consistent results.

        Parameters
        ----------
        bbox : list of floats (optional)
            The region of the simulation which should be updated. If None, then
            the whole region is updated. Format: [xmin, xmax, ymin, ymax, zmin,
            zmax]
        """
        if(bbox == None):
            verbose = self.verbose
            self.verbose = 0
            self.build()
            self.verbose = verbose

        else:
            eps = self._eps
            mu = self._mu

            pos, lens = self._da.getCorners()
            k0, j0, i0 = pos
            K, J, I = lens

            bbox = DomainCoordinates(bbox[0], bbox[1], bbox[2], bbox[3],
                                     bbox[4], bbox[5], self._dx,
                                     self._dy, self._dz)
            g_inds, l_inds, d_inds, sizes = self.__get_local_domain_overlap(bbox)
            if(g_inds == None): return # no overlap between source and this chunk

            temp = np.zeros([sizes[0], sizes[1], sizes[2]], dtype=np.complex128)

            li = slice(l_inds[0], l_inds[0]+sizes[0])
            lj = slice(l_inds[1], l_inds[1]+sizes[1])
            lk = slice(l_inds[2], l_inds[2]+sizes[2])

            # update eps_x
            eps_x = self._eps_x.getArray()
            eps_x = np.reshape(eps_x, [I,J,K])
            eps.get_values(g_inds[2], g_inds[2]+sizes[2],
                           g_inds[1], g_inds[1]+sizes[1],
                           g_inds[0], g_inds[0]+sizes[0],
                           sx=0.5, sy=0.0, sz=-0.5,
                           arr=temp)
            eps_x[li, lj, lk] = temp

            # update eps_y
            eps_y = self._eps_y.getArray()
            eps_y = np.reshape(eps_y, [I,J,K])
            eps.get_values(g_inds[2], g_inds[2]+sizes[2],
                           g_inds[1], g_inds[1]+sizes[1],
                           g_inds[0], g_inds[0]+sizes[0],
                           sx=0.0, sy=0.5, sz=-0.5,
                           arr=temp)
            eps_y[li, lj, lk] = temp

            # update eps_z
            eps_z = self._eps_z.getArray()
            eps_z = np.reshape(eps_z, [I,J,K])
            eps.get_values(g_inds[2], g_inds[2]+sizes[2],
                           g_inds[1], g_inds[1]+sizes[1],
                           g_inds[0], g_inds[0]+sizes[0],
                           sx=0.0, sy=0.0, sz=0.0,
                           arr=temp)
            eps_z[li, lj, lk] = temp

            # update mu_x
            mu_x = self._mu_x.getArray()
            mu_x = np.reshape(mu_x, [I,J,K])
            mu.get_values(g_inds[2], g_inds[2]+sizes[2],
                          g_inds[1], g_inds[1]+sizes[1],
                          g_inds[0], g_inds[0]+sizes[0],
                          sx=0.0, sy=0.5, sz=0.0,
                          arr=temp)
            mu_x[li, lj, lk] = temp

            # update mu_y
            mu_y = self._mu_y.getArray()
            mu_y = np.reshape(mu_y, [I,J,K])
            mu.get_values(g_inds[2], g_inds[2]+sizes[2],
                          g_inds[1], g_inds[1]+sizes[1],
                          g_inds[0], g_inds[0]+sizes[0],
                          sx=0.5, sy=0.0, sz=0.0,
                          arr=temp)
            mu_y[li, lj, lk] = temp

            # update mu_z
            mu_z = self._mu_z.getArray()
            mu_z = np.reshape(mu_z, [I,J,K])
            mu.get_values(g_inds[2], g_inds[2]+sizes[2],
                          g_inds[1], g_inds[1]+sizes[1],
                          g_inds[0], g_inds[0]+sizes[0],
                          sx=0.5, sy=0.5, sz=-0.5,
                          arr=temp)
            mu_z[li, lj, lk] = temp

    def __solve(self):
        ## Solve Maxwell's equations. This process is identical for the forward
        # and adjoint simulation. The only difference is the specific sources
        # and auxillary arrays used for saving the final frequency-domain
        # fields.
        # setup spatial derivative factors
        R = self._wavelength/(2*pi)
        odx = R/self._dx
        ody = R/self._dy
        odz = R/self._dz
        Nx, Ny, Nz = self._Nx, self._Ny, self._Nz

        COMM.Barrier()

        # define time step
        dt = self._dt

        pos, lens = self._da.getCorners()
        k0, j0, i0 = pos
        K, J, I = lens

        # Reset field values, pmls, etc
        libFDTD.FDTD_reset_pml(self._libfdtd)
        self._Ex.fill(0); self._Ey.fill(0); self._Ez.fill(0)
        self._Hx.fill(0); self._Hy.fill(0); self._Hz.fill(0)

        da = self._da
        Tn = np.int(self._Ncycle*3/4)
        p = 0

        Ex0 = np.zeros(self._nconv); Ey0 = np.zeros(self._nconv); Ez0 = np.zeros(self._nconv)
        Ex1 = np.zeros(self._nconv); Ey1 = np.zeros(self._nconv); Ez1 = np.zeros(self._nconv)
        Ex2 = np.zeros(self._nconv); Ey2 = np.zeros(self._nconv); Ez2 = np.zeros(self._nconv)

        phi0 = np.zeros(self._nconv)
        phi1 = np.zeros(self._nconv)
        phi2 = np.zeros(self._nconv)

        A0 = np.zeros(self._nconv)
        A1 = np.zeros(self._nconv)
        A2 = np.zeros(self._nconv)

        t0 = 0; t1 = 0; t2 = 0

        A_change = 1
        phi_change = 1

        recvbuff = []

        # Note: ultimately we care about the error in the real and imaginary
        # part of the fields. The error in the real/imag parts goes as the
        # square of the phase error.
        amp_rtol = self._rtol
        phi_rtol = np.sqrt(self._rtol)

        n = 0
        while(A_change > amp_rtol or phi_change > phi_rtol or \
              np.isnan(A_change) or np.isinf(A_change) or \
              np.isnan(phi_change) or np.isinf(phi_change)):

            if(n > self.Nmax):
                warning_message('Maximum number of time steps exceeded.',
                                'emopt.fdtd')
                break


            libFDTD.FDTD_update_H(self._libfdtd, n, n*dt)

            self._gc.update_local_vector(self._Hx)
            self._gc.update_local_vector(self._Hy)
            self._gc.update_local_vector(self._Hz)

            libFDTD.FDTD_update_E(self._libfdtd, n, (n+0.5)*dt)

            self._gc.update_local_vector(self._Ex)
            self._gc.update_local_vector(self._Ey)
            self._gc.update_local_vector(self._Ez)

            if(p == Tn-1):
                # Update times of field snapshots
                t0 = t1
                t1 = t2
                t2 = n*dt

                phi0[:] = phi1
                A0[:] = A1
                for q in range(self._nconv):
                    conv_index = self._conv_pts[q]

                    # start with Ex
                    Ex0[q] = Ex1[q]
                    Ex1[q] = Ex2[q]
                    Ex2[q] = np.real(self._Ex[conv_index])

                    Ey0[q] = Ey1[q]
                    Ey1[q] = Ey2[q]
                    Ey2[q] = np.real(self._Ey[conv_index])

                    Ez0[q] = Ez1[q]
                    Ez1[q] = Ez2[q]
                    Ez2[q] = np.real(self._Ez[conv_index])

                    phasex = libFDTD.FDTD_calc_phase_3T(t0, t1, t2,
                                                        Ex0[q], Ex1[q], Ex2[q])
                    phasey = libFDTD.FDTD_calc_phase_3T(t0, t1, t2,
                                                        Ey0[q], Ey1[q], Ey2[q])
                    phasez = libFDTD.FDTD_calc_phase_3T(t0, t1, t2,
                                                        Ez0[q], Ez1[q], Ez2[q])

                    ampx = libFDTD.FDTD_calc_amplitude_3T(t0, t1, t2, Ex0[q],
                                                          Ex1[q], Ex2[q], phasex)
                    ampy = libFDTD.FDTD_calc_amplitude_3T(t0, t1, t2, Ey0[q],
                                                          Ey1[q], Ey2[q], phasey)
                    ampz = libFDTD.FDTD_calc_amplitude_3T(t0, t1, t2, Ez0[q],
                                                          Ez1[q], Ez2[q], phasez)

                    if(ampx < 0):
                        ampx *= -1
                        phasex += pi
                    if(ampy < 0):
                        ampy *= -1
                        phasey += pi
                    if(ampz < 0):
                        ampz *= -1
                        phasez += pi

                    phi1[q] = phasex + phasey + phasez
                    A1[q] = ampx + ampy + ampz


                # broadcast amplitudes and phases so that "net" change can be
                # calculated
                phi1s = COMM.gather(phi1, root=0)
                phi0s = COMM.gather(phi0, root=0)
                A1s   = COMM.gather(A1, root=0)
                A0s   = COMM.gather(A0, root=0)

                if(NOT_PARALLEL):
                    phi0s = np.concatenate(phi0s)
                    phi1s = np.concatenate(phi1s)
                    A0s = np.concatenate(A0s)
                    A1s = np.concatenate(A1s)

                    A_change = np.linalg.norm(A1s-A0s)/np.linalg.norm(A0s)
                    phi_change = np.linalg.norm(phi1s-phi0s)/np.linalg.norm(phi0s)
                else:
                    A_change = 1
                    phi_change = 1

                A_change = COMM.bcast(A_change, root=0)
                phi_change = COMM.bcast(phi_change, root=0)

                if(NOT_PARALLEL and self.verbose > 1 and n > 2*Tn):
                    print('time step: {0: <8d} Delta A: {1: <12.4e} ' \
                          'Delta Phi: {2: <12.4e}'.format(n, A_change, phi_change))

                p = 0
            else:
                p += 1

            n += 1

        libFDTD.FDTD_capture_t0_fields(self._libfdtd)

        # perform a couple more iterations to get a second time point
        n0 = n
        for n in range(Tn):
            libFDTD.FDTD_update_H(self._libfdtd, n+n0, (n+n0)*dt)

            # Note: da.localToLocal seems to have the same performance?
            self._gc.update_local_vector(self._Hx)
            self._gc.update_local_vector(self._Hy)
            self._gc.update_local_vector(self._Hz)

            libFDTD.FDTD_update_E(self._libfdtd, n+n0, (n+n0+0.5)*dt)

            self._gc.update_local_vector(self._Ex)
            self._gc.update_local_vector(self._Ey)
            self._gc.update_local_vector(self._Ez)


        libFDTD.FDTD_capture_t1_fields(self._libfdtd)

        for n in range(Tn):
            libFDTD.FDTD_update_H(self._libfdtd, n+n0+Tn, (n+n0+Tn)*dt)

            # Note: da.localToLocal seems to have the same performance?
            self._gc.update_local_vector(self._Hx)
            self._gc.update_local_vector(self._Hy)
            self._gc.update_local_vector(self._Hz)

            libFDTD.FDTD_update_E(self._libfdtd, n+n0+Tn, (n+n0+Tn+0.5)*dt)

            self._gc.update_local_vector(self._Ex)
            self._gc.update_local_vector(self._Ey)
            self._gc.update_local_vector(self._Ez)

        t0 = n0*dt
        t1 = (n0+Tn)*dt
        t2 = (n0+2*Tn)*dt
        libFDTD.FDTD_calc_complex_fields_3T(self._libfdtd, t0, t1, t2)

    def solve_forward(self):
        """Run a forward simulation.

        A forward simulation is just a solution to Maxwell's equations.
        """
        if(self.verbose > 0 and NOT_PARALLEL):
            info_message('Solving forward simulation.')

        # Reset fourier-domain fields
        self._Ex_fwd_t0.set(0); self._Ey_fwd_t0.set(0); self._Ez_fwd_t0.set(0)
        self._Hx_fwd_t0.set(0); self._Hy_fwd_t0.set(0); self._Hz_fwd_t0.set(0)

        self._Ex_fwd_t1.set(0); self._Ey_fwd_t1.set(0); self._Ez_fwd_t1.set(0)
        self._Hx_fwd_t1.set(0); self._Hy_fwd_t1.set(0); self._Hz_fwd_t1.set(0)

        # make sure we are recording forward fields
        libFDTD.FDTD_set_t0_arrays(self._libfdtd,
                                   self._Ex_fwd_t0.getArray(),
                                   self._Ey_fwd_t0.getArray(),
                                   self._Ez_fwd_t0.getArray(),
                                   self._Hx_fwd_t0.getArray(),
                                   self._Hy_fwd_t0.getArray(),
                                   self._Hz_fwd_t0.getArray())

        libFDTD.FDTD_set_t1_arrays(self._libfdtd,
                                   self._Ex_fwd_t1.getArray(),
                                   self._Ey_fwd_t1.getArray(),
                                   self._Ez_fwd_t1.getArray(),
                                   self._Hx_fwd_t1.getArray(),
                                   self._Hy_fwd_t1.getArray(),
                                   self._Hz_fwd_t1.getArray())

        # set the forward simulation sources
        libFDTD.FDTD_clear_sources(self._libfdtd)
        for src in self._sources:
            libFDTD.FDTD_add_source(self._libfdtd,
                                    src.Jx, src.Jy, src.Jz,
                                    src.Mx, src.My, src.Mz,
                                    src.i0, src.j0, src.k0,
                                    src.I, src.J, src.K,
                                    False)

        self.__solve()

        self.update_saved_fields()

        # calculate source power
        Psrc = self.get_source_power()
        if(NOT_PARALLEL):
            self._source_power = Psrc
        else:
            self._source_power = MathDummy()

        # update count (for diagnostics)
        self.forward_count += 1

    def solve_adjoint(self):
        """Run an adjoint simulation.

        In FDTD, this is a solution to Maxwell's equations but using a
        different set of sources than the forward simulation.
        """
        if(self.verbose > 0 and NOT_PARALLEL):
            info_message('Solving adjoint simulation...')

        # Reset fourier-domain fields
        self._Ex_adj_t0.set(0); self._Ey_adj_t0.set(0); self._Ez_adj_t0.set(0)
        self._Hx_adj_t0.set(0); self._Hy_adj_t0.set(0); self._Hz_adj_t0.set(0)

        self._Ex_adj_t1.set(0); self._Ey_adj_t1.set(0); self._Ez_adj_t1.set(0)
        self._Hx_adj_t1.set(0); self._Hy_adj_t1.set(0); self._Hz_adj_t1.set(0)

        # make sure we are recording adjoint fields
        libFDTD.FDTD_set_t0_arrays(self._libfdtd,
                                    self._Ex_adj_t0.getArray(),
                                    self._Ey_adj_t0.getArray(),
                                    self._Ez_adj_t0.getArray(),
                                    self._Hx_adj_t0.getArray(),
                                    self._Hy_adj_t0.getArray(),
                                    self._Hz_adj_t0.getArray())

        libFDTD.FDTD_set_t1_arrays(self._libfdtd,
                                    self._Ex_adj_t1.getArray(),
                                    self._Ey_adj_t1.getArray(),
                                    self._Ez_adj_t1.getArray(),
                                    self._Hx_adj_t1.getArray(),
                                    self._Hy_adj_t1.getArray(),
                                    self._Hz_adj_t1.getArray())

        # set the adjoint simulation sources
        # The phase of these sources has already been calculated,
        # so we tell the C++ library to skip the phase calculation
        libFDTD.FDTD_clear_sources(self._libfdtd)
        for src in self._adj_sources:
            libFDTD.FDTD_add_source(self._libfdtd,
                                    src.Jx, src.Jy, src.Jz,
                                    src.Mx, src.My, src.Mz,
                                    src.i0, src.j0, src.k0,
                                    src.I, src.J, src.K,
                                    False)

        self.__solve()

        # update count (for diagnostics)
        self.adjoint_count += 1

    def update_saved_fields(self):
        """Update the fields contained in the regions specified by
        self.field_domains.

        This function is called internally by solve_forward and should not need
        to be called otherwise.
        """
        # clean up old fields
        self._saved_fields = []

        for domain in self._field_domains:
            Ex = self.get_field_interp(FieldComponent.Ex, domain)
            Ey = self.get_field_interp(FieldComponent.Ey, domain)
            Ez = self.get_field_interp(FieldComponent.Ez, domain)
            Hx = self.get_field_interp(FieldComponent.Hx, domain)
            Hy = self.get_field_interp(FieldComponent.Hy, domain)
            Hz = self.get_field_interp(FieldComponent.Hz, domain)

            self._saved_fields.append((Ex, Ey, Ez, Hx, Hy, Hz))

    def __get_field(self, component, domain=None, adjoint=False):
        ##Get the uninterpolated field component in the specified domain.
        # The process is nearly identical for forward/adjoint
        if(domain == None):
            domain = DomainCoordinates(0, self._X, 0, self._Y, 0,
                                       self._Z, self._dx, self._dy, self._dz)

        if(component == FieldComponent.Ex):
            if(adjoint): field = self._Ex_adj_t0
            else: field = self._Ex_fwd_t0
        elif(component == FieldComponent.Ey):
            if(adjoint): field = self._Ey_adj_t0
            else: field = self._Ey_fwd_t0
        elif(component == FieldComponent.Ez):
            if(adjoint): field = self._Ez_adj_t0
            else: field = self._Ez_fwd_t0
        elif(component == FieldComponent.Hx):
            if(adjoint): field = self._Hx_adj_t0
            else: field = self._Hx_fwd_t0
        elif(component == FieldComponent.Hy):
            if(adjoint): field = self._Hy_adj_t0
            else: field = self._Hy_fwd_t0
        elif(component == FieldComponent.Hz):
            if(adjoint): field = self._Hz_adj_t0
            else: field = self._Hz_fwd_t0

        # get a "natural" representation of the appropriate field vector,
        # gather it on the rank 0 node and return the appropriate piece
        self._da.globalToNatural(field, self._vn)
        scatter, fout = PETSc.Scatter.toZero(self._vn)
        scatter.scatter(self._vn, fout, False, PETSc.Scatter.Mode.FORWARD)

        if(NOT_PARALLEL):
            fout = np.array(fout, dtype=np.complex128)
            fout = np.reshape(fout, [self._Nz, self._Ny, self._Nx])
            return fout[domain.i, domain.j, domain.k]
        else:
            return MathDummy()

    def get_field(self, component, domain=None):
        """Get the (raw, uninterpolated) field.

        In most cases, you should use :def:`get_field_interp` instead.

        Parameters
        ----------
        component : str
            The field component to retrieve
        domain : misc.DomainCoordinates (optional)
            The domain in which the field is retrieved. If None, retrieve the
            field in the whole 3D domain (which you probably should avoid doing
            for larger problems). (default = None)

        Returns
        -------
        numpy.ndarray
            An array containing the field.
        """
        return self.__get_field(component, domain, adjoint=False)

    def get_adjoint_field(self, component, domain=None):
        """Get the adjoint field.

        Parameters
        ----------
        component : str
            The adjoint field component to retrieve.
        domain : misc.DomainCoordinates (optional)
            The domain in which the field is retrieved. If None, retrieve the
            field in the whole 3D domain (which you probably should avoid doing
            for larger problems). (default = None)

        Returns
        -------
        numpy.ndarray
            An array containing the field.
        """
        return self.__get_field(component, domain, adjoint=True)

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
        if(self._w_pml[0] > 0): xmin = self._w_pml[0]+dx
        else: xmin = 0.0

        if(self._w_pml[1] > 0): xmax = self._X - self._w_pml[1]-dx
        else: xmax = self._X - self._dx

        if(self._w_pml[2] > 0): ymin = self._w_pml[2]+dy
        else: ymin = 0.0

        if(self._w_pml[3] > 0): ymax = self._Y - self._w_pml[3]-dy
        else: ymax = self._Y - self._dy

        if(self._w_pml[4] > 0): zmin = self._w_pml[4]+dz
        else: zmin = 0.0

        if(self._w_pml[5] > 0): zmax = self._Z - self._w_pml[5]-dz
        else: zmax = self._Z - self._dz

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
            Py = 0.5*dx*dz*np.sum(np.real(Ex*np.conj(Hz)-Ez*np.conj(Hx)))
            #print Py
            Psrc += Py
        del Ex; del Ez; del Hx; del Hz

        # calculate power transmitter through ymax boundary
        Ex = self.get_field_interp('Ex', y2)
        Ez = self.get_field_interp('Ez', y2)
        Hx = self.get_field_interp('Hx', y2)
        Hz = self.get_field_interp('Hz', y2)

        if(NOT_PARALLEL):
            Py = -0.5*dx*dz*np.sum(np.real(Ex*np.conj(Hz)-Ez*np.conj(Hx)))
            #print Py
            Psrc += Py
        del Ex; del Ez; del Hx; del Hz

        # calculate power transmitter through zmin boundary
        Ex = self.get_field_interp('Ex', z1)
        Ey = self.get_field_interp('Ey', z1)
        Hx = self.get_field_interp('Hx', z1)
        Hy = self.get_field_interp('Hy', z1)

        if(NOT_PARALLEL and self._bc[2] != 'E' and self._bc[2] != 'H'):
            Pz = -0.5*dx*dy*np.sum(np.real(Ex*np.conj(Hy)-Ey*np.conj(Hx)))
            #print Pz
            Psrc += Pz
        del Ex; del Ey; del Hx; del Hy

        # calculate power transmitter through zmin boundary
        Ex = self.get_field_interp('Ex', z2)
        Ey = self.get_field_interp('Ey', z2)
        Hx = self.get_field_interp('Hx', z2)
        Hy = self.get_field_interp('Hy', z2)

        if(NOT_PARALLEL):
            Pz = 0.5*dx*dy*np.sum(np.real(Ex*np.conj(Hy)-Ey*np.conj(Hx)))
            #print Pz
            Psrc += Pz
        del Ex; del Ey; del Hx; del Hy

        return Psrc

    def get_A_diag(self):
        """Get a representation of the diagonal of the discretized Maxwell's
        equations assuming they are assembled in a matrix in the frequency
        domain.

        For the purposes of this implementation, this is just a copy of the
        permittivity and permeability distribution. In reality, there should be
        a factor of 1j and -1j for the permittivities and permeabilities,
        respectively, however we handle these prefactors elsewhere.

        Returns
        -------
        tuple of numpy.ndarrays
            A copy of the set of permittivity and permeability distributions used
            internally.
        """
        eps_x = np.copy(self._eps_x.getArray())
        eps_y = np.copy(self._eps_y.getArray())
        eps_z = np.copy(self._eps_z.getArray())
        mu_x = np.copy(self._mu_x.getArray())
        mu_y = np.copy(self._mu_y.getArray())
        mu_z = np.copy(self._mu_z.getArray())

        return (eps_x, eps_y, eps_z, mu_x, mu_y, mu_z)

    def calc_ydAx(self, Adiag0):
        """Calculate the product y^T*dA*x.

        Parameters
        ----------
        Adiag0 : tuple of 6 numpy.ndarray
            The 'initial' diag[A] obtained from self.get_A_diag()

        Returns
        -------
        complex
            The product y^T*dA*x
        """
        eps_x0, eps_y0, eps_z0, mu_x0, mu_y0, mu_z0 = Adiag0

        ydAx = np.zeros(eps_x0.shape, dtype=np.complex128)
        ydAx = ydAx + self._Ex_adj_t0[...] *  1j * (self._eps_x[...]-eps_x0) * self._Ex_fwd_t0[...]
        ydAx = ydAx + self._Ey_adj_t0[...] *  1j * (self._eps_y[...]-eps_y0) * self._Ey_fwd_t0[...]
        ydAx = ydAx + self._Ez_adj_t0[...] *  1j * (self._eps_z[...]-eps_z0) * self._Ez_fwd_t0[...]
        ydAx = ydAx + self._Hx_adj_t0[...] * -1j * (self._mu_x[...]-mu_x0)   * self._Hx_fwd_t0[...]
        ydAx = ydAx + self._Hy_adj_t0[...] * -1j * (self._mu_y[...]-mu_y0)   * self._Hy_fwd_t0[...]
        ydAx = ydAx + self._Hz_adj_t0[...] * -1j * (self._mu_z[...]-mu_z0)   * self._Hz_fwd_t0[...]

        return np.sum(ydAx)

    @run_on_master
    def test_src_func(self):
        import matplotlib.pyplot as plt

        time = np.arange(0,3000,1)*self._dt

        ramp = np.zeros(3000)
        for i in range(len(time)):
            ramp[i] = libFDTD.FDTD_src_func_t(self._libfdtd, i, time[i], 0)

        plt.plot(time,ramp)
        plt.show()


class GhostComm(object):

    def __init__(self, k0, j0, i0, K, J, I, Nx, Ny, Nz):
        """Create a GhostComm object.

        The GhostComm object mediates the communication of ghost (edge) values
        in shared vectors which store values on a divided rectangular grid.

        Parameters
        ----------
        k0 : int
            The starting x grid index of a block of the grid.
        j0 : int
            The starting y grid index of a block of the grid.
        i0 : int
            The starting z grid index of a block of the grid.
        K : int
            The number of grid points along x of the block.
        J : int
            The number of grid points along y of the block.
        I : int
            The number of grid points along z of the block.

        """
        # Inputs describe the block in the grid which has ghost values
        self._i0 = i0
        self._j0 = j0
        self._k0 = k0
        self._I = I
        self._J = J
        self._K = K
        self._Nx = Nx
        self._Ny = Ny
        self._Nz = Nz

        # define the number of boundary points and ghost points
        # for simplicity, the edge values of  the boundary elements are
        # duplicated. This should minimally impact performance.
        nbound = K*J*2 + K*I*2 + I*J*2
        nghost = nbound
        self._nbound = nbound
        self._nghsot = nghost

        # Create the vector that will store edge and ghost values
        gvec = PETSc.Vec().createMPI((nbound, None))
        self._gvec = gvec

        # determine the global indices for ghost values. To do this, each
        # process needs to know (1) the starting index of each block in the
        # global ghost vector, (2) the position of each block in the physical
        # grid, and (3) the size of each block.
        start, end = gvec.getOwnershipRange()
        pos_data = (start, i0, j0, k0, I, J, K)

        # collect and then share all of the position data
        pos_data = COMM.gather(pos_data, root=0)
        pos_data = COMM.bcast(pos_data, root=0)

        ighosts = []

        # Find the global indices of the different boundaries
        # xmin boundary
        k = k0-1
        if(k < 0): k = Nx-1
        for pd in pos_data:
            gindex, i0n, j0n, k0n, In, Jn, Kn = pd
            if(k == k0n + Kn-1 and i0 == i0n and j0 == j0n):
                ighosts += list(range(gindex+In*Jn, gindex+2*In*Jn))

        # xmax boundary
        k = k0+K
        if(k > Nx-1): k = 0
        for pd in pos_data:
            gindex, i0n, j0n, k0n, In, Jn, Kn = pd
            if(k ==  k0n and i0 == i0n and j0 == j0n):
                ighosts += list(range(gindex, gindex + In*Jn))

        # ymin boundary
        j = j0-1
        if(j < 0): j = Ny-1
        for pd in pos_data:
            gindex, i0n, j0n, k0n, In, Jn, Kn = pd
            if(j == j0n + Jn - 1 and k0 == k0n and i0 == i0n):
                ighosts += list(range(gindex+2*In*Jn+In*Kn, gindex+2*In*Jn+2*In*Kn))

        # ymax boundary
        j = j0+J
        if(j > Ny-1): j = 0
        for pd in pos_data:
            gindex, i0n, j0n, k0n, In, Jn, Kn = pd
            if(j == j0n and k0 == k0n and i0 == i0n):
                ighosts += list(range(gindex+2*In*Jn, gindex+2*In*Jn+In*Kn))

        # zmin boundary
        i = i0-1
        if(i < 0): i = Nz-1
        for pd in pos_data:
            gindex, i0n, j0n, k0n, In, Jn, Kn = pd
            if(i == i0n + In - 1 and j0 == j0n and k0 == k0n):
                ighosts += list(range(gindex+2*In*Jn+2*In*Kn+Jn*Kn,
                                 gindex+2*In*Jn+2*In*Kn+2*Jn*Kn))

        # zmax boundary
        i = i0+I
        if(i > Nz-1): i = 0
        for pd in pos_data:
            gindex, i0n, j0n, k0n, In, Jn, Kn = pd
            if(i == i0n and j0 == j0n and k0 == k0n):
                ighosts += list(range(gindex+2*In*Jn+2*In*Kn,
                                 gindex+2*In*Jn+2*In*Kn+Jn*Kn))

        # Set the ghost indices = finish constructing ghost vector
        self._ighosts = np.array(ighosts, dtype=np.int32)
        ghosts = PETSc.IS().createGeneral(self._ighosts)
        self._gvec.setMPIGhost(ghosts)

    def update_local_vector(self, vl):
        """Update the ghost values of a local vector.

        Parameters
        ----------
        vl : np.ndarray
            The local vector
        """
        I = self._I; J = self._J; K = self._K
        i0 = self._i0; j0 = self._j0; k0 = self._k0
        nbound = self._nbound

        garr = self._gvec.getArray()

        libFDTD.FDTD_copy_to_ghost_comm(vl, garr, I, J, K)

        # do the ghost update
        self._gvec.ghostUpdate()

        # copy the distributed ghost values to the correct positions
        nstart = nbound
        with self._gvec.localForm() as gloc:
            gloc_arr = gloc.getArray()

            libFDTD.FDTD_copy_from_ghost_comm(vl, garr, I, J, K)
