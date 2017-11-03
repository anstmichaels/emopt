"""
Miscellanious functions useful for simulation and optimization.
"""

import numpy as np
from scipy import interpolate
import os

from petsc4py import PETSc
#import decorator # so that sphinx will document decorated functions :S
import warnings, inspect

__author__ = "Andrew Michaels"
__license__ = "Apache License, Version 2.0"
__version__ = "0.2"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

# functions and variables useful for MPI stuff
RANK = PETSc.COMM_WORLD.getRank()
NOT_PARALLEL = (RANK == 0)

def run_on_master(func):
    """Prevent a decorated function from running on any node but the master
    node
    """

    def wrapper(*args, **kwargs):
        if(NOT_PARALLEL):
            return func(*args, **kwargs)
        else:
            return

    return wrapper

def n_silicon(wavelength):
    """Load silicon refractive index vs wavlength and interpolate at desired wavelength.
    A piecewise cubic fit is used for the interpolation.

    Parameters
    ----------
    wavelength : float
        The wavlenegth in [um] between 1.2 um and 14 um

    Returns
    -------
        Refractive index of silicon at desired wavelength.
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = ''.join([dir_path, '/data/silicon.csv'])

    data_si = np.loadtxt(data_path, delimiter=',', skiprows=1)
    wlen_si = data_si[:,0]
    n_si = data_si[:,1]
    n_interp = interpolate.interp1d(wlen_si, n_si, kind='cubic')

    return n_interp(wavelength)

@run_on_master
def plot_iterations(params, fom_list, sim, am, field_comp, \
                    fname='./current_result.pdf', show_plot=False):
    """Update a plot during the course of optimization.
    """
    print('Finished iteration %d. Updating plot' % (len(fom_list)))
    import matplotlib.pyplot as plt

    current_fom = am.calc_fom(sim, params)
    print('Current FOM = %0.4f' % (-1*current_fom))
    fom_list.append(-1*current_fom)

    w_pml_l = sim._w_pml_left
    w_pml_r = sim._w_pml_right
    w_pml_t = sim._w_pml_top
    w_pml_b = sim._w_pml_bottom
    M = sim.M
    N = sim.N

    eps = sim.eps.get_values(w_pml_b, M-w_pml_t, w_pml_l, N-w_pml_r)
    Ez = sim.get_field_interp(field_comp)
    Ez = Ez[w_pml_b:M-w_pml_t, w_pml_l:N-w_pml_r]
    xmin = 0
    xmax = Ez.shape[1]*sim.dx
    ymin = 0
    ymax = Ez.shape[0]*sim.dy
    extent = [xmin, xmax, ymin, ymax]

    f = plt.figure()
    ax1 = f.add_subplot(311)
    ax2 = f.add_subplot(312)
    ax3 = f.add_subplot(313)

    ax1.imshow(np.flipud(eps.real), vmin=np.min(eps.real),\
               vmax=np.max(eps.real), extent=extent, cmap='Blues')

    ax2.imshow(np.flipud(Ez.real), vmin=-np.max(Ez.real), vmax=np.max(Ez.real),\
               extent=extent, cmap='seismic')

    ax3.plot(fom_list, 'b.-', markersize=10, alpha=0.75)
    plt.savefig(fname, format='pdf', dpi=300, bbox_inches='tight')
    if(show_plot):
        plt.show()
    plt.close()

def save_results(fname, data, additional=None):
    """Save an hdf5 file containing common simulation and optimization results.

    The input to this function is a fileneam and a dictionary which
    contains the following possible items:
    W - Width of simulation
    H - Height of simulation
    dx - x grid spacing
    dy - y grid spacing
    M - number of rows in field matrices
    N - number of columns in field matrices
    w_pml_x - PML width in x
    w_pml_y - PML height in y
    Ex - x component of electric field
    Ey - y component of electric field
    Ez - z component of electric field
    Hx - x component of magnetic field
    Hy - y component of magnetic field
    Hz - z component of magnetic field
    eps - The permittivity of the system
    mu - The permeability of the system
    params - The design parameters of the system
    foms - List of figure of merits achieved during optimization

    A second optional dict can be passed as well which contains additional data to store
    that is not recognized as a typical simulation or optimization result

    Notes
    -----
    This function depends on h5py

    Parameters
    ----------
    fname : string
        The name and path of file which will be saved (Note: a file extention is added automatically)
    data : dict
        The simulation and optimization results to be saved
    additional : dict
        Any addtional data to save
    """
    import h5py

    fname_full = ''.join([fname, '.h5'])
    with h5py.File(fname_full, "w") as hf:
        group_sim = hf.create_group("simulation")
        group_opt = hf.create_group("optimization")
        group_misc = hf.create_group("misc")

        # simulation attributes
        if 'W' in data:
            group_sim.attrs['W'] = data['W']
        if 'H' in data:
            group_sim.attrs['H'] = data['H']
        if 'dx' in data:
            group_sim.attrs['dx'] = data['dx']
        if 'dy' in data:
            group_sim.attrs['dy'] = data['dy']
        if 'M' in data:
            group_sim.attrs['M'] = data['M']
        if 'N' in data:
            group_sim.attrs['N'] = data['M']
        if 'w_pml_x' in data:
            group_sim.attrs['w_pml_x'] = data['w_pml_x']
        if 'w_pml_y' in data:
            group_sim.attrs['w_pml_y'] = data['w_pml_y']

        # Simulation results
        if 'Ex' in data:
            group_sim.create_dataset('Ex', data=data['Ex'])
        if 'Ey' in data:
            group_sim.create_dataset('Ey', data=data['Ey'])
        if 'Ez' in data:
            group_sim.create_dataset('Ez', data=data['Ez'])
        if 'Hx' in data:
            group_sim.create_dataset('Hx', data=data['Hx'])
        if 'Hy' in data:
            group_sim.create_dataset('Hy', data=data['Hy'])
        if 'Hz' in data:
            group_sim.create_dataset('Hz', data=data['Hz'])
        if 'eps' in data:
            group_sim.create_dataset('eps', data=data['eps'])
        if 'mu' in data:
            group_sim.create_dataset('mu', data=data['mu'])

        # Optimization results
        if 'params' in data:
            group_opt.create_dataset('params', data=data['params'])
        if 'foms' in data:
            group_opt.create_dataset('foms', data=data['foms'])

        # any additional data
        if(additional is not None):
            for key in additional:
                group_misc.create_dataset(key, data=additional[key])


def load_results(fname):
    """
    Load data that has been saved with the :func:`save_results` function.

    Parameters
    ----------
    fname : string
        The file name and path of file from which data is loaded.

    Returns
    -------
    dict
        A dictionary containing the loaded data.
    """
    import h5py

    data = {}

    fname_full = ''.join([fname, '.h5'])
    with h5py.File(fname_full, "r") as fh5:

        for key in fh5['simulation'].keys():
            data[key] = fh5['simulation'][key][...]

        for key in fh5['simulation'].attrs.keys():
            data[key] = fh5['simulation'].attrs[key][...]

        for key in fh5['optimization'].keys():
            data[key] = fh5['optimization'][key][...]

        for key in fh5['misc'].keys():
            data[key] = fh5['misc'][key][...]

    return data

def info_message(message):
    """Print a formatted, easily-distinguishable message.

    Parameters
    ----------
    message : str
        The message to print.
    """
    print(u'\u001b[46;1m[INFO]\u001b[0m %s' % (message))

def _warning_message(message, category=UserWarning, filename='', lineno=-1):
    # Override python's warning message by adding a colored [WARNING] flag in
    # front to make it more noticeable.
    if(filename is not '' and lineno != -1):
        print(u'\u001b[43;1m[WARNING]\u001b[0m in %s at line %d: %s' % \
              (filename, lineno, message))
    else:
        print(u'\u001b[43;1m[WARNING]\u001b[0m %s' % (message))
warnings.showwarning = _warning_message

def warning_message(message, module):
    # Produce a warning message to warn the user that a problem has occurred.
    # This is primarily intended for internal use within emopt
    warnings.warn('in %s: %s' % (module, message), category=RuntimeWarning)

def error_message(message):
    """Print a formatted, easily-distinguishable error message.

    In general, exceptions are probably preferable, but if you ever want to
    throw a non-disrupting error whose format is consistent with info_message
    and warning_message, use this!

    Parameters
    ----------
    message : str
        The message to print.
    """
    print(u'\u001b[41;1m[ERROR]\u001b[0m %s' % (message))




class DomainCoordinates(object):
    """Define a domain coordinate.

    A DomainCoordinate is a class which manages accessing data on a rectangular
    grid. It stores both the indexed positions and real-space coordinates of a
    desired line, plane, or volume.

    Attributes
    ----------
    x : numpy.ndarray
        The real-space x coordinates of the domain
    y : numpy.ndarray
        The real-space y coordinates of the domain
    z : numpy.ndarray
        The real-space z coordinates of the domain
    i : numpy.ndarray
        The array index corresponding to the z direction
    j : numpy.ndarray
        The array index corresponding to the y direction
    k : numpy.ndarray
        The array index corresponding to the x direction
    """

    def __init__(self):
        self._x = np.array([0])
        self._y = np.array([0])
        self._z = np.array([0])

        self._i = np.array([0])
        self._j = np.array([0])
        self._k = np.array([0])

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @x.setter
    def x(self, value):
        warning_message('x cannot be reassigned in this way.', 'emopt.misc')

    @y.setter
    def y(self, value):
        warning_message('y cannot be reassigned in this way.', 'emopt.misc')

    @z.setter
    def z(self, value):
        warning_message('z cannot be reassigned in this way.', 'emopt.misc')

    @property
    def i(self):
        return self._i

    @property
    def j(self):
        return self._j

    @property
    def k(self):
        return self._k

    @i.setter
    def i(self, value):
        warning_message('i cannot be reassigned in this way.', 'emopt.misc')

    @j.setter
    def j(self, value):
        warning_message('j cannot be reassigned in this way.', 'emopt.misc')

    @k.setter
    def k(self, value):
        warning_message('k cannot be reassigned in this way.', 'emopt.misc')

    def get_bounding_box(self):
        return [np.min(self._x), np.max(self._x), np.min(self._y),
                np.max(self._y), np.min(self._z), np.max(self._z),]

class LineCoordinates(DomainCoordinates):
    """Define a line in a rectangular grid.

    Parameters
    ----------
    orientation : str
        The cartesian direction along to which the line is parallel.
    *args
        Depending on the specified orientation, a LineCoordinate is initialized
        with the following set of floating point values:

            orientation='x' : y, z, xmin, xmax, dx, dy, dz

            orientation='y' : x, z, ymin, ymax, dx, dy, dz

            orientation='z' : x, y, zmin, zmax, dx, dy, dz

        Furthermore, for 2D problems, a simplified list of parameters may be
        used:

            orientation='x' : y, xmin, xmax, dx, dy

            orientation='y' : x, ymin, ymax, dx, dy

        Note: dx, dy, and dz MUST be non-zero.

    Attributes
    ----------
    x : numpy.ndarray
        The real-space x coordinates of the domain
    y : numpy.ndarray
        The real-space y coordinates of the domain
    z : numpy.ndarray
        The real-space z coordinates of the domain
    i : numpy.ndarray
        The array index corresponding to the z direction
    j : numpy.ndarray
        The array index corresponding to the y direction
    k : numpy.ndarray
        The array index corresponding to the x direction
    """

    def __init__(self, orientation, *args):
        super(LineCoordinates, self).__init__()

        if(orientation == 'x'):
            if(len(args) == 5):
                y, xmin, xmax, dx, dy = args
                z = 0; dz = 1.0
            elif(len(args) == 7):
                y, z, xmin, xmax, dx, dy, dz = args
            else:
                raise ValueError('Incorrect number of arguments. Accepted'\
                                 'arguments:' \
                                 'y, xmin, xmax, dx, dy or ' \
                                 'y, z, xmin, xmax, dx, dy, dz')

            self._j = np.array([int(y/dy)], dtype=np.int)
            self._y = dy * self._j

            self._i = np.array([int(z/dz)], dtype=np.int)
            self._z = dz * self._i

            k1 = int(xmin/dx)
            k2 = int(xmax/dx)+1
            self._k = np.arange(k1, k2, 1, dtype=np.int)
            self._x = dx * self._k.astype(np.double)

        elif(orientation == 'y'):
            if(len(args) == 5):
                x, ymin, ymax, dx, dy = args
                z = 0; dz = 1.0
            elif(len(args) == 7):
                x, z, ymin, ymax, dx, dy, dz = args
            else:
                raise ValueError('Incorrect number of arguments. Accepted'\
                                 'arguments:' \
                                 'x, ymin, ymax, dx, dy or ' \
                                 'x, z, ymin, ymax, dx, dy, dz')

            self._k = np.array([int(x/dx)], dtype=np.int)
            self._x = dx * self._k

            self._i = np.array([int(z/dz)], dtype=np.int)
            self._z = dz * self._i

            j1 = int(ymin/dy)
            j2 = int(ymax/dy)+1
            self._j = np.arange(j1, j2, 1, dtype=np.int)
            self._y = dy * self._j.astype(np.double)

        elif(orientation == 'z'):
            x, y, zmin, zmax, dx, dy, dz = args

            self._k = np.array([int(x/dx)], dtype=np.int)
            self._x = dx * self._k

            self._j = np.array([int(y/dx)], dtype=np.int)
            self._y = dy * self._j

            i1 = int(zmin/dy)
            i2 = int(zmax/dy)+1
            self._i = np.arange(i1, i2, 1, dtype=np.int)
            self._z = dy * self._i.astype(np.double)
        else:
            raise ValueError('Unknown line orientation "%s"' % (orientation))

class PlaneCoordinates(DomainCoordinates):
    """Define a plane in a rectangular grid.

    Parameters
    ----------
    normal_dir: str
        The cartesian direction along to which the line is parallel.
    *args
        Depending on the specified orientation, a PlaneCoordinate is initialized
        with the following set of floating point values:

            normal_dir='x' : x, ymin, ymax, zmin, zmax, dx, dy, dz

            normal_dir='y' : y, xmin, xmax, zmin, zmax, dx, dy, dz

            normal_dir='z' : z, xmin, xmax, ymin, ymax, dx, dy, dz

        Additionally, for 2D grids, a simplified parameter list may be used

           xmin, xmax, ymin, ymax, dx, dy

        Note: dx, dy, and dz MUST be non-zero.

    Attributes
    ----------
    x : numpy.ndarray
        The real-space x coordinates of the domain
    y : numpy.ndarray
        The real-space y coordinates of the domain
    z : numpy.ndarray
        The real-space z coordinates of the domain
    i : numpy.ndarray
        The array index corresponding to the z direction
    j : numpy.ndarray
        The array index corresponding to the y direction
    k : numpy.ndarray
        The array index corresponding to the x direction
    """

    def __init__(self, normal_dir, *args):

        if(normal_dir == 'x'):
            if(len(args) != 8):
                raise ValueError('Incorrect number of parameters. The following' \
                                 'parameters are required: x, ymin, ymax, ' \
                                 'zmin, zmax, dx, dy, dz.')
            x, ymin, ymax, zmin, zmax, dx, dy, dz = args

            self._k = np.array([int(x/dx)], dtype=np.int)
            # "snap" to nearest grid plane
            self._x = dx * self._k.astype(np.double)

            j1 = int(ymin/dy)
            j2 = int(ymax/dy)+1
            self._j = np.arange(j1, j2, 1, dtype=np.int)
            self._y = dy * self._j.astype(np.double)

            i1 = int(zmin/dz)
            i2 = int(zmax/dz)+1
            self._i = np.arange(i1, i2, 1, dtype=np.int)
            self._z = dz * self._i.astype(np.double)

        elif(normal_dir == 'y'):
            if(len(args) != 8):
                raise ValueError('Incorrect number of parameters. The following' \
                                 'parameters are required: y, xmin, xmax, ' \
                                 'zmin, zmax, dx, dy, dz.')
            y, xmin, xmax, zmin, zmax, dx, dy, dz = args

            self._j = np.array([int(y/dy)], dtype=np.int)
            # "snap" to nearest grid plane
            self._y = dy * self._j.astype(np.double)

            k1 = int(xmin/dx)
            k2 = int(xmax/dx)+1
            self._k = np.arange(k1, k2, 1, dtype=np.int)
            self._x = dx * self._k.astype(np.double)

            i1 = int(zmin/dz)
            i2 = int(zmax/dz)+1
            self._i = np.arange(i1, i2, 1, dtype=np.int)
            self._z = dz * self._i.astype(np.double)

        elif(normal_dir == 'z'):
            if(len(args) == 6):
                xmin, xmax, ymin, ymax, dx, dy = args

                self._i = np.array([0], dtype=np.int)
                self._z = np.array([0.0])

                k1 = int(xmin/dx)
                k2 = int(xmax/dx)+1
                self._k = np.arange(k1, k2, 1, dtype=np.int)
                self._k.resize((1,k2-k1))
                self._x = dx * self._k.astype(np.double)

                j1 = int(ymin/dy)
                j2 = int(ymax/dy)+1
                self._j = np.arange(j1, j2, 1, dtype=np.int)
                self._j.resize((j2-j1,1))
                self._y = dy * self._j.astype(np.double)

            elif(len(args) == 8):
                z, xmin, xmax, ymin, ymax, dx, dy, dz = args

                self._i = np.array([int(z/dz)], dtype=np.int)
                # "snap" to nearest grid plane
                self._z = dz * self._i.astype(np.double)

                k1 = int(xmin/dx)
                k2 = int(xmax/dx)+1
                self._k = np.arange(k1, k2, 1, dtype=np.int)
                self._x = dx * self._k.astype(np.double)

                j1 = int(ymin/dy)
                j2 = int(ymax/dy)+1
                self._j = np.arange(j1, j2, 1, dtype=np.int)
                self._y = dy * self._j.astype(np.double)
            else:
                raise ValueError('Incorrect number of parameters. Either 6 (for' \
                                 '2D space) or 8 (for 3D space) parameters are' \
                                 'required. In each case, the required ' \
                                 'parameters are: \n' \
                                 'xmin, xmax, ymin, ymax, dx, dy \n'
                                 'or\n'
                                 'z, xmin, xmax, ymin, ymax, dx, dy, dz')
        else:
            raise ValueError("Normal direction '%s' not recognized. Options "\
                             "are: 'x', 'y', 'z'" % (normal_dir))

####################################################################################
# Define a MathDummy
####################################################################################

class MathDummy(np.ndarray):
    """Define a MathDummy.

    A MathDummy is an empty numpy.ndarray which devours all mathematical
    operations done by it or on it and just spits itself back out. This is
    used by emopt in order simplify its interface in the presence of MPI. For
    example, in many instances, you will need to calculate a quantity which
    need only be known on the master node, however the function performing the
    computation will be run on all nodes. Rather than having to worry about
    putting in if(NOT_PARALLEL) statements everywhere, we can just sneakily
    replace quantities involved in the calculation with MathDummies on all
    nodes but the master node.  You can then do any desired calculations
    without worying about what's going on in the other nodes.
    """
    def __new__(cls):
        obj = np.asarray([]).view(cls)
        return obj

    def __add__(self, other): return self
    def __sub__(self, other): return self
    def __mul__(self, other): return self
    def __matmul__(self, other): return self
    def __truediv__(self, other): return self
    def __floordiv__(self, other): return self
    def __mod__(self, other): return self
    def __divmod__(self, other): return self
    def __pow__(self, other, modulo=2): return self
    def __lshift__(self, other): return self
    def __rshift__(self, other): return self
    def __and__(self, other): return self
    def __xor__(self, other): return self
    def __or__(self, other): return self
    def __radd__(self, other): return self
    def __rsub__(self, other): return self
    def __rmul__(self, other): return self
    def __rmatmul__(self, other): return self
    def __rtruediv__(self, other): return self
    def __rfloordiv__(self, other): return self
    def __rmod__(self, other): return self
    def __rdivmod__(self, other): return self
    def __rpow__(self, other): return self
    def __rlshift__(self, other): return self
    def __rrshift__(self, other): return self
    def __rand__(self, other): return self
    def __rxor__(self, other): return self
    def __ror__(self, other): return self
    def __iadd__(self, other): return self
    def __isub__(self, other): return self
    def __imul__(self, other): return self
    def __imatmul__(self, other): return self
    def __itruediv__(self, other): return self
    def __ifloordiv__(self, other): return self
    def __imod__(self, other): return self
    def __ipow__(self, other, modulo=2): return self
    def __ilshift__(self, other): return self
    def __irshift__(self, other): return self
    def __iand__(self, other): return self
    def __ixor__(self, other): return self
    def __ior__(self, other): return self
    def __neg__(self): return self
    def __pos__(self): return self
    def __abs__(self): return self
    def __invert__(self): return self
    def __complex__(self): return self
    def __int__(self): return self
    def __float__(self): return self
    def __round__(self, n): return self
    def __index__(self): return 0
    def __getitem__(self, key): return self
    def __setitem__(self, key, value): return self
