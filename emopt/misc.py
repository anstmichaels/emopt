"""Miscellanious functions useful for simulation and optimization.
"""

import numpy as np
from scipy import interpolate
import os

from petsc4py import PETSc
#import decorator # so that sphinx will document decorated functions :S
import warnings, inspect

__author__ = "Andrew Michaels"
__license__ = "GPL License, Version 3.0"
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



def info_message(message):
    """Print a formatted, easily-distinguishable message.

    Parameters
    ----------
    message : str
        The message to print.
    """
    print(u'\u001b[46;1m[INFO]\u001b[0m %s' % (message))

class EMOptWarning(RuntimeWarning):
    pass

@run_on_master
def _warning_message(message, category=UserWarning, filename='', lineno=-1):
    # Override python's warning message by adding a colored [WARNING] flag in
    # front to make it more noticeable.
    if(type(category) == EMOptWarning):
        print(u'\u001b[43;1m[WARNING]\u001b[0m %s' % (message))
    else:
        print(u'\u001b[43;1m[WARNING]\u001b[0m in %s at line %d: %s' % \
              (filename, lineno, message))
warnings.showwarning = _warning_message


def warning_message(message, module):
    # Produce a warning message to warn the user that a problem has occurred.
    # This is primarily intended for internal use within emopt
    warnings.warn('in %s: %s' % (module, message), category=EMOptWarning)

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
    i : slice
        The slice along the z direction
    j : slice
        The slice along the y direction
    k : numpy.ndarray
        The slice along the x direction
    """

    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz):
        i1 = int(zmin/dz)
        i2 = int(zmax/dz)+1
        ilist = np.arange(i1, i2, 1, dtype=np.int)
        self._z = dz * ilist.astype(np.double)
        self._i = slice(i1, i2)

        j1 = int(ymin/dy)
        j2 = int(ymax/dy)+1
        jlist = np.arange(j1, j2, 1, dtype=np.int)
        self._y = dy * jlist.astype(np.double)
        self._j = slice(j1, j2)

        k1 = int(xmin/dx)
        k2 = int(xmax/dx)+1
        klist = np.arange(k1, k2, 1, dtype=np.int)
        self._x = dx * klist.astype(np.double)
        self._k = slice(k1, k2)


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
