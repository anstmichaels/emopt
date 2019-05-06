"""Python interface for grid smoothing code.

Grid smoothing refers to the process of mapping a continuously-defined shapes with
sharp boundaries to a rectangular grid in a continuous manner.  This process results
in a rectangular grid whose material distribution matches the defined shape
except at the boundaries where the effective material value is between the
material value within the bound shape and that of the surrounding medium, hence
making the boundaries appear as if they have been 'smoothed' out.

Ensuring that this mapping from 'real' space to the discretized rectangular
grid is continuous (i.e. small changes in the underlying boundaries produce
small changes in the material distribution of the grid) is very important to
sensitivity analysis. Gradients computed using an adjoint method, in
particular, will be inaccurate if changes to the material distribution occur in
discrete jumps which are too large.

Grid smoothing can be accomplished in a variety of ways.  The implementation
here is very general for sets of shapes which do not require changes in
topology (creation and elimination of holes, etc). It relies on representing
boundaries using polygons and then computing the smoothed grid by applying a
series of boolean subtraction operations (in c++).
"""
from __future__ import absolute_import

from builtins import range
from builtins import object
from .grid_ctypes import libGrid
import numpy as np
import scipy
from ctypes import c_int, c_double

from .misc import DomainCoordinates
from .misc import warning_message

__author__ = "Andrew Michaels"
__license__ = "GPL License, Version 3.0"
__version__ = "2019.5.6"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

class Material2D(object):
    """Define a general interface for 2D Material distributions.

    Notes
    -----
    Currently material distributions only support 2D systems.

    Methods
    -------
    get_value(self, x, y)
        Get the value of the material distribution at (x,y)
    get_values(self, m1, m2, n1, n2, arr=None)
        Get the values of the material distribution within a set of array
        indicesa set of array indices.
    get_values_on(self, domain)
        Get the values of the material distribution within a domain.
    """

    def __init__(self):
        self._object = None

    def get_value(self, x, y):
        """Get the value of the material distribution at (x,y).

        Parameters
        ----------
        x : int or float
            The (fractional) x index.
        y : int or float
            The (fractional) y index.

        Returns
        -------
        complex128
            The complex material value at the desired location.
        """
        value = np.array([0], dtype=np.complex128)
        libGrid.Material2D_get_value(self._object, value, x, y)

        return value

    def get_values(self, k1, k2, j1, j2, sx=0.0, sy=0.0, arr=None):
        """Get the values of the material distribution within a set of array
        indicesa set of array indices.

        Parameters
        ----------
        k1 : int
            The lower integer bound on x of the desired region
        k2 : int
            The upper integer bound on x of the desired region
        j1 : int
            The lower integer bound on y of the desired region
        j2 : int
            The upper integer bound on y of the desired region
        arr : numpy.ndarray (optional)
            The array with dimension (m2-m1)x(n2-n1) with type np.complex128
            which will store the retrieved material distribution. If None, a
            new array will be created. (optional = None)

        Returns
        -------
        numpy.ndarray
            The retrieved complex material distribution.
        """
        Nx = k2-k1
        Ny = j2-j1

        if(type(arr) == type(None)):
            arr = np.zeros(Nx*Ny, dtype=np.complex128)
        else:
            arr = np.ravel()

        libGrid.Material2D_get_values(self._object, arr, k1, k2, j1, j2, sx, sy)

        # This might result in an expensive copy operation, unfortunately
        arr = np.reshape(arr, [Ny, Nx])

        return arr

    def get_values_in(self, domain, sx=0, sy=0, squeeze=False, arr=None):
        """Get the values of the material distribution within a domain.

        Parameters
        ----------
        domain : emopt.misc.DomainCoordinates
            The domain in which the material distribution is retrieved.

        Returns
        -------
        numpy.ndarray
            The retrieved material distribution which lies in the domain.
        """
        j1 = domain.j.start
        j2 = domain.j.stop
        k1 = domain.k.start
        k2 = domain.k.stop

        arr = self.get_values(k1, k2, j1, j2, sx, sy, arr)
        if(squeeze): return np.squeeze(arr)
        else: return arr

class ConstantMaterial2D(Material2D):
    """A uniform constant material.

    Parameters
    ----------
    value : complex
        The constant material value.

    Attributes
    ----------
    material_value : complex
        The constant material value
    """
    def __init__(self, value):
        self._material_value = value
        self._object = libGrid.ConstantMaterial2D_new(value.real, value.imag)

    @property
    def material_value(self):
        return self._material_value

    @material_value.setter
    def material_value(self, new_value):
        libGrid.ConstantMaterial2D_set_material(self._object,
                                              new_value.real,
                                              new_value.imag)
        self._material_value = new_value

class GridMaterial2D(Material2D):
    """Define a simple rectangular-grid-based Material distribution.

    This is the simplest form of :class:`.Material` object which defines a
    material distribution as a simple rectangular grid. On its own, there is
    no grid smoothing performed by a :class:`.GridMaterial`, however it can be
    used to implement custom grid smoothing subroutines.

    Parameters
    ----------
    M : int
        Number of rows in the grid.
    N : int
        Number of columns in the grid.
    grid : numpy.ndarray
        The complex (dtype=np.complex128) material distribution

    Attributes
    ----------
    M : int
        The number of rows in the grid.
    N : int
        The number of cols in the grid.
    grid : numpy.ndarray
        The complex material distribution.
    """

    def __init__(self, M, N, grid):
        grid = grid.astype(np.complex128, copy=False)
        self._object = libGrid.GridMaterial2D_new(M, N, grid)

        self._M = M
        self._N = N
        self._grid = grid

    def __del__(self):
        libGrid.GridMaterial2D_delete(self._object)

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, newM):
        libGrid.GridMaterial2D_delete(self._object)
        self._object = libGrid.GridMaterial2D_new(newM, self._N, grid)
        self._M = newM

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, newN):
        libGrid.GridMaterial_delete(self._object)
        self._object = libGrid.GridMaterial_new(self._M, newN, grid)
        self._N = newN

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, new_grid):
        self._grid = new_grid
        self._object = libGrid.GridMaterial2D_new(self._M, self._N, new_grid)

class GridMaterial3D(object):

    def __init__(self, X, Y, Z, Nx, Ny, Nz, grid):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.grid = grid

    def get_value(self, k, j ,i):
        return self.grid[i,j,k]


class MaterialPrimitive(object):
    """Define a MaterialPrimitive.

    A MaterialPrimitive is a material distribution belonging to shapes like
    rectangles, circules, polygons, etc.

    TODO
    ----
    Pythonify this function using properties.

    Methods
    -------
    contains_point(self, x, y)
        Check if a material primitive contains the supplied (x,y) coordinate

    Attributes
    ----------
    layer : int
        The layer of the material primitive. Lower means higher priority in
        terms of visibility.
    """

    def __init__(self):
        self._object = None
        self._layer = 1

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, newlayer):
        self._layer = newlayer
        libGrid.MaterialPrimitive_set_layer(self._object, c_int(newlayer))

    def get_layer(self):
        """Get the layer of the primitive.

        Returns
        -------
        int
            The layer.
        """
        warning_message('get_layer() is deprecated. Use property ' \
            'myprim.layer instead.', 'emopt.grid')
        return libGrid.MaterialPrimitive_get_layer(self._object)

    def set_layer(self, layer):
        """Set the layer of the primitive.

        Parameters
        ----------
        layer : int
            The new layer.
        """
        warning_message('set_layer(...) is deprecated. Use property ' \
            'myprim.layer=... instead.', 'emopt.grid')
        libGrid.MaterialPrimitive_set_layer(self._object, c_int(layer))

    def contains_point(self, x, y):
        """Check if a material primitive contains the supplied (x,y) coordinate

        Parameters
        ----------
        x : float
            The real-space x coordinate
        y : float
            The real-space y coordinate

        Returns
        -------
        bool
            True if the (x,y) point is contained within the primitive, false
            otherwise.
        """
        return libGrid.MaterialPrimitive_contains_point(self._object, x, y)

class Circle(MaterialPrimitive):
    """Define a circle primitive.

    Notes
    -----
    This is not fully implemented.

    TODO
    ----
    Actually fully implement this.
    """
    def __init__(self, x0, y0, r):
        self._object = libGrid.Circle_new(c_double(x0),
                                         c_double(y0),
                                         c_double(r))

    def __del__(self):
        libGrid.Circle_delete(self._object)

    def set_material(self, mat):
        libGrid.Circle_set_material(self._object, mat.real, mat.imag)

    def set_position(self, x0, y0):
        libGrid.Circle_set_position(self._object, x0, y0)

    def set_radius(self, r):
        libGrid.Circle_set_radius(self._object, r)

    def get_x0(self):
        return libGrid.Circle_get_x0(self._object)

    def get_y0(self):
        return libGrid.Circle_get_y0(self._object)

    def get_r(self):
        return libGrid.Circle_get_r(self._object)

class Rectangle(MaterialPrimitive):
    """Define a rectangular material primitive.

    A rectangle is defined by its center position and its width and height.

    Parameters
    ----------
    x0 : float
        The x coordinate of the center of the Rectangle
    y0 : float
        The y coordinate of the center of the Rectangle
    xspan : float
        The width of the Rectangle
    yspan : float
        The height of the Rectangle

    Attributes
    ----------
    x0 : float
        The x coordinate of the center of the Rectangle
    y0 : float
        The y coordinate of the center of the Rectangle
    width : float
        The width of the Rectangle
    height : float
        The height of the Rectangle
    material_value : complex128
        The material value of the Rectangle's interior

    Methods
    -------
    set_material(self, mat)
        (Deprecated) Set the material value of the Rectangle's interior.
    set_position(self, x0, y0)
        Set the (x,y) position of the center of the Rectangle.
    set_width(self, width)
        (Deprecated) Set the width of the rectangle.
    set_height(self, height)
        (Deprecated) Set the height of the rectangle.
    """

    def __init__(self, x0, y0, xspan, yspan):
        self._object = libGrid.Rectangle_new(x0, y0, xspan, yspan)
        self._x0 = x0
        self._y0 = y0
        self._xspan = xspan
        self._yspan = yspan

        self._mat = 1.0

    def __del__(self):
        libGrid.Rectangle_delete(self._object)

    @property
    def x0(self):
        return self._x0

    @property
    def y0(self):
        return self._y0

    @property
    def width(self):
        return self._xspan

    @property
    def height(self):
        return self._yspan

    @property
    def material_value(self):
        return self._mat

    @x0.setter
    def x0(self, x):
        self._x0 = x
        libGrid.Rectangle_set_position(self._object, self._x0, self._y0)

    @y0.setter
    def y0(self, y):
        self._y0 = y
        libGrid.Rectangle_set_position(self._object, self._x0, self._y0)

    @width.setter
    def width(self, w):
        self._xspan = w
        libGrid.Rectangle_set_width(self._object, w)

    @height.setter
    def height(self, h):
        self._yspan = h
        libGrid.Rectangle_set_height(self._object, h)

    @material_value.setter
    def material_value(self, mat):
        self._mat = mat
        libGrid.Rectangle_set_material(self._object, mat.real, mat.imag)

    def set_material(self, mat):
        """(Deprecated) Set the material value of the Rectangle's interior.
        """
        warning_message('set_mateiral(...) is deprecated. Use property ' \
                        'myrect.material_value=... instead.', 'emopt.grid')
        libGrid.Rectangle_set_material(self._object, mat.real, mat.imag)

    def set_position(self, x0, y0):
        """Set the (x,y) position of the center of the Rectangle.

        Parameters
        ----------
        x0 : float
            The x coordinate of the center of the Rectangle.
        y0 : float
            The y coordinate of the center of the Rectangle.
        """
        libGrid.Rectangle_set_position(self._object, x0, y0)
        self._x0 = x0
        self._y0 = y0

    def set_width(self, width):
        """(Deprecated) Set the width of the rectangle.
        """
        warning_message('set_width(...) is deprecated. Use property ' \
                        'myrect.width=... instead.', 'emopt.grid')

        libGrid.Rectangle_set_width(self._object, width)
        self._xspan = width

    def set_height(self, height):
        """(Deprecated) Set the height of the rectangle."""
        warning_message('set_width(...) is deprecated. Use property ' \
                        'myrect.width=... instead.', 'emopt.grid')

        libGrid.Rectangle_set_height(self._object, height)
        self._yspan = height

class Polygon(MaterialPrimitive):

    def __init__(self, xs=None, ys=None):
        self._object = libGrid.Polygon_new()

        if(xs is None or ys is None):
            self._xs = []
            self._ys = []
            self._Np = []
        else:
            self.set_points(xs,ys)

        self._value = 1.0

    @property
    def xs(self):
        return np.copy(self._xs)

    @property
    def ys(self):
        return np.copy(self._ys)

    @property
    def Np(self):
        return self._Np

    @property
    def material_value(self):
        return self._value

    @xs.setter
    def xs(self, vals):
        warning_message('Polygon.xs cannot be set on its own. Use ' \
                        'set_points(x,y) instead', 'emopt.grid.Polygon')

    @ys.setter
    def ys(self, vals):
        warning_message('Polygon.ys cannot be set on its own. Use ' \
                        'set_points(x,y) instead', 'emopt.grid.Polygon')

    @Np.setter
    def Np(self, N):
        warning_message('Polygon.Np cannot be modified in this way.', \
                        'emopt.grid.Polygon')

    @material_value.setter
    def material_value(self, value):
        libGrid.Polygon_set_material(self._object, value.real, value.imag)
        self._value = value

    def __del__(self):
        libGrid.Polygon_delete(self._object)

    def add_point(self, x, y):
        libGrid.Polygon_add_point(self._object, x, y)

        self._Np += 1
        self._xs = np.concatenate(self._xs, [x])
        self._ys = np.concatenate(self._ys, [y])

    def add_points(self, x, y):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        libGrid.Polygon_add_points(self._object, x, y, len(x))

        self._Np += len(x)
        self._xs = np.concatenate(self._xs, x)
        self._ys = np.concatenate(self._ys, y)

    def set_points(self, x, y):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        libGrid.Polygon_set_points(self._object, x, y, len(x))

        self._Np = len(x)
        self._xs = np.copy(x)
        self._ys = np.copy(y)

    def set_point(self, index, x, y):
        libGrid.Polygon_set_point(self._object, x, y, index)

        self._xs[index] = x
        self._ys[index] = y

    def set_material(self, mat):
        warning_message('set_material(...) is deprecated. Use property ' \
                        'mypoly.material_value=... instead.', 'emopt.grid')
        libGrid.Polygon_set_material(self._object, mat.real, mat.imag)
        self._value = mat

    @staticmethod
    def populate_lines(xs, ys, ds):
        """Populate one or more line segments with points.

        This is useful when defining polygons that will be manipulated by an
        optimization. Given one or more line segments, a new set of line
        segments is created which are filled with points spaced approximately
        by ds.

        For example, let's say you start with this line segment:

            *-------------------------------*

        defined by xs=[x1, x2] and ys=[y1,y1]. Using a value of ds = (x2-x1)/4
        would yield a new set of line segments:

            *-------*-------*-------*-------*

        Notes
        -----
        This function assumes that the supplied line segments fit together
        end-to-end!

        Parameters
        ----------
        xs : list or numpy.array
            The list of x coordinates of the *connected* line segments
        ys : list or numpy.array
            The list of y coordinates of the *connected* line segments
        ds : float
            The approximate point spacing in the new set of line segments

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Arrays containing the new x and y coordinates
        """
        Np = len(xs)

        xf = np.array([xs[0]])
        yf = np.array([ys[0]])

        for i in range(Np-1):
            x2 = xs[i+1]
            x1 = xs[i]
            y2 = ys[i+1]
            y1 = ys[i]

            s = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            Ns = np.ceil(s/ds)
            if(Ns < 2): Ns = 2

            xnew = np.linspace(x1, x2, Ns); xnew = xnew[1:]
            ynew = np.linspace(y1, y2, Ns); ynew = ynew[1:]

            xf = np.concatenate([xf, xnew])
            yf = np.concatenate([yf, ynew])

        return xf, yf

class StructuredMaterial2D(Material2D):
    """Create a 2D material consisting of one or more primitive shapes
    (rectangles, polyongs, etc)

    Notes
    -----
    When used for defining the material distribution for a simulation, the
    dimensions supplied will typically match the dimensions of the simulation.

    Parameters
    ----------
    w : float
        The width of the underlying grid.
    h : float
        The height of the underlying grid.
    dx : float
        The grid spacing of the underlying grid in the x direction.
    dy : float
        The grid spacing of the underlying grid in the y direction.

    Attributes
    ----------
    primitives : list
        The list of primitives used to define the material distribution.
    """
    def __init__(self, w, h, dx, dy):
        self._object = libGrid.StructuredMaterial2D_new(w, h, dx, dy)
        self._primitives = []

    @property
    def primitives(self):
        return self._primitives

    @primitives.setter
    def primitive(self):
        warning_message('The primitive list cannot be modified in this way.',
                        'emopt.grid')

    def __del__(self):
        libGrid.StructuredMaterial2D_delete(self._object)

    def add_primitive(self, prim):
        """Add a primitive to the StructuredMaterial.

        This could be an emopt.grid.Rectangle, emopt.grid.Polygon,
        etc--anything that extends emopt.grid.MaterialPrimitive.

        Parameters
        ----------
        prim : MaterialPrimitive
            The MaterialPrimitive to add.
        """
        libGrid.StructuredMaterial2D_add_primitive(self._object, prim._object)
        self._primitives.append(prim)

    def add_primitives(self, prims):
        """Add multiple primitives from a list.

        Parameters
        ----------
        prims : list of MaterialPrimitives
            The list of MaterialPrimitives to add.
        """
        for p in prims:
            self.add_primitive(p)

def row_wise_A_update(eps, mu, ib, ie, M, N, x1, x2, y1, y2, vdiag):
    libGrid.row_wise_A_update(eps._object, mu._object, ib, ie, M, N, x1, x2, y1, y2, vdiag)
    return vdiag

class Material3D(object):
    """Define a general interface for 3D Material distributions.

    Methods
    -------
    get_value(self, x, y, z)
        Get the value of the material distribution at (x,y,z)
    get_values(self, k1, k2, j1, j2, i1, i2, arr=None)
        Get the values of the material distribution within a set of array
        indicesa set of array indices.
    get_values_on(self, domain)
        Get the values of the material distribution within a domain.
    """

    def __init__(self):
        self._object = None

    def get_value(self, x, y, z):
        """Get the value of the material distribution at (x,y,z).

        Parameters
        ----------
        x : int or float
            The (fractional) x index.
        y : int or float
            The (fractional) y index.
        z : int or float
            The (fractional) z index

        Returns
        -------
        complex128
            The complex material value at the desired location.
        """
        value = np.array([0], dtype=np.complex128)
        libGrid.Material3D_get_value(self._object, value, x, y, z)

        return value

    def get_values(self, k1, k2, j1, j2, i1, i2, sx=0, sy=0, sz=0, arr=None,
                   reshape=True):
        """Get the values of the material distribution within a set of array
        indicesa set of array indices.

        Parameters
        ----------
        k1 : int
            The lower integer bound on x of the desired region
        k2 : int
            The upper integer bound on x of the desired region
        j1 : int
            The lower integer bound on y of the desired region
        j2 : int
            The upper integer bound on y of the desired region
        i1 : int
            The lower integer bound on y of the desired region
        i2 : int
            The upper integer bound on y of the desired region
        arr : numpy.ndarray (optional)
            The array with dimension (m2-m1)x(n2-n1) with type np.complex128
            which will store the retrieved material distribution. If None, a
            new array will be created. (optional = None)

        Returns
        -------
        numpy.ndarray
            The retrieved complex material distribution.
        """
        Nx = k2-k1
        Ny = j2-j1
        Nz = i2-i1

        if(type(arr) == type(None)):
            arr = np.zeros(Nx*Ny*Nz, dtype=np.complex128)
        else:
            arr = np.ravel(arr)

        libGrid.Material3D_get_values(self._object, arr, k1, k2, j1, j2, i1,
                                      i2, sx, sy, sz)

        # This might result in an expensive copy operation, unfortunately
        if(reshape):
            arr = np.reshape(arr, [Nz, Ny, Nx])

        return arr

    def get_values_in(self, domain, sx=0, sy=0, sz=0, arr=None, squeeze=False):
        """Get the values of the material distribution within a domain.

        Parameters
        ----------
        domain : emopt.misc.DomainCoordinates
            The domain in which the material distribution is retrieved.
        sx : float (optional)
            The partial index shift in the x direction
        sy : float (optional)
            The partial index shift in the y direction
        sz : float (optional)
            The partial index shift in the z direction
        arr : np.ndarray (optional)
            The array in which the retrieved material distribution is stored.
            If None, a new array is instantiated (default = None)
        squeeze : bool (optional)
            If True, eliminate length-1 dimensions from the resulting array.
            This only affects 1D and 2D domains. (default = False)

        Returns
        -------
        numpy.ndarray
            The retrieved material distribution which lies in the domain.
        """
        i1 = domain.i.start
        i2 = domain.i.stop
        j1 = domain.j.start
        j2 = domain.j.stop
        k1 = domain.k.start
        k2 = domain.k.stop

        vals = self.get_values(k1, k2, j1, j2, i1, i2, sx, sy, sz, arr)

        if(squeeze): return np.squeeze(vals)
        else: return vals

class ConstantMaterial3D(Material3D):
    """A uniform constant 3D material.

    Parameters
    ----------
    value : complex
        The constant material value.

    Attributes
    ----------
    material_value : complex
        The constant material value
    """
    def __init__(self, value):
        self._material_value = value
        self._object = libGrid.ConstantMaterial3D_new(value.real, value.imag)

    @property
    def material_value(self):
        return self._material_value

    @material_value.setter
    def material_value(self, new_value):
        libGrid.ConstantMaterial3D_set_material(self._object,
                                              new_value.real,
                                              new_value.imag)
        self._material_value = new_value

class StructuredMaterial3D(Material3D):
    """Create a 3D material consisting of one or more primitive shapes
    (rectangles, polygons, etc) which thickness along z.

    Currently StructuredMaterial3D only supports layered slab structures.

    Notes
    -----
    When used for defining the material distribution for a simulation, the
    dimensions supplied will typically match the dimensions of the simulation.

    Parameters
    ----------
    X : float
        The x width of the underlying grid.
    Y : float
        The y width of the underlying grid.
    Z : float
        The z width of the underlying grid.
    dx : float
        The grid spacing of the underlying grid in the x direction.
    dy : float
        The grid spacing of the underlying grid in the y direction.
    dz : float
        The grid spacing of the underlying grid in the z direction.

    Attributes
    ----------
    primitives : list
        The list of primitives used to define the material distribution.
    """
    def __init__(self, X, Y, Z, dx, dy, dz):
        self._object = libGrid.StructuredMaterial3D_new(X, Y, Z, dx, dy, dz)
        self._primitives = []
        self._zmins = []
        self._zmaxs = []

    @property
    def primitives(self):
        return self._primitives

    @primitives.setter
    def primitive(self):
        warning_message('The primitive list cannot be modified in this way.',
                        'emopt.grid')

    @property
    def zmins(self):
        return self._zmins

    @zmins.setter
    def zmins(self):
        warning_message('The list of minimum z coordinates cannot be changed in this way.',
                        'emopt.grid')

    @property
    def zmaxs(self):
        return self._zmaxs

    @zmaxs.setter
    def zmaxs(self):
        warning_message('The list of maximum z coordinates cannot be changed in this way.',
                        'emopt.grid')

    def __del__(self):
        libGrid.StructuredMaterial3D_delete(self._object)

    def add_primitive(self, prim, z1, z2):
        """Add a primitive to the StructuredMaterial.

        This could be an emopt.grid.Rectangle, emopt.grid.Polygon,
        etc--anything that extends emopt.grid.MaterialPrimitive.

        Parameters
        ----------
        prim : MaterialPrimitive
            The MaterialPrimitive to add.
        z1 : float
            The minimum z-coordinate of the primitive to add.
        z2 : float
            The maximum z-coordinate of the primitive to add.
        """
        self._primitives.append(prim)
        self._zmins.append(z1)
        self._zmaxs.append(z2)
        libGrid.StructuredMaterial3D_add_primitive(self._object, prim._object,
                                                  z1, z2)

