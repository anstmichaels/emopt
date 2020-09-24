import numpy as np
from math import pi
from .misc import warning_message, error_message, NOT_PARALLEL
from ._grid_ctypes import libGrid, c_int

__author__ = "Andrew Michaels"
__license__ = "GPL License, Version 3.0"

class MaterialPrimitive(object):
    """Define any primitive object that contains a material.

    Common examples include shapes like rectangles, circles, or more generally polygons that
    are filled with a specified material.

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

    def __init__(self, label=None):
        self._object = None
        self._layer = 1

        if(label == None):
            label = 'Primitive'

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, newlayer):
        self._layer = newlayer
        libGrid.MaterialPrimitive_set_layer(self._object, c_int(newlayer))

    @property
    def label(self): return self._label

    @label.setter
    def label(self, val): self._label = val

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

class Polygon(MaterialPrimitive):

    def __init__(self, xs, ys, material_value=1.0, layer=1, label=None):
        if(label == None):
            label = 'Polygon'
        super().__init__(label)

        self._object = libGrid.Polygon_new()

        self.set_points(xs,ys)
        self.material_value = material_value
        self.layer = layer

        # Store transformation data -- may be useful
        self._transformations = []

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

    @property
    def transformations(self): return self._transformations

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

    def refine_edges(self, ds, refine_box=None):
        """Refine polygon edges by adding additional evenly spaced vertices.

        Parameters
        ----------
        ds : float
            The approximate spacing of the populated points.
        refine_box : list of tuple
            Only populate points within the box [xmin, xmax, ymin, ymax]. If None,
            populate all lines (default = None)

        Returns
        -------
        numpy.array, numpy.array
            The x and y coordinates of the newly populated line segments.
        """
        xs = np.copy(self._xs)
        ys = np.copy(self._ys)
        Np = len(xs)

        xf = np.array([xs[0]])
        yf = np.array([ys[0]])

        if(refine_box is not None):
            xmin, xmax, ymin, ymax = refine_box
        else:
            xmin = np.min(xs) - 1
            xmax = np.max(xs) + 1
            ymin = np.min(ys) - 1
            ymax = np.max(ys) + 1

        for i in range(Np-1):

            x2 = xs[i+1]
            x1 = xs[i]
            y2 = ys[i+1]
            y1 = ys[i]

            p1in = x1 >= xmin and x1 <= xmax and y1 >= ymin and y1 <= ymax
            p2in = x2 >= xmin and x2 <= xmax and y2 >= ymin and y2 <= ymax

            if(not p1in or not p2in):
                xf = np.concatenate([xf, [x1, x2]])
                yf = np.concatenate([yf, [y1, y2]])
            else:
                s = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                Ns = np.ceil(s/ds)
                if(Ns < 2): Ns = 2

                xnew = np.linspace(x1, x2, Ns)
                ynew = np.linspace(y1, y2, Ns)

                xf = np.concatenate([xf, xnew])
                yf = np.concatenate([yf, ynew])

        # remove duplicate points
        x_reduc = []
        y_reduc = []
        for x,y in zip(xf, yf):
            if(len(x_reduc) == 0):
                x_reduc.append(x)
                y_reduc.append(y)
            elif(x != x_reduc[-1] or y != y_reduc[-1]):
                x_reduc.append(x)
                y_reduc.append(y)

        self.set_points(np.array(x_reduc), np.array(y_reduc))

    def fillet(self, R, make_round=None, points_per_90=10, equal_thresh=1e-8,
               ignore_roc_lim=False, points_per_bend=None):
        """Round corners of a polygon.

        This function replaces sharp corners with circular arcs. The radius of
        these arcs will be equal to the specified radius as long as the line
        segments which make up the corner are sufficiently long. If they are too
        short, the arc will be made with a smaller radius.

        If a non-closed polygon is supplied, the end points will be ignored. A
        polygon is considered closed if the first and last point in the provided
        x,y coordinate lists are the same.

        Parameters
        ----------
        R : float
            The desired fillet radius
        make_round : list or None (optional)
            A list of boolean values which specifies which points should be
            rounded. If None, then all roundable points are rounded. If supplying a
            list, the list must have the same length as x and y. (default = None)
        points_per_90 : int (optional)
            The number of points to generate per 90 degrees of the arc. (default =
            10)
        equal_thresh : float
            The threshold used for comparing values. If the |difference| between
            two values is less than this value, they are considered equal.
        points_per_bend : int (optional)
            The number of points to use to define the bend. This parameter
            overrides points_per_90. Use this if you need the number of points in a
            fillet to remain fixed regardless of the corner angle.

        Returns
        -------
        list, list
            The x and y coordinates of the new set of lines segments or polygon.
        """
        x = self._xs
        y = self._ys

        xn = []
        yn = []
        i = 0
        N = len(x)
        N0 = N

        # we always ignore the last point
        N -= 1

        closed = True
        i = 0
        if(x[0] != x[-1] or y[0] != y[-1]):
            closed = False
            xn.append(x[0])
            yn.append(y[0])
            i = 1

        while(i < N):
            if(make_round is None or make_round[i]):
                # get current and adjacent points
                x1 = x[i]
                y1 = y[i]
                if(i == 0):
                    x0 = x[-2]
                    y0 = y[-2]
                else:
                    x0 = x[i-1]
                    y0 = y[i-1]
                if(i == N0-1):
                    x2 = x[1]
                    y2 = y[1]
                else:
                    x2 = x[i+1]
                    y2 = y[i+1]

                # calc angle
                dx10 = x0-x1
                dy10 = y0-y1
                dx12 = x2-x1
                dy12 = y2-y1
                d10 = np.sqrt(dx10**2 + dy10**2)
                d12 = np.sqrt(dx12**2 + dy12**2)

                theta = np.arccos((dx10*dx12 + dy10*dy12)/(d10*d12))

                if(theta != 0 and theta != pi):
                    dxc = (dx10/d10 + dx12/d12)
                    dyc = (dy10/d10 + dy12/d12)
                    dc = np.sqrt(dxc**2 + dyc**2)
                    nxc = dxc/dc
                    nyc = dyc/dc

                    nx10 = dx10/d10
                    ny10 = dy10/d10

                    nx12 = dx12/d12
                    ny12 = dy12/d12

                    # reduce fillet radius if necessary
                    Ri = R
                    iprev = i-1
                    inext = i+1
                    if(iprev < 0): iprev = N0-1
                    if(inext > N0-1): inext = 0

                    if( (d10 < 2*R or d12 < 2*R) and not ignore_roc_lim):
                        Ri = np.min([d12, d10])/2.0
                        if(NOT_PARALLEL):
                            warning_message('Warning: Desired radius of curvature too large at ' \
                                            'point %d. Reducing to maximum allowed.' % \
                                            (i), 'emopt.geometry')

                    # figure out where the circle "fits" in the corner
                    S10 = Ri / np.tan(theta/2.0)
                    S12 = S10
                    Sc = np.sqrt(S10**2 + Ri**2)

                    # generate the fillet
                    theta1 = np.arctan2((S10*ny10 - Sc*nyc), (S10*nx10 - Sc*nxc))
                    theta2 = np.arctan2((S12*ny12 - Sc*nyc), (S12*nx12 - Sc*nxc))

                    if(theta1 < 0): theta1 += 2*pi
                    if(theta2 < 0): theta2 += 2*pi

                    theta1 = np.mod(theta1, 2*pi)
                    theta2 = np.mod(theta2, 2*pi)

                    if(theta2 - theta1 > pi): theta2 -= 2*pi
                    elif(theta1 - theta2 > pi): theta2 += 2*pi

                    if(points_per_bend == None):
                        Np = int(np.abs(theta2-theta1) / (pi/2) * points_per_90)
                    else:
                        Np = points_per_bend
                    if(Np < 1):
                        Np = 1
                    thetas = np.linspace(theta1, theta2, Np)
                    for t in thetas:
                        xfil = x[i] + Sc*nxc + Ri*np.cos(t)
                        yfil = y[i] + Sc*nyc + Ri*np.sin(t)

                        # only add point if not duplicate (this can happen if
                        # the desired radis of curvature equals or exceeds the
                        # maximum allowed)
                        if(np.abs(xfil - xn[-1]) > equal_thresh or
                           np.abs(yfil - yn[-1]) > equal_thresh):
                            xn.append(xfil)
                            yn.append(yfil)

                else:
                    xn.append(x[i])
                    yn.append(y[i])
            else:
                xn.append(x[i])
                yn.append(y[i])
            i += 1

        if(not closed):
            xn.append(x[-1])
            yn.append(y[-1])
        else:
            xn.append(xn[0])
            yn.append(yn[0])

        self.set_points(np.array(xn), np.array(yn)) 


    def translate(self, dx, dy):
        """Move the polygon by the specified amount along x and y.
        """
        xs = self.xs + dx
        ys = self.ys + dy

        self.set_points(xs, ys)

        # update list of transformations
        self._transformations.append( ('translate', dx, dy) )

    def rotate(self, angle, x0=0, y0=0):
        """Rotate the polygon about a point.

        Parameters
        ----------
        angle : float
            Counterclockwise angle of rotation in degrees.
        """
        theta = angle / 180 * np.pi
        x = (self.xs - x0) * np.cos(theta) - (self.ys - y0) * np.sin(theta) + x0
        y = (self.xs - x0) * np.sin(theta) + (self.ys - y0) * np.cos(theta) + y0

        self.set_points(x, y)
    
        self._transformations.append( ('rotate', angle, x0, y0) )

    def scale(self, sx, sy, x0=0, y0=0):
        """Scale the polygon.
        """
        x = (self.xs-x0)*sx + x0
        y = (self.ys-y0)*sy + y0

        self.set_points(x, y)

        self._transformations( ('scale', sx, sy, x0, y0) )

    def mirror(self, mx, my, x0=0, y0=0):
        """Mirror the polygon about a point
        """
        if(mx):
            x = x0 - self.xs
        else:
            x = self.xs

        if(my):
            y = y0 - self.ys
        else:
            y = self.ys

        self.set_points(x, y)

        self._transformations.append( ('mirror', mx, my, x0, y0) )

class Rectangle(Polygon):
    """Define a rectangle.
    """
    def __init__(self, x0, y0, w, h, mat_val=1.0, layer=1, label=None):
        if(label == None):
            label = 'Rectangle'
        super().__init__([], [], mat_val, layer=1, label=label)

        self._x0 = x0
        self._y0 = y0
        self._w = w
        self._h = h

        self._update_points()

    @property
    def x0(self): return self._x0

    @x0.setter
    def x0(self, x):
        self._x0 = x
        self._update_points()

    @property
    def y0(self): return self._y0

    @y0.setter
    def y0(self, val):
        self._y0 = val
        self._update_points()

    @property
    def width(self): return self._w

    @width.setter
    def width(self, val):
        self._w = val
        self._update_points()

    @property
    def height(self): return self._h

    @height.setter
    def height(self, val):
        self._h = val
        self._update_points()

    def _update_points(self):
        x0 = self._x0
        y0 = self._y0
        w = self._w
        h = self._h
        x = [x0-w/2, x0+w/2, x0+w/2, x0-w/2, x0-w/2]
        y = [y0-h/2, y0-h/2, y0+h/2, y0+h/2, y0-h/2]

        super().set_points(x, y)


class Circle(Polygon):
    """Define a circle filled with a particular material
    """
    def __init__(self, x0, y0, r, mat_val=1.0, layer=1, resolution=72, label=None):
        if(label == None):
            label = 'Circle'
        super().__init__([], [], mat_val, layer, label=label)


        self._x0 = x0
        self._y0 = y0
        self._r = r
        self._res = resolution

        self._update_points()

    @property
    def x0(self): return self._x0

    @x0.setter
    def x0(self, x):
        self._x0 = x
        self._update_points()

    @property
    def y0(self): return self._y0

    @y0.setter
    def y0(self, val):
        self._y0 = val
        self._update_points()

    @property
    def r(self): return self._r

    @r.setter
    def r(self, val):
        self._r = val
        self._update_points()

    @property
    def resolution(self): return self._resolution

    @resolution.setter
    def resolution(self, val):
        self._resolution = val
        self._update_points()

    def _update_points(self):
        theta = np.linspace(0,2*pi,self._res+1)
        x = self._r*np.cos(theta) + self._x0
        y = self._r*np.sin(theta) + self._y0

        super().set_points(x, y)


class Ellipse(Polygon):
    """Define a circle filled with a particular material
    """
    def __init__(self, x0, y0, a, b, mat_val=1.0, layer=1, resolution=72, label=None):
        if(label == None):
            label = 'Ellipse'
        super().__init__([], [], mat_val, layer, label=label)

        self._x0 = x0
        self._y0 = y0
        self._a = a
        self._b = b
        self._res = resolution

        # Rectangle is internally repesented as a polygon
        self._object = libGrid.Polygon_new()

        self._update_points()

    @property
    def x0(self): return self._x0

    @x0.setter
    def x0(self, x):
        self._x0 = x
        self._update_points()

    @property
    def y0(self): return self._y0

    @y0.setter
    def y0(self, val):
        self._y0 = val
        self._update_points()

    @property
    def a(self): return self._a

    @a.setter
    def a(self, val):
        self._a = val
        self._update_points()

    @property
    def b(self): return self._b

    @b.setter
    def b(self, val):
        self._b = val
        self._update_points()

    @property
    def resolution(self): return self._resolution

    @resolution.setter
    def resolution(self, val):
        self._resolution = val
        self._update_points()

    def _update_points(self):
        theta = np.linspace(0,2*pi,self._res+1)
        x = self._a*np.cos(theta) + self._x0
        y = self._b*np.sin(theta) + self._y0

        super().set_points(x,y)

class IndexSet(object):

    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._indices = []

    def __getitem__(self, key):
        return self._indices[key]

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def indices(self):
        return self._indices

    @x.setter
    def x(self, newx):
        self._x = newx

    @y.setter
    def y(self, newy):
        self._y = newy

    def append(self, xmin, xmax, ymin, ymax, reverse=False):
        x = self._x
        y = self._y

        Np = len(x)
        inds = []
        for i in range(Np):
            if(x[i] >= xmin and x[i] <= xmax and
               y[i] >= ymin and y[i] <= ymax):
                inds.append(i)

        if(reverse):
            self._indices += inds[::-1]
        else:
            self._indices += inds

    def remove(self, xmin, xmax, ymin, ymax):
        x = self._x
        y = self._y

        inds = []
        for i in self._indices:
            if(x[i] < xmin or x[i] > xmax or
               y[i] < ymin or y[i] > ymax):
                inds.append(i)

        self._indices = inds

    def clear(self):
        self._indices = []

class NURBS(object):
    """Create a NURBS curve.

    This class wraps a NURBS-python NURBS object and provides some additional
    useful functionality (like radius of curvature calculation).

    Notes
    -----
    The underlying NURBS object can be accessed at any time using the curve
    attribute.

    Attributes
    ----------
    x : numpy.ndarray
        The x coordinates of the control points (array of size N).
    y : numpy.ndarray
        The y coordinates of the control points (array of size N).
    N : int
        The number of control points.
    w : numpy.ndarray
        Weight values (array of size N).
    curve : geomdl.NURBS.Curve
        The wrapped NURBS object
    knot_vec : list
        The knot vector
    degree : int
        The degree of the NURBS curve

    Parameters
    ----------
    x : numpy.ndarray
        The x coordinates of the control points (array of size N).
    y : numpy.ndarray
        The y coordinates of the control points (array of size N).
    w : numpy.ndarray (optional)
        Weight values (array of size N). By default, uniform unity weights are used. (default = None)
    degree : int (optional)
        The degree of the curve. Recommended: degree >= 3 (default = 3)
    """

    def __init__(self, x, y, degree=3, w=None):
        try:
            from geomdl import NURBS, utilities
            self.utilities = utilities
        except ImportError:
            error_message('The module `geomdl` is not installed, but it is required for NURBS!')

        self._x = np.array(x)
        self._y = np.array(y)
        self._n = len(x)

        if(w is None):
            w = np.ones(self._x.shape)
        self._w = w

        # Create the curve
        curve = NURBS.Curve()
        curve.degree = degree
        self._degree = degree

        cpoints = np.zeros((self._n,2))
        cpoints[:,0] = x
        cpoints[:,1] = y
        self._cpoints = cpoints

        curve.ctrlpts = cpoints
        curve.knotvector = utilities.generate_knot_vector(degree, len(cpoints))
        curve.weights = w
        self._curve = curve

    @property
    def x(self): return self._x

    @property
    def y(self): return self._y

    @property
    def N(self): return self._n

    @property
    def w(self): return self._w

    @property
    def curve(self): return self._curve

    @property
    def knot_vec(self): return self._curve.kotvector

    @property
    def degree(self): return self._degree

    @property
    def bbox(self): return [coordinate for xy in self._curve.bbox \
                            for coordinate in xy]

    @w.setter
    def w(self, ww):
        self._w = ww
        self._curve.weights = ww

    @degree.setter
    def degree(self, d):
        self._degree = d
        self._curve.degree = d

    def set_cpoints(self, x, y):
        """Set new control points

        Parameters
        ----------
        x : numpy.ndarray
            The new x coordinates.
        y : numpy.ndarray
            The new y coordinates.
        """
        self._x = x
        self._y = y
        self._n = len(x)

        cpoints = np.zeros((self._n,2))
        cpoints[:,0] = x
        cpoints[:,1] = y
        self._cpoints = cpoints

        self._curve.ctrlpts = cpoints
        self._curve.knotvector = self.utilities.generate_knot_vector(self._degree, len(cpoints))

    def evaluate(self, u=None, Neval=10):
        """Evaluate the NURBS curve.

        The NURBS curve can be evaluated either at a single specified value or at a
        specified number of values which span the whole curve. If a single value
        is provided, then the curve is evaluated at that single point. Otherwise,
        the curve is evaluated at the specified Neval points.

        Parameters
        ----------
        u : float (optional)
            A value between 0 and 1 when the x,y coordinate of the curve is evaluated. (default = None)
        Neval : int
            The number of points along the full length of the curve to evaluate the x,y
            coordinates of the curve. This is only evaluated if u=None. (default = 10)

        Returns
        -------
        float, float or numpy.ndarray, numpy.ndarray
            The x,y coordinate(s) of the evaluated curve
        """
        if(u is not None):
            xy = self._curve.curvept(u)
            return xy[0], xy[1]
        else:
            self._curve.sample_size = Neval
            self._curve.evaluate()
            points = np.array(self._curve.evalpts)
            return points[:,0], points[:,1]

    def radius_of_curvature(self, u=None, Neval=10):
        """Evaluate the radius of curvature along the NURBS curve.

        The ROC can be evaluated either at a single specified value or at a
        specified number of values which span the whole curve. If a single value
        is provided, then the ROC is evaluated at that single point. Otherwise,
        the curve is evaluated at the specified Neval points.

        Parameters
        ----------
        u : float (optional)
            A value between 0 and 1 when the x,y coordinate of the curve is evaluated. (default = None)
        Neval : int
            The number of points along the full length of the curve to evaluate the x,y
            coordinates of the curve. This is only evaluated if u=None. (default = 10)

        Returns
        -------
        float or numpy.ndarray
            The radius of curvature evaluated along the curve.
        """
        if(u is not None):
            us = [u]
            Neval = 1
        else:
            us = np.linspace(0,1,Neval)

        # Calculate 1st and second derivative
        xdot = np.zeros(Neval)
        ydot = np.zeros(Neval)
        xddot = np.zeros(Neval)
        yddot = np.zeros(Neval)
        for i in range(Neval):
            xydots = self._curve.derivatives(us[i],order=2)
            xdot[i] = xydots[1][0]
            ydot[i] = xydots[1][1]
            xddot[i] = xydots[2][0]
            yddot[i] = xydots[2][1]

        roc = (xdot**2 + ydot**2)**(3.0/2.0) / np.abs(xdot*yddot - xddot*ydot)
        return roc

    def evaluate_tangent(self, u=None, Neval=None):
        """Evaluate the tangent direction along the curve.

        Notes
        -----
        1. This is the normalized first derivative of x(u) and y(u)

        returns
        -------
        numpy.ndarray, numpy.ndarray
            The tangential vectors at each point
        """
        if(u is not None):
            us = [u]
            Neval = 1
        else:
            us = np.linspace(0,1,Neval)

        # Calculate 1st and second derivative
        xdot = np.zeros(Neval)
        ydot = np.zeros(Neval)
        for i in range(Neval):
            xydots = self._curve.derivatives(us[i],order=2)
            dx = xydots[1][0]
            dy = xydots[1][1]

            norm = np.sqrt(dx**2 + dy**2)
            xdot[i] = dx/norm
            ydot[i] = dy/norm

        return xdot, ydot
