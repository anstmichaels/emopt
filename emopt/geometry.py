from __future__ import division, print_function, absolute_import
import numpy as np
import scipy
from math import pi
from .misc import warning_message, NOT_PARALLEL

def fillet(x, y, R, make_round=None, points_per_90=10, equal_thresh=1e-8,
           ignore_roc_lim=False):
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
    x : list
        The x coordinates of the chain of line segments to round.
    y : list
        The y coordinates of the chain of line segments to round.
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

    Returns
    -------
    list, list
        The x and y coordinates of the new set of lines segments or polygon.
    """
    xn = []
    yn = []
    i = 0
    N = len(x)
    N0 = N

    # we always ignore the last point
    N -= 1

    closed = True
    if(x[0] != x[-1] or y[0] != y[-1]):
        closed = False
        i += 1
        xn.append(x[0])
        yn.append(y[0])

    i = 0
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

                Np = int(np.abs(theta2-theta1) / (pi/2) * points_per_90)
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

    return np.array(xn), np.array(yn)

def populate_lines(xs, ys, ds, refine_box=None):
    """Populate a chain of line segments with additional points.

    Given a set of connected line segments, add evenly distributed points to
    the line segments. This is currently done in a very approximate manner.

    Parameters
    ----------
    xs : list or numpy.array
        The list of x coordinates of the line segments.
    ys : list or numpy.array
        The list of y coordinates of the line segments.
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

    return np.array(x_reduc), np.array(y_reduc)

class IndexSet(object):

    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._indices = []

    def __getitem__(self, key):
        return self._indices[key]

    @property
    def x(self):
        return x

    @property
    def y(self):
        return y

    @property
    def indices(self):
        return self._indices

    @x.setter
    def x(self, newx):
        x = newx

    @y.setter
    def y(self, newy):
        y = newy

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

class FourierDisplacer(object):
    """Displace a points which define a path using a radius-of-curvature-limited
    Fourier series.
    """

    def __init__(self, xs, ys, Rmin):
        self._xs = xs
        self._ys = ys
        self._Npts = len(xs)
        self._Rmin = Rmin

        # calculate path lengths
        ts = [0]
        for i in range(i, self._Npts):
            t_tot = ts[i-1]
            t_next = np.sqrt((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2)
            ts.append(t_tot+t_next)

        self._ts = ts
        Tmax = ts[-1]
        self._Tmax = Tmax

        # calculate number of Fourier sin coefficients = # half periods which
        # fit along path
        Nf = np.floor(Tmax/(2*Rmin))
        self._Nf

        # calculate the fourier periods
        indices = np.arange(0, Nf, 1) + 1
        periods = Tmax/indices
        self._P = periods

        # generate the x and y Fourier coefficients
        self._Ax = np.zeros(Nf)
        self._Ay = np.zeros(Nf)

        # calculate the Fourier amplitude limits based on radius of curvature.
        # Note: this is approximate
        self._Amax = (periods/pi)**2 / Rmin

    @property
    def xs(self):
        return self._xs

    @property
    def ys(self):
        return self._ys

    @property
    def Nf(self):
        return self._Nf

    @property
    def periods(self):
        return self._P

    @property
    def max_amplitude(self):
        return self._Amax

    def get_x(self, Ax):
        """Calculate the displaced x coordinates.

        Parameters
        ----------
        Ax : numpy.ndarray
            The list of Fourier coefficients used for x displacements. This
            list should have length equal to Nf.

        Returns
        -------
        numpy.ndarray
            The list of displaced x coordinates.
        """
        xs = np.copy(self._xs)

        for i in range(self._Nf):
            xs = xs + Ax[i] * np.sin(pi*self._ts/self._P[i])

        return xs

    def get_y(self, Ay):
        """Calculate the displaced y coordinates.

        Parameters
        ----------
        Ay : numpy.ndarray
            The list of Fourier coefficients used for y displacements. This
            list should have length equal to Nf.

        Returns
        -------
        numpy.ndarray
            The list of displaced y coordinates.
        """
        ys = np.copy(self._ys)

        for i in range(self._Nf):
            ys = ys + Ay[i] * np.sin(pi*self._ts/self._P[i])

        return ys
