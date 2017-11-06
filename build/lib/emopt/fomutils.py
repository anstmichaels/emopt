"""
Common functions useful for calculating figures of merit and their derivatives.
"""
import fdfd
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Andrew Michaels"
__license__ = "Apache License, Version 2.0"
__version__ = "0.2"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

#===================================================================================
# Miscellanious FOM helper functions
#===================================================================================

def radius_of_curvature(x1, x2, x3, y1, y2, y3):
    """Compute the approximate radius of curvature of three points.

    This is achieved by first fitting a parabola to the three points and then
    finding the radius of cruvature of that parabola.

    Notes
    -----
    To find the derivative of this function, it is easiest to simply perform a
    finite difference.

    Parameters
    ----------
    x1 : float
        The x coordinate of the first point
    x2 : float
        The x coordinate of the second point
    x3 : float
        The x coordinate of the third point
    y1 : float
        The y coordinate of the first point
    y2 : float
        The y coordinate of the second point
    y3 : float
        The y coordinate of the third point

    Returns
    float
        The approximate radius of curvature of the set of points
    """
    # speed things up by detecting the special case of three points in a line
    #if((x1 == x2 and x2 == x3) or (y1 == y2 and y2 == y3)):
    #   return np.inf

    t0 = 0;
    t1 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    t2 = np.sqrt((x3-x2)**2 + (y3-y2)**2) + t1

    A = np.matrix([[0, 0, 1], [t1**2, t1, 1], [t2**2, t2, 1]])

    Ainv = np.matrix([[A[1,1]*1.0 -A[1,2]*A[2,1], 1.0*A[2,1], - 1.0*A[1,1]],\
                     [1.0*A[2,0]-A[1,0]*1.0,-1.0*A[2,0], 1.0*A[1,0]],\
                     [A[1,0]*A[2,1]-A[1,1]*A[2,0], 0.0,0.0]])
    Adet = A[0,0]*(A[1,1]*A[2,2] - A[1,2]*A[2,1]) - A[0,1]*(A[1,0]*A[2,2] - \
                   A[1,2]*A[2,0]) + A[0,2]*(A[1,0]*A[2,1] - A[1,1]*A[2,0])
    Ainv = Ainv/Adet

    b1 = np.matrix([x1,x2,x3]).T
    b2 = np.matrix([y1,y2,y3]).T

    x1 = Ainv*b1 #np.linalg.solve(A,b1) 
    x2 = Ainv*b2 #np.linalg.solve(A,b2)

    a = x1[0]; b = x1[1]; c = x1[2]
    d = x2[0]; e = x2[1]; f = x2[2]

    # We calculate the radius of curvature at point x2
    R = np.power((2*a*t1+b)**2 + (2*d*t1+e)**2, 1.5) / np.abs((2*a*t1+b)*2*d - (2*d*t1+e)*2*a)

    return float(R)

def step(x, k, y0=0, A=1.0):
    """Compute the value of a smooth and analytic step function.

    The step function is approximated using a logistic function which can be
    scaled and shifted:

    .. math::
        \\Pi(x) = \\frac{A}{1 + e^{-k x}} + y_0

    This function has the property that :math:`\\Pi \\rightarrow y_0` as :math:`x
    \\rightarrow -\infty` and :math:`\\Pi \\rightarrow y_0 + A` as :math:`x
    \\rightarrow \infty`.

    Parameters
    ----------
    x : float or numpy.ndarray
        The input values
    k : float
        The steepness of step function
    y0 : float (optional)
        The shift of the step function (default = 0)
    A : float (optional)
        The scale factor of the step function (default = 1.0)

    Returns
    -------
    float or numpy.ndarray
        The step function applied to x.
    """
    return A / (1 + np.exp(-k*x)) + y0

def step_derivative(x, k, A=1.0):
    pass

def calc_ROC_foms(x, y, Rmin, k):
    """Calculate a figure of merit which imposes a minimum radius of curvature
    constraint.

    A radius of curvature constraint can be imposed by first calculating the
    approximate radius of curvature at every point and then penalizing a figure
    of merit when radii of curvature fall below a minimum value.  Penalization
    is achieved by applying a (smooth) step function to the radii of curvature;
    when a radius is below a specified minimum, the resulting output of
    the function drops below zero, reducing the figure of merit.

    Parameters
    ----------
    x : numpy.ndarray
        The x coordinates of a polygon or connected set of points
    y : numpy.ndarray
        The y coordinates of a polygon or connected set of points
    Rmin : float
        The minimum radius of curvature
    k : float
        The steepness of the step function used to determine violation of Rmin

    Returns
    -------
    numpy.ndarray
        The list of ROC foms computed for each point.
    """
    N_pts = len(x)

    foms = np.zeros(N_pts)

    # Original Unvectorized version
    for i in range(N_pts):
         j1 = i-1
         j2 = i
         j3 = i+1

         if(i == 0):
             j1 = N_pts-1
         if(i == N_pts - 1):
             j3 = 0

         x1 = x[j1]; x2 = x[j2]; x3 = x[j3]
         y1 = y[j1]; y2 = y[j2]; y3 = y[j3]

         Ri = radius_of_curvature(x1,x2,x3,y1,y2,y3)
         fom_i = logistic(Ri-Rmin, k) - 1
         foms[i] = fom_i

    return np.array(foms)

def rect(x, w1p, ws):
    """Apply a smooth rect function.

    This function is centered at zero.

    Parameters
    ----------
    x : float or numpy.ndarray
        The values of the independent variables passed to the rect function.
    w1p : float
        The `one percent` width; i.e. the width w for which rect(+/- w/2) =
        0.01.
    ws : float
        The steepness of the sides of the rect function.

    Returns
    -------
    float or numpy.ndarray
        The value of rect(x).
    """
    k = 2*np.log(99.0)/ws
    x1 = -w1p/2.0 + 1/k*np.log(99.0)
    x2 = w1p/2.0 - 1/k*np.log(99.0)

    return 1/(1 + np.exp(-k*(x-x1))) - 1/(1 + np.exp(-k*(x-x2)))

def rect_derivative(x, w1p, ws):
    """Calculate the derivative of the smoothed rect function.

    Parameters
    ----------
    x : float or numpy.ndarray
        The values of the independent variables passed to the rect function.
    w1p : float
        The `one percent` width; i.e. the width w for which rect(+/- w/2) =
        0.01.
    ws : float
        The steepness of the sides of the rect function.

    Returns
    -------
    float or numpy.ndarray
        The value of :math:`d \mathrm{rect} / dt |_x`.
    """
    k = 2*np.log(99.0)/ws
    x1 = - w1p/2.0 + 1/k*np.log(99.0)
    x2 = + w1p/2.0 - 1/k*np.log(99.0)

    return k*np.exp(-k*(x-x1))/(1 + np.exp(-k*(x-x1)))**2 - k*np.exp(-k*(x-x2))/(1 + np.exp(-k*(x-x2)))**2

#===================================================================================
# Mode Match
#===================================================================================

class ModeMatch:
    """Compute the mode match between two sets of electromagnetic fields.

    The mode match is essentially a projection of one set of fields onto a
    second set of fields. It defines the fraction of power in field 1 which
    propagates in field 2.

    When normalized with respect to the total source power injected into a system,
    this function can be used to compute coupling efficiencies.

    See [1] for a detailed derivation of the mode match equation.

    References
    ----------
    [1] A. Michaels, E. Yablonovitch, "Gradient-Based Inverse Electromagnetic Design
    Using Continuously-Smoothed Boundaries," Arxiv, 2017

    Parameters
    ----------
    normal : list or tuple
        The normal direction of the plane in which the mode match is computed
        in the form (x,y,z).
    ds1 : float
        The grid spacing along the first dimension of the supplied fields
    ds2 : float (optional)
        The grid spacing along the second dimension of the supplied fields. For
        1D mode matches, leave this untouched. (default = 1.0)
    Exm : numpy.ndarray (optional)
        The x component of the reference electric field
    Eym : numpy.ndarray (optional)
        The y component of the reference electric field
    Ezm : numpy.ndarray (optional)
        The z component of the reference electric field
    Hxm : numpy.ndarray (optional)
        The x component of the reference magnetic field
    Hym : numpy.ndarray (optional)
        The y component of the reference magnetic field
    Hzm : numpy.ndarray (optional)
        The z component of the reference magnetic field

    Methods
    -------
    compute(Ex=None, Ey=None, Ez=None, Hx=None, Hy=None, Hz=None)
        Compute the mode match and other underlying quantities.
    get_mode_match_forward(P_in)
        Get the mode match in the forward direction normalized with respect to
        a desired power.
    get_mode_match_back(P_in)
        Get the mode match in the backwards direction normalized with respect
        to a desired power.
    get_dFdEx()
        Get the derivative of unnormalized mode match with respect to Ex
    get_dFdEy()
        Get the derivative of unnormalized mode match with respect to Ey
    get_dFdEz()
        Get the derivative of unnormalized mode match with respect to Ez
    get_dFdHx()
        Get the derivative of unnormalized mode match with respect to Hx
    get_dFdHy()
        Get the derivative of unnormalized mode match with respect to Hy
    get_dFdHz()
        Get the derivative of unnormalized mode match with respect to Hz
    """

    def __init__(self, normal, ds1, ds2=1.0, Exm=None, Eym=None, Ezm=None,
                                             Hxm=None, Hym=None, Hzm=None):

        input_fields = [Exm, Eym, Ezm, Hxm, Hym, Hzm]

        self.fshape = [0,0]
        for f in input_fields:
            if(f is not None):
                self.fshape = f.shape
                break

        if(self.fshape == [0,0]):
            raise ValueError('No fields were passed to ModeMatch.  Mode matching is impossible without fields!')

        self.Exm = Exm if Exm is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Eym = Eym if Eym is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Ezm = Ezm if Ezm is not None else np.zeros(self.fshape, dtype=np.complex128)

        self.Hxm = Hxm if Hxm is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Hym = Hym if Hym is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Hzm = Hzm if Hzm is not None else np.zeros(self.fshape, dtype=np.complex128)

        # Note: In the case of fields along a line, we need to correct their
        # matrix dimensions to make sure multiplications due not produce
        # higher-dimensional arrays
        if(len(self.fshape) == 2):
            for f in [self.Exm, self.Eym, self.Ezm, self.Hxm, self.Hym,
                      self.Hzm]:
                f.resize(self.fshape[0])
        self.fshape = (self.fshape[0],)

        self.normal = np.array(normal)
        self.ds1 = ds1
        self.ds2 = ds2
        ds = ds1*ds2

        # cartesian basis vectors
        self.xhat = np.array([1, 0, 0])
        self.yhat = np.array([0, 1, 0])
        self.zhat = np.array([0, 0, 1])

        self.x_dot_s = self.xhat.dot(self.normal)
        self.y_dot_s = self.yhat.dot(self.normal)
        self.z_dot_s = self.zhat.dot(self.normal)

        # Calculate the mode field power normalization
        Pxm = self.Eym * np.conj(self.Hzm) - self.Ezm * np.conj(self.Hym)
        Pym = -self.Exm * np.conj(self.Hzm) + self.Ezm * np.conj(self.Hxm)
        Pzm = self.Exm * np.conj(self.Hym) - self.Eym * np.conj(self.Hxm)

        self.Pm = ds*np.sum(self.x_dot_s * Pxm + \
                            self.y_dot_s * Pym + \
                            self.z_dot_s * Pzm )

        self.efficiency = 0.0

        self.Ex = np.zeros(self.fshape, dtype=np.complex128)
        self.Ey = np.zeros(self.fshape, dtype=np.complex128)
        self.Ez = np.zeros(self.fshape, dtype=np.complex128)
        self.Hx = np.zeros(self.fshape, dtype=np.complex128)
        self.Hy = np.zeros(self.fshape, dtype=np.complex128)
        self.Hz = np.zeros(self.fshape, dtype=np.complex128)

    ## Some of the calculations are redundant, so we calculate most things in advance and 
    # save them for future access.
    def compute(self, Ex=None, Ey=None, Ez=None, Hx=None, Hy=None, Hz=None):
        """Compute the mode match and other underlying quantities.

        Notes
        -----
        This function must be called befor getting the mode match efficiency.

        If a NULL field is passed, it is assumed to be zero and have the
        correct dimensions.

        Parameters
        ----------
        Ex : numpy.ndarray (optional)
            The x component of the electric field. (default = None)
        Ey : numpy.ndarray (optional)
            The y component of the electric field. (default = None)
        Ez : numpy.ndarray (optional)
            The z component of the electric field. (default = None)
        Hx : numpy.ndarray (optional)
            The x component of the magnetic field. (default = None)
        Hy : numpy.ndarray (optional)
            The y component of the magnetic field. (default = None)
        Hz : numpy.ndarray (optional)
            The z component of the magnetic field. (default = None)
        """
        self.Ex = Ex if Ex is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Ey = Ey if Ey is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Ez = Ez if Ez is not None else np.zeros(self.fshape, dtype=np.complex128)

        self.Hx = Hx if Hx is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Hy = Hy if Hy is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Hz = Hz if Hz is not None else np.zeros(self.fshape, dtype=np.complex128)

        ds = self.ds1*self.ds2

        self.am1 = 0.5*ds*np.sum( (self.Ey*np.conj(self.Hzm) - self.Ez*np.conj(self.Hym))*self.x_dot_s - \
                                  (self.Ex*np.conj(self.Hzm) - self.Ez*np.conj(self.Hxm))*self.y_dot_s + \
                                  (self.Ex*np.conj(self.Hym) - self.Ey*np.conj(self.Hxm))*self.z_dot_s ) / self.Pm

        self.am2 = 0.5*ds*np.sum( (np.conj(self.Eym)*self.Hz - np.conj(self.Ezm)*self.Hy)*self.x_dot_s - \
                                  (np.conj(self.Exm)*self.Hz - np.conj(self.Ezm)*self.Hx)*self.y_dot_s + \
                                  (np.conj(self.Exm)*self.Hy - np.conj(self.Eym)*self.Hx)*self.z_dot_s ) / np.conj(self.Pm)

        self.am = self.am1 + self.am2
        self.bm = self.am1 - self.am2

        self.mode_match_fwd = 0.5 * self.am * np.conj(self.am) * np.real(self.Pm)
        self.mode_match_back = 0.5 * self.bm * np.conj(self.bm) * np.real(self.Pm)

    def get_mode_match_forward(self, P_in):
        """Get the mode match in the forward direction normalized with respect to
        a desired power.

        Parameters
        ----------
        P_in : float
            The power used to normalize the mode match.

        Returns
        -------
        float
            The mode match for forward-propagating fields.
        """
        return self.mode_match_fwd.real / P_in

    def get_mode_match_back(self, P_in):
        """Get the mode match in the backwards direction normalized with respect to
        a desired power.

        Parameters
        ----------
        P_in : float
            The power used to normalize the mode match.

        Returns
        -------
        float
            The mode match for backward-propagating fields.
        """
        return self.mode_match_back.real / P_in

    def get_dFdEx(self):
        """Get the derivative of unnormalized mode match with respect to Ex.

        Returns
        -------
        numpy.ndarray
            The derivative of the unnormalized mode match with respect to the x
            component of the electric field.
        """
        ds = self.ds1*self.ds2
        return 1/4.0 * ds * np.real(self.Pm) * (-np.conj(self.Hzm)*self.y_dot_s + np.conj(self.Hym)*self.z_dot_s) * np.conj(self.am)/self.Pm

    def get_dFdEy(self):
        """Get the derivative of unnormalized mode match with respect to Ey.

        Returns
        -------
        numpy.ndarray
            The derivative of the unnormalized mode match with respect to the y
            component of the electric field.
        """
        ds = self.ds1*self.ds2
        return 1/4.0 * ds * np.real(self.Pm) * (np.conj(self.Hzm)*self.x_dot_s - np.conj(self.Hxm)*self.z_dot_s) * np.conj(self.am)/self.Pm

    def get_dFdEz(self):
        """Get the derivative of unnormalized mode match with respect to Ez.

        Returns
        -------
        numpy.ndarray
            The derivative of the unnormalized mode match with respect to the z
            component of the electric field.
        """
        ds = self.ds1*self.ds2
        return 1/4.0 * ds * np.real(self.Pm) * (-np.conj(self.Hym)*self.x_dot_s + np.conj(self.Hxm)*self.y_dot_s) * np.conj(self.am)/self.Pm

    def get_dFdHx(self):
        """Get the derivative of unnormalized mode match with respect to Hx.

        Returns
        -------
        numpy.ndarray
            The derivative of the unnormalized mode match with respect to the x
            component of the magnetic field.
        """
        ds = self.ds1*self.ds2
        return 1/4.0 * ds * np.real(self.Pm) * (np.conj(self.Ezm)*self.y_dot_s - np.conj(self.Eym)*self.z_dot_s) * np.conj(self.am)/np.conj(self.Pm)

    def get_dFdHy(self):
        """Get the derivative of unnormalized mode match with respect to Hy.

        Returns
        -------
        numpy.ndarray
            The derivative of the unnormalized mode match with respect to the y
            component of the magnetic field.
        """
        ds = self.ds1*self.ds2
        return 1/4.0 * ds * np.real(self.Pm) * (-np.conj(self.Ezm)*self.x_dot_s + np.conj(self.Exm)*self.z_dot_s) * np.conj(self.am)/np.conj(self.Pm)

    def get_dFdHz(self):
        """Get the derivative of unnormalized mode match with respect to Hz.

        Returns
        -------
        numpy.ndarray
            The derivative of the unnormalized mode match with respect to the z
            component of the magnetic field.
        """
        ds = self.ds1*self.ds2
        return 1/4.0 * ds * np.real(self.Pm) * (np.conj(self.Eym)*self.x_dot_s - np.conj(self.Exm)*self.y_dot_s) * np.conj(self.am)/np.conj(self.Pm)

class ModeOverlap:

    def __init__(self, normal, ds, Exm=None, Eym=None, Ezm=None, Hxm=None, Hym=None, Hzm=None):

        input_fields = [Exm, Eym, Ezm, Hxm, Hym, Hzm]

        self.fshape = [0,0]
        for f in input_fields:
            if(f is not None):
                self.fshape = f.shape
                break

        if(self.fshape == [0,0]):
            raise ValueError('No fields were passed to ModeMatch.  Mode matching is impossible without fields!')

        self.Exm = Exm if Exm is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Eym = Eym if Eym is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Ezm = Ezm if Ezm is not None else np.zeros(self.fshape, dtype=np.complex128)

        self.Hxm = Hxm if Hxm is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Hym = Hym if Hym is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Hzm = Hzm if Hzm is not None else np.zeros(self.fshape, dtype=np.complex128)

        self.normal = np.array(normal)
        self.ds = ds

        # cartesian basis vectors
        self.xhat = np.array([1, 0, 0])
        self.yhat = np.array([0, 1, 0])
        self.zhat = np.array([0, 0, 1])

        self.x_dot_s = self.xhat.dot(self.normal)
        self.y_dot_s = self.yhat.dot(self.normal)
        self.z_dot_s = self.zhat.dot(self.normal)

        # Calculate the mode field power normalization
        Pxm = self.Eym * np.conj(self.Hzm) - self.Ezm * np.conj(self.Hym)
        Pym = -self.Exm * np.conj(self.Hzm) + self.Ezm * np.conj(self.Hxm)
        Pzm = self.Exm * np.conj(self.Hym) - self.Eym * np.conj(self.Hxm)

        self.Pm = ds*np.sum(self.x_dot_s * Pxm + \
                            self.y_dot_s * Pym + \
                            self.z_dot_s * Pzm )

        self.F1 = 0.0 + 1j*0.0
        self.F2 = 0.0 + 1j*0.0
        self.efficiency = 0.0

        self.Ex = np.zeros(self.fshape, dtype=np.complex128)
        self.Ey = np.zeros(self.fshape, dtype=np.complex128)
        self.Ez = np.zeros(self.fshape, dtype=np.complex128)
        self.Hx = np.zeros(self.fshape, dtype=np.complex128)
        self.Hy = np.zeros(self.fshape, dtype=np.complex128)
        self.Hz = np.zeros(self.fshape, dtype=np.complex128)

    ## Some of the calculations are redundant, so we calculate most things in advance and 
    # save them for future access.
    def compute(self, Ex=None, Ey=None, Ez=None, Hx=None, Hy=None, Hz=None):
        self.Ex = Ex if Ex is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Ey = Ey if Ey is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Ez = Ez if Ez is not None else np.zeros(self.fshape, dtype=np.complex128)

        self.Hx = Hx if Hx is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Hy = Hy if Hy is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Hz = Hz if Hz is not None else np.zeros(self.fshape, dtype=np.complex128)

        F1x =  self.Ey * np.conj(self.Hzm) - self.Ez * np.conj(self.Hym)
        F1y = -self.Ex * np.conj(self.Hzm) + self.Ez * np.conj(self.Hxm)
        F1z =  self.Ex * np.conj(self.Hym) - self.Ey * np.conj(self.Hxm)

        F2x =  self.Eym * np.conj(self.Hz) - self.Ezm * np.conj(self.Hy)
        F2y = -self.Exm * np.conj(self.Hz) + self.Ezm * np.conj(self.Hx)
        F2z =  self.Exm * np.conj(self.Hy) - self.Eym * np.conj(self.Hx)

        self.F1 = self.ds * np.sum(F1x * self.x_dot_s + F1y * self.y_dot_s + F1z * self.z_dot_s)
        self.F2 = self.ds * np.sum(F2x * self.x_dot_s + F2y * self.y_dot_s + F2z * self.z_dot_s)
        F = self.F1*self.F2 / self.Pm

        print self.F1/self.Pm, np.conj(self.F2/self.Pm)

        self.efficiency = 0.5 * (F + np.conj(F)).real

    def F1(self):
        return self.F1

    def F2(self):
        return self.F2

    def get_efficiency(self, P_in):
        return self.efficiency/2.0/P_in

    def get_dFdEx(self):
        return 0.5 * self.F2 / self.Pm * (-np.conj(self.Hzm)*self.y_dot_s + np.conj(self.Hym)*self.z_dot_s)

    def get_dFdEy(self):
        return 0.5 * self.F2 / self.Pm * (np.conj(self.Hzm)*self.x_dot_s - np.conj(self.Hxm)*self.z_dot_s)

    def get_dFdEz(self):
        return 0.5 * self.F2 / self.Pm * (-np.conj(self.Hym)*self.x_dot_s - np.conj(self.Hxm)*self.y_dot_s)

    def get_dFdHx(self):
        return 0.5 * np.conj(self.F1 / self.Pm) * (np.conj(self.Ezm)*self.y_dot_s - np.conj(self.Eym)*self.z_dot_s)

    def get_dFdHy(self):
        return 0.5 * np.conj(self.F1 / self.Pm) * (-np.conj(self.Ezm)*self.x_dot_s + np.conj(self.Exm)*self.z_dot_s)

    def get_dFdHz(self):
        return 0.5 * np.conj(self.F1 / self.Pm) * (-np.conj(self.Exm)*self.y_dot_s + np.conj(self.Eym)*self.x_dot_s)


def interpolated_dFdx_2D(dFdEz, dFdHx, dFdHy):
    """Account for interpolated fields in a 'naive' derivative of a figure of
    merit.

    In order to calculate any sort of quantity that involves power flow, we
    must interpolate the fields such that they are all known at the same point
    in space.  This process of interpolation must be handled very carefully in
    the contex of calculating gradients of a figure of merit.  In order to
    simplify this process and minimize the number of errors made, you can
    naively calculate the derivatives with respect to the interpolated fields
    and then compensate in order to ensure that the derivatives are correct
    with respect to the 'True' undelying shifted fields.

    Notes
    -----
    1. This function modifies the input arrays.

    2. dFdEz is not modified in the TE case

    Parameters
    ----------
    dFdEz : numpy.array
        2D numpy array containing derivative of FOM with respect to
        INTERPOLATED Ez
    dFdHx : numpy.array
        2D numpy array containing derivative of FOM with respect to
        INTERPOLATED Hx
    dFdHy : numpy.array
        2D numpy array containing derivative of FOM with respect to
        INTERPOLATED Hy
    x_fom : int or numpy.array
        x indices of figure of merit
    y_fom : int or numpy.array
        t indices of figure of merit

    Returns
    -------
    numpy.array, numpy.array, numpy.array
        The modified derivatives which account for interpolation
    """
    #dFdHx[y_fom-1, x_fom] += dFdHx[y_fom, x_fom]
    #dFdHy[y_fom, x_fom+1] += dFdHy[y_fom, x_fom]

    dFdHx[0:-2, 1:-1] += dFdHx[1:-1, 1:-1]
    dFdHy[1:-1, 2:] += dFdHy[1:-1, 1:-1]

    return dFdEz, dFdHx/2.0, dFdHy/2.0

def power_norm_dFdx_TE(sim, f, dfdEz, dfdHx, dfdHy):
    """Compute the derivative of a figure of merit which has power
    normalization.

    In many if not most cases of electromagnetic optimization, we will consider
    optimization problems in which we want to maximize some quantity which is
    normalized with respect to the total source power leaving a source,
    i.e.

            F(E,H) = f(E,H)/Psrc

    In some cases (e.g. a dipole emitting into a dielectric structure), the
    source power will depend on the shape of the structure itself. As a result,
    we need to account for this in our gradient calculations.

    Fortunately, we can easily account for this by writing our merit function
    as

            F(E,H) = f(E,H)/Psrc(E,H)

    and taking the necessary derivatives of Psrc. This process requires no deep
    knowledge about f.  It only needs f and its (numerical) derivative.

    Notes
    -----
    1. This function assumes f(E,H) is a function of the INTERPOLATED fields.
    All interpolation compensation is taken care of here (so don't call
    interp_dFdx on the dfdx's!!)

    Parameters
    ----------
    sim : gremlin.FDFD_TE
        simulation object which is needed in order to access field components
        as well as grid parameters
    f : float
        current value of merit function
    dfdEz : numpy.ndarray
        derivative w.r.t. Ez of non-normalized figure of merit
    dfdHx : numpy.ndarray
        derivative w.r.t. Hx of non-normalized figure of merit
    dfdHy : numpy.ndarray
        derivative w.r.t. Hy of non-normalized figure of merit

    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        The derivative of the full power-normalized figure of merit with
        interpolation accounted for.
    """
    Ezc = np.conj(sim.get_field_interp('Ez'))
    Hxc = np.conj(sim.get_field_interp('Hx'))
    Hyc = np.conj(sim.get_field_interp('Hy'))
    Psrc = sim.get_source_power()
    dx = sim.dx
    dy = sim.dy
    M = sim.M
    N = sim.N

    eps = sim.eps.get_values(0,M,0,N)
    mu = sim.mu.get_values(0,M,0,N)

    # get the planes through which power leaves the system
    w_pml_l = sim._w_pml_left
    w_pml_r = sim._w_pml_right
    w_pml_t = sim._w_pml_top
    w_pml_b = sim._w_pml_bottom

    x_bot = np.arange(w_pml_l, N-w_pml_r)
    y_bot = w_pml_b
    x_top = np.arange(w_pml_l, N-w_pml_r)
    y_top = M-w_pml_t

    x_left = w_pml_l
    y_left = np.arange(w_pml_b, M-w_pml_t)
    x_right = N-w_pml_r
    y_right = np.arange(w_pml_b, M-w_pml_t)

    x_all = np.arange(w_pml_l, N-w_pml_r)
    y_all = np.arange(w_pml_b, M-w_pml_t)
    y_all = y_all.reshape(y_all.shape[0], 1).astype(np.int)

    dPdEz = np.zeros([M, N], dtype=np.complex128)
    dPdHx = np.zeros([M, N], dtype=np.complex128)
    dPdHy = np.zeros([M, N], dtype=np.complex128)

    dPdEz[y_left, x_left]   += 0.25*dy*Hyc[y_left, x_left]
    dPdEz[y_top, x_top]     += 0.25*dx*Hxc[y_top, x_top]
    dPdEz[y_right, x_right] += -0.25*dy*Hyc[y_right, x_right]
    dPdEz[y_bot, x_bot]     += -0.25*dx*Hxc[y_bot, x_bot]
    dPdEz[y_all, x_all]     += 0.25*dx*dy*eps[y_all,x_all].imag*Ezc[y_all, x_all]

    dPdHx[y_top, x_top] += 0.25*dx*Ezc[y_top, x_top]
    dPdHx[y_bot, x_bot] += -0.25*dx*Ezc[y_bot, x_bot]
    dPdHx[y_all, x_all] += 0.25*dx*dy*mu[y_all,x_all].imag*Hxc[y_all, x_all]

    dPdHy[y_left, x_left]   += 0.25*dy*Ezc[y_left, x_left]
    dPdHy[y_right, x_right] += -0.25*dy*Ezc[y_right, x_right]
    dPdHy[y_all, x_all] += 0.25*dx*dy*mu[y_all,x_all].imag*Hyc[y_all, x_all]

    dFdEz = (Psrc * dfdEz - f * dPdEz) / Psrc**2
    dFdHx = (Psrc * dfdHx - f * dPdHx) / Psrc**2
    dFdHy = (Psrc * dfdHy - f * dPdHy) / Psrc**2

    dFdEz, dFdHx, dFdHy = interpolated_dFdx_2D(dFdEz, dFdHx, dFdHy)

    return dFdEz, dFdHx, dFdHy

def power_norm_dFdx_TM(sim, f, dfdHz, dfdEx, dfdEy):
    """Compute the derivative of a figure of merit which has power
    normalization.

    In many if not most cases of electromagnetic optimization, we will consider
    optimization problems in which we want to maximize some quantity which is
    normalized with respect to the total source power leaving a source,
    i.e.

            F(E,H) = f(E,H)/Psrc

    In some cases (e.g. a dipole emitting into a dielectric structure), the
    source power will depend on the shape of the structure itself. As a result,
    we need to account for this in our gradient calculations.

    Fortunately, we can easily account for this by writing our merit function
    as

            F(E,H) = f(E,H)/Psrc(E,H)

    and taking the necessary derivatives of Psrc. This process requires no deep
    knowledge about f.  It only needs f and its (numerical) derivative.

    Notes
    -----
    1. This function assumes f(E,H) is a function of the INTERPOLATED fields.
    All interpolation compensation is taken care of here (so don't call
    interp_dFdx on the dfdx's!!)

    Parameters
    ----------
    sim : gremlin.FDFD_TM
        simulation object which is needed in order to access field components
        as well as grid parameters
    f : float
        current value of merit function
    dfdEz : numpy.ndarray
        derivative w.r.t. Ez of non-normalized figure of merit
    dfdHx : numpy.ndarray
        derivative w.r.t. Hx of non-normalized figure of merit
    dfdHy : numpy.ndarray
        derivative w.r.t. Hy of non-normalized figure of merit

    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        The derivative of the full power-normalized figure of merit with
        interpolation accounted for.
    """
    Hzc = np.conj(sim.get_field_interp('Hz'))
    Exc = np.conj(sim.get_field_interp('Ex'))
    Eyc = np.conj(sim.get_field_interp('Ey'))
    Psrc = sim.get_source_power()
    dx = sim.dx
    dy = sim.dy
    M = sim.M
    N = sim.N

    eps = sim.eps.get_values(0,M,0,N)
    mu = sim.mu.get_values(0,M,0,N)

    # get the planes through which power leaves the system
    w_pml_l = sim._w_pml_left
    w_pml_r = sim._w_pml_right
    w_pml_t = sim._w_pml_top
    w_pml_b = sim._w_pml_bottom

    x_bot = np.arange(w_pml_l, N-w_pml_r)
    y_bot = w_pml_b
    x_top = np.arange(w_pml_l, N-w_pml_r)
    y_top = M-w_pml_t

    x_left = w_pml_l
    y_left = np.arange(w_pml_b, M-w_pml_t)
    x_right = N-w_pml_r
    y_right = np.arange(w_pml_b, M-w_pml_t)

    x_all = np.arange(w_pml_l, N-w_pml_r)
    y_all = np.arange(w_pml_b, M-w_pml_t)
    y_all = y_all.reshape(y_all.shape[0], 1).astype(np.int)

    dPdHz = np.zeros([M, N], dtype=np.complex128)
    dPdEx = np.zeros([M, N], dtype=np.complex128)
    dPdEy = np.zeros([M, N], dtype=np.complex128)

    dPdHz[y_left, x_left]   += -0.25*dy*Eyc[y_left, x_left]
    dPdHz[y_top, x_top]     += -0.25*dx*Exc[y_top, x_top]
    dPdHz[y_right, x_right] += 0.25*dy*Eyc[y_right, x_right]
    dPdHz[y_bot, x_bot]     += 0.25*dx*Exc[y_bot, x_bot]
    dPdHz[y_all, x_all]     += 0.25*dx*dy*mu[y_all,x_all].imag*Hzc[y_all, x_all]

    dPdEx[y_top, x_top] += -0.25*dx*Hzc[y_top, x_top]
    dPdEx[y_bot, x_bot] += +0.25*dx*Hzc[y_bot, x_bot]
    dPdEx[y_all, x_all] += 0.25*dx*dy*eps[y_all,x_all].imag*Exc[y_all, x_all]

    dPdEy[y_left, x_left]   += -0.25*dy*Hzc[y_left, x_left]
    dPdEy[y_right, x_right] += 0.25*dy*Hzc[y_right, x_right]
    dPdEy[y_all, x_all] += 0.25*dx*dy*eps[y_all,x_all].imag*Eyc[y_all, x_all]

    dFdHz = (Psrc * dfdHz - f * dPdHz) / Psrc**2
    dFdEx = (Psrc * dfdEx - f * dPdEx) / Psrc**2
    dFdEy = (Psrc * dfdEy - f * dPdEy) / Psrc**2

    dFdHz, dFdEx, dFdEy = interpolated_dFdx_2D(dFdHz, dFdEx, dFdEy)

    return dFdHz, dFdEx, dFdEy

def power_norm_dFdx(sim, f, dfdA1, dfdA2, dfdA3):
    if(type(sim) == fdfd.FDFD_TM):
        power_norm_dFdx_TM(sim, f, dfdAi, dfdA2, dfdA3)
    elif(type(sim) == fdfd.FDFD_TE):
        power_norm_dFdx_TE(sim, f, dfdAi, dfdA2, dfdA3)
