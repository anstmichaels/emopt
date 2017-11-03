"""Calculate 2D waveguide modes and the corresponding current sources.
"""

import numpy as np
from math import pi
import scipy
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt

import pickle

from fdfd import *
from grid import *

__author__ = "Andrew Michaels"
__license__ = "Apache License, Version 2.0"
__version__ = "0.2"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

class WaveguideMode2D:

    eps0 = 8.854e-12
    mu0 = 4*pi*1e-7
    C = 3e8

    def __init__(self, eps1, eps2, eps3, d, wavelength):
        C = WaveguideMode2D.C
        eps0 = WaveguideMode2D.eps0
        mu0 = WaveguideMode2D.mu0

        self.eps1 = eps1*eps0
        self.eps2 = eps2*eps0
        self.eps3 = eps3*eps0
        self.d = d
        self.wavelength = wavelength
        self.omega = 2*pi*C/wavelength

        self.mu1 = mu0
        self.mu2 = mu0
        self.mu3 = mu0

    def get_TE_mode(self, ys, dx, dy, m=0):
        eps0 = WaveguideMode2D.eps0
        mu0 = WaveguideMode2D.mu0

        eps1 = self.eps1
        eps2 = self.eps2
        eps3 = self.eps3

        mu1 = self.mu1
        mu2 = self.mu2
        mu3 = self.mu3
        omega = self.omega

        alpha1 = lambda k2y : np.sqrt(omega**2 * (mu2*eps2 - mu1*eps1) - k2y**2 + 0j)
        alpha3 = lambda k2y : np.sqrt(omega**2 * (mu2*eps2 - mu3*eps3) - k2y**2 + 0j)
        k2y_condition = lambda k2y : np.abs(k2y*self.d - np.arctan(mu2*alpha1(k2y)/(mu1*k2y)) - \
                                            np.arctan(mu2*alpha3(k2y)/(mu3*k2y)) - m*pi)

        k2y0 = np.array([1e-4])
        result = minimize(k2y_condition, k2y0, method='Nelder-Mead', tol=1e-8)

        k2y = result.x[0]
        kz = np.sqrt(omega**2 * mu2 * eps2 - k2y**2)
        neff = kz / (2*pi/self.wavelength)

        phi = np.arctan(mu2/mu1*alpha1(k2y)/k2y)
        C1 = np.cos(phi)
        C3 = np.cos(phi - k2y*self.d)

        Ez = np.zeros(len(ys), dtype=np.complex128)
        Hx = np.zeros(len(ys), dtype=np.complex128)
        Hy = np.zeros(len(ys), dtype=np.complex128)

        Ez[ys>0] = C1 * np.exp(-alpha1(k2y)*ys[ys > 0])
        Ez[(ys<=0) & (ys >= -self.d)] = np.cos(k2y*ys[(ys<=0) & (ys >= -self.d)] + phi)
        Ez[ys<-self.d] = C3 * np.exp(alpha3(k2y) * (ys[ys<-self.d]+self.d))

        Hx[ys>0] = -1 / (1j*omega*mu1) * C1 * alpha1(k2y) * np.exp(-alpha1(k2y)*(ys[ys > 0]+dy/2.0))
        Hx[(ys<=0) & (ys >= -self.d)] = -1 / (1j*omega*mu2) * k2y * np.sin(k2y*(ys[(ys<=0) & (ys >= -self.d)]+dy/2.0) + phi)
        Hx[ys<-self.d] = 1 / (1j*omega*mu3) * C3 * alpha3(k2y) * np.exp(alpha3(k2y) * (ys[ys<-self.d]+self.d+dy/2.0))

        Hy[ys>0] = -C1 * np.exp(-alpha1(k2y)*ys[ys > 0]) * 1 / (1j*omega*mu1) * 1j * kz * np.exp(-1j*kz*dx/2.0)
        Hy[(ys<=0) & (ys >= -self.d)] = -np.cos(k2y*ys[(ys<=0) & (ys >= -self.d)] + phi) * 1 / (1j*omega*mu2) * 1j * kz * np.exp(-1j*kz*dx/2.0)
        Hy[ys<-self.d] = -C3 * np.exp(alpha3(k2y) * (ys[ys<-self.d]+self.d)) * 1 / (1j*omega*mu3) * 1j * kz * np.exp(-1j*kz*dx/2.0)

        # Danger!!! These fields have units.  We need to de-unit-ify them!
        # We set the electric field to have an amplitude of 1 V/m so the unitless is equal to the unit version
        # unitless H and normal H are related by the impedance of free space
        # TODO: Reevaluate the previous expressions using unitless time harmonic Maxwell's equations
        Hx *= np.sqrt(mu0/eps0)
        Hy *= np.sqrt(mu0/eps0)

        return Ez, Hx, Hy, neff

    def get_TM_mode(self, ys, dx, dy, m=0):
        eps0 = WaveguideMode2D.eps0
        mu0 = WaveguideMode2D.mu0

        # Take advantage of symmetry of Maxwell's equations: Swap eps and mu and E with H and H with -E
        # and use TE mode solution
        eps1_prev = self.eps1
        eps2_prev = self.eps2
        eps3_prev = self.eps3

        self.eps1 = mu0
        self.eps2 = mu0
        self.eps3 = mu0

        self.mu1 = eps1_prev
        self.mu2 = eps2_prev
        self.mu3 = eps3_prev

        Hz, Ex, Ey, neff = self.get_TE_mode(ys, dx, dy, m=m)

        # fix the unit normalization
        Ex /= np.sqrt(mu0/eps0)
        Ey /= np.sqrt(mu0/eps0)
        Hz *= np.sqrt(mu0/eps0)

        # return eps and mu to original values
        self.eps1 = eps1_prev
        self.eps2 = eps2_prev
        self.eps3 = eps3_prev

        self.mu1 = mu0
        self.mu2 = mu0
        self.mu3 = mu0

        return Hz, -Ex, -Ey, neff


## Generate mode data for a 2D TE waveguide 
# Parameters:
# eps_core     = permitivity of core
# eps_clad     = permitivity of cladding
# wgh          = waveguide height
# wavelength   = wavelength [um]
# dx           = grid size along x [um]
# dy           = grid size along y [um]
# srcg         = height of source (> wavelength/2.0) [um]
#
# NOTE: unlike most other code, this function assumes all length units are in micrometers
def gen_mode_data_TE(eps_core, eps_clad, wgh, wavelength, dx, dy, srch, modenum=0):
    H = 2.0*srch
    W = wavelength*4.0
    PML_W = wavelength/np.sqrt(eps_clad)
    srch = int(srch/dy)

    M = int(W/dx)
    N = int(H/dy)

    # Define a single mode waveguide that we will excite
    wg = Rectangle(W/2, H/2, W, wgh)

    # set the background material using a rectangle equal in size to the system
    background = Rectangle(W/2,H/2,W,H)

    # set the relative layers of the permitivity primitives
    wg.set_layer(1)
    background.set_layer(2)

    # set the complex permitivies of each shape
    # the waveguide is Silicon clad in SiO2
    wg.set_material(eps_core)
    background.set_material(eps_clad)

    # assembled the primitives in a StructuredMaterial to be used by the FDFD solver
    eps = StructuredMaterial(W,H,dx,dy)
    eps.add_primitive(wg)
    eps.add_primitive(background)

    # set up the magnetic permeability -- just 1.0 everywhere
    mu_background = Rectangle(W/2,H/2,W,H)
    mu_background.set_material(1.0)
    mu_background.set_layer(1)
    mu = StructuredMaterial(W,H,dx,dy)
    mu.add_primitive(mu_background)

    H = (N-1)*dy
    ys = np.arange(0, N, 1)*dy - H/2.0 - wgh/2.0

    mode = WaveguideMode2D(eps_clad, eps_core, eps_clad, wgh*1e-6, wavelength*1e-6)
    Ezm, Hxm, Hym, neff = mode.get_TE_mode(ys*1e-6, dx*1e-6, dy*1e-6, m=modenum)

    # copy the generated mode field profile into a matrix and "extrude" the field profile

    # we need to get the permittivity
    #eps_grid = np.zeros([N,M], dtype=np.complex128)
    #for y in range(N):
    #    for x in range(M):
    #        eps_grid[y,x] = eps.get_value(x,y)


    JEz = np.zeros([N,M], dtype=np.complex64)
    JHx = np.zeros([N,M], dtype=np.complex64)
    JHy = np.zeros([N,M], dtype=np.complex64)

    #dHxdy = -np.copy(Hxm)
    #dHxdy[0:-1] += Hxm[1:]
    #dHxdy = dHxdy / dy

    x_src = int(PML_W/dx)+1
    y_src = np.arange(int(N/2-srch/2.0), int(N/2+srch/2.0), 1)

    #JEz[:, i] = Hym/dx - dHxdy + 1j*eps_grid[:,i]*Ezm
    #JHy[:, i-1] = -Ezm/dx
    JEz[y_src, x_src] = Hym[y_src]/dx
    JHy[y_src, x_src] = -Ezm[y_src]/dx

    return Ezm[y_src], Hxm[y_src], Hym[y_src], JEz[y_src, x_src], JHx[y_src, x_src], JHy[y_src, x_src], neff

## Generate mode data for a 2D TM waveguide 
# Parameters:
# eps_core     = permitivity of core
# eps_clad     = permitivity of cladding
# wgh          = waveguide height
# wavelength   = wavelength [desired unit]
# dx           = grid size along x [desired unit]
# dy           = grid size along y [desired unit]
# srcg         = height of source (> wavelength/2.0) [desired unit]
def gen_mode_data_TM(eps_core, eps_clad, wgh, wavelength, dx, dy, srch, plot_result=False, modenum=0):
    H = 2.0*srch
    W = wavelength*4.0
    PML_W = wavelength/np.sqrt(eps_clad)
    srch = int(srch/dy)

    M = int(W/dx)
    N = int(H/dy)

    # Define a single mode waveguide that we will excite
    wg = Rectangle(W/2, H/2, W, wgh)

    # set the background material using a rectangle equal in size to the system
    background = Rectangle(W/2,H/2,W,H)

    # set the relative layers of the permitivity primitives
    wg.set_layer(1)
    background.set_layer(2)

    # set the complex permitivies of each shape
    # the waveguide is Silicon clad in SiO2
    wg.set_material(eps_core)
    background.set_material(eps_clad)

    # assembled the primitives in a StructuredMaterial to be used by the FDFD solver
    eps = StructuredMaterial(W,H,dx,dy)
    eps.add_primitive(wg)
    eps.add_primitive(background)

    # set up the magnetic permeability -- just 1.0 everywhere
    mu_background = Rectangle(W/2,H/2,W,H)
    mu_background.set_material(1.0)
    mu_background.set_layer(1)
    mu = StructuredMaterial(W,H,dx,dy)
    mu.add_primitive(mu_background)


    H = (N-1)*dy
    ys = np.arange(0, N, 1)*dy - H/2.0 - wgh/2.0

    mode = WaveguideMode2D(eps_clad, eps_core, eps_clad, wgh*1e-6, wavelength*1e-6)
    Hzm, Exm, Eym, neff = mode.get_TM_mode(ys*1e-6, dx*1e-6, dy*1e-6, m=modenum)

    #ys += dy/2.0

    #_, _, Eym, neff = mode.get_TM_mode(ys*1e-6, dx*1e-6, dy*1e-6, m=modenum)
    # copy the generated mode field profile into a matrix and "extrude" the field profile

    # we need to get the permittivity
    #eps_grid = np.zeros([N,M], dtype=np.complex128)
    #for y in range(N):
    #    for x in range(M):
    #        eps_grid[y,x] = eps.get_value(x,y)

    JHz = np.zeros([N,M], dtype=np.complex128)
    JEx = np.zeros([N,M], dtype=np.complex128)
    JEy = np.zeros([N,M], dtype=np.complex128)

    #dExdy = -np.copy(Exm)
    #dExdy[0:-1] += Exm[1:]
    #dExdy = dExdy / dy

    x_src = int(PML_W/dx)+10
    y_src = np.arange(int(N/2-srch/2.0), int(N/2+srch/2.0), 1)

    #JHz[y_src, x_src] = Eym[y_src]/dx - dExdy[y_src] - 1j * 1.0 * Hzm[y_src]
    JHz[y_src, x_src] = Eym[y_src]/dx
    #JEy[y_src, x_src] = Hzm[y_src]/dx - 1j*eps_grid[y_src, x_src]*Eym[y_src]
    JEy[y_src, x_src] = Hzm[y_src]/dx

    return  Hzm[y_src], Exm[y_src], Eym[y_src], JHz[y_src, x_src], JEx[y_src, x_src], JEy[y_src, x_src], neff
