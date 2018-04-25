"""
    File name: mode_data.py
    Author: Andrew Michaels
    Date last modified: 5/21/2017
    Python Version: 2.7+

    Generate electric and magnetic field for Gaussian beam.
"""

import numpy as np
from math import pi
import scipy
import matplotlib.pyplot as plt
import time

import pickle

# Define the desired Gaussian modes of the system
def Ez_Gauss(x, x0, w0, theta, wavelength, n):
    k = 2*pi*n/wavelength
    sinphi = np.sin(pi/2 - theta)
    mu0 = 4*pi*1e-7
    c = 3e8

    E0 = np.exp(-(x-x0)**2 * sinphi**2 / w0**2)

    #E0 = np.sqrt(2*mu0*c / (b*np.sqrt(2*pi))) * np.exp(-(x-x0)**2 *
    #                                                   sinphi**2 / w0**2)

    Ezm = E0 * np.exp(1j * k * np.sin(theta) * x)

    return Ezm

def Hx_Gauss(x, x0, w0, theta, wavelength, n):
    mu0 = 4*pi*1e-7
    c = 3e8

    Hxm = Ez_Gauss(x, x0, w0, theta, wavelength, n) * np.cos(theta) * n
    return Hxm 

def Hy_Gauss(x, x0, w0, theta, wavelength, n):
    mu0 = 4*pi*1e-7
    c = 3e8

    Hym = -1*Ez_Gauss(x, x0, w0, theta, wavelength, n) * np.sin(theta) * n
    return Hym
