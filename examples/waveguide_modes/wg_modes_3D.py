"""Solve for the fundamental TE-like mode without symmetry.

On most *nix-based machines, run the script with:

    $ mpirun -n 8 python wg_modes_3D.py

If you wish to increase the number of cores that the example is executed on,
change 8 to the desired number of cores.
"""
from __future__ import division, print_function, absolute_import
import emopt
from emopt.misc import NOT_PARALLEL

import numpy as np
from math import pi

####################################################################################
# Set up the size of the problem
####################################################################################
W = 3.0
H = 2.0
dx = 0.01
dy = 0.01
N = int(np.ceil(W/dx)+1)
M = int(np.ceil(H/dy)+1)
W = (N-1)*dx
H = (M-1)*dy

wavelength = 1.31

####################################################################################
# Define the material diststriputions!
####################################################################################
eps_Si = 3.506**2
eps_SiO2 = 1.4467**2

w_wg = 1.0
h_SOI = 0.22

# we need to set up the geometry of the cross section for which the mode will
# be computed. Here we use emopt.grid objects to define the permittivity and
# permeability of waveguiding structure for which the modes will be calculated
strip = emopt.grid.Rectangle(W/2, H/2, w_wg, h_SOI)
eps_bg = emopt.grid.Rectangle(W/2, H/2, 2*W, 2*H)

strip.layer = 1; strip.material_value = eps_Si
eps_bg.layer = 2; eps_bg.material_value = eps_SiO2

eps = emopt.grid.StructuredMaterial2D(W,H,dx,dy)
eps.add_primitive(strip); eps.add_primitive(eps_bg)

mu = emopt.grid.ConstantMaterial2D(1.0)

domain = emopt.misc.DomainCoordinates(0, W, 0, H, 0, 0, dx, dy, 1.0)

####################################################################################
# setup the mode solver
####################################################################################
neigs = 4
modes = emopt.modes.ModeFullVector(wavelength, eps, mu, domain, n0=np.sqrt(eps_Si), neigs=neigs)
modes.build() # build the eigenvalue problem internally
modes.solve() # solve for the effective indices and mode profiles

Ex = modes.get_field_interp(0, 'Ex')
if(NOT_PARALLEL):
    import matplotlib.pyplot as plt

    print(modes.neff[0])

    eps_arr = eps.get_values_in(domain)

    vmin = np.min(np.abs(Ex))
    vmax = np.max(np.abs(Ex))

    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(np.abs(Ex), extent=[0,W,0,H], vmin=vmin,
                     vmax=vmax, cmap='hot', origin='lower')
    ax.contour(eps_arr.real, levels=[eps_Si,], extent=[0,W,0,H],
                linewidths=[1,], colors=['w'], alpha=0.5)
    f.colorbar(im)
    plt.show()
