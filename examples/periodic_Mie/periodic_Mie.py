"""Demonstrate how to use periodic boundary conditions to simulate Mie
scattering of a plane wave off of a cylinder.

On most *nix-based machines, run the script with:

    $ mpirun -n 8 python periodic_Mie_x.py

If you wish to increase the number of cores that the example is executed on,
change 8 to the desired number of cores.
"""

import emopt.fdfd
from emopt.fdfd import FDFD_TE

from emopt.grid import StructuredMaterial, Rectangle, Polygon
from emopt.misc import info_message, warning_message, error_message, RANK, \
NOT_PARALLEL, run_on_master, PlaneCoordinates, LineCoordinates

import numpy as np
from math import pi

####################################################################################
#Simulation Region parameters
####################################################################################
W = 5.0
H = 15.0
dx = 0.02
dy = 0.02
wlen = 1.55
sim = FDFD_TE(W, H, dx, dy, wlen)

# planewave incident along x
sim.w_pml = [0.75, 0.75, 0., 0.]
sim.bc = '0P'

# planewave incident along y
#sim.w_pml = [0., 0., 0.75, 0.75]
#sim.bc = 'P0'

# Get the actual width and height
# The true width/height will not necessarily match what we used when
# initializing the solver. This is the case when the width is not an integer
# multiple of the grid spacing used.
W = sim.W
H = sim.H
M = sim.M
N = sim.N

####################################################################################
# Setup system materials
####################################################################################
# Materials
n0 = 1.0
n1 = 1.444

# set a background permittivity of 1
eps_background = Rectangle(W/2, 0, 2*W, H)
eps_background.layer = 2
eps_background.material_value = n0**2

# Create a high index circle in the middle of the simulation
R = 0.5
x0 = W/2
y0 = H/2
theta = np.linspace(0,2*pi,100)
xs = x0 + R*np.cos(theta)
ys = y0 + R*np.sin(theta)
cyl = Polygon()
cyl.set_points(xs,ys)
cyl.layer = 1
cyl.material_value = n1**2

eps = StructuredMaterial(W, H, dx, dy)
eps.add_primitive(cyl)
eps.add_primitive(eps_background)

mu_background = Rectangle(W/2, H/2, 2*W, H)
mu_background.layer = 2
mu_background.material_value = 1.0

mu = StructuredMaterial(W, H, dx, dy)
mu.add_primitive(mu_background)

# set the materials used for simulation
sim.set_materials(eps, mu)

####################################################################################
# setup the sources
####################################################################################
# setup the sources -- just a dipole in the center of the waveguide
Jz = np.zeros([M,N], dtype=np.complex128)
Mx = np.zeros([M,N], dtype=np.complex128)
My = np.zeros([M,N], dtype=np.complex128)

# planewave incident along x
Jz[sim.w_pml_bottom:M-sim.w_pml_top, sim.w_pml_left+1] = 1.0
My[sim.w_pml_bottom:M-sim.w_pml_top, sim.w_pml_left+1] = 1.0

# planewave incident along y
#Jz[sim.w_pml_bottom+1, sim.w_pml_left:N-sim.w_pml_right] = 1.0
#Mx[sim.w_pml_bottom+1, sim.w_pml_left:N-sim.w_pml_right] = -1.0


sim.set_sources((Jz, Mx, My))

####################################################################################
# Build and simulate
####################################################################################
sim.build()
sim.solve_forward()

# Get the fields we just solved for
sim_area = PlaneCoordinates('z', .0, W-.0, .0, H-.0, dx, dy)
Ez = sim.get_field_interp('Ez', sim_area)

# Simulate the field.  Since we are running this using MPI, we only generate
# plots in the master process (otherwise we would end up with a bunch of
# plots...). This is accomplished using the NOT_PARALLEL flag which is defined
# by emopt.misc
if(NOT_PARALLEL):
    import matplotlib.pyplot as plt

    extent = sim_area.get_bounding_box()[0:4]
    Ez = np.flipud(Ez)
    eps_grid = eps.get_values_on(sim_area)

    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(Ez.real, extent=extent,
                            vmin=-np.max(Ez.real)/1.0,
                            vmax=np.max(Ez.real)/1.0,
                            cmap='seismic')
    ax.plot(xs,ys,linewidth=1)
    ax.set_title('E$_z$', fontsize=18)
    ax.set_xlabel('x [um]', fontsize=14)
    ax.set_ylabel('y [um]', fontsize=14)
    f.colorbar(im)
    plt.show()
