"""Demonstrate how to set up a simple simulation in emopt consisting of a
waveguide which is excited by a dipole current located at the center of the
waveguide.

On most *nix-based machines, run the script with:

    $ mpirun -n 8 python simple_waveguide.py

If you wish to increase the number of cores that the example is executed on,
change 8 to the desired number of cores.
"""

import emopt
from emopt.misc import NOT_PARALLEL

import numpy as np

####################################################################################
#Simulation Region parameters
####################################################################################
X = 10.0
Y = 3.5
wlen = 1.55
dx = 0.03
dy = 0.03
sim = emopt.solvers.Maxwell2DTE(X, Y, dx, dy, wlen)
sim.w_pml = [0.75, 0.75, 0.75, 0]
sim.bc = '0E'

# Get the actual width and height
# The true width/height will not necessarily match what we used when
# initializing the solver. This is the case when the width is not an integer
# multiple of the grid spacing used.
X = sim.X
Y = sim.Y
M = sim.M
N = sim.N

####################################################################################
# Setup system materials
####################################################################################
# Materials
n0 = 1.0
n1 = 3.0

# set a background permittivity of 1
eps_background = emopt.grid.Rectangle(X/2, 0, 2*X, Y)
eps_background.layer = 2
eps_background.material_value = n0**2

# Create a high index waveguide through the center of the simulation
h_wg = 0.5
waveguide = emopt.grid.Rectangle(X/2, 0, 2*X, h_wg)
waveguide.layer = 1
waveguide.material_value = n1**2

eps = emopt.grid.StructuredMaterial2D(X, Y, dx, dy)
eps.add_primitive(waveguide)
eps.add_primitive(eps_background)

mu = emopt.grid.ConstantMaterial2D(1.0)

# set the materials used for simulation
sim.set_materials(eps, mu)

####################################################################################
# setup the sources
####################################################################################
# setup the sources -- just a dipole in the center of the waveguide
Jz = np.zeros([M,N], dtype=np.complex128)
Mx = np.zeros([M,N], dtype=np.complex128)
My = np.zeros([M,N], dtype=np.complex128)
Jz[0, N//2] = 1.0

sim.set_sources((Jz, Mx, My))

####################################################################################
# Build and simulate
####################################################################################
sim.build()
sim.solve_forward()

# Get the fields we just solved for
sim_area = emopt.misc.DomainCoordinates(1.0, X-1.0, 0.0, Y-1.0, 0.0, 0.0, dx, dy, 1.0)
Ez = sim.get_field_interp('Ez', sim_area)

# Simulate the field.  Since we are running this using MPI, we only generate
# plots in the master process (otherwise we would end up with a bunch of
# plots...). This is accomplished using the NOT_PARALLEL flag which is defined
# by emopt.misc
if(NOT_PARALLEL):
    import matplotlib.pyplot as plt

    extent = sim_area.get_bounding_box()[0:4]
    Ez = np.flipud(Ez)

    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(Ez.real, extent=extent,
                            vmin=-np.max(Ez.real)/1.0,
                            vmax=np.max(Ez.real)/1.0,
                            cmap='seismic')
    ax.set_title('E$_z$', fontsize=18)
    ax.set_xlabel('x [um]', fontsize=14)
    ax.set_ylabel('y [um]', fontsize=14)
    f.colorbar(im)
    plt.show()
