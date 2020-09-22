"""Demonstrate how to set up a simple simulation in emopt consisting of a
waveguide which is excited by a mode source which injects the fundamental mode
of the waveguide. This example also demonstrates how to use symmetry boundary
conditions to cut the simulation region in half.

On most *nix-based machines, run the script with:

    $ mpirun -n 8 python simple_waveguide_mode_symmetric.py

If you wish to increase the number of cores that the example is executed on,
change 8 to the desired number of cores.
"""
import emopt
from emopt.misc import NOT_PARALLEL

import numpy as np
from math import pi

####################################################################################
#Simulation Region parameters
####################################################################################
X = 10.0
Y = 2.0
dx = 0.02
dy = 0.02
wlen = 1.55

# set up TE simulation. TE refers to the field polarization which has E
# strictly perpendicular to the direction of propagation, i.e. E = Ez
sim = emopt.solvers.Maxwell2DTM(X, Y, dx, dy, wlen)
sim.w_pml = [wlen/2, wlen/2, wlen/2, 0]
sim.bc = '0H'

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
# Material constants
n0 = 1.44
n1 = 3.45

# set a background permittivity of 1
eps_background = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y)
eps_background.layer = 2
eps_background.material_value = n0**2

# Create a high index waveguide through the center of the simulation
h_wg = 0.5
waveguide = emopt.grid.Rectangle(X/2, 0, X*2, h_wg)
waveguide.layer = 1
waveguide.material_value = n1**2

# Create the a structured material which holds the waveguide and background
eps = emopt.grid.StructuredMaterial2D(X, Y, dx, dy)
eps.add_primitive(waveguide)
eps.add_primitive(eps_background)

# Basic magnetic materials are supported, however in most situations the
# permeability will be 1.
mu = emopt.grid.ConstantMaterial2D(1.0)

# set the materials used for simulation
sim.set_materials(eps, mu)

####################################################################################
# setup the sources
####################################################################################
# Specify a line of coordinates where the source is defined
src_line = emopt.misc.DomainCoordinates(2.0, 2.0, 0, 1.5, 0.0, 0.0, dx, dy, 1.0)

# setup, build the system, and solve
mode = emopt.solvers.Mode1DTM(wlen, eps, mu, src_line, n0=3.0, neigs=8)
mode.bc = 'H'
mode.build()
mode.solve()

# after solving, we cannot be sure which of the generated modes is the one we
# want.  We find the desired TE_X mode
mindex = mode.find_mode_index(0)
sim.set_sources(mode, src_domain=src_line, mindex=mindex)

####################################################################################
# Build and simulate
####################################################################################
sim.build()
sim.solve_forward()

# Get the fields we just solved for
sim_area = emopt.misc.DomainCoordinates(1.0, X-1.0, 0.0, Y-1.0, 0.0, 0.0, dx, dy, 1.0)
Hz = sim.get_field_interp('Hz', sim_area)

# Visualize the field.  Since we are running this using MPI, we only generate
# plots in the master process (otherwise we would end up with a bunch of
# plots...)
if(NOT_PARALLEL):
    import matplotlib.pyplot as plt

    extent = sim_area.get_bounding_box()[0:4]

    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(np.flipud(Hz.real), extent=extent,
                            vmin=-np.max(Hz.real)/1.0,
                            vmax=np.max(Hz.real)/1.0,
                            cmap='seismic')
    f.colorbar(im)
    ax.set_title('E$_z$', fontsize=18)
    ax.set_xlabel('x [um]', fontsize=14)
    ax.set_ylabel('y [um]', fontsize=14)
    plt.show()
