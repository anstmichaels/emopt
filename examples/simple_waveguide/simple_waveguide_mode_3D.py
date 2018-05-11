"""Demonstrates how to run a simple 3D simulation using emopt.

This script sets up a waveguide and mode source and then plots a slice of the
resulting field.

To run the script run:

    $ mpirun -n 16 python simple_waveguide_mode_3D.py

If you want to run the script on a different number of processors, change 16 to
the desired value.

Furthermore, if you would like to monitor the process of the solver, you can
add the '-ksp_monitor_true_residual' command line argument:

    $ mpirun -n 16 python simple_waveguide_mode_3D.py -ksp_monitor_true_residual

and look at the last number that is printed each iteration. The simulation
terminates when this value drops below the specified rtol (which defaults to
1e-6)
"""

import emopt
from emopt.misc import NOT_PARALLEL

import numpy as np
from math import pi

from petsc4py import PETSc
from mpi4py import MPI

####################################################################################
# Simulation parameters
####################################################################################
X = 5.0
Y = 3.0
Z = 2.5
dx = 0.04
dy = 0.04
dz = 0.04

wavelength = 1.55

#####################################################################################
# Setup simulation
#####################################################################################
sim = emopt.fdfd.FDFD_3D(X,Y,Z,dx,dy,dz,wavelength, rtol=1e-5)
w_pml = dx * 15
sim.w_pml = [w_pml, w_pml, w_pml, w_pml, w_pml, w_pml]

X = sim.X
Y = sim.Y
Z = sim.Z

Nx = sim.Nx
Ny = sim.Ny
Nz = sim.Nz

#####################################################################################
# Define the geometry/materials
#####################################################################################
r1 = emopt.grid.Rectangle(X/2, Y/2, 2*X, 0.5); r1.layer = 1
r2 = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y); r2.layer = 2

r1.material_value = 3.45**2
r2.material_value = 1.444**2

eps = emopt.grid.StructuredMaterial3D(X, Y, Z, dx, dy, dz)
eps.add_primitive(r2, -Z, Z)
eps.add_primitive(r1, Z/2-0.11, Z/2+0.11)

mu = emopt.grid.ConstantMaterial3D(1.0)

sim.set_materials(eps, mu)
sim.build()

#####################################################################################
# Setup the sources
#####################################################################################
mode_slice = emopt.misc.DomainCoordinates(0.8, 0.8, w_pml, Y-w_pml, w_pml, Z-w_pml, dx, dy, dz)

mode = emopt.modes.ModeFullVector(wavelength, eps, mu, mode_slice, n0=3.45,
                                   neigs=4)
mode.build()
mode.solve()
sim.set_sources(mode, mode_slice)

#####################################################################################
# Simulate and view results
#####################################################################################
sim.solve_forward()

field_monitor = emopt.misc.DomainCoordinates(w_pml, X-w_pml, w_pml, Y-w_pml, Z/2, Z/2,
                                  dx, dy, dz)
Ey = sim.get_field_interp('Ey', domain=field_monitor, squeeze=True)

if(NOT_PARALLEL):
    import matplotlib.pyplot as plt

    eps_arr = eps.get_values_in(field_monitor, squeeze=True)

    vmax = np.max(np.real(Ey))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(np.real(Ey), extent=[0,X-2*w_pml,0,Y-2*w_pml], vmin=-vmax, vmax=vmax, cmap='seismic')
    plt.show()

