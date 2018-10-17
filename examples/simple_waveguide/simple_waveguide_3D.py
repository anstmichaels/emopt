from __future__ import division, print_function, absolute_import
import emopt
from emopt.misc import NOT_PARALLEL

import numpy as np
from math import pi

from petsc4py import PETSc

####################################################################################
# Simulation parameters
####################################################################################
X = 5.0
Y = 5.0
Z = 2.5
dx = 0.04
dy = dx
dz = dx

wavelength = 1.55

#####################################################################################
# Setup simulation
#####################################################################################
sim = emopt.fdtd.FDTD(X,Y,Z,dx,dy,dz,wavelength, rtol=1e-5, min_rindex=1.44)
w_pml = sim.w_pml[0]

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
# The source is a single dipole
src_point = emopt.misc.DomainCoordinates(X/2, X/2, Y/2, Y/2, Z/2, Z/2,
                                         dx, dy, dz)
Jx = np.zeros([1,1,1], dtype=np.complex128)
Jy = np.zeros([1,1,1], dtype=np.complex128)
Jz = np.zeros([1,1,1], dtype=np.complex128)
Mx = np.zeros([1,1,1], dtype=np.complex128)
My = np.zeros([1,1,1], dtype=np.complex128)
Mz = np.zeros([1,1,1], dtype=np.complex128)

Jy[0,0,0] = 1.0

src = [Jx, Jy, Jz, Mx, My, Mz]
sim.set_sources(src, src_point)

#####################################################################################
# Run the simulation and plot the results 
#####################################################################################
sim.solve_forward()

field_monitor = emopt.misc.DomainCoordinates(w_pml, X-w_pml, w_pml, Y-w_pml, Z/2, Z/2,
                                  dx, dy, dz)
Ey = sim.get_field_interp('Ey', domain=field_monitor, squeeze=True)

if(NOT_PARALLEL):
    import matplotlib.pyplot as plt

    vmax = np.max(np.real(Ey))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(np.real(Ey), extent=[0,X-w_pml*2,0,Y-w_pml*2], vmin=-vmax, vmax=vmax, cmap='seismic')
    plt.show()
