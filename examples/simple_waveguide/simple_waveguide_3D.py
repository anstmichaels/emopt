import emopt.fdfd
from emopt.fdfd import FDFD_3D

from emopt.grid import GridMaterial3D
from emopt.misc import info_message, warning_message, error_message, RANK, \
NOT_PARALLEL, run_on_master, DomainCoordinates

import numpy as np
from math import pi

X = 2.2
Y = 2.2
Z = 3.2
dx = 0.02
dy = dx
dz = dx

wavelength = 1.55

sim = FDFD_3D(X,Y,Z,dx,dy,dz,wavelength, rtol=1e-6)
#sim = FDFD_3D(X,Y,Z,dx,dy,dz,wavelength)
sim.w_pml = [0,0,0,0,0,0]

X = sim.X
Y = sim.Y
Z = sim.Z

Nx = sim.Nx
Ny = sim.Ny
Nz = sim.Nz

eps_grid = np.ones([Nz, Ny, Nx], dtype=np.complex128)*(-1j*10.0)#(1.444+0.0j)**2
mu_grid = np.ones([Nz, Ny, Nx], dtype=np.complex128)

h_wg = 0.3
w_wg = 0.4

wg_vol = DomainCoordinates(X/2.0-w_wg/2.0, X/2.0+w_wg/2.0, Y/2.0-h_wg/2.0,
                           Y/2.0+h_wg/2.0, 0, Z, dx, dy, dz)

wg_slice = DomainCoordinates(0, X, 0, Y, Z/2.0, Z/2.0, dx, dy, dz)
f_slice = DomainCoordinates(0, X, Y/2.0, Y/2.0, 0, Z, dx, dy, dz)

#eps_grid[ wg_vol.i, wg_vol.j, wg_vol.k] = 3.45**2+0.0j
eps = GridMaterial3D(X, Y, Z, Nx, Ny, Nz, eps_grid)
mu = GridMaterial3D(X, Y, Z, Nx, Ny, Nz, mu_grid)

sim.set_materials(eps, mu)
sim.build()

Jx = np.zeros([Nz, Ny, Nx], dtype=np.complex128)
Jy = np.zeros([Nz, Ny, Nx], dtype=np.complex128)
Jz = np.zeros([Nz, Ny, Nx], dtype=np.complex128)
Mx = np.zeros([Nz, Ny, Nx], dtype=np.complex128)
My = np.zeros([Nz, Ny, Nx], dtype=np.complex128)
Mz = np.zeros([Nz, Ny, Nx], dtype=np.complex128)

Jx[Nz/2, Ny/2, Nx/2] = 1.0

src = [Jx, Jy, Jz, Mx, My, Mz]
sim.set_sources(src)

sim.solve_forward()

if(NOT_PARALLEL):
    fields = sim.fields
    Ez = fields[0::6]
    Ez = np.reshape(Ez, (Nz, Ny, Nx))

#if(NOT_PARALLEL):
#    import matplotlib.pyplot as plt
#
#    eps_slice = eps_grid[wg_slice.i, wg_slice.j, wg_slice.k].real
#    eps_slice = eps_slice[0, :, :]
#
#    Ez_slice = Ez[:, f_slice.j, :]
#    Ez_slice = Ez_slice[:, 0, :]
#
#
#    vmax = np.max(np.real(Ez_slice))
#    f = plt.figure()
#    ax = f.add_subplot(111)
#    ax.imshow(np.real(Ez_slice), extent=[0,X,0,Z], vmin=-vmax, vmax=vmax, cmap='seismic')
#    plt.show()
