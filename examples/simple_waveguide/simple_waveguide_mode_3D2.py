import emopt
from emopt.misc import NOT_PARALLEL

import numpy as np
from math import pi

from petsc4py import PETSc
from mpi4py import MPI

####################################################################################
# Simulation parameters
####################################################################################
X = 3.0
Y = 3.0
Z = 3.0
dx = 0.03
dy = dx
dz = dx

wavelength = 1.55

#####################################################################################
# Setup simulation
#####################################################################################
sim = emopt.fdfd.FDFD_3D(X,Y,Z,dx,dy,dz,wavelength, rtol=1e-5)
sim.w_pml = [0.3, 0.3, 0.3, 0.3]

X = sim.X
Y = sim.Y
Z = sim.Z

Nx = sim.Nx
Ny = sim.Ny
Nz = sim.Nz

#####################################################################################
# Define the geometry/materials
#####################################################################################
r1 = emopt.grid.Rectangle(X/2, Y/2, 0.5, 0.22); r1.layer = 1
r2 = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y); r2.layer = 2

r1.material_value = 3.45**2
r2.material_value = 1.444**2

eps = emopt.grid.StructuredMaterial3D(X, Y, Z, dx, dy, dz)
eps.add_primitive(r2, -Z, Z)
eps.add_primitive(r1, -Z, Z)

mu = emopt.grid.ConstantMaterial3D(1.0)

sim.set_materials(eps, mu)
sim.build()

#####################################################################################
# Setup the sources
#####################################################################################
mode_slice = emopt.misc.DomainCoordinates(0.3, X-0.3, 0.3, Y-0.3, Z/2, Z/2, dx, dy, dz)

mode = emopt.modes.Mode_FullVector(wavelength, eps, mu, mode_slice, n0=3.45,
                                   neigs=4)
mode.build()
mode.solve()
Jxs, Jys, Jzs, Mxs, Mys, Mzs = mode.get_source(0, dx, dy ,dz)

Jxs = emopt.misc.COMM.bcast(Jxs, root=0)
Jys = emopt.misc.COMM.bcast(Jys, root=0)
Jzs = emopt.misc.COMM.bcast(Jzs, root=0)
Mxs = emopt.misc.COMM.bcast(Mxs, root=0)
Mys = emopt.misc.COMM.bcast(Mys, root=0)
Mzs = emopt.misc.COMM.bcast(Mzs, root=0)

Ex = mode.get_field_interp(0, 'Ex')
Ey = mode.get_field_interp(0, 'Ey')
Ez = mode.get_field_interp(0, 'Ez')
Hx = mode.get_field_interp(0, 'Hx')
Hy = mode.get_field_interp(0, 'Hy')
Hz = mode.get_field_interp(0, 'Hz')

#if(NOT_PARALLEL):
#    import pickle
#    eps_arr = eps.get_values_in(mode_slice, squeeze=True)
#
#    dout = {}
#    dout['eps'] = eps_arr
#    dout['Ex'] = Ex
#    dout['Ey'] = Ey
#    dout['Ez'] = Ez
#    dout['Hx'] = Hx
#    dout['Hy'] = Hy
#    dout['Hz'] = Hz
#    dout['Jx'] = Jxs
#    dout['Jy'] = Jys
#    dout['Jz'] = Jzs
#    dout['Mx'] = Mxs
#    dout['My'] = Mys
#    dout['Mz'] = Mzs
#
#    with open('debug_mode_2.p', 'wb') as fout:
#        pickle.dump(dout, fout)

#Ex = mode.get_field_interp(0, 'Ex')
#
#if(NOT_PARALLEL):
#    import matplotlib.pyplot as plt
#
#    #plt.imshow(Mxs.real, extent=[0, Y, 0, Z], cmap='seismic')
#    print mode.neff[0]
#    plt.imshow(np.abs(Ex), extent=[0, Y, 0, Z], cmap='hot')
#    plt.show()

Jx = np.zeros([Nz, Ny, Nx], dtype=np.complex128)
Jy = np.zeros([Nz, Ny, Nx], dtype=np.complex128)
Jz = np.zeros([Nz, Ny, Nx], dtype=np.complex128)
Mx = np.zeros([Nz, Ny, Nx], dtype=np.complex128)
My = np.zeros([Nz, Ny, Nx], dtype=np.complex128)
Mz = np.zeros([Nz, Ny, Nx], dtype=np.complex128)

Jx[mode_slice.i, mode_slice.j, mode_slice.k] = np.reshape(Jxs, [1, mode_slice.Ny, mode_slice.Nx])
Jy[mode_slice.i, mode_slice.j, mode_slice.k] = np.reshape(Jys, [1, mode_slice.Ny, mode_slice.Nx])
Jz[mode_slice.i, mode_slice.j, mode_slice.k] = np.reshape(Jzs, [1, mode_slice.Ny, mode_slice.Nx])
Mx[mode_slice.i, mode_slice.j, mode_slice.k] = np.reshape(Mxs, [1, mode_slice.Ny, mode_slice.Nx])
My[mode_slice.i, mode_slice.j, mode_slice.k] = np.reshape(Mys, [1, mode_slice.Ny, mode_slice.Nx])
Mz[mode_slice.i, mode_slice.j, mode_slice.k] = np.reshape(Mzs, [1, mode_slice.Ny, mode_slice.Nx])
#Jy[Nz/2, Ny/2, Nx/2] = 1.0
#Jx[Nz/2, Ny/2-8:Ny/2+8, Nx/2-8:Nx/2+8] = 3.5
#My[Nz/2, Ny/2-8:Ny/2+8, Nx/2-8:Nx/2+8] = 1.0

src = [Jx, Jy, Jz, Mx, My, Mz]
sim.set_sources(src)

sim.solve_forward()


#R = sim._Rst
#Interp = sim._Interp
#flr = Interp*(R*sim.x)
#scatter, flr_full = PETSc.Scatter.toZero(flr)
#scatter.scatter(flr, flr_full,False, PETSc.Scatter.Mode.FORWARD)

if(NOT_PARALLEL):
    #flr = flr_full
    fields = sim.fields

    Ez = fields[0::6]
    Ez = np.reshape(Ez, (Nz, Ny, Nx))

    #Ezlr = x2h[0::6]
    #Ezlr = np.reshape(Ezlr, (int(Nz/2), int(Ny/2), int(Nx/2)))

if(NOT_PARALLEL):
    import matplotlib.pyplot as plt

    #eps_slice = eps_grid[wg_slice.i, wg_slice.j, wg_slice.k].real
    #eps_slice = eps_slice[0, :, :]

    Ez_slice = Ez[:, Ny/2, :]
    #Ez_slice = Ez_slice[:, 0, :]

    vmax = np.max(np.real(Ez_slice))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(np.real(Ez_slice), extent=[0,X,0,Z], vmin=-vmax, vmax=vmax, cmap='seismic')
    plt.show()

    #Ez_slice = Ezlr[:, int(Ny/4), :]

    #vmax = np.max(np.real(Ez_slice))
    #f = plt.figure()
    #ax = f.add_subplot(111)
    #ax.imshow(np.real(Ez_slice), extent=[0,Nx,0,Nz], vmin=-vmax, vmax=vmax, cmap='seismic')
    #plt.show()
