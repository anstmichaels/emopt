"""Demonstrates how optimize an MMI 1x2 splitter in 3D.

This optimization involves varying the width and height of a silicon slab in
order to fine tune multimode interference with the ultimate goal of splitting
light from a single input waveguide equally between two output waveguides:

        --------------------
        |                  -------------
        |                  -------------
---------        MMI       |
---------      Splitter    |
        |                  -------------
        |                  -------------
        --------------------

This structure should have pretty well-defined local optima. For a given MMI
splitter height, there will be a corresponding optimal length which produces
the desired relative phase between the fundamental and higher order modes.
Furthermore, we expect these optima to improve as the splitter gets larger
since the field concentrated at the outer edges/corners should be reduced
leading to less unwanted scattering. This behavior can be explored by choosing
different starting MMI widths and heights.

This optimization is setup to for the TE-like polarization. In order to design
a TM-like device, you should be able to just modify the symmetric boundary
conditions.

To run the script run:

    $ mpirun -n 16 python mmi_1x2_splitter_3D.py

If you want to run the script on a different number of processors, change 16 to
the desired value.

Furthermore, if you would like to monitor the process of the solver, you can
add the '-ksp_monitor_true_residual' command line argument:

    $ mpirun -n 16 python mmi_1x2_splitter_3D.py -ksp_monitor_true_residual

and look at the last number that is printed each iteration. The simulation
terminates when this value drops below the specified rtol (which defaults to
1e-6)
"""

import emopt
from emopt.misc import NOT_PARALLEL, run_on_master
from emopt.adjoint_method import AdjointMethod

import numpy as np
from math import pi

from petsc4py import PETSc
from mpi4py import MPI

class MMISplitterAdjointMethod(AdjointMethod):

    def __init__(self, sim, mmi, fom_domain, mode_match):
        super(MMISplitterAdjointMethod, self).__init__(sim, step=1e-5)
        self.mmi = mmi
        self.mode_match = mode_match
        self.fom_domain = fom_domain

    def update_system(self, params):
        self.mmi.height = params[0]

    @run_on_master
    def calc_fom(self, sim, params):
        Ex, Ey, Ez, Hx, Hy, Hz = sim.saved_fields[0]
        Psrc = sim.source_power

        self.mode_match.compute(Ex, Ey, Ez, Hx, Hy, Hz)
        fom = self.mode_match.get_mode_match_forward(1.0)
        print fom

        return fom

    @run_on_master
    def calc_dFdx(self, sim, params):
        Ex, Ey, Ez, Hx, Hy, Hz = sim.saved_fields[0]

        self.mode_match.compute(Ex, Ey, Ez, Hx, Hy, Hz)
        dFdEx = self.mode_match.get_dFdEx()
        dFdEy = self.mode_match.get_dFdEy()
        dFdEz = self.mode_match.get_dFdEz()
        dFdHx = self.mode_match.get_dFdHx()
        dFdHy = self.mode_match.get_dFdHy()
        dFdHz = self.mode_match.get_dFdHz()

        src = emopt.fomutils.interpolated_dFdx_3D(sim, self.fom_domain,
                                                  dFdEx, dFdEy, dFdEz,
                                                  dFdHx, dFdHy, dFdHz)


        return [[src[1:]], [src[0]]]

    def calc_grad_y(self, sim, params):
        return np.array([0])


####################################################################################
# Simulation parameters
####################################################################################
X = 5.0
Y = 4.0/2
Z = 2.5/2
dx = 0.04
dy = 0.04
dz = 0.04

wavelength = 1.55

#####################################################################################
# Setup simulation
#####################################################################################
sim = emopt.fdfd.FDFD_3D(X,Y,Z,dx,dy,dz,wavelength, rtol=5e-5)
w_pml = dx * 15
sim.w_pml = [w_pml, w_pml, 0, w_pml, 0, w_pml]
sim.bc = '0HE'

X = sim.X
Y = sim.Y
Z = sim.Z

Nx = sim.Nx
Ny = sim.Ny
Nz = sim.Nz

#####################################################################################
# Define the geometry/materials
#####################################################################################
w_wg = 0.45
L_in = X/2+1
L_out = X/2+1
L_mmi = 2.5
w_mmi = 1.75
h_si = 0.22

wg_in = emopt.grid.Rectangle(X/4, 0, L_in, w_wg); wg_in.layer = 1
mmi = emopt.grid.Rectangle(X/2, 0, L_mmi, w_mmi); mmi.layer = 1
wg_out = emopt.grid.Rectangle(3*X/4, w_wg, L_out, w_wg); wg_out.layer = 1
rbg = emopt.grid.Rectangle(X/2, 0, 2*X, 2*Y); rbg.layer = 2

wg_in.material_value = 3.45**2
mmi.material_value = 3.45**2
wg_out.material_value = 3.45**2
rbg.material_value = 1.444**2

eps = emopt.grid.StructuredMaterial3D(X, Y, Z, dx, dy, dz)
eps.add_primitive(wg_in, -h_si/2, h_si/2)
eps.add_primitive(mmi, -h_si/2, h_si/2)
eps.add_primitive(wg_out, -h_si/2, h_si/2)
eps.add_primitive(rbg, -Z, Z)

mu = emopt.grid.ConstantMaterial3D(1.0)

sim.set_materials(eps, mu)
sim.build()

#####################################################################################
# Setup the sources
#####################################################################################
input_slice = emopt.misc.DomainCoordinates(16*dx, 16*dx, 0, Y-w_pml, 0, Z-w_pml, dx, dy, dz)

mode = emopt.modes.ModeFullVector(wavelength, eps, mu, input_slice, n0=3.45,
                                   neigs=4)
mode.bc = 'HE'
mode.build()
mode.solve()

sim.set_sources(mode, input_slice)

#####################################################################################
# Mode match for optimization
#####################################################################################
fom_slice = emopt.misc.DomainCoordinates(X-w_pml-dx, X-w_pml-dx, 0, Y-w_pml, 0, Z-w_pml, dx, dy, dz)

fom_mode = emopt.modes.ModeFullVector(wavelength, eps, mu, fom_slice, n0=3.45,
                                   neigs=4)
fom_mode.bc = 'HE'
fom_mode.build()
fom_mode.solve()

Exm = fom_mode.get_field_interp(0, 'Ex')
Eym = fom_mode.get_field_interp(0, 'Ey')
Ezm = fom_mode.get_field_interp(0, 'Ez')
Hxm = fom_mode.get_field_interp(0, 'Hx')
Hym = fom_mode.get_field_interp(0, 'Hy')
Hzm = fom_mode.get_field_interp(0, 'Hz')

# In the current version of emopt, we need to manually reshape things to make
# the mode match compatible with set_adjoint_sources
if(NOT_PARALLEL):
    Nz, Ny = Exm.shape
    Exm = np.reshape(Exm, (Nz, Ny, 1))
    Eym = np.reshape(Eym, (Nz, Ny, 1))
    Ezm = np.reshape(Ezm, (Nz, Ny, 1))
    Hxm = np.reshape(Hxm, (Nz, Ny, 1))
    Hym = np.reshape(Hym, (Nz, Ny, 1))
    Hzm = np.reshape(Hzm, (Nz, Ny, 1))

mode_match = emopt.fomutils.ModeMatch([1,0,0], dy, dz, Exm, Eym, Ezm,
                                      Hxm, Hym, Hzm)

#####################################################################################
# Simulate and view results
#####################################################################################
sim.field_domains = [fom_slice]

am = MMISplitterAdjointMethod(sim, mmi, fom_slice, mode_match)
params = np.array([w_mmi])
#am.gradient(params)
am.check_gradient(params, plot=True)
#am.fom(params)

field_monitor = emopt.misc.DomainCoordinates(w_pml, X-w_pml, 0, Y-w_pml, 0, 0,
                                  dx, dy, dz)

Ey = sim.get_adjoint_field('Ey', domain=field_monitor, squeeze=True)
if(NOT_PARALLEL):
    import matplotlib.pyplot as plt

    # Mirror the electric field for nicer plotting :)
    Ey = np.concatenate([Ey[::-1], Ey], axis=0)

    eps_arr = eps.get_values_in(field_monitor, squeeze=True)
    vmax = np.max(np.real(Ey))
    f = plt.figure()
    ax1 = f.add_subplot(111)
    ax1.imshow(np.real(Ey), extent=[0,X-2*w_pml,0,2*Y-2*w_pml], vmin=-vmax, vmax=vmax, cmap='seismic')
    plt.show()

