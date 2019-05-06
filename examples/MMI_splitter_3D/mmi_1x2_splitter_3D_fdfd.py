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
from emopt.adjoint_method import AdjointMethodPNF3D

import numpy as np
from math import pi

from petsc4py import PETSc
from mpi4py import MPI

class MMISplitterAdjointMethod(AdjointMethodPNF3D):
    """Define a figure of merit and its derivative for adjoint sensitivity
    analysis.

    Our goal is to optimize the the dimensions of an MMI 1x2 splitter using
    gradients computed using the adjoint method. In this problem, we choose the
    figure of merit (FOM) to be the overlap of the simulated fields with the
    super mode which has power equally split between the two waveguides. Both
    the evaluation of the FOM and its derivative with respect to E and H is
    handled by the emopt.fomutils.ModeMatch class.
    """

    def __init__(self, sim, mmi, fom_domain, mode_match):
        super(MMISplitterAdjointMethod, self).__init__(sim, step=1e-5)
        self.mmi = mmi
        self.mode_match = mode_match
        self.fom_domain = fom_domain

    def update_system(self, params):
        """Update the geometry of the system based on the current design
        parameters.

        The design parameter vector has the following format:

            params = [mmi_width, mmi_length]

        We use these values to modify the structure dimensions
        """
        self.mmi.height = params[0]
        self.mmi.width = params[1]

    @run_on_master
    def calc_f(self, sim, params):
        """Calculate the figure of merit.

        The FOM is the mode overlap between the simulated fields and the
        fundamental super mode of the output waveguides.
        """
        Ex, Ey, Ez, Hx, Hy, Hz = sim.saved_fields[0]

        self.mode_match.compute(Ex, Ey, Ez, Hx, Hy, Hz)
        fom = -1*self.mode_match.get_mode_match_forward(1.0)

        return fom

    @run_on_master
    def calc_dfdx(self, sim, params):
        """Calculate the figure of merit with respect to E and H.

        Note: our function is normalized with respect to the total source
        power and our derivative needs to account for this`*`. Currently,
        AdjointMethodPNF does not handle 3D simulations. For now, instead use
        emopt.fomutils.power_norm_dFdx_3D directly.

        `*` One might think that no power should couple back into the source.
        Unfortunately, due to the finite extent of the source, discretization,
        etc, this is not the case and modifications to the geometry will modify
        the source power.
        """
        Psrc = sim.source_power

        dfdEx = -1*self.mode_match.get_dFdEx()
        dfdEy = -1*self.mode_match.get_dFdEy()
        dfdEz = -1*self.mode_match.get_dFdEz()
        dfdHx = -1*self.mode_match.get_dFdHx()
        dfdHy = -1*self.mode_match.get_dFdHy()
        dfdHz = -1*self.mode_match.get_dFdHz()

        return [(dfdEx, dfdEy, dfdEz, dfdHx, dfdHy, dfdHz)]

    def get_fom_domains(self):
        """We must return the DomainCoordinates object that corresponds to our
        figure of merit. In theory, we could have many of these.
        """
        return [self.fom_domain]

    def calc_grad_p(self, sim, params):
        """Our FOM does not depend explicitly on the design parameters so we
        return zeros."""
        return np.zeros(len(params))


####################################################################################
# Simulation parameters
####################################################################################
X = 5.0   # simulation size along x
Y = 4.0/2 # simulation size along y
Z = 2.5/2 # simulation size along z
dx = 0.04 # grid spacing along x
dy = 0.04 # grid spacing along y
dz = 0.04 # grid spacing along z

wavelength = 1.55

#####################################################################################
# Setup simulation
#####################################################################################
# Setup the simulation--rtol tells the iterative solver when to stop. 5e-5
# yields reasonably accurate results/gradients
sim = emopt.fdfd.FDFD_3D(X,Y,Z,dx,dy,dz,wavelength, rtol=5e-5)
w_pml = dx * 15 # set the PML width

# we use symmetry boundary conditions at y=0 and z=0 to speed things up. We
# need to make sure to set the PML widths at these boundaries to zero!
sim.w_pml = [w_pml, w_pml, 0, w_pml, 0, w_pml]
sim.bc = '0HE'

# get actual simulation dimensions
X = sim.X
Y = sim.Y
Z = sim.Z

#####################################################################################
# Define the geometry/materials
#####################################################################################
# Geometry consists of input waveguide, output waveguide, and MMI splitting
# section. Structure is silicon clad in SiO2
w_wg = 0.45
L_in = X/2+1
L_out = X/2+1
L_mmi = 2.4
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

# Set the materials and build the system
sim.set_materials(eps, mu)
sim.build()

#####################################################################################
# Setup the sources
#####################################################################################
# We excite the system by injecting the fundamental mode of the input waveguide
input_slice = emopt.misc.DomainCoordinates(16*dx, 16*dx, 0, Y-w_pml, 0, Z-w_pml, dx, dy, dz)

mode = emopt.modes.ModeFullVector(wavelength, eps, mu, input_slice, n0=3.45,
                                   neigs=4)

# The mode boundary conditions should match the simulation boundary conditins.
# Mode is in the y-z plane, so the boundary conditions are HE
mode.bc = 'HE'
mode.build()
mode.solve()

sim.set_sources(mode, input_slice)

#####################################################################################
# Mode match for optimization
#####################################################################################
# we need to calculate the field used as the reference field in our mode match
# figure of merit calculation. This is the fundamental super mode of the output
# waveguides.
fom_slice = emopt.misc.DomainCoordinates(X-w_pml-dx, X-w_pml-dx, 0, Y-w_pml, 0, Z-w_pml, dx, dy, dz)

fom_mode = emopt.modes.ModeFullVector(wavelength, eps, mu, fom_slice, n0=3.45,
                                   neigs=4)

# Need to be consistent with boundary conditions!
fom_mode.bc = 'HE'
fom_mode.build()
fom_mode.solve()

# Retrieve the fields for the mode match
Exm = fom_mode.get_field_interp(0, 'Ex')
Eym = fom_mode.get_field_interp(0, 'Ey')
Ezm = fom_mode.get_field_interp(0, 'Ez')
Hxm = fom_mode.get_field_interp(0, 'Hx')
Hym = fom_mode.get_field_interp(0, 'Hy')
Hzm = fom_mode.get_field_interp(0, 'Hz')

mode_match = emopt.fomutils.ModeMatch([1,0,0], dy, dz, Exm, Eym, Ezm,
                                      Hxm, Hym, Hzm)

#####################################################################################
# Setup the AdjointMethod object needed for gradient calculations
#####################################################################################
sim.field_domains = [fom_slice]

am = MMISplitterAdjointMethod(sim, mmi, fom_slice, mode_match)
params = np.array([w_mmi, L_mmi])
am.check_gradient(params)

#####################################################################################
# Setup and run the optimization
#####################################################################################
# L-BFGS-B will print out the iteration number and FOM value
opt = emopt.optimizer.Optimizer(am, params, Nmax=10, opt_method='L-BFGS-B')
fom, pfinal = opt.run()

# run a simulation to make sure we visualize the correct data
am.fom(pfinal)

field_monitor = emopt.misc.DomainCoordinates(w_pml, X-w_pml, 0, Y-w_pml, 0, 0,
                                  dx, dy, dz)

# visualize the final results!
Ey = sim.get_field_interp('Ey', domain=field_monitor, squeeze=True)
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

