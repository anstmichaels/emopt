"""Demonstrate how to use the EMopt mode solver (for 2D problems with 1D
slices) with symmeric boundary conditions.

On most *nix-based machines, run the script with:

    $ mpirun -n 8 python wg_modes_2D_symmetry.py

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
H = 3.0
dy = 0.01
N = int(np.ceil(H/dy)+1)
H = (N-1)*dy

wavelength = 1.55

####################################################################################
# Define the material distributions
####################################################################################
w_wg_out = 2.0
w_wg_in = 0.4

# Define rectangles for the waveguide structure and cladding
wg_out = emopt.grid.Rectangle(0, 0, 1.0, w_wg_out)
wg_out.layer = 2; wg_out.material_value = 2.5**2

wg_in = emopt.grid.Rectangle(0, 0, 1.0, w_wg_in)
wg_in.layer = 1; wg_in.material_value = 3.45**2

bg = emopt.grid.Rectangle(0, 0, 1.0, 2*H)
bg.layer = 3; bg.material_value = 1.444**2

# Create a structured material which is just the ensemble of rectangles created above
# A slice from this StructuredMaterial will be used in the mode calculation
eps = emopt.grid.StructuredMaterial2D(1.0, H, dy, dy) # W and dx do not matter much
eps.add_primitive(wg_out); eps.add_primitive(wg_in); eps.add_primitive(bg)

mu = emopt.grid.ConstantMaterial2D(1.0)

# define a line along which a slice of the material distribution will be taken
# The modes will be solved for the slice.
mode_line = emopt.misc.DomainCoordinates(0.0, 0.0, 0, H, 0.0, 0.0, 1.0, dy, 1.0)

####################################################################################
# setup the mode solver
####################################################################################
# Solving for the modes of an electromagnetic structure involves expressing
# Maxwell's equations for a field with a known form as an eigenvalue problem.
# Because of the way that eigenvalues are compute numerically, we cannot be
# absolutely sure that the highest order *physical* modes will be the first 3
# eigenvectors that we find.  We thus solve for more vectors than we really
# need to be sure that we can pick out the desired modes.
neigs = 8
modes = emopt.modes.ModeTE(wavelength,eps, mu, mode_line, n0=3.0, neigs=neigs)

# set the boundary condition type. 'E' refers to electric field symmetry on the
# bottom y=0 boundary (i.e. the electric field is mirrored across the
# boundary). This is distinct from magnetic field symmetry ('H') which mirrors
# the magnetic field.
modes.bc = 'E'

modes.build() # build the eigenvalue problem internally
modes.solve() # solve for the effective indices and mode profiles

# Finally, we visualize the results. Because emopt relies on MPI to do
# parallelization, this script is actually duplicated and run many times in
# parallel. If we are not careful, we could end up generating a separate plot
# for every single processor. In order to avoid this, we only run on the
# "master" node, which is achieved using the NOT_PARALLEL flag.
if(NOT_PARALLEL):
    import matplotlib.pyplot as plt

    print('          n_eff          ')
    print('-------------------------')
    for j in range(neigs):
        n = modes.neff[j]
        print('%d : %0.4f  +  %0.4f i' % (j, n.real, n.imag))


    f, axes = plt.subplots(3,1)
    for j in range(3):
        i = modes.find_mode_index(j*2) # 2x factor because we want even modes
        Ez = modes.get_field_interp(i, 'Ez')
        x = np.linspace(0, H, N)
        eps_arr = eps.get_values_in(mode_line, squeeze=True)

        ax = axes[j]
        #ax = f.add_subplot(3,1,j+1)
        ax.plot(x, np.abs(Ez), linewidth=2)
        ax.set_ylabel('E$_z$ (TE$_%d$)' % j, fontsize=12)
        ax.set_xlim([x[0], x[-1]])

        ax2 = ax.twinx()
        ax2.plot(x, np.sqrt(eps_arr.real), 'r--', linewidth=1, alpha=0.5)
        ax2.set_ylabel('Refractive Index')

    axes[2].set_xlabel('x [um]', fontsize=12)
    plt.show()

