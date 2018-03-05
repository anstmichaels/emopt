"""Demonstrate how to use the EMopt mode solver (for 2D problems with 1D
slices).

On most *nix-based machines, run the script with:

    $ mpirun -n 8 python wg_modes_2D.py

If you wish to increase the number of cores that the example is executed on,
change 8 to the desired number of cores.
"""
import emopt
from emopt.misc import NOT_PARALLEL

import numpy as np
from math import pi

####################################################################################
# Set up the size of the problem
####################################################################################
H = 6.0
dy = 0.01
N = int(np.ceil(H/dy)+1)
H = (N-1)*dy

wavelength = 1.55

####################################################################################
# Define the material distributions
####################################################################################
eps = np.ones(N, dtype=np.complex128)*1.444**2
mu = np.ones(N, dtype=np.complex128)

# Setup a waveguide by inserting values into eps
# Geometry is represented simply as an array of values. The physical distance
# between values in the array is given by dy.
w_wg = 2.0
x = np.arange(N)*dy
eps[(x >= H/2-w_wg/2) & (x <= H/2+w_wg/2)] = 2.5**2
eps[(x >= H/2-w_wg/6) & (x <= H/2+w_wg/6)] = 3.45**2

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
modes = emopt.modes.Mode_TE(wavelength, dy, eps, mu, n0=3.0, neigs=neigs)
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
        i = modes.find_mode_index(j)
        Ez = modes.get_field_interp(i, 'Ez')

        ax = axes[j]
        #ax = f.add_subplot(3,1,j+1)
        ax.plot(x, np.abs(Ez), linewidth=2)
        ax.set_ylabel('E$_z$ (TE$_%d$)' % j, fontsize=12)
        ax.set_xlim([x[0], x[-1]])

        ax2 = ax.twinx()
        ax2.plot(x, np.sqrt(eps.real), 'r--', linewidth=1, alpha=0.5)
        ax2.set_ylabel('Refractive Index')

    axes[2].set_xlabel('x [um]', fontsize=12)
    plt.show()

