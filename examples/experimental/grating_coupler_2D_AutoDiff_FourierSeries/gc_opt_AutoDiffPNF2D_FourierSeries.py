"""Fourier-series parameterized grating coupler optimized with
AutoDiff-compatible geometry definitions.

Because we use rectangles to define the scattering elements, we may
exactly approximate conventional boundary smoothing with AutoDiff-
compatible geometry elements, using a piecewise linear boundary function.

To test the corresponds, the user may specify either the standard EMopt
representation or the AutoDiff representation of the blazed grating in
the command line using the --version flag. Furthermore, the permittivty
distributions can be directly compared using the --test flag.

Example usage:
mpirun -n 8 python g_opt_2D_AutoDiffPNF2D_BlazedGrating.py
mpirun -n 8 python g_opt_2D_AutoDiffPNF2D_BlazedGrating.py --version 'AutoDiff'
mpirun -n 8 python g_opt_2D_AutoDiffPNF2D_BlazedGrating.py --version 'Standard'
mpirun -n 8 python g_opt_2D_AutoDiffPNF2D_BlazedGrating.py --version 'AutoDiff' --test True
"""
import time
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import torch

import emopt
from emopt.misc import NOT_PARALLEL
from emopt.experimental.adjoint_method import AutoDiffPNF2D
from emopt.experimental.fdfd import FDFD_TE
from emopt.experimental.grid import HybridMaterial2D, AutoDiffMaterial2D
import emopt.experimental.autodiff_geometry as adg
from emopt.experimental.optimizer import TimedOptimizer

STEP = 1e-8

# Import the library
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Ng', type=int, default=30)
parser.add_argument('--Nc', type=int, default=5)
args = parser.parse_args()

###################################################################
# EMopt-AutoDiff
###################################################################

nl = adg.nl_lin

def fseries(coeffs, Nc, Ng):
    # Compute Fourier series sums given coefficients
    ii = torch.arange(Ng).view(-1,1)
    jj = torch.arange(Nc).view(1,-1)
    sins = torch.sin(np.pi/2.0 * ii * jj * 1.0 / Ng)
    sins[:,0] = 1.0
    coss = torch.cos(np.pi/2.0 * ii * jj * 1.0 / Ng)
    retval = coeffs[:Nc].view(1,-1) * sins + coeffs[Nc:].view(1,-1) * coss
    return retval.sum(-1)

def build_rects(x, periods, widths, Ng, k, w_wg_input):
    # build rects corresponding to etches in grating
    rects = []
    pos = w_wg_input
    for i in range(Ng):
        rects.append(adg.rect1d(k, x, [pos, pos+widths[i]], nl=nl))
        pos = pos + periods[i]
    return adg.union(rects)

def create_eps_grid(v, coords, Nc, Ng, k, w_in, eps_l, eps_h, wg_y_min, wg_y_max, bg=None):
    y, x = coords

    # Get widths and periods from Fourier decomposition over parameters
    widths = fseries(v[:2*Nc], Nc, Ng)
    periods = fseries(v[2*Nc:-3], Nc, Ng)

    # Get other parameters
    wg_input_x = w_in + v[-1]
    h_etch = v[-2]
    box_height = v[-3]

    # Define the substrate
    subs = adg.step1d(k, y, wg_y_min - box_height, reverse=True, nl=nl).view(-1,1)

    # Define the waveguide (extruded 1d step function)
    wg_xmax = wg_input_x + periods.sum()
    wg = adg.step1d(k, x, wg_xmax, reverse=True, nl=nl)
    wg = adg.depth(wg, k, y, [wg_y_min, wg_y_max], nl=nl)

    # Define the etches (each constrained to have same depth)
    SiO2_rects = build_rects(x, periods, widths, Ng, k, wg_input_x)
    SiO2_rects = adg.depth(SiO2_rects, k, y, [wg_y_max - h_etch, wg_y_max], nl=nl)

    # Put the total shape together
    shape = subs + wg - SiO2_rects
    eps = eps_l + (eps_h - eps_l) * shape

    return eps

class SiliconGratingAutograd(AutoDiffPNF2D):
    def __init__(self, sim, domain, mm_line):
        super().__init__(sim, domain=domain, update_mu=False)
        self.current_fom = 0.0

        # save the variables for later
        self.mm_line = mm_line

        theta = 8.0/180.0*pi
        match_w0 = 5.2
        match_center = 13.0

        Ezm, Hxm, Hym = emopt.misc.gaussian_mode(mm_line.x-match_center,
                                                 0.0, match_w0,
                                                 theta, sim.wavelength,
                                                 np.sqrt(eps_clad))

        self.mode_match = emopt.fomutils.ModeMatch([0,1,0], sim.dx, Ezm=Ezm, Hxm=Hxm, Hym=Hym)

        self.current_fom = 0.0

    def calc_f(self, sim, params):
        # Get the fields which were recorded
        Ez, Hx, Hy = sim.saved_fields[0]

        # compute the mode match efficiency
        self.mode_match.compute(Ez=Ez, Hx=Hx, Hy=Hy)

        # we want to maximize the efficiency, so we minimize the negative of the efficiency
        self.current_fom = -self.mode_match.get_mode_match_forward(1.0)
        return self.current_fom

    def calc_dfdx(self, sim, params):
        dFdEz = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dFdHx = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dFdHy = np.zeros([sim.M, sim.N], dtype=np.complex128)

        # Get the fields which were recorded
        Ez, Hx, Hy = sim.saved_fields[0]

        self.mode_match.compute(Ez=Ez, Hx=Hx, Hy=Hy)

        dFdEz[self.mm_line.j, self.mm_line.k] = -self.mode_match.get_dFdEz()
        dFdHx[self.mm_line.j, self.mm_line.k] = -self.mode_match.get_dFdHx()
        dFdHy[self.mm_line.j, self.mm_line.k] = -self.mode_match.get_dFdHy()

        return (dFdEz, dFdHx, dFdHy)

    def calc_grad_p(self, sim, params):
        return np.zeros(params.shape)


def plot_update(params, fom_list, sim, am):
    """Save a snapshot of the current state of the structure.

    This function is passed to an Optimizer object and is called after each
    iteration of the optimization. It plots the current refractive index
    distribution, the electric field, and the full figure of merit history.
    """
    print('Finished iteration %d' % (len(fom_list)+1))
    current_fom = -1*am.calc_fom(sim, params)
    fom_list.append(current_fom)

    Ez, Hx, Hy = sim.saved_fields[1]
    eps = sim.eps.get_values_in(sim.field_domains[1])

    Ng = args.Ng; Nc=args.Nc
    foms = {'Insertion Loss' : fom_list}
    emopt.io.plot_iteration(np.flipud(Ez.real), np.flipud(eps.real), sim.Xreal,
                            sim.Yreal, foms,
                            fname='current_result_Ng{}_Nc{}.pdf'.format(Ng,Nc),
                            dark=False)

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    torch.set_default_tensor_type('torch.DoubleTensor')

    ####################################################################################
    # define the system parameters
    ####################################################################################
    wavelength = 1.55
    X = 28.0
    Y = 8.0
    dx = 0.03
    dy = dx

    # create the simulation object.
    # TE => Ez, Hx, Hy
    sim = FDFD_TE(X, Y, dx, dy, wavelength)

    # Get the actual width and height
    X = sim.X
    Y = sim.Y
    M = sim.M
    N = sim.N
    w_pml = sim.w_pml[0] # PML width which is the same on all boundaries by default

    ####################################################################################
    # Define the structure
    ####################################################################################
    eps_si = 3.4757**2
    eps_clad = 1.444**2
    n_si = np.sqrt(eps_si)
    n_clad = np.sqrt(eps_clad)

    neff = 3.0
    neff_etched = 2.3
    n0 = np.sqrt(eps_clad)

    h_wg = 0.3
    h_etch = 0.18 # etch depth
    w_wg_input = 5.0
    h_BOX = 2.0
    Ng = args.Ng # number of grating teeth (default 30)

    y_ts = Y/2.0

    df = 0.8
    theta = 8.0/180.0*pi
    period = wavelength / (df * neff + (1-df)*neff_etched - n0*np.sin(theta))

    Nc = args.Nc # number of Fourier coefficients (default 5)
    design_params = np.zeros(4*Nc+3) # position and width
    design_params[0] = (1.0-df)*period
    design_params[2*Nc] = period
    design_params[-3] = h_BOX + 1e-3
    design_params[-2] = h_etch + 1e-3
    design_params[-1] = 0.0 + 1e-3 # offset relative to w_wg_input
    k = 1.0/dx

    wg_y_min = y_ts - h_wg/2.0
    wg_y_max = y_ts + h_wg/2.0

    func = partial(create_eps_grid,
                   Nc=Nc, Ng=Ng, k=k, w_in=w_wg_input,
                   eps_l=eps_clad, eps_h=eps_si, wg_y_min=wg_y_min, wg_y_max=wg_y_max)

    eps = AutoDiffMaterial2D(dx, dy, func, torch.as_tensor(design_params).squeeze())
    mu = emopt.grid.ConstantMaterial2D(1.0)

    # add the materials and build the system
    sim.set_materials(eps, mu)
    fdomain = emopt.misc.DomainCoordinates(w_pml, X-w_pml, w_pml, Y-w_pml, 0, 0, dx, dy, 1.0)

    ####################################################################################
    # Setup the sources
    ####################################################################################
    w_src= 3.5

    # place the source in the simulation domain
    src_line = emopt.misc.DomainCoordinates(w_pml+2*dx, w_pml+2*dx, Y/2-w_src/2,
                                 Y/2+w_src/2, 0, 0, dx, dy, 1.0)

    # Setup the mode solver.
    mode = emopt.modes.ModeTE(wavelength, eps, mu, src_line, n0=n_si, neigs=4)

    if(NOT_PARALLEL):
        print('Generating mode data...')

    mode.build()
    mode.solve()

    # at this point we have found the modes but we dont know which mode is the
    # one we fundamental mode.  We have a way to determine this, however
    mindex = mode.find_mode_index(0)

    # set the current sources using the mode solver object
    sim.set_sources(mode, src_line, mindex)

    ####################################################################################
    # Setup the mode match domain
    ####################################################################################
    mm_line = emopt.misc.DomainCoordinates(w_pml, X-w_pml, Y/2.0+2.0, Y/2.0+2.0, 0, 0,
                                           dx, dy, 1.0)
    full_field = emopt.misc.DomainCoordinates(w_pml, X-w_pml, w_pml, Y-w_pml, 0.0, 0.0,
                                              dx, dy, 1.0)
    sim.field_domains = [mm_line, full_field]

    ####################################################################################
    # Build the system
    ####################################################################################
    sim.build()

    ####################################################################################
    # Setup the optimization
    ####################################################################################

    am = SiliconGratingAutograd(sim, fdomain, mm_line)
    am.update_system(design_params)
    am.check_gradient(design_params, fd_step=1e-8)

    fom_list = []
    callback = lambda x : plot_update(x, fom_list, sim, am)

    # setup and run the optimization!
    opt = emopt.optimizer.Optimizer(am, design_params, tol=1e-5,
                                    callback_func=callback,
                                    opt_method='L-BFGS-B',
                                    #opt_method=adam,
                                    Nmax=100)

    # Run the optimization
    final_fom, final_params = opt.run()
