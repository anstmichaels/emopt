"""Blazed grating coupler optimized with AutoDiff-compatible geometry
definitions.

This blazed grating coupler is defined by scattering elements that
consist of 2 etches with different depths. Thus, we represent each
scattering element with 2 rectangles with variable length and height,
constrained to share a boundary.

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
parser.add_argument('--version', type=str, default='AutoDiff')
parser.add_argument('--test', type=bool, default=False)
args = parser.parse_args()

TEST = args.test

###################################################################
# EMopt-AutoDiff
###################################################################

nl = adg.nl_lin

def build_rects(x, periods, widths1, widths2, Ng, k, w_wg_input):
    # Function that builds two sets of rectangles, representing the two etches of the
    # blazed grating. They are built as 1d rect objects. We extrude them in the
    # create_eps_grid function below.
    rects1 = []
    rects2 = []
    pos = w_wg_input
    for i in range(Ng):
        # Rectangles are constrained to share a boundary
        rects1.append(adg.rect1d(k, x, [pos, pos+widths1[i]], nl=nl))
        rects2.append(adg.rect1d(k, x, [pos+widths1[i], pos+widths1[i]+widths2[i]], nl=nl))
        pos = pos + periods[i]
    return adg.union(rects1), adg.union(rects2)

def create_eps_grid(v, coords, Nc, Ng, k, w_in, eps_l, eps_h, wg_y_min, wg_y_max, bg=None):
    y, x = coords

    widths1 = v[:Ng]
    widths2 = v[Ng:2*Ng]
    periods = v[2*Ng:-4]

    wg_input_x = w_in + v[-1]
    h_etch1 = v[-3]
    h_etch2 = v[-2]
    box_height = v[-4]

    wg_xmax = wg_input_x + sum(periods)

    # Define the substrate
    subs = adg.step1d(k, y, wg_y_min - box_height, reverse=True, nl=nl).view(-1,1) # defines subs

    # Define the waveguide
    wg = adg.step1d(k, x, wg_xmax, reverse=True, nl=nl)
    wg = adg.depth(wg, k, y, [wg_y_min, wg_y_max], nl=nl)

    # Define SiO2 rectangles for the etch
    SiO2_rects1, SiO2_rects2 = build_rects(x, periods, widths1, widths2, Ng, k, wg_input_x)

    # Extrude the rectangles to their respective depths
    SiO2_rects1 = adg.depth(SiO2_rects1, k, y, [wg_y_max - h_etch1, wg_y_max], nl=nl)
    SiO2_rects2 = adg.depth(SiO2_rects2, k, y, [wg_y_max - h_etch2, wg_y_max], nl=nl)

    # Take the union
    SiO2_rects = adg.union([SiO2_rects1, SiO2_rects2])

    # Subtract the etches from the waveguide
    shape = subs + wg - SiO2_rects

    # Scale to desired material values
    eps = eps_l + (eps_h - eps_l) * shape

    return eps

class BlazedGratingAutograd(AutoDiffPNF2D):
    # The AutoDiff version of the blazed grating
    def __init__(self, sim, domain, mm_line):
        super().__init__(sim, domain=domain, update_mu=False)
        self.current_fom = 0.0

        # save the variables for later
        self.mm_line = mm_line

        theta = 8.0/180.0*pi
        match_w0 = 5.2
        match_center = 13.0
        eps_clad = 1.444**2

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
    
###################################################################
# EMopt-Standard
###################################################################

class BlazedGratingStandard(emopt.adjoint_method.AdjointMethodPNF2D):
    # The Standard EMopt version of the blazed grating
    def __init__(self, sim, rects1, rects2, wg, substrate, y_ts, w_in, h_wg, Y,
                 Ng, Nc, eps_clad, mm_line):
        super().__init__(sim, step=STEP)
        self.current_fom = 0.0

        # save the variables for later
        self.rects1 = rects1
        self.rects2 = rects2
        self.y_ts = y_ts
        self.w_in = w_in
        self.h_wg = h_wg
        self.Ng = Ng
        self.Nc = Nc
        self.wg = wg
        self.substrate = substrate

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

    def update_system(self, params):
        Ng = self.Ng

        widths1 = params[:Ng]
        widths2 = params[Ng:2*Ng]
        periods = params[2*Ng:-4]

        x0 = self.w_in + params[-1]
        h_etch1 = params[-3]
        h_etch2 = params[-2]
        box_height = params[-4]


        for i in range(Ng):
            w_etch1 = widths1[i]
            self.rects1[i].width  = w_etch1
            self.rects1[i].height = h_etch1
            self.rects1[i].x0     = x0 + w_etch1/2.0
            self.rects1[i].y0     = self.y_ts + self.h_wg/2.0 - h_etch1/2.0

            w_etch2 = widths2[i]
            self.rects2[i].width  = w_etch2
            self.rects2[i].height = h_etch2
            self.rects2[i].x0     = x0 + w_etch1 + w_etch2/2.0
            self.rects2[i].y0     = self.y_ts + self.h_wg/2.0 - h_etch2/2.0

            x0 += periods[i]

        # update the BOX/Substrate
        h_subs = self.y_ts - self.h_wg/2.0 - box_height
        self.substrate.height = 2*h_subs

        # update the width of the unetched grating
        w_in = x0
        self.wg.width = 2*w_in

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

def plot_update_test(params, fom_list, sim, am, version):
    pass

def plot_update_full(params, fom_list, sim, am, version, time_list):
    current_fom = -1*am.calc_fom(sim, params)
    fom_list.append(current_fom)
    #end = time.time()
    time_list.append(time.time())

def plot_update(params, fom_list, sim, am, version, time_list):
    iternum = len(fom_list)+1
    print('Finished iteration %d' % (iternum))
    current_fom = -1*am.calc_fom(sim, params)
    fom_list.append(current_fom)
    time_list.append(time.time())

    Ez, Hx, Hy = sim.saved_fields[1]
    eps = sim.eps.get_values_in(sim.field_domains[1])

    foms = {'Insertion Loss' : fom_list}
    emopt.io.plot_iteration(np.flipud(Ez.real), np.flipud(eps.real), sim.Xreal,
                            sim.Yreal, foms, fname='current_result_{}_{}.pdf'.format(iternum, version),
                            dark=False)

###################################################################
# define grating coupler
###################################################################

def blazed_coupler(Ng, design_params, version):
    widths1 = design_params[:Ng]
    widths2 = design_params[Ng:2*Ng] 
    periods = design_params[2*Ng:-4]
    h_BOX = design_params[-4]
    h_etch1 = design_params[-3]
    h_etch2 = design_params[-2]
    offset = design_params[-1]
    Nc = 5

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
    if version == 'AutoDiff':
        sim = FDFD_TE(X, Y, dx, dy, wavelength)
    else:
        sim = emopt.fdfd.FDFD_TE(X, Y, dx, dy, wavelength)

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

    # set up the initial dimensions of the waveguide structure that we are exciting
    h_wg = 0.3
    w_wg_input = 5.0

    # set the center position of the top silicon and the etches
    y_ts = Y/2.0
    y_etch1 = y_ts + h_wg/2.0 - h_etch1/2.0
    y_etch2 = y_ts + h_wg/2.0 - h_etch2/2.0

    if version == 'Standard' or TEST:
        rects1 = []
        for i in range(Ng):
            w_etch1 = widths1[i]
            rect_etch = emopt.grid.Rectangle(w_wg_input+offset+i*periods[i]+w_etch1/2, y_etch1,
                                             w_etch1, h_etch1)
            rect_etch.layer = 1
            rect_etch.material_value = eps_clad
            rects1.append(rect_etch)

        rects2 = []
        for i in range(Ng):
            w_etch1 = widths1[i]
            w_etch2 = widths2[i]
            rect_etch = emopt.grid.Rectangle(w_wg_input+offset+i*periods[i]+w_etch1/2+w_etch2, y_etch2,
                                             w_etch2, h_etch2)
            rect_etch.layer = 1
            rect_etch.material_value = eps_clad
            rects2.append(rect_etch)

        # grating waveguide
        Lwg = np.sum(periods) + w_wg_input + offset
        wg = emopt.grid.Rectangle(0, y_ts, 2*Lwg, h_wg)
        wg.layer = 2
        wg.material_value = eps_si

        # define substrate
        h_subs = Y/2.0 - h_wg/2.0 - h_BOX
        substrate = emopt.grid.Rectangle(X/2.0, 0, 2*X, 2*h_subs)
        substrate.layer = 2
        substrate.material_value = eps_si # silicon

        # set the background material using a rectangle equal in size to the system
        background = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y)
        background.layer = 3
        background.material_value = eps_clad

        eps_struct = emopt.grid.StructuredMaterial2D(X, Y, dx, dy)
        eps_struct.add_primitive(background)
        eps_struct.add_primitive(wg)
        eps_struct.add_primitive(substrate)
        for ee in rects1:
            eps_struct.add_primitive(ee)
        for ee in rects2:
            eps_struct.add_primitive(ee)

        eps = eps_struct

    elif version=='AutoDiff':
        wg_y_min = y_ts - h_wg/2.0
        wg_y_max = y_ts + h_wg/2.0
        k = 1.0/dx

        func = partial(create_eps_grid,
                       Nc=Nc, Ng=Ng, k=k, w_in=w_wg_input, eps_l=eps_clad, eps_h=eps_si, wg_y_min=wg_y_min, wg_y_max=wg_y_max)

        eps_f = AutoDiffMaterial2D(dx, dy, func, torch.as_tensor(design_params).squeeze())
        #fdomain = emopt.misc.DomainCoordinates(0, X, 0, Y, 0, 0, dx, dy, 1.0)
        fdomain = emopt.misc.DomainCoordinates(w_pml, X-w_pml, w_pml, Y-w_pml, 0, 0, dx, dy, 1.0)

        eps = eps_f


    if TEST:
        wg_y_min = y_ts - h_wg/2.0
        wg_y_max = y_ts + h_wg/2.0
        k = 1.0/dx

        func = partial(create_eps_grid,
                       Nc=Nc, Ng=Ng, k=k, w_in=w_wg_input, eps_l=eps_clad, eps_h=eps_si, wg_y_min=wg_y_min, wg_y_max=wg_y_max)

        eps_f = AutoDiffMaterial2D(dx, dy, func, torch.as_tensor(design_params).squeeze())

        fdomain = emopt.misc.DomainCoordinates(0, X, 0, Y, 0, 0, dx, dy, 1.0)

        eps1 = eps_f
        eps2 = eps

        if NOT_PARALLEL:
            import matplotlib.pyplot as plt
            f, axes = plt.subplots(3, 3, sharex=True, sharey=True)
            domain = emopt.misc.DomainCoordinates(0, X, 0, Y, 0, 0, dx, dy, 1.0)
            e11 = eps1.get_values_in(domain, sx=0, sy=0, squeeze=True).real
            e12 = eps1.get_values_in(domain, sx=-0.5, sy=0, squeeze=True).real
            e13 = eps1.get_values_in(domain, sx=0, sy=0.5, squeeze=True).real
            e21 = eps2.get_values_in(domain, sx=0, sy=0, squeeze=True).real
            e22 = eps2.get_values_in(domain, sx=-0.5, sy=0, squeeze=True).real
            e23 = eps2.get_values_in(domain, sx=0, sy=0.5, squeeze=True).real

            axes[0,0].imshow(e11, extent=[0,X,0,Y], origin='lower', cmap='Blues')
            axes[1,0].imshow(e12, extent=[0,X,0,Y], origin='lower', cmap='Blues')
            axes[2,0].imshow(e13, extent=[0,X,0,Y], origin='lower', cmap='Blues')

            axes[0,1].imshow(e21, extent=[0,X,0,Y], origin='lower', cmap='Blues')
            axes[1,1].imshow(e22, extent=[0,X,0,Y], origin='lower', cmap='Blues')
            axes[2,1].imshow(e23, extent=[0,X,0,Y], origin='lower', cmap='Blues')

            print(np.unravel_index(np.argmax(np.abs(e11-e21)), e11.shape))
            print(np.unravel_index(np.argmax(np.abs(e12-e22)), e11.shape))
            print(np.unravel_index(np.argmax(np.abs(e13-e23)), e11.shape))

            im1 = axes[0,2].imshow((e11-e21), extent=[0,X,0,Y], origin='lower', cmap='bwr')
            im2 = axes[1,2].imshow((e12-e22), extent=[0,X,0,Y], origin='lower', cmap='bwr')
            im3 = axes[2,2].imshow((e13-e23), extent=[0,X,0,Y], origin='lower', cmap='bwr')
            plt.colorbar(im1, ax=axes[0,2])
            plt.colorbar(im2, ax=axes[1,2])
            plt.colorbar(im3, ax=axes[2,2])
            plt.show()
        
        if version == 'AutoDiff':
            eps = eps1
            del eps2
        elif version == 'Standard':
            del eps1



    # set up the magnetic permeability -- just 1.0 everywhere
    mu = emopt.grid.ConstantMaterial2D(1.0)

    # add the materials and build the system
    sim.set_materials(eps, mu)

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

    if version == 'AutoDiff':
        am = BlazedGratingAutograd(sim, fdomain, mm_line)
    else:
        am = BlazedGratingStandard(sim, rects1, rects2, wg, substrate, y_ts, w_wg_input, h_wg, Y,
                     Ng, Nc, eps_clad, mm_line)

    am.update_system(design_params)
    if TEST:
        am.check_gradient(design_params, fd_step=STEP)

    fom_list = []
    time_list = []
    #callback = lambda x : plot_update_full(x, fom_list, sim, am, version, time_list)
    callback = lambda x : plot_update(x, fom_list, sim, am, version, time_list)
    time_list.append(time.time())

    # setup and run the optimization!
    opt = TimedOptimizer(am, design_params, tol=1e-8,
                         callback_func=callback,
                         opt_method='L-BFGS-B',
                         Nmax=500)

    # Run the optimization
    final_fom, final_params = opt.run()
    if NOT_PARALLEL:
        grad_times = np.array(opt.grad_times)
        grad_f_times = np.array(opt.grad_f_times)
        fom_times = np.array(opt.fom_times)
        nit = opt.nit
        nfev = opt.nfev
        njev = opt.njev
        total_time = opt.total_time

        data_to_save = {
            'final_fom':final_fom,
            'final_params':final_params,
            'foms': fom_list,
            'iter_times': time_list,
            'grad_times':grad_times,
            'grad_f_times':grad_f_times,
            'fom_times':fom_times,
            'nit':nit,
            'nfev':nfev,
            'njev':njev,
            'total_time':total_time,
            'version':version,
            'test':TEST,
            'Ng': Ng
        }
        import scipy.io
        scipy.io.savemat('data_{}_{}.mat'.format(version, Ng), data_to_save)

        callback(final_params)

    return final_fom, final_params

def main():
    wavelength = 1.55
    neff = 3.0
    neff_etched = 2.3
    h_etch = 0.101 # etch depth
    n0 = 1.444
    df = 0.8
    theta = 8.0/180.0*pi
    period = wavelength / (df * neff + (1-df)*neff_etched - n0*np.sin(theta))
    h_BOX = 2.001

    #Ng = 30
    Ng = args.Ng
    version = args.version
    design_params = np.zeros(3*Ng+4) # position and width
    design_params[:Ng] = (1.0-df)*period/2
    design_params[Ng:2*Ng] = (1.0-df)*period/2
    design_params[2*Ng:-3] = period
    design_params[-4] = h_BOX
    design_params[-3] = 2*h_etch
    design_params[-2] = h_etch
    design_params[-1] = -0.5 # offset relative to w_wg_input

    final_fom, final_params = blazed_coupler(Ng, design_params, version)
    return final_fom, final_params


if __name__=='__main__':
    torch.set_default_dtype(torch.float64)
    torch.set_default_tensor_type(torch.DoubleTensor)
    f,p = main()