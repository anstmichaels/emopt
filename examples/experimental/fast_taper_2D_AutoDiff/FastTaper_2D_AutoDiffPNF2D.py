"""Optimize a non-adiabatic (fast) transition taper from
a 0.5um width input to 10.5um width output waveguide with a
taper length of 23um. We use AutoDiff to accelerate the
optimization. The script allows you to compare "standard"
boundary smoothing to the AutoDiff version, which will have
small differences depending on choice of nonlinearity.

Note that we only parameterize the taper boundary. We use a
Fourier series parameterization that only permits low spatial
frequencies for less sensitive, fabrication compatible features
"""
import time
from functools import partial
from math import pi

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import torch

import emopt
from emopt.misc import NOT_PARALLEL, run_on_master, MathDummy
from emopt.adjoint_method import AdjointMethodPNF2D

from emopt.experimental.adjoint_method import AutoDiffPNF2D
from emopt.experimental.fdfd import FDFD_TM
from emopt.experimental.grid import HybridMaterial2D, AutoDiffMaterial2D
from emopt.experimental.autodiff_geometry import step1d, rect1d, rect2d, union, intersection, nl_sin, nl_lin
from emopt.experimental.optimizer import TimedOptimizer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='AutoDiff')
parser.add_argument('--test', type=bool, default=False)
args = parser.parse_args()

TEST = args.test
version = args.version

STEP = 1e-6
nl = nl_lin

def sin_series(i, Nv, Nc, params):
    iss = torch.as_tensor(i)
    js = torch.arange(1,Nc+1).view(1,-1)
    sins = torch.sin(pi * js * iss.view(-1,1) / Nv).numpy()
    retval = params[:Nc] * sins
    return retval.sum(-1)

def sin_series_torch(x, L, Nc, v):
    js = torch.arange(1, Nc+1).view(1,-1)
    sins = torch.sin(pi * js * x.view(-1,1) / L)
    retval = v * sins
    return retval.sum(-1)

def wavy(x, y, k, v, Nc, L, yt, reverse=False):
    envelope = 0.45*(1.0 - torch.cos(pi * x / L * 2)) + 0.1 # the constant 0.1 helps reduce sensitivity near input
    ybound = yt + envelope * sin_series_torch(x, L, Nc, v[:Nc])
    if reverse:
        shape = nl(k, -(ybound.view(1,-1) - y.view(-1,1)))
    else:
        shape = nl(k, ybound.view(1,-1) - y.view(-1,1))
    return shape



def create_eps_grid(v, coords, k_wg, k_tap, eps_l, eps_h,
                    Ycen, w_pml, Nc, L, X, x_in, wg_i_ymax,
                    wg_i_ymin, wg_o_ymax, wg_o_ymin, dx, bg=None):
    y, x = coords
    v = torch.as_tensor(v)

    # define input and output waveguides
    wg_i = rect2d(k_wg, x, y, [-1, x_in+w_pml, wg_i_ymin, wg_i_ymax], nl=nl_lin)
    wg_o = rect2d(k_wg, x, y, [X-(x_in+w_pml), X+1, wg_o_ymin, wg_o_ymax], nl=nl_lin)
    shape = wg_i + wg_o

    # find y coords of linear taper boundary connecting input to output
    x0 = x_in + w_pml
    x1 = X - x_in - w_pml
    y0 = wg_i_ymax - Ycen
    y1 = wg_o_ymax - Ycen
    m = (y1 - y0)/(x1 - x0)
    line_y = y0 + m * (x - x0)

    # we perturb the linear taper boundary using Fourier series
    # defining both upper and lower boundary
    taper_t = wavy(x-x0, y, k_tap, v[:2*Nc], Nc, L, Ycen + line_y)
    taper_b = wavy(x-x0, y, k_tap, -v[:2*Nc], Nc, L, Ycen - line_y, reverse=True)
    #taper = taper_t * taper_b
    #taper = taper_t * taper_b

    # two options here:
    # (1) we can use logic to combine input and output waveguides plus taper structure
    #   This requires a bit more work, because we need to truncate the taper in its
    #   current form before taking union with the waveguides
    ## mask_rect = rect1d(k_wg, x, [x_in+w_pml, X-(x_in+w_pml)], nl=nl_lin).view(1,-1)
    ## taper = intersection([taper_t, taper_b, mask_rect])
    ## shape = union([shape, taper])

    # (2) we can just mask over the eps array where the taper is positioned.
    #   We can do this because only this part of the array contributes to the
    #   gradient anyway. Note: this might cause trouble for some values of
    #   sim.dx, due to half-steps needed in Yee cells.
    taper = intersection([taper_t, taper_b])
    mask = (x+0.5*dx >= x_in+w_pml) * (X-x_in-w_pml >= x-0.5*dx)
    shape[:,mask] = taper[:,mask]

    return eps_l + (eps_h - eps_l) * shape


class TaperAutoDiff(AutoDiffPNF2D):
    def __init__(self, sim, optdomain, fom_domain, mode_match, mm_line):
        super().__init__(sim, domain=optdomain, update_mu=False)
        self.mode_match = mode_match
        self.fom_domain = fom_domain
        self.mm_line = mm_line

    @run_on_master
    def calc_f(self, sim, params):
        Hz, Ex, Ey = sim.saved_fields[0]
        self.mode_match.compute(Hz=Hz, Ex=Ex, Ey=Ey)

        return -1*self.mode_match.get_mode_match_forward(1.0)

    @run_on_master
    def calc_dfdx(self, sim, params):
        Hz, Ex, Ey = sim.saved_fields[0]
        self.mode_match.compute(Hz=Hz, Ex=Ex, Ey=Ey)

        dFdHz = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dFdEx = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dFdEy = np.zeros([sim.M, sim.N], dtype=np.complex128)

        dFdHz[self.mm_line.j, self.mm_line.k] = -self.mode_match.get_dFdHz()
        dFdEx[self.mm_line.j, self.mm_line.k] = -self.mode_match.get_dFdEx()
        dFdEy[self.mm_line.j, self.mm_line.k] = -self.mode_match.get_dFdEy()

        return dFdHz, dFdEx, dFdEy

    def calc_grad_p(self, sim, params):
        return np.zeros(len(params))


class TaperAdjointMethod(AdjointMethodPNF2D):
    def __init__(self, sim, fom_domain, mode_match, x_in, wg_w_i, wg_w_o, w_pml, Nv, Nc, taper, mm_line):
        super().__init__(sim, step=STEP)
        self.mode_match = mode_match
        self.fom_domain = fom_domain
        self.x_in = x_in
        self.wg_w_i = wg_w_i
        self.wg_w_o = wg_w_o
        self.w_pml = w_pml
        self.Nv = Nv
        self.Nc = Nc
        self.taper = taper
        self.mm_line = mm_line

    def update_system(self, params):
        p = params
        x_in = self.x_in
        wg_w_i = self.wg_w_i
        wg_w_o = self.wg_w_o
        w_pml = self.w_pml
        X = self.sim.X
        Ycen = self.sim.Y/2.0
        Nv = self.Nv
        Nc = self.Nc

        i = np.arange(Nv+1) # number of vertices

        # get x and y coordinates of vertices of a line
        x0 = x_in+w_pml
        x1 = X-x_in-w_pml
        y0 = wg_w_i/2.0
        y1 = wg_w_o/2.0
        m = (y1 - y0)/(x1 - x0)
        line_x = x0 + (x1 - x0) * i / Nv
        line_y = y0 + m * (line_x - x0)

        # modify y vertices with our Fourier series
        j = sin_series(i, Nv, Nc, p)
        envelope = 0.45*(1.0 - np.cos(pi * i / Nv * 2)) + 0.1
        xs = line_x
        ys = line_y + j * envelope

        # mirror our vertices for symmetry
        xs = np.concatenate([xs, xs[::-1]], axis=0)
        ys = Ycen + np.concatenate([ys, -ys[::-1]], axis=0)

        # update geo
        self.taper.set_points(xs, ys)

    @run_on_master
    def calc_f(self, sim, params):
        Hz, Ex, Ey = sim.saved_fields[0]
        self.mode_match.compute(Hz=Hz, Ex=Ex, Ey=Ey)

        return -1*self.mode_match.get_mode_match_forward(1.0)

    @run_on_master
    def calc_dfdx(self, sim, params):
        Hz, Ex, Ey = sim.saved_fields[0]
        self.mode_match.compute(Hz=Hz, Ex=Ex, Ey=Ey)

        dFdHz = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dFdEx = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dFdEy = np.zeros([sim.M, sim.N], dtype=np.complex128)

        dFdHz[self.mm_line.j, self.mm_line.k] = -self.mode_match.get_dFdHz()
        dFdEx[self.mm_line.j, self.mm_line.k] = -self.mode_match.get_dFdEx()
        dFdEy[self.mm_line.j, self.mm_line.k] = -self.mode_match.get_dFdEy()

        return dFdHz, dFdEx, dFdEy

    def calc_grad_p(self, sim, params):
        return np.zeros(len(params))

def plot_update_full(params, fom_list, sim, am, version, time_list):
    current_fom = -1*am.calc_fom(sim, params)
    fom_list.append(current_fom)
    time_list.append(time.time())

def plot_update(params, fom_list, sim, am, version):
    print('Finished iteration %d' % (len(fom_list)+1))
    current_fom = -1*am.calc_fom(sim, params)
    fom_list.append(current_fom)
    Hz, Ex, Ey = sim.saved_fields[1]
    eps = sim.eps.get_values_in(sim.field_domains[1])
    foms = {'Insertion Loss' : fom_list}
    emopt.io.plot_iteration(np.flipud(Ey.real.squeeze()), np.flipud(eps.real.squeeze()), sim.X - 2*sim.w_pml[0],
                            sim.Y - 2*sim.w_pml[0], foms, fname='current_result_{}_{}.pdf'.format(len(fom_list), version),
                            dark=False)

def taper_opt():
    x_in = 1.0 # input/output waveguide lengths
    wg_w_i = 0.5 # input waveguide width
    wg_w_o = 10.5 # output waveguide width
    taper_l = 23.0 # length of taper
    Nc = 100 # num Fourier coefficients
    params = np.zeros(Nc) # define design vector
    Nv = 200 # num vertices on one side of taper (for standard version)

    nSi = 3.2 # effective index for 300nm thick Si
    nSiO2 = 1.45

    ####################################################################################
    # Simulation parameters
    ####################################################################################
    wavelength = 1.31
    dx = 0.03 # grid spacing along x
    dy = dx # grid spacing along y
    w_pml = 13*dx # pml length

    X = 2*x_in + taper_l + 2*w_pml   # simulation size along x
    Y = wg_w_o + 2*w_pml + 1.0 # simulation size along y

    #####################################################################################
    # Setup simulation
    #####################################################################################
    if version == 'AutoDiff':
        sim = FDFD_TM(X, Y, dx, dy, wavelength)
    else:
        sim = emopt.fdfd.FDFD_TM(X, Y, dx, dy, wavelength)

    sim.w_pml = [w_pml, w_pml, w_pml, w_pml]
    sim.bc = '00'
    X = sim.X
    Y = sim.Y
    Ycen = Y/2.0

    #####################################################################################
    # Define the geometry/materials
    #####################################################################################
    if version != 'AutoDiff' or TEST:
        wg_in = emopt.grid.Rectangle(0, Ycen, 2*(x_in+w_pml), wg_w_i); wg_in.layer = 1
        wg_out = emopt.grid.Rectangle(X, Ycen, 2*(x_in+w_pml), wg_w_o); wg_out.layer = 1

        i = np.arange(Nv+1)
        x0 = x_in + w_pml
        x1 = X - x_in - w_pml
        y0 = wg_w_i/2.0
        y1 = wg_w_o/2.0
        m = (y1 - y0)/(x1 - x0)

        line_x = x0 + (x1 - x0) * i / Nv
        line_y = y0 + m * (line_x - x0)

        xs = line_x
        ys = line_y

        xs = np.concatenate([xs, xs[::-1]], axis=0)
        ys = Ycen + np.concatenate([ys, -ys[::-1]], axis=0)

        taper = emopt.grid.Polygon(xs, ys); taper.layer = 1

        if TEST and NOT_PARALLEL:
            # quick plot of vertices
            f = plt.figure()
            ax = f.add_subplot(111)
            ax.plot(xs, ys, '-o')
            plt.show()

        bg = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y); bg.layer = 6
        wg_in.material_value = nSi**2
        wg_out.material_value = nSi**2
        taper.material_value = nSi**2
        bg.material_value = nSiO2**2

        eps_struct = emopt.grid.StructuredMaterial2D(X, Y, dx, dy)
        eps_struct.add_primitive(wg_in)
        eps_struct.add_primitive(wg_out)
        eps_struct.add_primitive(taper)
        eps_struct.add_primitive(bg)

        eps = eps_struct

    else:
        func = partial(create_eps_grid,
                       k_wg=1.0/dx, k_tap=1./dx, eps_l=nSiO2**2, eps_h=nSi**2,
                       Ycen=Ycen, w_pml=w_pml, Nc=Nc, L=taper_l, X=X, x_in=x_in,
                       wg_i_ymax=Ycen+wg_w_i/2.0, wg_i_ymin=Ycen-wg_w_i/2.0,
                       wg_o_ymax=Ycen+wg_w_o/2.0, wg_o_ymin=Ycen-wg_w_o/2.0, dx=dx
                       )

        eps_f = AutoDiffMaterial2D(dx, dy, func, torch.as_tensor(params).squeeze())
        fdomain = emopt.misc.DomainCoordinates(w_pml, X-w_pml, w_pml, Y-w_pml, 0, 0, dx, dy, 1.0)

        eps = eps_f


    if TEST:
        func = partial(create_eps_grid,
                       k_wg=1.0/dx, k_tap=1./dx, eps_l=nSiO2**2, eps_h=nSi**2,
                       Ycen=Ycen, w_pml=w_pml, Nc=Nc, L=taper_l, X=X, x_in=x_in,
                       wg_i_ymax=Ycen+wg_w_i/2.0, wg_i_ymin=Ycen-wg_w_i/2.0,
                       wg_o_ymax=Ycen+wg_w_o/2.0, wg_o_ymin=Ycen-wg_w_o/2.0, dx=dx
                       )

        eps_f = AutoDiffMaterial2D(dx, dy, func, torch.as_tensor(params).squeeze())
        fdomain = emopt.misc.DomainCoordinates(w_pml, X-w_pml, w_pml, Y-w_pml, 0, 0, dx, dy, 1.0)

        eps1 = eps_struct
        eps2 = eps_f

        if NOT_PARALLEL:
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
            eps = eps2
            del eps1
        elif version != 'AutoDiff':
            eps = eps1
            del eps2

    # set up the magnetic permeability -- just 1.0 everywhere
    mu = emopt.grid.ConstantMaterial2D(1.0)

    # Set the materials and build the system
    sim.set_materials(eps, mu)
    sim.build()

    #####################################################################################
    # Setup the sources
    #####################################################################################
    # We excite the system by injecting the fundamental mode of the input waveguide
    input_slice = emopt.misc.DomainCoordinates(w_pml+x_in/2.0, w_pml+x_in/2.0, w_pml, Y-w_pml, 0, 0, dx, dy, 1)
    mode = emopt.modes.ModeTM(wavelength, eps, mu, input_slice, n0=nSi, neigs=4)

    mode.bc = '0'
    mode.build()
    mode.solve()
    sim.set_sources(mode, input_slice)

    Ey = mode.get_field_interp(0, 'Ey')
    if TEST and NOT_PARALLEL:
        # check input mode
        print('Effective index = {:.4}'.format(mode.neff[0].real))
        eps_arr = eps.get_values_in(input_slice)
        vmin = np.min(np.abs(Ey))
        vmax = np.max(np.abs(Ey))
        f, ax = plt.subplots(2,1)
        im1 = ax[0].plot(Ey.real)
        im2 = ax[1].plot(np.real(eps_arr))
        plt.show()

    #####################################################################################
    # Mode match for optimization
    #####################################################################################
    fom_slice = emopt.misc.DomainCoordinates(X-(w_pml+x_in/2.0), X-(w_pml+x_in/2.0), w_pml, Y-w_pml,
                                             0, 0, dx, dy, 1)
    fom_mode = emopt.modes.ModeTM(wavelength, eps, mu, fom_slice, n0=nSi, neigs=4)
    fom_mode.bc = '0'
    fom_mode.build()
    fom_mode.solve()

    Exm = fom_mode.get_field_interp(0, 'Ex')
    Eym = fom_mode.get_field_interp(0, 'Ey')
    Hzm = fom_mode.get_field_interp(0, 'Hz')
    if not NOT_PARALLEL:
        Exm = MathDummy()
        Eym = MathDummy()
        Hzm = MathDummy()

    mode_match = emopt.fomutils.ModeMatch([1,0,0], dy, Exm=Exm, Eym=Eym, Hzm=Hzm)

    Ey = fom_mode.get_field_interp(0, 'Ey')
    if TEST and NOT_PARALLEL:
        # check output mode
        print('Effective index = {:.4}'.format(fom_mode.neff[0].real))
        eps_arr = eps.get_values_in(fom_slice)
        vmin = np.min(np.abs(Ey))
        vmax = np.max(np.abs(Ey))
        f = plt.figure()
        ax1 = f.add_subplot(211)
        ax2 = f.add_subplot(212)
        im1 = ax1.plot(Ey.real)
        im2 = ax2.plot(np.real(eps_arr))
        plt.show()

    #####################################################################################
    # Setup the AdjointMethod object needed for gradient calculations
    #####################################################################################
    field_slice = emopt.misc.DomainCoordinates(w_pml, X-w_pml, w_pml, Y-w_pml,
                                                0, 0, dx, dy, 1.0)
    sim.field_domains = [fom_slice, field_slice]

    eps_check = eps.get_values_in(field_slice, squeeze=True)
    if TEST and NOT_PARALLEL:
        # Check total geometry
        f = plt.figure()
        ax1 = f.add_subplot(111)
        im1 = ax1.imshow(np.real(eps_check),
                         cmap='Blues', origin='lower')
        plt.show()


    if version == 'AutoDiff':
        am = TaperAutoDiff(sim, fdomain, fom_slice, mode_match, fom_slice)
    else:
        am = TaperAdjointMethod(sim, fom_slice, mode_match,
                                x_in, wg_w_i, wg_w_o, w_pml, Nv, Nc, taper, fom_slice)

    am.update_system(params)


    #####################################################################################
    # Setup and run the optimization
    #####################################################################################
    # L-BFGS-B will print out the iteration number and FOM value
    fom_list = []
    if TEST:
        callback = lambda pars: plot_update(pars, fom_list, sim, am, version)
    else:
        time_list = []
        callback = lambda pars: plot_update_full(pars, fom_list, sim, am, version, time_list)
        time_list.append(time.time())

    opt = TimedOptimizer(am, params, Nmax=100, opt_method='L-BFGS-B', callback_func=callback)
    final_fom, final_params = opt.run()

    if NOT_PARALLEL:
        grad_times = np.array(opt.grad_times)
        grad_f_times = np.array(opt.grad_f_times)
        fom_times = np.array(opt.fom_times)
        nit = opt.nit
        nfev = opt.nfev
        njev = opt.njev
        total_time = opt.total_time

        if TEST:
            time_list = []

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
        }
        print(data_to_save)
        scipy.io.savemat('data_{}.mat'.format(version), data_to_save)

        callback = lambda x : plot_update(x, fom_list, sim, am, version)
        callback(final_params)

    return final_fom, final_params

if __name__=='__main__':
    torch.set_default_dtype(torch.float64)
    torch.set_default_tensor_type(torch.DoubleTensor)
    f, p = taper_opt()
