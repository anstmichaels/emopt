import emopt
from emopt.misc import NOT_PARALLEL
from emopt.experimental.adjoint_method import AutoDiffPNF2D
from emopt.experimental.fdfd import FDFD_TE
from emopt.experimental.grid import HybridMaterial2D, AutoDiffMaterial2D
import matplotlib.pyplot as plt
from functools import partial

import numpy as np
from math import pi

import torch

def fseries(i, coeffs, Nc, Ng):
    sins = np.zeros(Nc)
    coss = np.zeros(Nc)
    for j in range(Nc):
        if j==0:
            sins[j] = 1.0
        else:
            sins[j] = np.sin(np.pi/2.0 * i * j * 1.0/Ng)
            sins[j] = np.sin(np.pi/2.0 * i * j * 1.0/Ng)

        coss[j] = np.cos(np.pi/2.0 * i * j * 1.0/Ng)

    sins = torch.as_tensor(sins)
    coss = torch.as_tensor(coss)
    retval = coeffs[:Nc] * sins + coeffs[Nc:] * coss
    return retval.sum()

def build_rects(x, y, periods, widths, ymin, ymax, Ng, k, w_wg_input):
    rects = []
    pos = w_wg_input
    for i in range(Ng):
       rects.append(rect(x, y, pos, widths[i], ymin, ymax, k))
       pos = pos + periods[i]
    return rects

def depth(shape, y, k, ymin, ymax):
    d = torch.sigmoid(k * (y - ymin)) * torch.sigmoid(-k * (y - ymax))
    return shape.view(1,-1) * d.view(-1,1)

def depth_1side(shape, y, k, ymax):
    d = torch.sigmoid(-k * (y - ymax))
    return shape.view(1,-1) * d.view(-1,1)

def rect(x, y, pos, width, ymin, ymax, k):
    b = pos+width/2.0
    a = pos-width/2.0
    #r = 0.5*(torch.erf(k*(b-x)) + torch.erf(k*(x-a)))
    r = torch.sigmoid(k*(b-x)) * torch.sigmoid(k*(x-a))
    if (ymin is not None) and (ymax is not None):
        r = depth(r, y, k, ymin, ymax)
    return r

def combine(list_of_rects, k):
    return torch.sigmoid(k*(sum(list_of_rects) - 0.5))

def create_eps_grid(v, coords, Nc, Ng, k, w_in, eps_l, eps_h, wg_y_min, wg_y_max, bg=None):
    y, x = coords

    widths = []
    periods = []
    for i in range(Ng):
        widths.append(fseries(i, v[:2*Nc], Nc, Ng))
        periods.append(fseries(i, v[2*Nc:-2], Nc, Ng))

    wg_input_x = w_in + v[-1]
    h_etch = v[-2]

    wg_xmax = wg_input_x + sum(periods)

    grat = torch.sigmoid(-k * (x - wg_xmax)) # defines wg in x
    SiO2_rects = build_rects(x, y, periods, widths, None, None, Ng, k, wg_input_x)
    grat = grat - sum(SiO2_rects) # this just gives us Si rectangles
    grat = depth(grat, y, k, wg_y_max - h_etch, wg_y_max)

    wg_bot = torch.sigmoid(-k * (x - wg_xmax)) # defines wg in x
    wg_bot = depth(wg_bot, y, k, wg_y_min, wg_y_max-h_etch) # defines wg in x

    k2 = 20.0
    #shape = torch.sigmoid(k2 * (wg_bot + grat - 0.5))
    shape = wg_bot + grat
    eps = eps_l + (eps_h - eps_l) * shape

    return eps

def create_eps_grid_sidewall(v, coords, Nc, Ng, k, w_in, eps_l, eps_h, wg_y_min, wg_y_max, bg=None):
    y, x = coords

    widths = []
    periods = []
    for i in range(Ng):
        widths.append(fseries(i, v[:2*Nc], Nc, Ng))
        periods.append(fseries(i, v[2*Nc:-2], Nc, Ng))

    wg_input_x = w_in + v[-1]
    h_etch = v[-2]

    wg_xmax = wg_input_x + sum(periods)

    grat = torch.sigmoid(-k * (x - wg_xmax)) # defines wg in x
    SiO2_rects = build_rects(x, y, periods, widths, None, None, Ng, k, wg_input_x)
    grat = grat - sum(SiO2_rects) # this just gives us Si rectangles
    grat = depth(grat, y, k, wg_y_max - h_etch, wg_y_max)

    wg_bot = torch.sigmoid(-k * (x - wg_xmax)) # defines wg in x
    wg_bot = depth(wg_bot, y, k, wg_y_min, wg_y_max-h_etch) # defines wg in x

    k2 = 20.0
    shape = torch.sigmoid(k2 * (wg_bot + grat - 0.5))
    #shape = wg_bot + grat
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

    foms = {'Insertion Loss' : fom_list}
    emopt.io.plot_iteration(np.flipud(Ez.real), np.flipud(eps.real), sim.Xreal,
                            sim.Yreal, foms, fname='current_result.pdf',
                            dark=False)

    data = {}
    data['Ez'] = Ez
    data['Hx'] = Hx
    data['Hy'] = Hy
    data['eps'] = eps
    data['params'] = params
    data['foms'] = fom_list

    i = len(fom_list)
    fname = 'data/gc_opt_results'
    emopt.io.save_results(fname, data)

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    torch.set_default_tensor_type('torch.DoubleTensor')
    torch.set_num_interop_threads(40)
    torch.set_num_threads(40)
    torch.set_anomaly_enabled(True)
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

    # the effective indices are precomputed for simplicity.  We can compute
    # these values using emopt.modes
    #neff = 2.86
    #neff_etched = 2.10
    neff = 3.0
    neff_etched = 2.3
    n0 = np.sqrt(eps_clad)

    # set up the initial dimensions of the waveguide structure that we are exciting
    h_wg = 0.3
    h_etch = 0.18 # etch depth
    w_wg_input = 5.0
    Ng = 30 #number of grating teeth

    # set the center position of the top silicon and the etches
    y_ts = Y/2.0

    # define the starting parameters of the partially-etched grating
    # notably the period and shift between top and bottom layers
    df = 0.8
    theta = 8.0/180.0*pi
    period = wavelength / (df * neff + (1-df)*neff_etched - n0*np.sin(theta))

    # grating waveguide
    #Lwg = Ng*period + w_wg_input + 1.5
    #wg = emopt.grid.Rectangle(Lwg/2.0, y_ts, Lwg, h_wg)
    #wg.layer = 2
    #wg.material_value = eps_si

    # define substrate
    h_BOX = 2.0
    h_subs = Y/2.0 - h_wg/2.0 - h_BOX
    substrate = emopt.grid.Rectangle(X/2.0, h_subs/2.0, X, h_subs)
    substrate.layer = 2
    substrate.material_value = eps_si # silicon

    # set the background material using a rectangle equal in size to the system
    background = emopt.grid.Rectangle(X/2, Y/2, X, Y)
    background.layer = 3
    background.material_value = eps_clad

    # assembled the primitives in a StructuredMaterial to be used by the FDFD solver
    # This Material defines the distribution of the permittivity within the simulated
    # environment
    eps_struct = emopt.grid.StructuredMaterial2D(X, Y, dx, dy)

    #eps_struct.add_primitive(wg)
    eps_struct.add_primitive(substrate)
    eps_struct.add_primitive(background)

    Nc = 5
    N_coeffs = 5
    design_params = np.zeros(4*Nc+2) # position and width
    design_params[0] = (1.0-df)*period
    design_params[2*Nc] = period
    design_params[-2] = h_etch
    design_params[-1] = 0.0 # offset relative to w_wg_input
    k = 200.0
    #k = 50.0

    wg_y_min = y_ts - h_wg/2.0
    wg_y_max = y_ts + h_wg/2.0

    func = partial(create_eps_grid,
                   Nc=Nc, Ng=Ng, k=k, w_in=w_wg_input, eps_l=eps_clad, eps_h=eps_si, wg_y_min=wg_y_min, wg_y_max=wg_y_max)

    eps_f = AutoDiffMaterial2D(dx, dy, func, torch.as_tensor(design_params).squeeze())

    # set up the magnetic permeability -- just 1.0 everywhere
    ymax = y_ts + h_wg/2 + 5*dy
    ymin = y_ts - h_wg/2 - 5*dy
    #fdomain = emopt.misc.DomainCoordinates(w_pml+5*dx, Lwg-2*dx, ymin, ymax, 0, 0, dx, dy, 1.0)
    fdomain = emopt.misc.DomainCoordinates(0, X-w_pml, ymin, ymax, 0, 0, dx, dy, 1.0)

    eps = HybridMaterial2D(eps_struct, eps_f, fdomain)
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

    am = SiliconGratingAutograd(sim, fdomain, mm_line)
    am.update_system(design_params)
    am.check_gradient(design_params, fd_step=1e-6)

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
