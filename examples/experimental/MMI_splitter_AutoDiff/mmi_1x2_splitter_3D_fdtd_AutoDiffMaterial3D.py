import emopt
from emopt.misc import NOT_PARALLEL, run_on_master
from emopt.experimental.fdtd import FDTD
from emopt.experimental.adjoint_method import AutoDiffPNF3D
from emopt.experimental.grid import AutoDiffMaterial3D
from functools import partial

import numpy as np
from math import pi
import torch

def rect(v, k, x, y):
    x_sig = torch.sigmoid(k * (x - v[0])) * torch.sigmoid(-k * (x - v[1]))
    y_sig = torch.sigmoid(k * (y - v[2])) * torch.sigmoid(-k * (y - v[3]))
    return x_sig.unsqueeze(0) * y_sig.unsqueeze(-1)

def depth(z, k, v):
    z = torch.sigmoid(k * (z - v[0])) * torch.sigmoid(-k * (z - v[1]))
    return z

def create_eps_grid(v, coords, k, zmin, zmax, wg_i, wg_o1, wg_o2,
                    eps_l, delta_eps, bg=None):
    z, y, x = coords
    zdepth = depth(z, 4*k, [zmin, zmax])

    in_wg = rect(wg_i, k, x, y)
    out_wg1 = rect(wg_o1, k, x, y)
    out_wg2 = rect(wg_o2, k, x, y)
    mmi = rect(v, k, x, y)

    shape = torch.sigmoid(k * (in_wg + out_wg1 + out_wg2 + mmi - 0.5))
    shape = shape.unsqueeze(0) * zdepth.unsqueeze(-1).unsqueeze(-1)
    eps = eps_l + delta_eps * shape
    return eps

class MMISplitterAdjointMethod(AutoDiffPNF3D):
    def __init__(self, sim, domain, fom_domain, mode_match, eps_l, eps_h, k, zmin, zmax, wg_i, wg_o1, wg_o2):
        super().__init__(sim, domain=domain, update_mu=False)
        self.mode_match = mode_match
        self.fom_domain = fom_domain

        self.eps_l = eps_l
        self.eps_h = eps_h
        self.delta_eps = eps_h - eps_l
        self.k = k

        self.zmin = zmin
        self.zmax = zmax

        self.wg_i = wg_i
        self.wg_o1 = wg_o1
        self.wg_o2 = wg_o2

    @run_on_master
    def calc_f(self, sim, params):
        Ex, Ey, Ez, Hx, Hy, Hz = sim.saved_fields[0]

        self.mode_match.compute(Ex, Ey, Ez, Hx, Hy, Hz)
        fom = -1*self.mode_match.get_mode_match_forward(1.0)

        return fom

    @run_on_master
    def calc_dfdx(self, sim, params):
        dfdEx = -1*self.mode_match.get_dFdEx()
        dfdEy = -1*self.mode_match.get_dFdEy()
        dfdEz = -1*self.mode_match.get_dFdEz()
        dfdHx = -1*self.mode_match.get_dFdHx()
        dfdHy = -1*self.mode_match.get_dFdHy()
        dfdHz = -1*self.mode_match.get_dFdHz()

        return [(dfdEx, dfdEy, dfdEz, dfdHx, dfdHy, dfdHz)]

    def get_fom_domains(self):
        return [self.fom_domain]

    def calc_grad_p(self, sim, params):
        return np.zeros(len(params))

def plot_update(params, fom_list, sim, am):
    print('Finished iteration %d' % (len(fom_list)+1))
    current_fom = -1*am.calc_fom(sim, params)
    fom_list.append(current_fom)

    #Ez, Hx, Hy = sim.saved_fields[1]
    Ex,Ey,Ez,Hx,Hy,Hz = sim.saved_fields[1]
    eps = sim.eps.get_values_in(sim.field_domains[1]).squeeze()
    Ey = np.squeeze(Ey)

    foms = {'eff' : fom_list}
    emopt.io.plot_iteration(np.flipud(Ey.real), np.flipud(eps.real), sim.X-2*sim.w_pml[0],
                            sim.Y-2*sim.w_pml[0], foms, fname='current_result_new.pdf',
                            dark=False)


if __name__=='__main__':
    torch.set_default_dtype(torch.double)
    torch.set_default_tensor_type(torch.DoubleTensor)

    ####################################################################################
    # Simulation parameters
    ####################################################################################
    X = 5.0   # simulation size along x
    Y = 4.0 # simulation size along y
    Z = 2.5   # simulation size along z
    dx = 0.04 # grid spacing along x
    dy = 0.04 # grid spacing along y
    dz = 0.04 # grid spacing along z

    wavelength = 1.55
    #wavelength = 1.31

    #####################################################################################
    # Setup simulation
    #####################################################################################
    # Setup the simulation--rtol tells the iterative solver when to stop. 5e-5
    # yields reasonably accurate results/gradients
    sim = FDTD(X,Y,Z,dx,dy,dz,wavelength, rtol=1e-5, min_rindex=1.44,
                          nconv=200)
    sim.Nmax = 1000*sim.Ncycle
    w_pml = dx * 20 # set the PML width

    # we use symmetry boundary conditions at y=0 to speed things up. We
    # need to make sure to set the PML width at the minimum y boundary is set to
    # zero. Currently, FDTD cannot compute accurate gradients using symmetry in z
    # :(
    #sim.w_pml = [w_pml, w_pml, w_pml, w_pml, w_pml, w_pml]
    sim.w_pml = [w_pml, w_pml, 0, w_pml, w_pml, w_pml]
    sim.bc = '0H0'

    # get actual simulation dimensions
    X = sim.X
    Y = sim.Y
    Z = sim.Z

    #domain = emopt.misc.DomainCoordinates(0, X, 0, Y, 0, Z, dx, dy, dz)
    domain = emopt.misc.DomainCoordinates(w_pml, X-w_pml, w_pml, Y-w_pml, w_pml, Z-w_pml, dx, dy, dz)

    #####################################################################################
    # Define the geometry/materials
    #####################################################################################
    # Geometry consists of input waveguide, output waveguide, and MMI splitting
    # section. Structure is silicon clad in SiO2
    n_si = 3.45
    n_clad = 1.444
    eps_si = n_si**2
    eps_clad = n_clad**2

    w_wg = 0.45
    #L_in = X/2+1
    #L_out = X/2+1
    L_in = X/2-0.5
    L_out = X/2-0.5
    L_mmi = 2.0
    w_mmi = 2.0
    h_si = 0.22
    #h_si = 0.3

    zmin = Z/2.0 - h_si/2.0
    zmax = Z/2.0 + h_si/2.0
    k = 20.0
    wg_i = [-5.0, L_in, Y/2-w_wg/2.0, Y/2+w_wg/2.0]
    #wg_i = [-5.0, X+5.0, Y/2-w_wg/2.0, Y/2+w_wg/2.0]
    wg_o1 = [X-L_out, X+5.0, Y/2 + w_wg - w_wg/2.0, Y/2 + w_wg + w_wg/2.0]
    wg_o2 = [X-L_out, X+5.0, Y/2 - w_wg - w_wg/2.0, Y/2 - w_wg + w_wg/2.0]

    design_params = np.array([X/2.0-L_mmi/2, X/2.0+L_mmi/2, Y/2.0-w_mmi/2.0, Y/2.0+w_mmi/2.0])

    func = partial(create_eps_grid, k=k, zmin=zmin, zmax=zmax, \
                   wg_i=wg_i, wg_o1=wg_o1, wg_o2=wg_o2, eps_l=eps_clad, delta_eps=eps_si-eps_clad)

    eps = AutoDiffMaterial3D(sim.dx, sim.dy, sim.dz, func, design_params)
    mu = emopt.grid.ConstantMaterial3D(1.0)

    sim.set_materials(eps, mu)
    sim.build()

    #eps = eps.get_values_in(domain).squeeze().real
    #if NOT_PARALLEL:
    #    import matplotlib.pyplot as plt
    #    f = plt.figure()
    #    ax1 = f.add_subplot(131)
    #    ax2 = f.add_subplot(132)
    #    ax3 = f.add_subplot(133)
    #    ax1.imshow(eps[:, :, eps.shape[2]//2].squeeze(), cmap='Blues')
    #    ax2.imshow(eps[:, eps.shape[1]//2, :].squeeze(), cmap='Blues')
    #    ax3.imshow(eps[eps.shape[0]//2, :, :].squeeze(), cmap='Blues')
    #    plt.tight_layout()
    #    plt.show()

    #####################################################################################
    # Setup the sources
    #####################################################################################
    # We excite the system by injecting the fundamental mode of the input waveguide
    input_slice = emopt.misc.DomainCoordinates(w_pml+3*dx, w_pml+3*dx, w_pml, Y-w_pml, w_pml, Z-w_pml, dx, dy, dz)

    mode = emopt.modes.ModeFullVector(wavelength, eps, mu, input_slice, n0=3.45,
                                       neigs=4)

    # The mode boundary conditions should match the simulation boundary conditins.
    # Mode is in the y-z plane, so the boundary conditions are HE
    mode.bc = 'H0'
    mode.build()
    mode.solve()
    sim.set_sources(mode, input_slice)

    #####################################################################################
    # Mode match for optimization
    #####################################################################################
    # we need to calculate the field used as the reference field in our mode match
    # figure of merit calculation. This is the fundamental super mode of the output
    # waveguides.
    fom_slice = emopt.misc.DomainCoordinates(X-w_pml-4*dx, X-w_pml-4*dx, w_pml, Y-w_pml,
                                             w_pml, Z-w_pml, dx, dy, dz)

    fom_mode = emopt.modes.ModeFullVector(wavelength, eps, mu, fom_slice, n0=3.45,
                                       neigs=4)

    # Need to be consistent with boundary conditions!
    fom_mode.bc = 'H0'
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
    full_field = emopt.misc.DomainCoordinates(w_pml, X-w_pml, w_pml, Y-w_pml,
                                              Z/2, Z/2, dx, dy, dz)
    sim.field_domains = [fom_slice, full_field]

    am = MMISplitterAdjointMethod(sim, domain, fom_slice, mode_match, eps_clad, eps_si, k, zmin, zmax, wg_i, wg_o1, wg_o2)
    am.update_system(design_params)
    am.check_gradient(design_params, fd_step=1e-6)

    #####################################################################################
    # Setup and run the optimization
    #####################################################################################
    # L-BFGS-B will print out the iteration number and FOM value
    fom_list = []
    callback = lambda x: plot_update(x, fom_list, sim, am)
    opt = emopt.optimizer.Optimizer(am, design_params, Nmax=50, opt_method='L-BFGS-B', callback_func=callback)
    fom, pfinal = opt.run()
