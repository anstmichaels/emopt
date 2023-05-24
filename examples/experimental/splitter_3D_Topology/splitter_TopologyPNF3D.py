import emopt
from emopt.misc import NOT_PARALLEL, run_on_master
from emopt.experimental.adjoint_method import TopologyPNF2D, Topology, TopologyPNF3D
from emopt.experimental.fdtd import FDTD
from emopt.experimental.adjoint_method import sigmoid
from emopt.experimental.grid import TopologyMaterial3D
import numpy as np

class TopologyAM(TopologyPNF3D):
    def __init__(self, sim, mode_match, mm_line, domain=None, update_mu=False, eps_bounds=None, mu_bounds=None, lam=0):
        super().__init__(sim, domain=domain, update_mu=update_mu, eps_bounds=eps_bounds, mu_bounds=mu_bounds, penalty_multiplier=lam)
        self.mode_match = mode_match
        self.current_fom = 0.0
        self.fom_domain = mm_line

    @run_on_master
    def calc_f(self, sim, params):
        Ex, Ey, Ez, Hx, Hy, Hz = sim.saved_fields[0]

        self.mode_match.compute(Ex, Ey, Ez, Hx, Hy, Hz)
        fom = -1*self.mode_match.get_mode_match_forward(1.0)

        return fom

    @run_on_master
    def calc_dfdx(self, sim, params):
        Psrc = sim.source_power

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
        return np.zeros(params.shape)

def plot_update(params, fom_list, sim, am):
    print('Finished iteration %d' % (len(fom_list)+1))
    current_fom = -1*am.calc_fom(sim, params)
    fom_list.append(current_fom)

    #Ez, Hx, Hy = sim.saved_fields[1]
    Ex, Ey, Ez, Hx, Hy, Hz = sim.saved_fields[1]
    eps = sim.eps.get_values_in(sim.field_domains[1]).squeeze()
    #Ez = np.squeeze(Ez)
    #Ey = np.squeeze(Ey)
    Hz = np.squeeze(Hz)

    foms = {'E2' : fom_list}
    emopt.io.plot_iteration(np.flipud(Hz.real), np.flipud(eps.real), sim.X,
                            sim.Y, foms, fname='current_result1.pdf',
                            dark=False)

    #data = {}
    #data['Ez'] = Ez
    #data['Hx'] = Hx
    #data['Hy'] = Hy
    #data['eps'] = eps
    #data['params'] = params
    #data['foms'] = fom_list

    #fname = 'data/topology_results'
    #emopt.io.save_results(fname, data)

if __name__ == '__main__':
    ####################################################################################
    # define the system parameters
    ####################################################################################
    wavelength = 1.31
    #X = 20.0
    #X = 16.0
    X = 12.0
    Y = 8.0
    Z = 4.0
    dx = 0.04
    dy = dx
    dz = dx

    # create the simulation object.
    # TE => Ez, Hx, Hy
    sim = FDTD(X,Y,Z,dx,dy,dz,wavelength, rtol=1e-5, min_rindex=1.44,
                      nconv=200)
    sim.Nmax = 1000*sim.Ncycle

    # Get the actual width and height
    X = sim.X
    Y = sim.Y
    Z = sim.Z
    Nx = sim.Nx
    Ny = sim.Ny
    Nz = sim.Nz
    w_pml = sim.w_pml[0] # PML width which is the same on all boundaries by default

    ####################################################################################
    # Define the structure
    ####################################################################################
    optdomain = emopt.misc.DomainCoordinates(0.5*X-2.2, 0.5*X+2.2, 0.5*Y-1.5, 0.5*Y+1.5, 0.5*Z-0.15, 0.5*Z+0.15, dx, dy, dz)

    n_si = emopt.misc.n_silicon(wavelength)
    n_sio2 = 1.444
    eps_si = n_si**2
    eps_clad = n_sio2**2

    in_wg = emopt.grid.Rectangle(0.0, Y/2.0, 10.0, 0.5)
    in_wg.layer = 2
    in_wg.material_value = eps_si

    out_wg1 = emopt.grid.Rectangle(X, Y/2.0+1.0, 10.0, 0.5)
    out_wg1.layer = 2
    out_wg1.material_value = eps_si

    out_wg2 = emopt.grid.Rectangle(X, Y/2.0-1.0, 10.0, 0.5)
    out_wg2.layer = 2
    out_wg2.material_value = eps_si

    init_rect = emopt.grid.Rectangle(X/2, Y/2, 4.4+2*dx, 3.0+2*dy)
    init_rect.layer = 1
    #init_rect.material_value = eps_si
    init_rect.material_value = eps_si/2

    bg = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y)
    bg.layer = 3
    bg.material_value = eps_clad

    eps = emopt.grid.StructuredMaterial3D(X, Y, Z, dx, dy, dz)

    eps.add_primitive(in_wg, 0.5*Z-0.15, 0.5*Z+0.15)
    eps.add_primitive(out_wg1, 0.5*Z-0.15, 0.5*Z+0.15)
    eps.add_primitive(out_wg2, 0.5*Z-0.15, 0.5*Z+0.15)
    eps.add_primitive(init_rect, 0.5*Z-0.15, 0.5*Z+0.15)
    eps.add_primitive(bg, -Z, 2*Z)

    eps_top = TopologyMaterial3D(eps, optdomain)

    # set up the magnetic permeability -- just 1.0 everywhere
    mu = emopt.grid.ConstantMaterial3D(1.0)

    # add the materials and build the system
    sim.set_materials(eps_top, mu)

    ####################################################################################
    # Setup the sources
    ####################################################################################
    w_src= 5.0

    # place the source in the simulation domain
    src_line = emopt.misc.DomainCoordinates(w_pml+5*dx, w_pml+5*dx, Y/2-w_src/2,
                                 Y/2+w_src/2, w_pml, Z-w_pml, dx, dy, dz)

    # Setup the mode solver.    
    mode = emopt.modes.ModeFullVector(wavelength, eps, mu, src_line, n0=n_si, neigs=4)

    if(NOT_PARALLEL):
        print('Generating mode data...')

    mode.build()
    mode.solve()

    # at this point we have found the modes but we dont know which mode is the
    # one we fundamental mode.  We have a way to determine this, however
    #mindex = mode.find_mode_index(0,0)

    # set the current sources using the mode solver object
    #sim.set_sources(mode, src_line, mindex)
    sim.set_sources(mode, src_line)

    ####################################################################################
    # Setup the field domains for FOM calc
    ####################################################################################
    full_field = emopt.misc.DomainCoordinates(w_pml, X-w_pml, w_pml, Y-w_pml, 0.5*Z, 0.5*Z,
                                              dx, dy, dz)
    w_mode = 6.0

    if NOT_PARALLEL:
        import matplotlib.pyplot as plt
        f = plt.figure()
        ax = f.add_subplot(111)
        ar = eps.get_values_in(full_field, squeeze=True)
        ax.imshow(ar.real)
        plt.show()

    # place the source in the simulation domain
    mode_line = emopt.misc.DomainCoordinates(X-(w_pml+5*dx), X-(w_pml+5*dx), Y/2-w_mode/2,
                                 Y/2+w_mode/2, w_pml, Z-w_pml, dx, dy, dz)

    # Setup the mode solver.    
    mode_fom = emopt.modes.ModeFullVector(wavelength, eps, mu, mode_line, n0=n_si, neigs=4)

    if(NOT_PARALLEL):
        print('Generating mode data...')

    mode_fom.build()
    mode_fom.solve()

    # at this point we have found the modes but we dont know which mode is the
    # one we fundamental mode.  We have a way to determine this, however
    #mindex = mode_fom.find_mode_index(0)
    Exm = mode_fom.get_field_interp(0, 'Ex')
    Eym = mode_fom.get_field_interp(0, 'Ey')
    Ezm = mode_fom.get_field_interp(0, 'Ez')
    Hxm = mode_fom.get_field_interp(0, 'Hx')
    Hym = mode_fom.get_field_interp(0, 'Hy')
    Hzm = mode_fom.get_field_interp(0, 'Hz')
    #if not NOT_PARALLEL:
    #    Exm = emopt.misc.MathDummy()
    #    Eym = emopt.misc.MathDummy()
    #    Ezm = emopt.misc.MathDummy()
    #    Hxm = emopt.misc.MathDummy()
    #    Hym = emopt.misc.MathDummy()
    #    Hzm = emopt.misc.MathDummy()

    mode_match = emopt.fomutils.ModeMatch([1,0,0], sim.dy, sim.dz, Exm, Eym, Ezm, Hxm, Hym, Hzm)

    sim.field_domains = [mode_line, full_field]

    ####################################################################################
    # Build the system
    ####################################################################################
    sim.build()

    ####################################################################################
    # Setup the optimization
    ####################################################################################
    am = TopologyAM(sim, mode_match, mode_line, domain=optdomain, update_mu=False, eps_bounds=[eps_clad, eps_si], lam=0.0)
    #am = TopologyAM(sim, mode_match, mode_line, domain=optdomain, update_mu=False, eps_bounds=None, lam=0.0)
    #am = TopologyAM(sim, mode_match, mode_line, domain=optdomain, update_mu=False, eps_bounds=None)
    design_params = am.get_params()
    #design_params = np.ones_like(design_params)*eps_si/2
    design_params = np.zeros_like(design_params)
    am.update_system(design_params)

    if NOT_PARALLEL:
        import matplotlib.pyplot as plt
        f = plt.figure()
        ax = f.add_subplot(111)
        ar = am.sim.eps.get_values_in(full_field, squeeze=True)
        ax.imshow(ar.real)
        plt.show()
    #assert False
    #am.check_gradient(design_params, indices=np.arange(100)[::50], fd_step=1e-6)
    am.check_gradient(design_params, indices=np.arange(100)[::50], fd_step=1e-3)

    fom_list = []
    callback = lambda x : plot_update(x, fom_list, sim, am)

    # setup and run the optimization!
    opt = emopt.optimizer.Optimizer(am, design_params, tol=1e-5,
                                    callback_func=callback,
                                    opt_method='L-BFGS-B',
                                    Nmax=300)

    # Run the optimization
    final_fom, final_params = opt.run()
