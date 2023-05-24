import emopt
from emopt.misc import NOT_PARALLEL, run_on_master
from emopt.experimental.adjoint_method import TopologyPNF2D
from emopt.experimental.fdfd import FDFD_TE
from emopt.experimental.grid import TopologyMaterial2D
import numpy as np

class TopologyAM(TopologyPNF2D):
    def __init__(self, sim, mode_match, mm_line, domain=None, update_mu=False, eps_bounds=None, mu_bounds=None, planar=False, lam=0):
        super().__init__(sim, domain=domain, update_mu=update_mu, eps_bounds=eps_bounds, mu_bounds=mu_bounds, planar=planar, penalty_multiplier=lam)
        self.mode_match = mode_match
        self.current_fom = 0.0
        self.mm_line = mm_line

    @run_on_master
    def calc_f(self, sim, params):
        Ez, Hx, Hy = sim.saved_fields[0]

        self.mode_match.compute(Ez=Ez, Hx=Hx, Hy=Hy)

        fom = -self.mode_match.get_mode_match_forward(1.0)

        return fom

    @run_on_master
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
    print('Finished iteration %d' % (len(fom_list)+1))
    current_fom = -1*am.calc_fom(sim, params)
    fom_list.append(current_fom)

    Ez, Hx, Hy = sim.saved_fields[1]
    eps = sim.eps.get_values_in(sim.field_domains[1])

    foms = {'E2' : fom_list}
    emopt.io.plot_iteration(np.flipud(Ez.real), np.flipud(eps.real), sim.Xreal,
                            sim.Yreal, foms, fname='current_result1.pdf',
                            dark=False)

    data = {}
    data['Ez'] = Ez
    data['Hx'] = Hx
    data['Hy'] = Hy
    data['eps'] = eps
    data['params'] = params
    data['foms'] = fom_list

    i = len(fom_list)
    fname = 'data/topology_results'
    emopt.io.save_results(fname, data)

if __name__ == '__main__':
    ####################################################################################
    # define the system parameters
    ####################################################################################
    wavelength = 1.31
    X = 12.0
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
    #optdomain = emopt.misc.DomainCoordinates(0.5*X-6.2, 0.5*X+6.2, 0.5*Y-2.2, 0.5*Y+2.2, 0.0, 0.0, dx, dy, 1.0)
    #optdomain = emopt.misc.DomainCoordinates(0.5*X-6.2, 0.5*X+6.2, 0.5*Y-2.2, 0.5*Y+2.2, 0.0, 0.0, dx, dy, 1.0)
    #optdomain = emopt.misc.DomainCoordinates(0.5*X-4.2, 0.5*X+4.2, 0.5*Y-2.2, 0.5*Y+2.2, 0.0, 0.0, dx, dy, 1.0)
    optdomain = emopt.misc.DomainCoordinates(0.5*X-2.2, 0.5*X+2.2, 0.5*Y-1.5, 0.5*Y+1.5, 0.0, 0.0, dx, dy, 1.0)

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

    init_rect = emopt.grid.Rectangle(X/2, Y/2, 4.4, 3.0)
    init_rect.layer = 1
    init_rect.material_value = eps_si/2

    bg = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y)
    bg.layer = 3
    bg.material_value = eps_clad

    eps = emopt.grid.StructuredMaterial2D(X, Y, dx, dy)

    eps.add_primitive(in_wg)
    eps.add_primitive(out_wg1)
    eps.add_primitive(out_wg2)
    eps.add_primitive(init_rect)
    eps.add_primitive(bg)

    eps_top = TopologyMaterial2D(eps, optdomain)

    # set up the magnetic permeability -- just 1.0 everywhere
    mu = emopt.grid.ConstantMaterial2D(1.0)

    # add the materials and build the system
    sim.set_materials(eps_top, mu)

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
    # Setup the field domains for FOM calc
    ####################################################################################
    full_field = emopt.misc.DomainCoordinates(w_pml, X-w_pml, w_pml, Y-w_pml, 0.0, 0.0,
                                              dx, dy, 1.0)
    w_mode = 6.0

    # place the source in the simulation domain
    mode_line = emopt.misc.DomainCoordinates(X-(w_pml+2*dx), X-(w_pml+2*dx), Y/2-w_mode/2,
                                 Y/2+w_mode/2, 0, 0, dx, dy, 1.0)

    # Setup the mode solver.
    mode_fom = emopt.modes.ModeTE(wavelength, eps, mu, mode_line, n0=n_si, neigs=4)

    if(NOT_PARALLEL):
        print('Generating mode data...')

    mode_fom.build()
    mode_fom.solve()

    # at this point we have found the modes but we dont know which mode is the
    # one we fundamental mode.  We have a way to determine this, however
    #mindex = mode_fom.find_mode_index(0)
    Ezm = mode_fom.get_field_interp(0, 'Ez')
    Hxm = mode_fom.get_field_interp(0, 'Hx')
    Hym = mode_fom.get_field_interp(0, 'Hy')
    if not NOT_PARALLEL:
        Ezm = emopt.misc.MathDummy()
        Hxm = emopt.misc.MathDummy()
        Hym = emopt.misc.MathDummy()

    mode_match = emopt.fomutils.ModeMatch([1,0,0], sim.dy, Ezm=Ezm, Hxm=Hxm, Hym=Hym)

    sim.field_domains = [mode_line, full_field]

    ####################################################################################
    # Build the system
    ####################################################################################
    sim.build()

    ####################################################################################
    # Setup the optimization
    ####################################################################################
    am = TopologyAM(sim, mode_match, mode_line, domain=optdomain, update_mu=False, eps_bounds=[eps_clad-0.1, eps_si+0.1], lam=0.0)
    design_params = am.get_params()
    am.update_system(design_params)
    am.check_gradient(design_params, indices=np.arange(100)[::2], fd_step=1e-4)

    fom_list = []
    callback = lambda x : plot_update(x, fom_list, sim, am)

    # setup and run the optimization!
    opt = emopt.optimizer.Optimizer(am, design_params, tol=1e-5,
                                    callback_func=callback,
                                    opt_method='L-BFGS-B',
                                    Nmax=300)

    # Run the optimization
    final_fom, final_params = opt.run()
