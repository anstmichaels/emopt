"""
This example demonstrates how to optimize a very simple waveguide bend in 2D.
This serves as a good starting point for learning how to optimize structures
that are defined using polygons.

To run the code, execute::

    mpirun -n 8 python wg_bend.py

in the command line which will run the optimization using 8 cores on the current
machine."""
import emopt
from emopt.adjoint_method import AdjointMethod
from emopt.misc import NOT_PARALLEL, run_on_master

import numpy as np
from math import pi

class WGBendAM(AdjointMethod):

    def __init__(self, sim, xs, ys, waveguide, Psrc):
        super(WGBendAM, self).__init__(sim, step=1e-12)

        self.xs = xs
        self.ys = ys
        self.waveguide = waveguide
        self.Psrc = Psrc

        self.current_fom = 0

    def get_current_points(self, params):
        x = np.copy(self.xs)
        y = np.copy(self.ys)

        Rin, Rout = params
        mr = np.zeros(len(x)); mr[1] = 1
        x, y = emopt.geometry.fillet(x, y, Rout, make_round=mr,
                                       points_per_bend=50, ignore_roc_lim=True)
        mr = np.zeros(len(x)); mr[53] = 1
        x, y = emopt.geometry.fillet(x, y, Rin, make_round=mr,
                                       points_per_bend=50, ignore_roc_lim=True)

        return x,y

    def update_system(self, params):
        x,y = self.get_current_points(params)
        self.waveguide.set_points(x,y)

    @run_on_master
    def calc_fom(self, sim, params):
        Hz, Ex, Ey = sim.saved_fields[0]

        dx = sim.dx
        dy = sim.dy

        # minus -1 so minimize = maximize
        self.current_fom = -0.5 * dx * np.sum( np.real(Ex*np.conj(Hz)) ) / self.Psrc
        return self.current_fom

    @run_on_master
    def calc_dFdx(self, sim, params):
        Hz, Ex, Ey = sim.saved_fields[0]
        src_plane = sim.field_domains[0]

        dx = sim.dx
        dy = sim.dy

        # derivative arrays
        dfdHz = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dfdEx = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dfdEy = np.zeros([sim.M, sim.N], dtype=np.complex128)

        dfdHz[src_plane.j, src_plane.k] = -0.25 * dx * np.conj(Ex) / self.Psrc
        dfdEx[src_plane.j, src_plane.k] = -0.25 * dx * np.conj(Hz) / self.Psrc

        dFdHz, dFdEx, dFdEy = emopt.fomutils.interpolated_dFdx_2D(sim, dfdHz,
                                                                  dfdEx, dfdEy)

        return (dFdHz, dFdEx, dFdEy)

    def calc_grad_p(self, sim, params):
        return np.zeros(params.shape)

def callback(params, sim, am, fom_history, Ts, Exm, Eym, Hzm):
    fom = am.current_fom
    fom_history.append(fom)

    Hz, Ex, Ey = sim.saved_fields[1]
    x, y = am.get_current_points(params)
    eps = sim.eps.get_values_in(sim.field_domains[1])

    Hzt, Ext, Eyt = sim.saved_fields[2]
    mm = emopt.fomutils.ModeMatch([0,-1,0], sim.dx, Exm=Exm, Eym=Eym, Hzm=Hzm)
    mm.compute(Ex=Ext, Ey=Eyt, Hz=Hzt)
    T = mm.get_mode_match_forward(sim.source_power)
    Ts.append(T)

    foms = {'Ts': Ts}
    emopt.io.plot_iteration(np.flipud(Hz.real), np.flipud(eps.real), sim.Xreal, sim.Yreal, foms, fname='current_result.pdf', dark=False)

    #Hza = sim.get_adjoint_field('Hz')

    #data = {}
    #data['Hz'] = Hz
    #data['Ex'] = Ex
    #data['Ey'] = Ey
    #data['foms'] = fom_history
    #data['dx'] = sim.dx
    #data['dy'] = sim.dy
    #data['X'] = sim.X
    #data['Y'] = sim.Y

    #additional = {}
    #additional['Ts'] = Ts
    #additional['Hza'] = Hza
    #additional['x'] = x
    #additional['y'] = y
    #additional['bbox'] = sim.field_domains[1].get_bounding_box()[0:4]
    #additional['source_power'] = sim.source_power
    #additional['w_pml'] = w_pml

    #fname = 'data/wg_bend_%d' % (len(fom_history))
    #emopt.io.save_results(fname, data, additional)


if __name__ == '__main__':
    ###########################################################################
    # Setup the simulation domain
    ###########################################################################
    X = 6.0
    Y = 6.0
    dx = dy = 0.03
    wavelength = 1.55

    sim = emopt.fdfd.FDFD_TM(X, Y, dx, dy, wavelength)
    X = sim.X
    Y = sim.Y
    M = sim.M
    N = sim.N
    w_pml = sim.w_pml

    ###########################################################################
    # Define the system geometry
    ###########################################################################
    n0 = 1.44
    n1 = 2.8 # approx effective index

    w_wg = 0.5
    Np = 150

    L_des = 1.5
    wg_pos = 4.0
    xs = [-1, wg_pos+w_wg/2, wg_pos+w_wg/2, wg_pos-w_wg/2,
          wg_pos-w_wg/2, -1, -1]
    ys = [wg_pos + w_wg/2, wg_pos+w_wg/2,-1.0, -1.0,
          wg_pos-w_wg/2, wg_pos-w_wg/2, wg_pos+w_wg/2]

    waveguide = emopt.grid.Polygon(xs, ys)
    waveguide.layer = 1
    waveguide.material_value = n1**2

    background = emopt.grid.Rectangle(wg_pos, wg_pos, 2*X, 2*Y)
    background.layer = 2
    background.material_value = n0**2

    eps = emopt.grid.StructuredMaterial2D(X, Y, dx, dy)
    eps.add_primitives([waveguide, background])

    mu = emopt.grid.ConstantMaterial2D(1.0)

    sim.set_materials(eps, mu)

    ###########################################################################
    # Set sources
    ###########################################################################
    h_src = 2.0
    src_plane = emopt.misc.DomainCoordinates(w_pml[0]+5*dx, w_pml[0]+5*dx,
                                             wg_pos-h_src/2, wg_pos+h_src/2,
                                             0, 0, dx, dy, 1.0)

    mode = emopt.modes.ModeTM(wavelength, eps, mu, src_plane, n0=n1, neigs=4)
    mode.build()
    mode.solve()

    sim.set_sources(mode, src_plane)

    # save mode fields for analysis
    T_area = emopt.misc.DomainCoordinates(wg_pos-h_src/2, wg_pos+h_src/2,
                                            w_pml[2]+5*dy, w_pml[2] + 5*dy,
                                            0, 0, dx, dy, 1.0)
    modem = emopt.modes.ModeTM(wavelength, eps, mu, T_area, n0=n1, neigs=4)
    modem.build()
    modem.solve()

    ## The mode solver assumes the mode propagates in the +x direction. Since
    # the actual field is propagating in the y direction, we need to permut the
    # Ex and Ey components. (This should really be handled by the mode
    # solver...)
    Hzm = modem.get_field_interp(0, 'Hz')
    Exm = modem.get_field_interp(0, 'Ey')
    Eym = modem.get_field_interp(0, 'Ex')

    ###########################################################################
    # Finish simulation setup
    ###########################################################################
    w_fom = w_wg
    fom_area = emopt.misc.DomainCoordinates(wg_pos-w_fom/2, wg_pos+w_fom/2,
                                            w_pml[2]+5*dy, w_pml[2] + 5*dy,
                                            0, 0, dx, dy, 1.0)
    full_field = emopt.misc.DomainCoordinates(w_pml[0], X-w_pml[1],
                                              w_pml[2], Y-w_pml[3],
                                              0, 0, dx, dy, 1.0)
    sim.build()
    sim.field_domains = [fom_area, full_field, T_area]

    # get starting source power--for simplicity, assume constant for rest of
    # opt
    sim.solve_forward()
    Psrc = sim.source_power

    ###########################################################################
    # Create the AdjointMethod object
    ###########################################################################
    # We will parameterize our structure in terms of the inner and outer radii
    # of the bend
    design_params = np.array([0.25, 0.25+w_wg])
    am = WGBendAM(sim, xs, ys, waveguide, Psrc)

    am.check_gradient(design_params)

    fom_history = []
    Ts = []
    callback_func = lambda p : callback(p, sim, am, fom_history, Ts, Exm, Eym, Hzm)

    bounds = [(0,wg_pos*0.75), (0,wg_pos*0.75)]
    opt = emopt.optimizer.Optimizer(am, design_params, opt_method='L-BFGS-B',
                                    Nmax=10, bounds=bounds,
                                    callback_func=callback_func)
    fom, params = opt.run()
    xs,ys = am.get_current_points(params)

    Ez = sim.saved_fields[1][0]
    if(NOT_PARALLEL):
        import matplotlib.pyplot as plt

        extent = full_field.get_bounding_box()[0:4]
        plt.imshow(np.flipud(np.abs(Ez)), extent=extent)
        plt.plot(xs,ys,'w',linewidth=0.5)
        plt.xlim(extent[0:2])
        plt.ylim(extent[2:])
        plt.show()
