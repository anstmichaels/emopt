import emopt
from emopt.adjoint_method import AdjointMethod
from emopt.misc import NOT_PARALLEL, run_on_master

import numpy as np
from math import pi

class NanoAntennaAdjointMethod(AdjointMethod):

    def __init__(self, sim, ant_top, ant_bot, n0):
        super(NanoAntennaAdjointMethod, self).__init__(sim, step=1e-7)

        self.ant_top = ant_top
        self.ant_bot = ant_bot
        self.n0 = n0

        self.W = sim.W
        self.H = sim.H

    def get_current_points(self, params):
        w_base, w_top, h_ant = params
        W = self.W
        H = self.H

        x = np.array([-w_base/2, -w_top/2, w_top/2, w_base/2, -w_base/2]) + W/2
        y = np.array([0, h_ant, h_ant, 0, 0]) + H/2 + h_gap/2

        return x,y

    def update_system(self, params):
        x,y = self.get_current_points(params)

        self.ant_top.set_points(x, y)
        self.ant_bot.set_points(x, H-y)

    @run_on_master
    def calc_fom(self, sim, params):
        Hz, Ex, Ey = sim.saved_fields[0]
        Hz2, Ex2, Ey2 = sim.saved_fields[1]

        eps0 = self.n0**2
        dx = sim.dx
        dy = sim.dy

        # minus -1 so minimize = maximize
        fom1 = -0.5 * dy * np.real(np.sum(Ey*np.conj(Hz)))
        fom2 = 0.5 * dy * np.real(np.sum(Ex2*np.conj(Hz2)))
        return (fom1 + fom2) / 1.76e-4

    @run_on_master
    def calc_dFdx(self, sim, params):
        Hz, Ex, Ey = sim.saved_fields[0]
        Hz2, Ex2, Ey2 = sim.saved_fields[1]
        fom_plane = sim.field_domains[0]
        fom_plane2 = sim.field_domains[1]

        eps0 = self.n0**2
        dx = sim.dx
        dy = sim.dy

        # derivative arrays
        dfdHz = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dfdEx = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dfdEy = np.zeros([sim.M, sim.N], dtype=np.complex128)

        dfdEy[fom_plane.j, fom_plane.k] = -0.25*dy*np.conj(Hz) / 1.76e-4
        dfdHz[fom_plane.j, fom_plane.k] = -0.25*dy*np.conj(Ey) / 1.76e-4

        dfdEx[fom_plane2.j, fom_plane2.k] = 0.25*dy*np.conj(Hz2) / 1.76e-4
        dfdHz[fom_plane2.j, fom_plane2.k] += 0.25*dy*np.conj(Ex2) / 1.76e-4

        dFdHz, dFdEx, dFdEy = emopt.fomutils.interpolated_dFdx_2D(sim, dfdHz,
                                                                  dfdEx, dfdEy)

        return (dFdHz, dFdEx, dFdEy)

    def calc_grad_y(self, sim, params):
        return np.zeros(params.shape)

if __name__ == '__main__':
    ###########################################################################
    # Setup the simulation domain
    ###########################################################################
    W = 3.0
    H = 3.0
    dx = dy = 0.0075
    wavelength = 1.55

    sim = emopt.fdfd.FDFD_TM(W, H, dx, dy, wavelength)
    W = sim.W
    H = sim.H
    M = sim.M
    N = sim.N
    w_pml = sim.w_pml

    ###########################################################################
    # Define the system geometry
    ###########################################################################
    n0 = 1.0
    n1 = 3.5

    # our antenna has a bowtie structure and is symmetric
    w_base = 0.1
    w_top = 0.1
    h_ant = 0.3
    h_gap = 0.03

    xs = np.array([-w_base/2, -w_top/2, w_top/2, w_base/2, -w_base/2]) + W/2
    ys = np.array([0, h_ant, h_ant, 0, 0]) + H/2 + h_gap/2

    ant_top = emopt.grid.Polygon(xs, ys)
    ant_top.layer = 1
    ant_top.material_value = -100.0 - 3j

    ant_bot = emopt.grid.Polygon(xs, H - ys)
    ant_bot.layer = 1
    ant_bot.material_value = -100.0 - 3j

    background = emopt.grid.Rectangle(W/2, H/2, 2*W, 2*H)
    background.layer = 2
    background.material_value = n0**2

    eps = emopt.grid.StructuredMaterial2D(W, H, dx, dy)
    eps.add_primitives([ant_top, ant_bot, background])

    mu = emopt.grid.ConstantMaterial2D(1.0)

    sim.set_materials(eps, mu)

    ###########################################################################
    # Set sources
    ###########################################################################
    # source = dipole at center of antenna
    src_point = emopt.misc.DomainCoordinates(W/2, W/2+dx, H/2, H/2, 0, 0,
                                             dx, dy, 1)

    Mz = np.zeros([M,N], dtype=np.complex128)
    Jx = np.zeros([M,N], dtype=np.complex128)
    Jy = np.zeros([M,N], dtype=np.complex128)

    # Technically we should take the appropriate derivative of these fields,
    # but it turns out that the fields themselves are close enough
    Jy[src_point.j, src_point.k] = 1e2

    sim.set_sources((Mz, Jx, Jy))

    ###########################################################################
    # Finish simulation setup
    ###########################################################################
    fom_area = emopt.misc.DomainCoordinates(W-w_pml[0], W-w_pml[1],
                                            w_pml[2], H-w_pml[3],
                                            0, 0, dx, dy, 1.0)
    fom_area2 = emopt.misc.DomainCoordinates(w_pml[0], W-w_pml[1], w_pml[2],
                                             w_pml[2], 0, 0,
                                             dx, dy, 1.0)
    full_field = emopt.misc.DomainCoordinates(w_pml[0], W-w_pml[1],
                                              w_pml[2], H-w_pml[3],
                                              0, 0, dx, dy, 1.0)
    sim.build()
    sim.field_domains = [fom_area, fom_area2, full_field]

    if(NOT_PARALLEL):
        epsg = eps.get_values_in(full_field)
        import matplotlib.pyplot as plt
        plt.imshow(np.abs(epsg), extent=full_field.get_bounding_box()[0:4],
                   cmap='hot')
        plt.show()

    ###########################################################################
    # Create the AdjointMethod object
    ###########################################################################
    # The design parameters of our structure will be the top and bottow widths
    # of the antenna and the length of the antenna.
    design_params = np.array([w_base, w_top, h_ant])
    am = NanoAntennaAdjointMethod(sim, ant_top, ant_bot, n0)

    am.check_gradient(design_params)

    bounds = [(None, None) for i in range(3)]
    opt = emopt.optimizer.Optimizer(am, design_params, opt_method='L-BFGS-B',
                                    Nmax=3, bounds=bounds)
    fom, params = opt.run()

    x,y = am.get_current_points(params)

    Ey = sim.saved_fields[2][0]
    if(NOT_PARALLEL):
        import matplotlib.pyplot as plt
        plt.plot(x,y)
        plt.axis('equal')
        plt.show()

        plt.imshow(np.abs(Ey), extent=[0,W-2,0,H-2])
        plt.show()
