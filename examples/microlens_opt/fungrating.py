import emopt
from emopt.adjoint_method import AdjointMethod
from emopt.misc import NOT_PARALLEL, run_on_master

import numpy as np
from math import pi

class LensAdjointMethod(AdjointMethod):

    def __init__(self, sim, xs, ys, xc, yc, Nc, Np, waveguide, gspline, n0):
        super(LensAdjointMethod, self).__init__(sim, step=1e-7)

        self.xs = xs
        self.ys = ys
        self.xc = xc
        self.yc = yc
        self.Nc = Nc
        self.Np = Np
        self.waveguide = waveguide
        self.gspline = gspline
        self.n0 = n0

    def get_current_points(self, params):
        x = np.copy(self.xs)
        y = np.copy(self.ys)

        xc = np.copy(self.xc)
        yc = np.copy(self.yc)
        Nc = self.Nc
        Np = self.Np

        yc[1:-1] += params

        self.gspline.set_cpoints(xc, yc)
        xs, ys = self.gspline.evaluate(Neval=Np)

        x[1:1+Np] = xs
        y[1:1+Np] = ys

        return x,y

    def update_system(self, params):
        x,y = self.get_current_points(params)
        self.waveguide.set_points(x,y)

    @run_on_master
    def calc_fom(self, sim, params):
        Ez, Hx, Hy = sim.saved_fields[0]

        eps0 = self.n0**2
        dx = sim.dx
        dy = sim.dy

        # minus -1 so minimize = maximize
        return -1 * eps0 * dx * dy * np.real(np.sum(Ez*np.conj(Ez)))

    @run_on_master
    def calc_dFdx(self, sim, params):
        Ez, Hx, Hy = sim.saved_fields[0]
        src_plane = sim.field_domains[0]

        eps0 = self.n0**2
        dx = sim.dx
        dy = sim.dy

        # derivative arrays
        dfdEz = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dfdHx = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dfdHy = np.zeros([sim.M, sim.N], dtype=np.complex128)

        dfdEz[src_plane.j, src_plane.k] = -1*eps0*dx*dy*np.conj(Ez)

        dFdEz, dFdHx, dFdHy = emopt.fomutils.interpolated_dFdx_2D(sim, dfdEz,
                                                                  dfdHx, dfdHy)

        return (dFdEz, dFdHx, dFdHy)

    def calc_grad_y(self, sim, params):
        return np.zeros(params.shape)

if __name__ == '__main__':
    ###########################################################################
    # Setup the simulation domain
    ###########################################################################
    W = 12.0
    H = 6.0
    dx = dy = 0.04
    wavelength = 1.55

    sim = emopt.fdfd.FDFD_TE(W, H, dx, dy, wavelength)
    W = sim.W
    H = sim.H
    M = sim.M
    N = sim.N
    w_pml = sim.w_pml

    ###########################################################################
    # Define the system geometry
    ###########################################################################
    n0 = 1.44
    n1 = 3.5

    h_wg = 0.22
    Np = 150

    # our lens consists initial of a rectangle.
    # one face of the rectangle will have many points to manipulate
    xs = [-1.0]
    xs += list(np.linspace(2.0,10.0,Np))
    xs += [13.0, 13.0, -1.0]
    xs += xs[0:1]

    ys = [H/4+h_wg/2]
    ys += list(np.ones(Np)*(H/4+h_wg/2))
    ys += [H/4+h_wg/2, H/4-h_wg/2, H/4-h_wg/2]
    ys += ys[0:1]

    waveguide = emopt.grid.Polygon(xs, ys)
    waveguide.layer = 1
    waveguide.material_value = n1**2

    background = emopt.grid.Rectangle(W/2, H/2, 2*W, 2*H)
    background.layer = 2
    background.material_value = n0**2

    eps = emopt.grid.StructuredMaterial2D(W, H, dx, dy)
    eps.add_primitives([waveguide, background])

    mu = emopt.grid.ConstantMaterial2D(1.0)

    sim.set_materials(eps, mu)

    ###########################################################################
    # Set sources
    ###########################################################################
    # We (approximately) inject a Gaussian beam
    h_src = 2.0
    src_plane = emopt.misc.DomainCoordinates(w_pml[0]+dx, w_pml[0]+dx,
                                             H/4-h_src/2, H/4+h_src/2,
                                             0, 0, dx, dy, 1.0)

    mode = emopt.modes.ModeTE(wavelength, eps, mu, src_plane, n0=n1, neigs=4)
    mode.build()
    mode.solve()

    sim.set_sources(mode, src_plane)

    ###########################################################################
    # Finish simulation setup
    ###########################################################################
    fom_area = emopt.misc.DomainCoordinates(W/2+2.5, W/2+2.7, H/2-0.1+1.0,
                                            H/2+0.1+1.0,
                                            0, 0, dx, dy, 1.0)
    full_field = emopt.misc.DomainCoordinates(w_pml[0], W-w_pml[1],
                                              w_pml[2], H-w_pml[3],
                                              0, 0, dx, dy, 1.0)
    sim.build()
    sim.field_domains = [fom_area, full_field]

    ###########################################################################
    # Create the AdjointMethod object
    ###########################################################################
    # We will parameterize our structure using a spline with a couple of
    # control points. The control points will be our design variables
    Nc = 100

    xc = np.linspace(2.0,10.0,Nc)
    yc = np.ones(Nc)*(H/4+h_wg/2); yc[::4] += 0.02
    gspline = emopt.geometry.NURBS(xc, yc)

    design_params = np.zeros(Nc-2)
    am = LensAdjointMethod(sim, xs, ys, xc, yc, Nc, Np, waveguide, gspline, n0)

    #am.check_gradient(design_params)

    bounds = [(-h_wg, 0) for i in range(Nc-2)]
    opt = emopt.optimizer.Optimizer(am, design_params, opt_method='L-BFGS-B',
                                    Nmax=20, bounds=bounds, tol=0)
    fom, params = opt.run()
    x,y = am.get_current_points(params)

    Ez = sim.saved_fields[1][0]
    if(NOT_PARALLEL):
        import matplotlib.pyplot as plt

        extent = full_field.get_bounding_box()[0:4]
        plt.imshow(np.flipud(np.abs(Ez)), extent=extent)
        plt.plot(x,y,'w',linewidth=0.5)
        plt.xlim(extent[0:2])
        plt.ylim(extent[2:])
        plt.show()
