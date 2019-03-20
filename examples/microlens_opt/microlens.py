import emopt
from emopt.adjoint_method import AdjointMethod
from emopt.misc import NOT_PARALLEL, run_on_master

import numpy as np
from math import pi

class LensAdjointMethod(AdjointMethod):

    def __init__(self, sim, xs, ys, xc, yc, Nc, Np, lens, lens_surf, n0):
        super(LensAdjointMethod, self).__init__(sim, step=1e-7)

        self.xs = xs
        self.ys = ys
        self.xc = xc
        self.yc = yc
        self.Nc = Nc
        self.Np = Np
        self.lens = lens
        self.spline = lens_surf
        self.n0 = n0

    def get_current_points(self, params):
        x = np.copy(self.xs)
        y = np.copy(self.ys)

        xc = np.copy(self.xc)
        yc = np.copy(self.yc)
        Nc = self.Nc
        Np = self.Np

        xc = xc + params[0:Nc]
        #yc += params[Nc:]

        self.spline.set_cpoints(xc, yc)
        xs, ys = self.spline.evaluate(Neval=Np)

        x[2:2+Np] = xs
        y[2:2+Np] = ys

        return x,y

    def update_system(self, params):
        x,y = self.get_current_points(params)
        self.lens.set_points(x,y)

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
    W = 13.0
    H = 14.0
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

    h_lens = 10.0
    w_lens = wavelength/n1
    x_lens = 3.0
    Np = 50

    # our lens consists initial of a rectangle.
    # one face of the rectangle will have many points to manipulate
    xs = [x_lens-w_lens/2, x_lens-w_lens/2]
    xs += list(np.ones(Np)*(x_lens+w_lens/2))
    xs += xs[0:1]

    ys = [H/2-h_lens/2, H/2+h_lens/2]
    ys += list(np.linspace(H/2+h_lens/2, H/2-h_lens/2, Np))
    ys += ys[0:1]

    lens = emopt.grid.Polygon(xs, ys)
    lens.layer = 1
    lens.material_value = n1**2

    background = emopt.grid.Rectangle(W/2, H/2, 2*W, 2*H)
    background.layer = 2
    background.material_value = n0**2

    eps = emopt.grid.StructuredMaterial2D(W, H, dx, dy)
    eps.add_primitives([lens, background])

    mu = emopt.grid.ConstantMaterial2D(1.0)

    sim.set_materials(eps, mu)

    ###########################################################################
    # Set sources
    ###########################################################################
    # We (approximately) inject a Gaussian beam
    h_src = 12.0
    src_plane = emopt.misc.DomainCoordinates(w_pml[0]+dx, w_pml[0]+dx,
                                             H/2-h_src/2, H/2+h_src/2,
                                             0, 0, dx, dy, 1.0)

    Jz = np.zeros([M,N], dtype=np.complex128)
    Mx = np.zeros([M,N], dtype=np.complex128)
    My = np.zeros([M,N], dtype=np.complex128)

    w_src = 3.0
    Ez, Hy, Hx = emopt.misc.gaussian_fields(src_plane.y-H/2, 0.0, 0.0, w_src, 0.0,
                                           wavelength, n0)

    # Technically we should take the appropriate derivative of these fields,
    # but it turns out that the fields themselves are close enough
    Jz[src_plane.j, src_plane.k] = np.reshape(Ez, src_plane.shape[1:]) * 1e2
    Mx[src_plane.j, src_plane.k] = np.reshape(Hx, src_plane.shape[1:]) * 1e2
    My[src_plane.j, src_plane.k] = np.reshape(Hy, src_plane.shape[1:]) * 1e2

    sim.set_sources((Jz, Mx, My))

    ###########################################################################
    # Finish simulation setup
    ###########################################################################
    fom_area = emopt.misc.DomainCoordinates(x_lens+8.0, x_lens+8.2, H/2-0.1+2.0,
                                            H/2+0.1+2.0,
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
    Nc = 15

    xc = np.ones(Nc)*(x_lens+w_lens/2)
    yc = np.linspace(H/2+h_lens/2, H/2-h_lens/2,Nc)
    lens_surf = emopt.geometry.NURBS(xc, yc)

    design_params = np.zeros(Nc)
    am = LensAdjointMethod(sim, xs, ys, xc, yc, Nc, Np, lens, lens_surf, n0)

    #am.check_gradient(design_params)

    bounds = [(-w_lens/2, w_lens*10) for i in range(Nc)]
    opt = emopt.optimizer.Optimizer(am, design_params, opt_method='L-BFGS-B',
                                    Nmax=20)
    fom, params = opt.run()
    x,y = am.get_current_points(params)

    Ez = sim.saved_fields[1][0]
    if(NOT_PARALLEL):
        import matplotlib.pyplot as plt

        extent = full_field.get_bounding_box()[0:4]
        plt.imshow(np.abs(Ez), extent=extent)
        plt.plot(x,y,'w',linewidth=0.5)
        plt.xlim(extent[0:2])
        plt.ylim(extent[2:])
        plt.show()
