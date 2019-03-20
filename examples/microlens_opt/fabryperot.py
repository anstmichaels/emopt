import emopt
from emopt.adjoint_method import AdjointMethod
from emopt.misc import NOT_PARALLEL, run_on_master

import numpy as np
from math import pi

class LensAdjointMethod(AdjointMethod):

    def __init__(self, sim, xs, ys, xc, yc, Nc, Np, mirror1, mirror2, mspline1,
                 mspline2, n0):
        super(LensAdjointMethod, self).__init__(sim, step=1e-8)

        self.xs = xs
        self.ys = ys
        self.xc = xc
        self.yc = yc
        self.Nc = Nc
        self.Np = Np
        self.mirror1 = mirror1
        self.mirror2 = mirror2
        self.mspline1 = mspline1
        self.mspline2 = mspline2
        self.n0 = n0

        self.W = sim.W
        self.H = sim.H

    def get_current_points(self, params):
        x1 = np.copy(self.xs)
        y1 = np.copy(self.ys)
        x2 = np.copy(self.xs)
        y2 = np.copy(self.ys)

        xc1 = np.copy(self.xc)
        yc1 = np.copy(self.yc)
        xc2 = np.copy(self.xc)
        yc2 = np.copy(self.yc)
        Nc = self.Nc
        Np = self.Np

        xc1 = xc1 + params[0:Nc]
        xc2 = xc2 + params[Nc:]

        self.mspline1.set_cpoints(xc1, yc1)
        xs1, ys1 = self.mspline1.evaluate(Neval=Np/2)

        self.mspline2.set_cpoints(xc2, yc2)
        xs2, ys2 = self.mspline2.evaluate(Neval=Np/2)

        x1[2:2+Np/2] = xs1
        y1[2:2+Np/2] = ys1

        x1[2+Np/2:2+Np] = xs1[::-1]
        y1[2+Np/2:2+Np] = self.H-ys1[::-1]

        x2[2:2+Np/2] = xs2
        y2[2:2+Np/2] = ys2

        x2[2+Np/2:2+Np] = xs2[::-1]
        y2[2+Np/2:2+Np] = self.H-ys2[::-1]

        return x1, y1, self.W-x2, y2

    def update_system(self, params):
        x1,y1,x2,y2 = self.get_current_points(params)
        self.mirror1.set_points(x1,y1)
        self.mirror2.set_points(x2,y2)

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
    H = 10.0
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

    h_mirror = 5.0
    w_mirror = wavelength/n1/4.0
    L_fp = wavelength*4.75/n0
    Np = 50

    # our lens consists initial of a rectangle.
    # one face of the rectangle will have many points to manipulate
    xs = [W/2-L_fp/2-w_mirror, W/2-L_fp/2-w_mirror]
    xs += list(np.ones(Np)*(W/2+L_fp))
    xs += xs[0:1]

    ys = [H/2-h_mirror/2, H/2+h_mirror/2]
    ys += list(np.linspace(H/2+h_mirror/2, H/2-h_mirror/2, Np))
    ys += ys[0:1]

    xs = np.array(xs); ys = np.array(ys)

    mirror1 = emopt.grid.Polygon(xs, ys)
    mirror1.layer = 1
    mirror1.material_value = n1**2

    mirror2 = emopt.grid.Polygon(W-xs, ys)
    mirror2.layer = 1
    mirror2.material_value = n1**2

    background = emopt.grid.Rectangle(W/2, H/2, 2*W, 2*H)
    background.layer = 2
    background.material_value = n0**2

    eps = emopt.grid.StructuredMaterial2D(W, H, dx, dy)
    eps.add_primitives([mirror1, background])

    mu = emopt.grid.ConstantMaterial2D(1.0)

    sim.set_materials(eps, mu)

    ###########################################################################
    # Set sources
    ###########################################################################
    # We (approximately) inject a Gaussian beam
    h_src = 8.0
    src_plane = emopt.misc.DomainCoordinates(w_pml[0]+dx, w_pml[0]+dx,
                                             H/2-h_src/2, H/2+h_src/2,
                                             0, 0, dx, dy, 1.0)

    Jz = np.zeros([M,N], dtype=np.complex128)
    Mx = np.zeros([M,N], dtype=np.complex128)
    My = np.zeros([M,N], dtype=np.complex128)

    w_src = 2.5
    Ez, Hy, Hx = emopt.misc.gaussian_fields(src_plane.y-H/2, 0.0, 0.0, w_src, 0.0,
                                           wavelength, n0)

    # Technically we should take the appropriate derivative of these fields,
    # but it turns out that the fields themselves are close enough
    Jz[src_plane.j, src_plane.k] = np.reshape(Ez, src_plane.shape[1:]) * 10
    Mx[src_plane.j, src_plane.k] = np.reshape(Hx, src_plane.shape[1:]) * 10
    My[src_plane.j, src_plane.k] = np.reshape(Hy, src_plane.shape[1:]) * 10

    sim.set_sources((Jz, Mx, My))

    ###########################################################################
    # Finish simulation setup
    ###########################################################################
    h_fom = 0.2
    w_fom = 0.2
    fom_area = emopt.misc.DomainCoordinates(W/2-w_fom/2.0, W/2+w_fom/2.0,
                                            H/2-h_fom/2.0, H/2+h_fom/2.0,
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
    Nc = 10

    xc = np.ones(Nc)*(W/2-L_fp/2.0)
    yc = np.linspace(H/2+h_mirror/2, H/2,Nc)
    mspline1 = emopt.geometry.NURBS(xc, yc)
    mspline2 = emopt.geometry.NURBS(W-xc, yc)

    design_params = np.zeros(2*Nc)
    am = LensAdjointMethod(sim, xs, ys, xc, yc, Nc, Np, mirror1, mirror2,
                           mspline1, mspline2, n0)

    #am.check_gradient(design_params)

    bounds = [(-w_mirror/2.0, None) for i in range(2*Nc)]
    opt = emopt.optimizer.Optimizer(am, design_params, opt_method='L-BFGS-B',
                                    Nmax=9, bounds=bounds)
    fom, params = opt.run()
    #params = design_params
    #am.fom(params)
    x1,y1,x2,y2 = am.get_current_points(params)

    Ez = sim.saved_fields[1][0]
    if(NOT_PARALLEL):
        import matplotlib.pyplot as plt

        extent = full_field.get_bounding_box()[0:4]
        plt.imshow(np.abs(Ez), extent=extent)
        plt.plot(x1,y1,'w',linewidth=0.5)
        plt.plot(x2,y2,'w',linewidth=0.5)
        plt.xlim(extent[0:2])
        plt.ylim(extent[2:])
        plt.show()
