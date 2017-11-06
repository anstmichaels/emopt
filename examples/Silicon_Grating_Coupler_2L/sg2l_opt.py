"""
This file demonstrates how to use EMOpt in order to simulate and optimize
more complex silicon photonic structures.  In particular, we optimize a two
layer silicon grating coupler in order to maximize the coupling efficiency
between the grating coupler and an optical fiber.  The design paramters of this
grating coupler are the widths of the gaps (etched parts) and teeth (unetched
parts) which define the two layers of the grating.

The optimization process is broken into three pieces. First, the simulation of
the structure is setup. Next, an AdjointMethod class is created which defines
the a mapping between the design parameters of the system and the material
distributions of the system, the figure of merit, and the quantities needed to
compute the gradient of the figure of merit with respect to the design
parameters of the system. Finally, the optimization is run using the
emopt.optimizer.Optimizer class. In order to monitor the progress of the
optimization, we define an optional callback function which saves a plot of the
current state of the optimization after each iteration in the optimization.

When working through this example, you should first scroll to the bottom of the
file to main section, i.e.

    if __name__ == '__main__':

and read through how the simulation is setup.  Then, look over the
:class:`.SiliconGrating2LAM` class which demonstrates the core components of an
adjoint-method-based optimization.

Finally, to run the code, execute::

    mpirun -n 16 python sg2l_opt.py

in the command line which will run the optimization using 16 cores on the current
machine.

"""
# We need to import a lot of things from emopt
import emopt.fdfd
from emopt.fdfd import FDFD_TE
from emopt.adjoint_method import AdjointMethodPNF
from emopt.optimizer import Optimizer
from emopt.grid import StructuredMaterial, Rectangle
from emopt.misc import info_message, warning_message, error_message, RANK, \
NOT_PARALLEL, run_on_master, n_silicon, LineCoordinates, plot_iterations
from emopt.modedata import gen_mode_data_TE
import emopt.fomutils as FOMUtils
from emopt.modes import Mode_TE

# We define the desired Gaussian modes in a separate file
from mode_data import Ez_Gauss, Hx_Gauss, Hy_Gauss

import numpy as np
from math import pi

class SiliconGrating2LAM(AdjointMethodPNF):
    """Compute the merit function and gradient of a two layer grating coupler.

    Parameters
    ----------
    sim : emopt.fdfd.FDFD
        The simulation object
    grating_top : list of Rectangle
        The list of rectangles which form the top half of the grating.
    grating_bot : list of Rectangle
        The list of rectangles which form the bottom half of the grating.
    y_top : float
        The y position of the top grating boxes.
    y_bot : float
        The y position of the bottom grating boxes.
    w_in : float
        The width of the input waveguide
    h_wg : float
        The thickness of the input waveguide.
    Ng : int
        The number of grating lines in the grating.
    eps_clad : complex128
        The refractive index of the cladding mateiral
    mm_line : emopt.misc.LineCoordinates
        The line where the mode match is computed.
    """
    def __init__(self, sim, grating_top, grating_bot, y_top, y_bot, w_in, h_wg, Ng,
                 eps_clad, mm_line):
        super(SiliconGrating2LAM, self).__init__(sim, step=1e-10)

        # save the variables for later
        self.grating_top = grating_top
        self.grating_bot = grating_bot
        self.y_top = y_top
        self.y_bot = y_bot
        self.w_in = w_in
        self.h_wg = h_wg
        self.Ng = Ng

        self.mm_line = mm_line

        # desired Gaussian beam properties used in mode match
        theta = 0.0/180.0*pi
        match_w0 = 5.2
        match_center = 10.

        # Define the desired field profiles
        # We want to generate a vertical Gaussian beam, so these are the fields
        # are the use in our calculation of the mode match
        Ezm = Ez_Gauss(mm_line.x, match_center, match_w0, theta, sim.wavelength, np.sqrt(eps_clad))
        Hxm = Hx_Gauss(mm_line.x, match_center, match_w0, theta, sim.wavelength, np.sqrt(eps_clad))
        Hym = Hy_Gauss(mm_line.x, match_center, match_w0, theta, sim.wavelength, np.sqrt(eps_clad))
        self.mode_match = FOMUtils.ModeMatch([0,1,0], sim.dx, Ezm=Ezm, Hxm=Hxm, Hym=Hym)

        self.current_fom = 0.0

    def update_system(self, params):
        """Update the geometry of the grating coupler based on the provided
        design parameters.

        The design parameters of the a list of fourier series coefficients
        which are used to compute the widths and duty factors along the
        grating as well as two parameters which define a horizontal shift for
        the top and bottom layers of the grating.

        A fourier series representation for the grating dimensions is chosen
        because it forces the grating to evolve in a smooth and gradual way
        (which is to be expected based on our physical intuition.) We could
        alternatively parameterize the individual tooth widths and gap sizes.
        This generally works pretty well, as well.
        """
        x0_top = self.w_in + params[-2]
        x0_bot = self.w_in + params[-1]
        coeffs = params
        N_coeffs = (len(coeffs) - 2) / 8

        # compute the periods and duty factors using a Fourier series
        fseries = lambda coeffs : coeffs[0] + np.sum([coeffs[j] *np.sin(pi/2*i*j*1.0/self.Ng) for j in range(1,N_coeffs)]) \
                                            + np.sum([coeffs[N_coeffs + j] * np.cos(pi/2*i*j*1.0/self.Ng) for j in range(0,N_coeffs)])
        for i in range(self.Ng):
            #period_top = coeffs[0] + np.sum([coeffs[j] * np.sin(pi/2*i*j*1.0/self.Ng) for j in range(1,N_coeffs)]) + np.sum([coeffs[N_coeffs + j] * np.cos(pi/2*i*j*1.0/self.Ng) for j in range(0,N_coeffs)])
            #df_top = coeffs[2*N_coeffs] + np.sum([coeffs[2*N_coeffs+j] * np.sin(pi/2*i*j*1.0/self.Ng) for j in range(1,N_coeffs)]) + np.sum([coeffs[3*N_coeffs + j] * np.cos(pi/2*i*j*1.0/self.Ng) for j in range(0,N_coeffs)])
            #period_bot = coeffs[4*N_coeffs] + np.sum([coeffs[4*N_coeffs+j] * np.sin(pi/2*i*j*1.0/self.Ng) for j in range(1,N_coeffs)]) + np.sum([coeffs[5*N_coeffs + j] * np.cos(pi/2*i*j*1.0/self.Ng) for j in range(0,N_coeffs)])
            #df_bot = coeffs[6*N_coeffs] + np.sum([coeffs[6*N_coeffs+j] * np.sin(pi/2*i*j*1.0/self.Ng) for j in range(1,N_coeffs)]) + np.sum([coeffs[7*N_coeffs + j] * np.cos(pi/2*i*j*1.0/self.Ng) for j in range(0,N_coeffs)])

            period_top = fseries(coeffs[0:2*N_coeffs])
            df_top     = fseries(coeffs[2*N_coeffs:4*N_coeffs])
            period_bot = fseries(coeffs[4*N_coeffs:6*N_coeffs])
            df_bot     = fseries(coeffs[6*N_coeffs:8*N_coeffs])

            # update the rectangles
            self.grating_top[i].width = period_top*df_top
            self.grating_top[i].x0 = x0_top+(1-df_top/2)*period_top
            self.grating_bot[i].width = period_bot*df_bot
            self.grating_bot[i].x0 = x0_bot+(1-df_bot/2)*period_bot

            x0_top += period_top
            x0_bot += period_bot

    def calc_f(self, sim, params):
        """
        Compute the figure of merit.

        The figure of merit is the mode match efficiency of the grating, i.e.
        the fraction of source power that can couple into the desired Gaussian
        fiber mode.

        In order to simplify working in a parallel environment, emopt will
        return 'MathDummy's on the non-master nodes in place of the actual
        fields. These objects will absorb any mathematical operations, and thus
        you do not need to worry about dealing with the parallelism when
        running the mode match calculation as long as the mode match object is
        instantiated on all of the nodes (which is the case in this example).
        """
        # Get the fields which were recorded 
        Ez, Hx, Hy = sim.saved_fields[0]

        # compute the mode match efficiency
        self.mode_match.compute(Ez=Ez, Hx=Hx, Hy=Hy)

        # we want to maximize the efficiency, so we minimize the negative of the efficiency
        self.current_fom = -self.mode_match.get_mode_match_forward(1.0)
        return self.current_fom

    def calc_dfdx(self, sim, params):
        """Calculate the derivative of the non-source-power-normalized figure
        of merit with respect to the electric and magnetic fields at each
        location in the grid.
        """
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

    def get_update_boxes(self, sim, params):
        """Define custom update boxes.

        When calculating gradients using the adjoint method, we need to compute
        the derivative of the system matrix A (running a simulation means
        solving Ax=b). The diagonal elements of this matrix contain the
        spatially-distributed permittivity and permeability and are thus
        modified when the geometry of the system is modified.  In many cases,
        changes to structure only locally modify the permittivity/permeability
        in a small region and thus only the correspondingly small number of
        elements in A need to be updated to compute the derivative. Limiting
        this update allows us to speed up the calculation of the gradient of
        the figure of merit.

        The only grid elements that are modified are in a rectangle that
        encompasses the grating. By specifying these update boxes, we can
        significantly speed up the calculation of the gradient.

        Note: This function is optional. By default, the whole grid is updated
        in the calculation of the derivative of A w.r.t. each design variable.
        """
        N = sim.N
        lenp = len(params)
        h_wg = int(self.h_wg/sim.dy)
        y_bot = int(self.y_bot/sim.dy)
        y_top = int(self.y_top/sim.dy)

        return [(0,N,y_bot-h_wg, y_top+h_wg) for i in range(lenp)]

    def calc_grad_y(self, sim, params):
        """Out figure of merit contains no additional non-field dependence on
        the design variables so we just return zeros here.

        See the AdjointMethod documentation for the mathematical details of
        grad y and to learn more about its use case.
        """
        return np.zeros(params.shape)

if __name__ == '__main__':
    ####################################################################################
    # define the system parameters
    ####################################################################################
    wavelength = 1.55
    W = 24.0
    H = 8.0
    dx = 0.02
    dy = dx
    w_pml = 1.0
    w_src= 5.0

    # create the simulation object.
    # Note: This is TE as is defined in electromagnetics (not the silly silicon
    # photonics definition)
    sim = FDFD_TE(W, H, dx, dy, wavelength, [1.0,1.0,1.0,1.0])

    # Get the actual width and height
    W = sim.W
    H = sim.H
    M = sim.M
    N = sim.N

    ####################################################################################
    # Define the structure
    ####################################################################################
    n_si = n_silicon(wavelength)
    eps_core = n_si**2
    eps_clad = 1.444**2

    # the effective indices are precomputed for simplicity.  We can compute
    # these values using emopt.modes
    neff = 2.8499235
    neff_etched = 2.27162265

    # set up the dimensions of the waveguide structure that we are exciting
    h_wg = 0.22
    h_etch = 0.11 # etch depth
    w_wg_input = 2.5 + 0.75
    y_center = H/2.0

    # input waveguide
    wg = Rectangle(w_wg_input/2.0, y_center, w_wg_input, h_wg)

    # define the starting parameters of the 2 layer grating
    # notably the period and shift between top and bottom layers
    ne1 = neff
    ne2 = neff_etched
    n0 = np.sqrt(eps_clad)
    df = 0.8
    period = wavelength / ((1-df)*(2*ne2) + (2*df - 1)*ne1 )
    shift_bot = 1/ne1 * (wavelength/4.0 + period*(1-df)*(ne1-ne2))
    shift0 = -(1-df)*period

    # We center the grating in the simulation region and specify the vertical
    # position of the top and bottom layers for future use
    y_top = y_center + h_wg/2 - h_etch/2
    y_bot = y_center - h_wg/2 + (h_wg - h_etch)/2.0

    # We now build up the grating using a bunch of rectangles
    Ng = 26
    grating_top = []
    grating_bot = []
    rect = None

    for i in range(Ng):
        rect_top = Rectangle(shift0 + period*i + df*period/2.0, \
                             y_top, period*df, h_etch)
        rect_top.layer = 1
        rect_top.material_value = eps_core
        grating_top.append(rect_top)

        rect_bot = Rectangle(shift_bot+shift0+period*i+df*period/2.0, \
                             y_bot, period*df, h_wg-h_etch)
        rect_bot.layer = 1
        rect_bot.material_value = eps_core
        grating_bot.append(rect_bot)

    # set the background material using a rectangle equal in size to the system
    background = Rectangle(W/2,H/2,W,H)

    # set the relative layers of the permitivity primitives
    wg.layer = 1
    background.layer = 2

    # set the complex permitivies of each shape
    # the waveguide is Silicon clad in SiO2
    wg.material_value = eps_core
    background.material_value = eps_clad

    # assembled the primitives in a StructuredMaterial to be used by the FDFD solver
    # This Material defines the distribution of the permittivity within the simulated
    # environment
    eps = StructuredMaterial(W,H,dx,dy)
    eps.add_primitive(wg)

    for g in grating_top:
        eps.add_primitive(g)

    for g in grating_bot:
        eps.add_primitive(g)

    eps.add_primitive(background)

    # set up the magnetic permeability -- just 1.0 everywhere
    mu_background = Rectangle(W/2,H/2,W,H)
    mu_background.material_value = 1.0
    mu_background.layer = 1
    mu = StructuredMaterial(W,H,dx,dy)
    mu.add_primitive(mu_background)

    # add the materials and build the system
    sim.set_materials(eps, mu)

    ####################################################################################
    # Setup the sources
    ####################################################################################

    if(NOT_PARALLEL):
        print('Generating mode data...')

    # We begin by setting up the source
    Jz = np.zeros([M,N], dtype=np.complex128)
    Mx = np.zeros([M,N], dtype=np.complex128)
    My = np.zeros([M,N], dtype=np.complex128)

    # place the source in the simulation domain
    src_line = LineCoordinates('y', w_pml+2*dx, H/2-w_src/2, H/2+w_src/2, dx, dy)

    # Setup the mode solver. This simply involves getting a slice of the
    # permittivity and permeability along our source line and passing them
    # along with some basic parameters to the Mode_TE class
    eps_slice = eps.get_values_on(src_line)
    mu_slice = mu.get_values_on(src_line)

    mode = Mode_TE(wavelength, dy, eps_slice, mu_slice, n0=2.5, neigs=4)
    mode.build()
    mode.solve()

    # at this point we have found the modes but we dont know which mode is the
    # one we fundamental mode.  We have a way to determine this, however
    mindex = mode.find_mode_index(0)

    # calculate the source from the mode fields
    msrc = mode.get_source(mindex, dx, dy)

    # set the source array explicitly. In the future, this might be managed in
    # a more high-level manner.
    Jz[src_line.j, src_line.k] = msrc[0]
    Mx[src_line.j, src_line.k] = msrc[1]
    My[src_line.j, src_line.k] = msrc[2]
    sim.set_sources((Jz, Mx, My))

    ####################################################################################
    # Setup the mode match domain
    ####################################################################################
    mm_line = LineCoordinates('x', H/2.0+2.0, w_pml, W-w_pml, dx, dy)
    sim.field_domains = [mm_line]

    ####################################################################################
    # Build the system
    ####################################################################################
    sim.build()

    ####################################################################################
    # Setup the optimization
    ####################################################################################

    # We initialize our application-specific adjoint method object which is
    # responsible for computing the figure of merit and its gradient with
    # respect to the design parameters of the problem
    am = SiliconGrating2LAM(sim, grating_top, grating_bot,
                            y_top, y_bot,
                            w_wg_input, h_wg, Ng, eps_clad, mm_line)

    # inital parameterization is a uniform grating defined by a truncated
    # Fourier series
    N_coeffs = 5
    design_params = np.zeros(N_coeffs*8+2)
    design_params[0] = period
    design_params[2*N_coeffs] = df
    design_params[4*N_coeffs] = period
    design_params[6*N_coeffs] = df
    design_params[-1] = -shift_bot -(1-df)*period
    design_params[-2] = -(1-df)*period

    am.check_gradient(design_params, indices=np.arange(0,len(design_params),4))

    fom_list = []
    callback = lambda x : plot_iterations(x, fom_list, sim, am, 'Ez')

    # setup and run the optimization!
    opt = Optimizer(am, design_params, tol=1e-5,
                    callback_func=callback, Nmax= 100)

    # Run the optimization
    # A good thing to do would be to save the results of the optimization. This
    # can be done using the emopt.misc.save_results function.
    opt.run()
