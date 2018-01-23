"""
This file demonstrates how to run a constrained optimization of a
partially-etched grating coupler. This example builds off of the gc_opt.py
example and must be run after gc_opt.py has completed.

The structure of the constrained optimization is almost identical to the
unconstrained optimization. The primary difference is that we introduce a
penalty term to the figure of merit which reduces the value of the figure of
merit when slots in the grating become too narrow. This penalty term is an
explicity function of the slot widths of the grating. Similarly, the gradient
must be modified accordingly. These modifications are introduced through the
implementation of the :func:`calc_y` and :func:`calc_grad_y` functions.

Because this penalty term is an explicit function of the slot widths, it
becomes convenient to parameterize the grating in terms of the individual slot
widths and periods. This has an added benefit that it would allow us to
optimize with bounds (we do not currently do this).

In addition to these mode fundamental changes, we tweak the optimization
process. First, we use the solution of the unconstrained optimization as the
starting point for this constrained optimization. Next, we run a sequence of
three separate optimizations. In the first optimization, the penalty term is
very weakly waited. In the second optimization, the penalty term is weighted
more heavily. In the third optimization, the penalty term is strongly weighted
and the minimum feature size is more strictly enforced. This process has proven
to yield good results, but tweaking may be required depending on the exact
application.

When working through this example, you should first scroll to the bottom of the
file to main section, i.e.

    if __name__ == '__main__':

and read through how the simulation is setup.  Then, look over the
:class:`.SiliconGratingAM` class which demonstrates the core components of an
adjoint-method-based optimization.

Finally, to run the code, execute::

    mpirun -n 16 python gc_opt_constrained.py

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
NOT_PARALLEL, run_on_master, n_silicon, LineCoordinates, plot_iterations, \
save_results, load_results, PlaneCoordinates
from emopt.modedata import gen_mode_data_TE
import emopt.fomutils as FOMUtils
from emopt.modes import Mode_TE

# We define the desired Gaussian modes in a separate file
from mode_data import Ez_Gauss, Hx_Gauss, Hy_Gauss

import numpy as np
from math import pi
from mpi4py import MPI
comm = MPI.COMM_WORLD

class SiliconGratingAM(AdjointMethodPNF):
    """Compute the merit function and gradient of a grating coupler.

    Parameters
    ----------
    sim : emopt.fdfd.FDFD
        The simulation object
    grating_etch : list of Rectangle
        The list of rectangles which define the grating etch.
    wg : Rectangle
        The rectangle which defines the waveguide.
    y_ts : float
        The y position of the grating coupler
    w_in : float
        The width of the input waveguide
    h_wg : float
        The thickness of the input waveguide.
    H : float
        The total height of the simulation.
    Ng : int
        The number of grating lines in the grating.
    eps_clad : complex128
        The refractive index of the cladding mateiral
    mm_line : emopt.misc.LineCoordinates
        The line where the mode match is computed.
    """
    def __init__(self, sim, grating_etch, wg, substrate, y_ts, w_in, h_wg, H, Ng,
                 eps_clad, mm_line):
        super(SiliconGratingAM, self).__init__(sim, step=1e-8)

        # save the variables for later
        self.grating_etch = grating_etch
        self.y_ts = y_ts
        self.w_in = w_in
        self.h_wg = h_wg
        self.H = H
        self.Ng = Ng
        self.wg = wg
        self.substrate = substrate

        self.mm_line = mm_line

        # constraints
        self.min_feature = 0.0
        self.c_steepness = 0.0
        self.penalty = 0.0

        # desired Gaussian beam properties used in mode match
        theta = 8.0/180.0*pi
        match_w0 = 5.2
        match_center = 11.0

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
        which are used to compute the etch widths and period along the grating.

        A fourier series representation for the grating dimensions is chosen
        because it forces the grating to evolve in a smooth and gradual way
        (which is to be expected based on our physical intuition.) We could
        alternatively parameterize the individual tooth widths and gap sizes.
        This generally works pretty well, as well.
        """
        coeffs = params
        N_coeffs = (len(coeffs) - 3) / 4

        h_etch = params[-3]
        x0 = self.w_in + params[-2]
        h_BOX = params[-1]

        for i in range(self.Ng):
            w_etch = params[i]
            period = params[i+Ng]

            # update the rectangles
            self.grating_etch[i].width  = w_etch
            self.grating_etch[i].height = h_etch
            self.grating_etch[i].x0     = x0 + w_etch/2.0
            self.grating_etch[i].y0     = self.y_ts + self.h_wg/2.0 - h_etch/2.0

            x0 += period

        # update the BOX/Substrate
        h_subs = self.H/2.0 - self.h_wg/2.0 - h_BOX
        self.substrate.height = h_subs
        self.substrate.y0 = h_subs/2.0

        # update the width of the unetched grating
        w_in = x0
        self.wg.width = w_in
        self.wg.x0 = w_in/2.0

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

    def calc_y(self, sim, params):
        """Calculate a penalty to the figure of merit due to etched slots which
        are too small.

        Typically, we chirp ("apodize") a grating coupler such that the
        intensity of scattered light matches a slowly-varying Gaussian profile.
        This slowly-varying shape requires that we scatter light very weakly at
        the beginning of the grating which requires that we have very narrow
        slots. In our constrained optimization, we will try to eliminate these
        slots without reducing the coupling
        """
        p = np.sum(self.penalty*FOMUtils.rect(params[0:self.Ng] -
                                              self.min_feature/2.0,
                                              self.min_feature,
                                              self.c_steepness))

        return p

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

        When modifying the etches of the grating, the only grid elements that
        need to be updated are contained in a rectangle that encompasses the
        grating coupler. This is equally true for the horizontal grating shift
        and the grating etch depth parameters. When updating the BOX thickness,
        the grid elements in a larger area must be updated.

        Note: This function is optional. By default, the whole grid is updated
        in the calculation of the derivative of A w.r.t. each design variable.
        """
        N = sim.M
        N = sim.N
        h_wg = int(self.h_wg*1.5/sim.dy)
        y_wg = int(self.y_ts/sim.dy)

        # define boxes surrounding grating
        boxes = [(0,N,y_wg-h_wg, y_wg+h_wg) for i in range(2*self.Ng+2)]

        # for BOX, update everything (easier)
        boxes.append((0,N,0,M))
        return boxes

    def calc_grad_y(self, sim, params):
        """Calculate the derivative of the penalty term in the FOM with respect
        to design variables of the system.

        Our figure of merit contains a non-power-normalized additive component
        which reduces the FOM's value when features are too small. We need to
        specify what the derivative of this function is.

        Note: In general, these functions are strictly functionals of the
        design parameters. In some cases, it may not be straightforward to
        analytically calculate the derivative of this function. In such cases,
        using a numerical approximation should be fine.
        """
        Ng = self.Ng
        dparams = np.zeros(params.shape)
        dparams[0:Ng] = self.penalty*FOMUtils.rect_derivative(params[0:Ng] -
                                                              self.min_feature/2.0,
                                                              self.min_feature,
                                                              self.c_steepness)
        return dparams


def plot_update(x, fom_list, fom_unconstrained, sim, am):
    """Save a snapshot of the current state of the structure.

    This function is passed to an Optimizer object and is called after each
    iteration of the optimization. It plots the current refractive index
    distribution, the electric field, and the full figure of merit history.
    """
    fom = am.calc_fom(sim, x)
    y = am.calc_y(sim, x)
    fom_unconstrained.append(-1*(fom-y))

    f, ax1, ax2, ax3 = plot_iterations(x, fom_list, sim, am, 'Ez',
                                       fom2=fom_unconstrained)

    data = {}
    Ez, Hx, Hy = sim.saved_fields[1]
    eps = sim.eps.get_values_on(sim.field_domains[1])
    data['Ez'] = Ez
    data['Hx'] = Hx
    data['Hy'] = Hy
    data['eps'] = eps
    data['params'] = x
    data['foms'] = fom_list

    i = len(fom_list)
    fname = 'data/gc_8deg_opt_constrained'
    save_results(fname, data)

if __name__ == '__main__':
    ####################################################################################
    # define the system parameters
    ####################################################################################
    wavelength = 1.55
    W = 26.0
    H = 8.0
    dx = 0.02
    dy = dx
    w_pml = 1.0
    w_src= 3.5

    # set the minimum feature size
    # here we choose 90 nm
    min_feature = 0.09

    # create the simulation object.
    # TE => Ez, Hx, Hy
    sim = FDFD_TE(W, H, dx, dy, wavelength)

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
    neff = 2.8548
    neff_etched = 2.0879

    # set up the initial dimensions of the waveguide structure that we are exciting
    h_wg = 0.28
    h_etch = 0.19 # etch depth
    w_wg_input = 5.0

    # define the starting parameters of the partially-etched grating
    # notably the period and shift between top and bottom layers
    ne1 = neff
    ne2 = neff_etched
    n0 = np.sqrt(eps_clad)
    df = 0.8
    theta = 8.0/180.0*pi

    period = wavelength / (df * ne1 + (1-df)*ne2 - n0*np.sin(theta))

    # set the center position of the top silicon and the etches
    y_ts = H/2.0
    y_etch = y_ts + h_wg/2.0 - h_etch/2.0

    # We now build up the grating using a bunch of rectangles
    Ng = 26
    grating_etch = []

    for i in range(Ng):
        rect_etch = Rectangle(0,y_etch, (1-df)*period, h_etch)
        rect_etch.layer = 1
        rect_etch.material_value = eps_clad
        grating_etch.append(rect_etch)

    # input waveguide
    Lwg = Ng*period + w_wg_input
    wg = Rectangle(Lwg/2.0, y_ts, Lwg, h_wg)

    # define substrate
    h_BOX = 2.0
    h_subs = H/2.0 - h_wg/2.0 - h_BOX
    substrate = Rectangle(W/2.0, h_subs/2.0, W, h_subs)

    # set the background material using a rectangle equal in size to the system
    background = Rectangle(W/2,H/2,W,H)

    # set the relative layers of the permitivity primitives
    wg.layer = 2
    substrate.layer = 2
    background.layer = 3

    # set the complex permitivies of each shape
    # the waveguide is Silicon clad in SiO2
    wg.material_value = eps_core
    substrate.material_value = eps_core
    background.material_value = eps_clad

    # assembled the primitives in a StructuredMaterial to be used by the FDFD solver
    # This Material defines the distribution of the permittivity within the simulated
    # environment
    eps = StructuredMaterial(W,H,dx,dy)

    for g in grating_etch:
        eps.add_primitive(g)

    eps.add_primitive(wg)
    eps.add_primitive(substrate)
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
    full_field = PlaneCoordinates('z', w_pml, W-w_pml, w_pml, H-w_pml, dx, dy)
    sim.field_domains = [mm_line, full_field]

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
    am = SiliconGratingAM(sim, grating_etch, wg, substrate, y_ts,
                          w_wg_input, h_wg, H, Ng, eps_clad, mm_line)

    # initial minimum feature size constraint parameters
    # these are chosen "empirically"
    am.min_feature = min_feature
    am.c_steepness = 0.45*min_feature
    am.penalty = 0.035

    # unconstrained parameterization is a uniform grating defined by a truncated
    # Fourier series
    N_coeffs = 5

    # load the results of the unconstrained parameterization
    params = None
    if(NOT_PARALLEL):
        data = load_results('data/gc_8deg_opt_final')
        params = data['params']

    params = comm.bcast(params, root=0)

    # The unconstrained Fourier parameterization needs to be converted to the
    # new parameterization in which all widths are separate variables
    fseries = lambda i, coeffs : coeffs[0] + np.sum([coeffs[j] *np.sin(pi/2*i*j*1.0/Ng) for j in range(1,N_coeffs)]) \
                                           + np.sum([coeffs[N_coeffs + j] * np.cos(pi/2*i*j*1.0/Ng) for j in range(0,N_coeffs)])

    design_params = np.zeros(Ng*2+3)

    for i in range(Ng):
        w_etch  = fseries(i, params[0*N_coeffs:2*N_coeffs])
        period = fseries(i, params[2*N_coeffs:4*N_coeffs])

        design_params[i] = w_etch
        design_params[i+Ng] = period

    design_params[-3] = params[-3]
    design_params[-2] = params[-2]
    design_params[-1] = params[-1]

    #am.check_gradient(design_params, indices=np.arange(0,len(design_params),1))

    fom_list = []
    fom_unconstrained = []
    callback = lambda x : plot_update(x, fom_list, fom_unconstrained, sim, am)

    # setup and run the optimization!
    # Note: For this optimization, it turns out that L-BFGS-B works a lot
    # better than normal BFGS (this is not totally unusual)
    opt = Optimizer(am, design_params, tol=1e-5,
                    callback_func=callback,
                    opt_method='L-BFGS-B', Nmax=30)

    # Run the optimization
    # A good thing to do would be to save the results of the optimization. This
    # can be done using the emopt.misc.save_results function.
    fom, params = opt.run()

    params = comm.bcast(params, root=0)

    # setup the second optimization with medium constraints
    am.c_steepness = 0.1*min_feature
    am.penalty = 0.02
    opt.p0 = params

    # Run the second optimization with medium constraints
    fom, params = opt.run()
    params = comm.bcast(params, root=0)

    # setup the third optimization with strict constraints
    # Forcefully impose the constraint
    for i in range(Ng):
        if(params[i] < am.min_feature/2.0 and params[i] > 0.0):
            l_temp = params[i]
            params[i+Ng] += l_temp
            params[i] = 0.0
        elif(params[i] < am.min_feature and params[i] > am.min_feature/2.0):
            l_temp = params[i]
            params[i+Ng] -= (am.min_feature-l_temp)
            params[i] = am.min_feature

    am.c_steepness = 0.05*min_feature
    am.penalty = 0.2
    opt.p0 = params

    fom, params = opt.run()
