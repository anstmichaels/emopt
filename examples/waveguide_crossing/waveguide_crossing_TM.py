"""Run a simple optimization of a MMI (multimode interference) waveguide
crossing.

This example demonstrates how to run a simple optimization of a crossing
structure which looks like:

                      |      |
                      |      |
           /----------        -----------\
    ------/                               \------
 input ->                                    output ->
    ------\                               /------
           \----------        -----------/
                      |      |
                      |      |

The goal of the optimization is to maximize the fraction of power which leaves
through the output waveguide. This is accomplished by modifying 3 design
variables:

    1) the width of the wider multimode crossing waveguide
    2) the length of the crossing waveguide
    3) length of the tapers on either side.

The input waveguide width is fixed.

It is worth noting that this example is a bit contrived as it can actually be
solved largely by hand. Nonetheless, it is a nice gentle introduction to
optimization with emopt.

On most *nix-based machines, run the script with:

    $ mpirun -n 8 python waveguide_crossing_TM.py

If you wish to increase the number of cores that the example is executed on,
change 8 to the desired number of cores.
"""
import emopt
from emopt.misc import NOT_PARALLEL, RANK, run_on_master
from emopt.adjoint_method import AdjointMethodPNF

import numpy as np
from math import pi

from mpi4py import MPI

class WGCrossAM_TM(AdjointMethodPNF):
    """Define an adjoint method class which calculates the figure of merit and
    its gradient for the waveguide crossing structure.

    In general, there is a lot of overhead associated with running
    optimizations of complex electromagnetic structures.  Gremlin attempts to
    alleviate the burden of dealing with some of this complexity by
    implementing a set of base classes which handle the problem-agnostic
    components of the adjoint method. This leaves only the problem-specific
    pieces which must be defined by the user. In particular, at a bare minimum,
    you must define:

        1) A function which updates your structure given a set of design variables
        2) A function which calculates the figure of merit
        3) A function which calculates the derivative of the figure of merit
        with respect to each of the relevant field components.

    With these three functions defined, we can optimize the electromagnetic
    structure.

    In emopt, the implementation of these functions is handled by extending a
    base AdjointMethod class and implementing the required abstract methods.
    There are a number of AdjointMethod classes to choose from:

        1) AdjointMethod : This is the simplest and should be used when your
        merit function is independent of the materials of the system and does
        not require power normalization.
        2) AdjointMethodPNF : This is for figures of merit which have source
        power normalization.
        3) AdjointMethodFM : This is for figures of merit which depend on the
        materials of the system
        4) AdjointMethodMO : This provides a simple way of combining multiple
        AdjointMethods to allow for optimizations of figures of merit which
        combine multiple objective functions (e.g. optimizing over multiple
        wavelengths, etc)

    It is highly recommended that you read the documentation for the
    AdjointMethod class and the relevant subclasses.
    """

    def __init__(self, sim, crossing_x, crossing_y, h_wg, mode_match, line_fom, step=1e-8):
        super(WGCrossAM_TM, self).__init__(sim, step)

        self.h_wg = h_wg
        self.crossing_x = crossing_x
        self.crossing_y = crossing_y

        self.y0 = sim.H/2.0
        self.x0 = sim.W/2.0

        self.mode_match = mode_match
        self.line_fom = line_fom

        self.W = sim.W

    def update_system(self, params):
        """Update the geometry of the system given a set of values for the
        design variables.

        The set of design variables is given by:
            [taper length, crossing height, crossing length]

        Updating the structure involves modifying the x,y coordinates of our
        polygon.
        """
        L_taper = params[0]
        cross_height = params[1]
        w_in_out = (self.W - params[2] - L_taper*2)/2.0
        h_wg = self.h_wg

        # Update the horizontal crossing structure
        self.crossing_x.set_point(1, w_in_out, self.y0+self.h_wg/2.0)
        self.crossing_x.set_point(2, w_in_out+L_taper, self.y0+cross_height/2.0)
        self.crossing_x.set_point(3, self.W-w_in_out-L_taper, self.y0+cross_height/2.0)
        self.crossing_x.set_point(4, self.W-w_in_out, self.y0+self.h_wg/2.0)
        self.crossing_x.set_point(7, self.W-w_in_out, self.y0-self.h_wg/2.0)
        self.crossing_x.set_point(8, self.W-w_in_out-L_taper, self.y0-cross_height/2.0)
        self.crossing_x.set_point(9, w_in_out+L_taper, self.y0-cross_height/2.0)
        self.crossing_x.set_point(10, w_in_out, self.y0-h_wg/2.0)

        # use the previous update to update the veritcal structure
        xs = np.copy(self.crossing_x.xs)
        ys = np.copy(self.crossing_x.ys)

        xs -= self.x0
        ys -= self.y0
        xs += self.y0
        ys += self.x0

        self.crossing_y.set_points(ys, xs)

    @run_on_master
    def calc_f(self, sim, params):
        """Calculate the figure of merit.

        The figure of merit for our crossing optimization is the
        source-power-normalized mode match between the simulated fields and the
        the mode of the output waveguide. This is a very common figure of merit
        in silicon photonics applications. As such, emopt.fomutils provides
        functions to aid in the calculation of mode matches.

        This function accepts the simulation object and current set of design
        parameters and returns a single real scalar which gives the figure of
        merit. Gremlin always minimizes the figure of merit, so we need to
        multiply the mode match by -1 in order to ensure that it is actually
        maximized.

        See AdjointMethodPNF's documentation for details about implementing
        this function.

        Note: because emopt is built on top of MPI, this function is called on
        every processor. The desired fields, meanwhile, are generally known
        only on the master node (broadcasting the same fields to all of the
        nodes is both unnecessary and expensive, in general).  To make dealing
        with this easier, we use the @run_on_master flag to ensure that this
        function is only run on the master node and then access the fields
        which were saved to the master node immediately after the simulation
        was performed.

        Alternatively, we could have allowed this function to run on all nodes,
        use the FDFD.get_fields_interp(...) function and then put the actual
        calculations in a if(NOT_PARALLEL) block.
        """
        # Get the fields which were recorded 
        Hz, Ex, Ey = sim.saved_fields[0]

        # Calculate the mode match
        self.mode_match.compute(Ex=Ex, Ey=Ey, Hz=Hz)

        fom = -1*self.mode_match.get_mode_match_forward(1.0)
        self.current_fom = fom
        return fom

    @run_on_master
    def calc_dfdx(self, sim, params):
        """Calculate the derivative of the figure of merit with respecto the
        fields.

        An essential part of computing gradients with the adjoint method is the
        calculation of the derivative of the figure of merit with respect to
        the fields at each spatial index, dF/dx.  Because power normalization is
        handled by the underlying adjoint method implementation that we have
        chosen, we only need to compute df/dx, where F = f/Psrc.

        Notice that f(x) is a functional (takes a vector and returns a scalar)
        of the vector x. The derivative df/dx is thus also a vector of the
        form df/dx = [df/dx1, df/dx2, ..., df/dxN] where x1, x2, ..., xN refer
        to the list of field components at the different spatial indices. In
        this case, we have non-zero Hz, Ex, Ey and thus df/dx will look like

            df/dx = [df/dHz11, df/dHz12, ..., df/dEx11, df/dEx12, ...,
                     df/dEy11, df/dEy12, ...]

        Because our figure of merit is an analytic function, it is
        straightforward to write down and compute its derivative. Note that
        when handling integral functions, it is important to cast them first as
        a discrete sum of indexed elements before differentiating.

        As with calc_f(...), we wrap the function in @run_on_master to ensure
        that the computation is only performed on the master node and retrieve
        the already-stored fields from our FDFD object.
        """
        dFdHz = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dFdEx = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dFdEy = np.zeros([sim.M, sim.N], dtype=np.complex128)

        Hz, Ex, Ey = sim.saved_fields[0]
        Psrc = sim.source_power

        self.mode_match.compute(Ex=Ex, Ey=Ey, Hz=Hz)

        j = self.line_fom.j
        k = self.line_fom.k
        dFdHz[j, k] = -self.mode_match.get_dFdHz()
        dFdEx[j, k] = -self.mode_match.get_dFdEx()
        dFdEy[j, k] = -self.mode_match.get_dFdEy()

        return (dFdHz, dFdEx, dFdEy)

    def calc_grad_y(self, sim, params):
        """Out figure of merit contains no additional non-field dependence on
        the design variables so we just return zeros here.

        See the AdjointMethod documentation for the mathematical details of
        grad y and to learn more about its use case.
        """
        return np.zeros(params.shape)

def opt_plot(params, sim, am, fom_hist):
    """Plot the current state of the optimization.

    This function is called after each iteration of the optimization
    """
    print('Finished iteration %d.' % (len(fom_hist)+1))
    current_fom = -1*am.calc_fom(sim, params)
    fom_hist.append(current_fom)
    foms = {'IL':fom_hist}

    Hz, Ex, Ey = sim.saved_fields[1]
    eps = sim.eps.get_values_in(sim.field_domains[1])

    emopt.io.plot_iteration(np.real(Hz), np.real(eps), sim.Wreal, sim.Hreal, foms,
                   fname='current_result.pdf')

if __name__ == '__main__':
    ####################################################################################
    # Define the system size
    ####################################################################################
    # define geometric parameters used when defining the structure.  We do this
    # first so that the width of the simulation region can grow or shrink to
    # accomodate these dimensions. It is worth noting that we can actually
    # calculate the near-optimal quantities "by hand", however that would
    # result in a less interesting optimization..
    w_in = 2.0
    L_crossing = 10.0
    L_taper = 1.25

    # Set the simulation wavelength. Gremlin uses non-dimensionalized
    # equations, so as long as all quantities with length units are chosen with
    # consistent units, things should work out.  Here we choose to express
    # everything in micrometers
    wlen = 1.55
    W = w_in*2.0 + L_crossing + L_taper*2
    H = 8.0
    dx = 0.04
    dy = 0.03
    sim = emopt.fdfd.FDFD_TM(W, H, dx, dy, wlen)
    pmls = [0.5,0.5,0.5,0.5]
    sim.w_pml = pmls

    # Get the actual width and height
    # The true width/height will not necessarily match what we used when
    # initializing the solver. This is the case when the width is not an integer
    # multiple of the grid spacing used.
    W = sim.W
    H = sim.H
    M = sim.M
    N = sim.N

    ####################################################################################
    # Set up the materials of the system
    ####################################################################################
    # Choose the desired materials of the system. We use silicon for the
    # crossing structure and SiO2 as the cladding
    n_si = emopt.misc.n_silicon(wlen)
    n_sio2 = 1.444
    eps_clad = n_sio2**2

    # Calculate effective indices so that we can apply the effective index
    # method and use a 2D simulation.
    si_thickness = 0.22
    n_eff = 2.85 # Precomputed. We could use the mode solver to do this too

    # set a background permittivity containing the cladding material (SiO2)
    eps_background = emopt.grid.Rectangle(W/2, H/2, 2*W, H)
    eps_background.layer = 2
    eps_background.material_value = eps_clad

    # Define the horizontal waveguide crossing structure
    h_wg = 0.75
    h_crossing = 1.75

    concat = lambda a, b : np.concatenate([a,b]) # make things less verbose

    # The horizontal crossing is defined as a single polygon.  We could also
    # choose to reprsent it using a combination of polygons and rectangles.
    #
    # It is very important to assemble polygons in the proper way. Points must
    # be added in a clock-wise manner. Furthermore, the polygon must be
    # "closed" meaning the first point must be added again at the very end. Not
    # following this convention can result in very unexpected consequences,
    # especially when running an optimization.
    crossing_x = emopt.grid.Polygon()

    # define the x coordinates
    xs = np.array([-w_in, w_in, w_in + L_taper])
    xs = concat(xs, W-np.array([w_in+L_taper, w_in, -w_in,
                                 -w_in, w_in, w_in + L_taper]))
    xs = concat(xs, [w_in + L_taper, w_in, -w_in, -w_in])

    # define the y-coordinates.
    ys = np.array([h_wg/2.0, h_wg/2.0, h_crossing/2.0])
    ys = concat(ys, [h_crossing/2.0, h_wg/2.0, h_wg/2.0,
                -h_wg/2.0, -h_wg/2.0,-h_crossing/2.0])
    ys = concat(ys, [-h_crossing/2.0, -h_wg/2.0, -h_wg/2.0, h_wg/2.0])
    ys += H/2.0

    # assemble the polygon
    crossing_x.set_points(xs, ys)
    crossing_x.layer = 1
    crossing_x.material_value = n_eff**2

    # vertical taper.  This has the same coordinates as the horizontal
    # crossing, except we need to rotate them (and make sure the
    # clockwise-convention is maintained!)
    xs = xs.copy()
    ys = ys.copy()

    xs -= W/2.0
    ys -= H/2.0

    xs += H/2.0
    ys += W/2.0

    crossing_y = emopt.grid.Polygon()

    crossing_y.set_points(ys, xs)
    crossing_y.layer = 1
    crossing_y.material_value = n_eff**2

    # All primitive components (polygons, rectangles, etc) must be added to a
    # StructuredMaterial. This layers the components in such a way that more
    # complicated material distributions can be formed
    eps = emopt.grid.StructuredMaterial2D(W, H, dx, dy)
    eps.add_primitive(crossing_x)
    eps.add_primitive(crossing_y)
    eps.add_primitive(eps_background)

    # Set the permeability --> it's uniformly 1
    mu = emopt.grid.ConstantMaterial2D(1.0)

    # set the materials used for simulation
    sim.set_materials(eps, mu)

    ####################################################################################
    # Set up the materials of the system
    ####################################################################################
    w_src = 5.0

    # To make accessing and initializing arrays easier, we create
    # DomainCoordinates. These manage the mapping between real-space coordinates
    # and array index coordinates.
    src_line = emopt.misc.DomainCoordinates(pmls[0]+0.1, pmls[0]+0.1, H/2-w_src/2,
                                            H/2+w_src/2, 0, 0, dx, dy, 1.0)

    Mz = np.zeros([M,N], dtype=np.complex128)
    Jx = np.zeros([M,N], dtype=np.complex128)
    Jy = np.zeros([M,N], dtype=np.complex128)

    # setup, build the system, and solve for the modes of the input waveguide
    mode = emopt.modes.ModeTM(wlen, eps, mu, src_line, n0=3.5, neigs=8)
    mode.build()
    mode.solve()

    mindex = mode.find_mode_index(0)

    # Get the current sources which excite the desired mode
    msrc = mode.get_source(mindex, dx, dy)

    # Currently, the emopt FDFD object accepts 3 big arrays that define the
    # current sources in space. The current sources are zero everywhere except
    # on the line where we calculated the mode.
    Mz[src_line.j, src_line.k] = msrc[0]
    Jx[src_line.j, src_line.k] = msrc[1]
    Jy[src_line.j, src_line.k] = msrc[2]

    sim.set_sources((Mz, Jx, Jy))

    ####################################################################################
    # Set up the Mode match which will be used by the figure of merit
    ####################################################################################
    # Get the mode fields to use as a mode match
    mode_match = None
    if(NOT_PARALLEL):
        # The mode fields are returns with size (N,) but the field slices used
        # in the future mode match calculations will be of size (N,1). We will
        # reshape the mode fields so that things are compatible
        Exm = mode.get_field_interp(mindex, 'Ex')
        Eym = mode.get_field_interp(mindex, 'Ey')
        Hzm = mode.get_field_interp(mindex, 'Hz')

        mode_match = emopt.fomutils.ModeMatch([1,0,0], sim.dy,
                                              Exm=Exm, Eym=Eym, Hzm=Hzm)

    # the mode match will be computed along a line intersecting the output
    # waveguide of the structure
    mm_line = emopt.misc.DomainCoordinates(W-pmls[1]-dx, W-pmls[1]-dx, H/2-w_src/2,
                                           H/2+w_src/2, 0, 0, dx, dy, 1.0)

    full_field = emopt.misc.DomainCoordinates(pmls[0], W-pmls[1], pmls[2], H-pmls[3], 0,
                                              0, dx, dy, 1.0)

    # we also tell our FDFD object to record the fields on this line
    # immediately after running a forward simulation. This will simplify the
    # calculation of the figure of merit
    sim.field_domains = [mm_line, full_field]

    ####################################################################################
    # Finalize the simulation
    ####################################################################################
    # Build the FDFD system. It is important to call this before running a
    # simulation.
    sim.build()

    ####################################################################################
    # Setup the optimization
    ####################################################################################
    # In order to run an optimization, we need to specify an initial set of
    # design parameters for the system and create an adjoint method object
    # which will compute the figure of merit that is minimized during
    # optimization and the gradient of that figure of merit
    design_params = np.array([L_taper, h_crossing, W-w_in*2-L_taper*2])

    # The adjoint method class for this problem is defined above!
    am = WGCrossAM_TM(sim, crossing_x, crossing_y, h_wg, mode_match, mm_line)

    # The first thing you should ALWAYS do before running an optimization is to
    # verify that the gradient of the figure of merit is accurate. Once the
    # gradient is correct, the optimization will work (although the quality of
    # the optimum found depends on your initial guess, parameterization, etc)
    am.check_gradient(design_params)

    # define a callback function which is executed after each iteration. This
    # function saves a plot of the structure the desired field component, and
    # the history of the figure of merit to a file called current_result.pdf
    fom_list = []
    callback = lambda x : opt_plot(x, sim, am, fom_list)

    # setup and run the optimization!
    opt = emopt.optimizer.Optimizer(am, design_params, tol=1e-5,
                                    callback_func=callback, Nmax=15)
    results = opt.run()
