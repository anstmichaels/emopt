.. _at_a_glance:

#################
EMopt at a Glance
#################

Before diving into the tutorials, it is a good idea to get acquainted with the
EMopt's philosophy. EMopt is a collection of different electromagnetic solvers
as well as optimization code which ties it all together. These different tools
have been designed to be perfectly usable on their own or together for the
purpose of optimizing structures. Furthermore, most components within EMopt
follow a relatively well-defined interface making them easy to modify or extend
without breaking the interoperability. Below we will introduce the most
important components of EMopt and explain how they can be used.

==============
Units in EMopt
==============

All equations solved in EMopt on non-dimensionalized. As such, you are free to
choose your own spatial units. As long as you express all dimensions in the
same using the same units as the wavelength, everything will work out. For
example, if we want everything to be expressed in micrometers, we would do:

.. code-block:: python

    wavelength = 1.55 # micron
    width = 3.0 # also in micron
    height = 3.0 # also in micron

Also as a result of the non-dimensionalized equatiosn used, all permittivity
and permeability values are *relative* values. That is, we work with
:math:`\epsilon_r` and :math:`\mu_r` where the actuall permittivity and
permeability are :math:`\epsilon = \epsilon_r \epsilon_0` and
:math:`\mu = \mu_r \mu_0`.

===========================
The EMopt Coordinate System
===========================

In EMopt, the origin is always assumed to be the minimum (x,y,z) coordinate of
the simulation domain. In other words, in 2D, the simulation domain is the
rectangle defined by the points (0,0) and (X,Y), and in 3D, the simulation
domain is the rectangular prism defined by the points (0,0,0) and (X,Y,Z).

===================================
Defining Structures with emopt.grid
===================================

A core part of emopt is the :ref:`emopt_grid` module which provides users with
a way of defining structures/material distributions on a rectangular grid. In
particular, :ref:`emopt_grid` provides an interface for defining collections of
:class:`grid.MaterialPrimitive` which are rectangles and polygons. In 2D, these
:class:`grid.MaterialPrimitive` can be overlayed with different priorities to form
complex 2D distributions of materials. In 3D, they can be prescribed a
thickness and stacked to form complex layered structures.

A key component of the :ref:`emopt_grid` module is the idea of *grid
smoothing*. All of the solvers included in EMopt solve for the electric and
magnetic field on a rectangular grid. When doing optimizations, parameter
sweeps, etc, it is highly desirable that structures be manipulatable in a
continuous way. For example, ideally a small perturbation to the width of a
waveguiding device would result in a correspondingly small perturbation to the
computed efficiency of the device. This continuous behavior is achieved by
mapping continuous polygons (defined with floating-point precision) onto a
rectangular grid and computing average material values in grid cells which
overlap the polyong's boundaries. The details of this process are described in
`[1] <https://arxiv.org/abs/1705.07188>`_.

==============================
Simulating Maxwell's Equations
==============================

EMopt contains two Finite Difference Frequency Domain (FDFD) solvers--one for
2D simulations and one for 3D simulations. These solvers simulate the frequency
domain Maxwell's equations on a rectangular grid. At this point in time, the
these solvers are ideally suited for simulating dielectric structures whereas
metallic structures may lead to issues (due to either problem size or grid
smoothing errors).

Using the EMopt's FDFD solvers involves creating an FDFD object (:class:`fdfd.FDFD_TE`
or :class:`fdfd.FDFD_TM` in 2D or :class:`fdfd.FDFD_3D` in 3D) which defines
the simulation size and resolution, defining the structure, defining the
sources, building the system, and then running the simulation. In 2D, you have
the option of running either a TE or TM simulation (full-vector + anisotropies
are not currently supported but may be in the future). The process of running
a 2D TE simulation is approximately as follows:

.. code-block:: python

    import emopt

    # define system size + resolution
    wavelength = 1.55 # micron => all distance units in micron
    W = 3.0 # simulation width
    H = 3.0 # simulation height
    dx = 0.02 # grid spacing along x
    dy = 0.02 # grid spacing along y

    # create FDFD object
    sim = emopt.fdfd.FDFD_TE(W, H, dx, dy, wavelength)

    # get "actual" simulation dimensions--W and H snap to nearest grid cell
    W = sim.W
    H = sim.H
    M = sim.M # number of grid cells along y
    N = sim.N # number of grid cells along z

    # define materisl
    eps = ... # permittivity distribution using emopt.grid
    mu  = ... # permeability distribution using emopt.grid
    sim.set_materials(eps, mu)

    # define the sources
    Jz, Mx, My = ... # set sources with arrays or emopt.modes
    sim.set_sources((Jz, Mx, My))

    # build and run
    sim.build()
    sim.solve_forward()

    # get resulting electric field
    Ez = sim.get_field_interp('Ez')

In 3D, the process is very similar, however we need to be a bit more careful
about how we specify sources and retrieve fields since the memory requirements
are increased. Specifically, we specify rectangular domains which are ideally
much smaller than the whole simulation region (e.g. a plane) and specify the
sources or retrieve the fields in these domains.

.. note:: 3D solver options

    Currently, EMopt provides two 3D solvers to choose from. The first is an
    FDFD solver which works well for smaller problems and the second is a
    CW-FDTD solver which works well for problems of any size.

    For very small problems, the FDFD solver may be a bit faster. Furthermore,
    the FDFD solver will typically produce more accurate gradients, regardless
    of how grid spacings are chosen or which boundary conditions are used.

    For modest to large problems (in terms of either size or resolution), the
    FDTD solver should be used. The 3D solver will scale to larger numbers of
    cores better and uses *considerably* less memory than the FDFD solver.
    The primary disadvantage of the FDTD solver is that it requires that the
    grid spacing be equal in all directions in order to calculate accurate
    gradients. Furthermore, symmetry boundary conditions can lead to
    inconsistent gradient calculations.

===========================
Calculating Waveguide Modes
===========================

EMopt provides 1D and 2D mode solvers for calculating propagating modes of 2D
and 3D structures. These mode solvers can be used on their own or in
conjunction with an FDFD object as a mode source.

The process of using the mode solvers is very similar to running FDFD
simulations. The basic process is as follows:

.. code-block:: python

   import emopt

   wavelength = 1.55 # micron
   H  = 3.0 # mode solver height
   dy = 0.01 # grid spacing

   # define the permittivity and permeability
   eps = ... # define permittivity distribution
   mu  = ... # define permeability distribution

   # define slice of structure to use in the mode calculation
   # Note: dx and dz dont matter here
   mode_slice = emopt.grid.DomainCoordinates(0, 0, 0, H, 0, 0, 1, dy, 1)

   # define the mode solver. n0 is the effective index to search around and
   # neigs is the number of modes to find.
   mode = emopt.modes.ModeTE(wavelength, eps, mu, mode_slice, n0=3.5, neigs=4)

   # build and solve
   mode.build()
   mode.solve()

   # get the result
   Ez = mode.get_field_interp('Ez')

The process for calculating 2D modes is almost identical. 

=========================
Calculating Sensitivities
=========================

A key component of EMopt is the calculation of sensitivities (i.e., gradients
of a figure of merit which describe an electromagnetic device's performance
with respect to design variables which describe the device's shape). In order
to effciently compute the sensitivities, EMopt implements the adjoint method.
This implementation has been designed in order to make it relatively easy to
use for any device, figure of merit, and set of design variables that you would
like to work with.

Sensitivity analysis in EMopt makes heavy use of object oriented programming.
In particular, in order to apply the adjoint method to a specific figure of
merit and set of design variables, you must implement your own class which
extends the :class:`adjoint_method.AdjointMethod` class defined in EMopt. In
this custom implementation, you tell EMopt how to update your structure given a
set of design variables, how to calculate your figure of merit, and how to take
its derivative with respect to the field quantities. For example:

.. code-block:: python

    from emopt.adjoint_method import AdjointMethod

    class MyAdjointMethod(AdjointMethod):
        def __init__(self, sim):
            super(MyAdjointMethod, self).__init__(sim)
            # define other variables you need

        def update_system(self, params):
            """Update the simulation geometry based on the list of design
            parameters given in params."""
            # e.g. self.rect.width = params[0]

        def calc_fom(self, sim, params):
            """Calculate the figure of merit."""
            Ex, Ey, Ez, Hx, Hy, Hz = sim.saved_fields[0] # get fields
            fom = ...# calculate fom using Ex, Ey, Ez, Hx, Hy, Hz
            return fom

        def calc_dFdx(self, sim, params):
            """Calculate the derivative of the figure of merit with respect to
            Ex, Ey, Ez, Hx, Hy, Hz"""
            Ex, Ey, Ez, Hx, Hy, Hz = sim.saved_fields[0] # get fields
            dFdEx = ... # calc derivative with respect to x
            dFdEy = ... # calc derivative with respect to y
            ...
            # the value we return will depend on the type of solver you are
            # using--see example projects
            return (dFdEx, dFdEy, dFdEz,
                    dFdHx, dFdHy, dFdHz)

        def calc_grad_y(self, sim, params):
            """Calculate the derivative of the figure of merit with respect to
            the design parameters themselves. This is useful for penalty
            functions."""
            dFdp = ... # calc using params
            return dFdp

After implementing your own :class:`AdjointMethod` class, it is straightforward
to use it to calculate the figure of merit and its gradient with respect to the
design variables. The base class implemented by EMopt will take care of all of
the dirty work internally:

.. code-block:: python

    am = MyAdjointMethod(sim, ...)
    design_params = # list of initial values for design params

    # compute figure of merit
    fom = am.fom(design_params)

    # verify the gradients are accurate
    am.check_gradient(design_params)

    # compute gradient of figure of merit
    gradient = am.gradient(design_params)

.. note::
    
    In the current version of EMopt, the value returned by :meth:`calc_dFdx`
    depends on whether you are working with a 2D or 3D solver. In 2D,
    :meth:`calc_dFdx` should return a tuple of arrays which specify dFdx in the
    whole simulation area. In 3D, it should return two lists. The first list
    should contain sets of 6 dFdx arrays (one for each field component) and the
    second should contain corresponding DomainCoordinates which specify where
    in the simulation those derivatives are from.

=====================
Running Optimizations
=====================

Optimizing electromagnetic structures is the bread and butter of EMopt. The
Maxwell solvers, mode solvers, and adjoint method implementation provided by
EMopt have all been created with optimization in mind. Technically, as soon as
you have set up a custom :class:`AdjointMethod` class, you are ready to
optimize your electromagnetic structure.

Unfortunately, because EMopt is written from the ground up based on MPI (for
parallelism) using the gradient information provided by :class:`AdjointMethod`
in conjunction with other optimization packages (e.g. scipy.optimize) is not
straightforward. In order to simplify this process, EMopt implements a small
class which interfaces between EMopt's parallelized components and scipy's
optimization library. This :class:`optimizer.Optimizer` class provides a very
simple interface for setting up and running an optimization. The process of
running an optimization is typically as follows:

.. code-block:: python

    import emopt

    # setup simulation, AdjointMethod, etc
    ...

    opt = emopt.optimizer.Optimizer(am, params, Nmax=100, opt_method='L-BFGS-B')
    fom, params_final = opt.run()

This snippet will run an optimization using the limited memory BFGS method
provided by scipy for a maximum of 100 iterations and then return the final
figure of merit and design parameters.

.. note:: 
    
    Before running a simulation, you are strongly encouraged to check that your
    gradients are accurate using :meth:`AdjointMethod.check_gradient`. The
    primary source of difficulties encountered when running gradient-based
    optimizations with EMopt is gradient inaccuracies which result from bugs in
    your code!

==========
References
==========

[1] A. Michaels and E. Yablonovitch, "
Gradient-Based Inverse Electromagnetic Design Using Continuously-Smoothed Boundaries," Arxiv 2017
