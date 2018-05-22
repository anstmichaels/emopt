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
:class:`MaterialPrimitive` which are rectangles and polygons. In 2D, these
:class:`MaterialPrimitive` can be overlayed with different priorities to form
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

Using the EMopt's FDFD solvers involves creating an FDFD object (:class:`FDFD_TE`
or :class:`FDFD_TM` in 2D or :class:`FDFD_3D` in 3D) which defines
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

=====================
Running Optimizations
=====================

==========
References
==========

[1] A. Michaels and E. Yablonovitch, "
Gradient-Based Inverse Electromagnetic Design Using Continuously-Smoothed Boundaries," Arxiv 2017
