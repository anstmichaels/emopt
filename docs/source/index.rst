.. _emopt_documentation_index:

###################
EMopt Documentation
###################

========
Overview
========

EMopt is a toolkit for shape (and topology) optimization of 2D and 3D electromagnetic
structures.

EMopt implements the adjoint method along with subroutines for smoothing of
material interfaces on a rectangular grid (grid smoothing) in order compute
gradients of a function of the electic and magnetic field. These gradients can
be used in conjunction with a number of minimization techniques in order to
optimize a complicated passive electromagnetic device. The included adjoint
method code relies on a small simple finite difference frequency domain (FDFD)
solver which solves Maxwell's equations on a rectangular grid. The grid
smoothing interface, meanwhile, is independent and can be used with the adjoint
method + FDFD libraries or with other 3rd party solvers.

.. toctree::
    :hidden:

    self

.. toctree::
    :maxdepth: 2
    :caption: Getting Started

    installation
    at_a_glance
    tutorials

Module Index
============

To see the modules which make up EMopt and view the class and function
documentation for each, head to the  :ref:`module index <modindex>`.

To view a complete index for all classes, head to the
:ref:`index page <genindex>`. Alternatively, the documentation can be 
searched using the side bar.
