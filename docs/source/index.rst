.. _emopt_documentation_index:

###################
EMopt Documentation
###################

========
Overview
========

EMopt is a toolkit for shape (and topology) optimization of 2D and 3D electromagnetic
structures.

EMopt offers a suite of tools for simulating and optimizing electromagnetic
structures. It includes 2D and 3D finite difference frequency domain solvers,
1D and 2D mode solvers, a flexible and *easily extensible* adjoint method
implementation, and a simple wrapper around scipy.minimize. Out of the box, it
provides just about everything needed to apply cutting-edge inverse design
techniques to your electromagnetic devices.

A key emphasis of EMopt's is shape optimization. Using boundary smoothing
techniques, EMopt allows you to compute sensitivities (i.e. gradient of a
figure of merit with respect to design variables which define an
electromagnetic device's shape) with very high accuracy. This allows you to
easily take adavantage of powerful minimization techniques in order to optimize
your electromagnetic device.

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
