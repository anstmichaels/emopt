"""
This modules provides the definition for the :class:`.AdjointMethod` class.  Given an
electromagnetic structure (which is simulated using the FDFD class) and a merit
function which describes the 'performance' of that electromagnetic structure,
the :class:`.AdjointMethod` class defines the methods needed in order to calculate the gradient
of that merit function with respect to a set of user-defined design variables.

Notes
-----

Mathematically, the adjoint method calculates the gradient of a function
:math:`F(\mathbf{E}, \mathbf{H})` which has an explicit dependence on the
electric and magnetic fields (:math:`\mathbf{E}` and :math:`\mathbf{H}`).
Assuming we have expressed Maxwell's equations as a discretized linear system
of equations, one can show [1]_ that the gradient of :math:`F` is given by

.. math::
    \\nabla_\\mathbf{p} F = -2\\Re\\left\{ \\lambda^T \\frac{\\partial A}{\\partial p_i} x\\right\\}

where :math:`x` contains the electric and magnetic fields, :math:`\lambda`
contains a second set of 'adjoint' fields which are found by solving a second
set of linear system of equations which consist of the transposed Maxwell's
equations, and :math:`\partial A / \partial p_i` describes how the materials in
the system change with respect to changes to the design variables of the
system.

The AdjointMethod class does most of the work needed to compute :math:`x`,
:math:`\lambda`, :math:`\partial A / \partial p_i`, and the gradient
:math:`\\nabla_\\mathbf{p}F`.

More generally, we may specify a function of not only the fields, but also of
additional quantities which depend in some way on the design parameters.  In
this case, the function is given by

.. math::
    F \\rightarrow F(\\mathbf{E}, \\mathbf{H}, y_1, y_2, \cdots, y_N)

where :math:`y_1`, ..., :math:`y_N` are functions of the design variables but
not of the fields.  The derivative of this function with respect to the i'th
design variable, :math:`p_i` is given by

.. math::
    \\frac{\\partial F}{\\partial p_i} = -2\\Re\\left\{ \\lambda^T
    \\frac{\\partial A}{\\partial p_i} x\\right\\} + \\frac{\\partial
    F}{\\partial y_1} \\frac{\\partial y_1}{\\partial p_i} + \\frac{\\partial
    F}{\\partial y_1^*} \\frac{\\partial y_1^*}{\\partial p_i} + \cdots +
    \\frac{\\partial F}{\\partial y_N} \\frac{\\partial y_N}{\\partial p_i} +
    \\frac{\\partial F}{\\partial y_N^*} \\frac{\\partial y_N^*}{\\partial p_i}

The derivatives of the :math:`y`s are assumed to be known, thus very general
figures of merit can be computed using the adjoint method.

Note: This file uses MPI for parallelism.  As a result, return types and values
will depend on the RANK of the node running the code.

Examples
--------

The AdjointMethod class is used by extending the AdjointMethod base class. At
a minimum, four methods must be defined in the inheriting class. As an
example, a custom AdjointMethod class might look like

.. doctest::

    class MyAdjointMethod(AdjointMethod):

        def __init__(self, sim, myparam, step=1e-8):
            super(MyAdjointMethod, self).__init__(sim, step=step)
            self._myparam = myparam

        def update_system(self, params):
            # update system based on values in params
            ...

        def calc_fom(self, sim, params):
            # calculate figure of merit. We assume a simulation has already
            # been run and the fields are contained in the sim object which is
            # of type FDFD
            ...
            return fom

        def calc_dFdx(self, sim, params):
            # calculate derivative of F with respect to fields which is used as
            # source in adjoint simulation
            ...
            return dFdx

        def calc_grad_y(self, sim, params):
            # calculte the gradient of F with respect to any non-field
            # quantities
            ...
            return grad_y

Here :meth:`.AdjointMethod.update_system` updates the the system based on the
current set of design parameters, :meth:`.AdjointMethod.calc_fom` calculates
the value of F(\\mathbf{E}, \\mathbf{H}, y_1, y_2, \cdots, y_N) for the specified
set of design parameters in :samp:`params`, :meth:`.AdjointMethod.calc_dFdx`
calculates the derivative of :math:`F` with respect to the relevant field
components, and :meth:`.AdjointMethod.calc_grad_y` calculates the gradient of F
with respect to the non-field-dependent quantities in F.

In order to verify that the :meth:`.AdjointMethod.calc_fom`,
:meth:`.AdjointMethod.calc_dFdx`, and :meth:`.AdjointMethod.calc_grad_y`
functions are consistent, the gradient accuracty should always be verified. The
AdjointMethod base class defines a function to do just this.  For example, using
the :samp:`MyAdjointMethod` that we have just defined, we might do:

.. doctest::

    # set up an FDFD simulation object called 'sim'
    ...

    # create the adjoint method object
    am = MyAdjointMethod(sim, myparam)

    # check the gradient
    init_params = ...
    am.check_gradient(init_params, indices=np.arange(0,10))

In this example, we check the accuracy of the gradients computed for a given
initial set of design parameters called :samp:`init_params`.  We restrict the
check to the first ten components of the gradient in order to speed things up.

In addition to the adjoint method base class, there are a number of
application-specific implementations which you may find useful. In particular,
the :class:`.AdjointMethodFM` class provides a simplified interface for
computing the gradient of a function that depends not only on the fields but
also the permittivity and permeability.  In addition to the functions specified
above, the user must implement an additional function
:meth:`.AdjointMethodFM.calc_dFdm` which must compute the derivative of the figure
of merit :math:`F` with respect to the permittivity and permeability,
:math:`\epsilon` and :math:`\mu`.  An example of such a function would be, for
example, the total absorption of electromagnetic energy in a domain.

Furthermore, in electromagnetics, efficiencies make common figures of merit.
In many cases, this efficiency is defined in terms of the ratio of a calculated
power to the total source power of the system.  Because differentiating these
power-normalized quantities (which depend on the fields and the
permittivity/permeability) is rather laborious, this functionality is
implemented in the :class:`.AdjointMethodPNF` class for convenience.  The
interface for this class is a bit different from the base
:class:`.AdjointMethod` class.  The relevant functions which must be overriden
are:

.. doctest::

    class MyAdjointMethod(AdjointMethodPNF):

        def __init__(self, sim, myparam, step=1e-8):
            super(MyAdjointMethod, self).__init__(sim, step=step)
            self._myparam = myparam

        def update_system(self, params):
            # update system based on values in params
            ...

        def calc_f(self, sim, params):
            # calculate the non-power-normalized figure of merit.  This
            # function is called by AdjointMethodPNF.calc_fom where power
            # normalization is taken care of
            ...
            return fom

        def calc_dfdx(self, sim, params):
            # calculate derivative of f (the non-power-normalized figure of
            # merit) with respect to the field components.  This function is
            # called by AdjointMethodPNF.calc_dFdx which handles the
            # differentiation of the full power-normalized function.
            ...
            return dFdx

        def calc_grad_y(self, sim, params):
            # calculte the gradient of F with respect to any non-field
            # quantities
            ...
            return grad_y

See Also
--------
emopt.fdfd.FDFD : Base class for simulators supported by :class:`.AdjointMethod`.

emopt.optimizer.Optimizer : Primary application of :class:`.AdjointMethod` to optimization.

References
----------
.. [1] A. Michaels, E. Yablonovitch, "Gradient-Based Inverse Electromagnetic Design Using Continuously-Smoothed Boundaries", arXiv:1705.07188, 2017
"""

import fdfd # this needs to come first
from grid import StructuredMaterial, Rectangle
from misc import info_message, warning_message, error_message, RANK, \
NOT_PARALLEL, run_on_master
import fomutils

import numpy as np
from math import pi
from abc import ABCMeta, abstractmethod
from petsc4py import PETSc
from mpi4py import MPI

__author__ = "Andrew Michaels"
__license__ = "Apache License, Version 2.0"
__version__ = "0.2"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

class AdjointMethod(object):
    """Adjoint Method Class

    Defines the core functionality needed to compute the gradient of a function
    of the form

    .. math:
        F \\rightarrow F(\\mathbf{E}, \\mathbf{H}, y_1, y_2, \cdots, y_M)

    with respect to an arbitrary set of design variables :math:`\\mathbf{p}`.
    In general, the gradient is given by

    .. math::
        \\nabla F = \\nabla_\mathrm{AM} F + \\frac{\partial F}{\partial y_1}
        \\nabla_\mathbf{p} y_1 + \\frac{\partial F}{\partial y_1^*}
        \\nabla_\\mathbf{p} y_1^* + \cdots + \\frac{\partial F}{\partial y_M}
        \\nabla_\mathrm{p} y_M + \\frac{\partial F}{\partial y_M^*}
        \\nabla_\\mathrm{p} y_M^*

    where :math:`\\nabla_\\mathrm{AM} F` is the gradient of :math:`F` computed
    using the adjoint method, and the remaining gradient terms correspond to
    any dependence of the figure of merit on quantities other than the fields.
    The derivatives of these quantities are assumed to be known and should be
    computed using :meth:`.AdjointMethod.calc_grad_y` function.

    In order to use the AdjointMethod class, it should extended and the
    abstract methods :meth:`.AdjointMethod.update_system`,
    :meth:`.AdjointMethod.calc_fom`, :meth:`.AdjointMethod.calc_dFdx`, and
    :meth:`.AdjointMethod.calc_grad_y` should be implemented for the desired
    application.

    Notes
    -----
    Currently source derivatives are not supported.  If needed, this should not
    be too difficult to achieve by extending :class:`.AdjointMethod`

    Parameters
    ----------
    sim : emopt.fdfd.FDFD
        Simulation object
    step : float
        Step sized used in the calculation of :math:`\partial A / \partial p_i`

    Attributes
    ----------
    step : float
        Step sized used in the calculation of :math:`\partial A / \partial p_i`

    Methods
    -------
    update_system(params)
        **(ABSTRACT METHOD)** Update the geometry of the system.
    calc_fom(sim, params)
        **(ABSTRACT METHOD)** Calculate the figure of merit.
    calc_dFdx(sim, params)
        **(ABSTRACT METHOD)** Calculate the derivative of the figure of merit
        with respect to the electric and magnetic field vectors.
    get_update_boxes(sim, params)
        Define update boxes which specify which portion of the underlying
        spatial grid is modified by each design variable.
    fom(params)
        Get the figure of merit.
    calc_gradient(sim, params)
        Calculate the figure of merit in a general way.
    gradient(params)
        Get the gradient for at the current set of design parameter values.
    """
    __metaclass__ = ABCMeta

    def __init__(self, sim, step=1e-8):
        self.sim = sim
        self.prev_params = []
        self._step = step

    @property
    def step(self):
        """
        Step size used for numerical differentiation of :math:`A`

        :getter: Returns the step size.
        :setter: Sets the step size
        :type: float
        """
        return self._step

    @step.setter
    def step(self, val):
        if(np.abs(val) > self.sim.dx/1e3):
            if(NOT_PARALLEL):
                warning_message('Step size used for adjoint method may be too '
                                'large.  Consider reducing it to ~1e-3*dx')
        self._step = val

    @abstractmethod
    def update_system(self, params):
        """Update the geometry/material distributions of the system.

        In order to calculate the gradient of a figure of merit, we need to
        define a mapping between the abstract design parameters of the system
        (which are contained in the vector :samp:`params`) and the underlying
        geometry or material distribution which makes up the physical system.
        We define this mapping here.

        Notes
        -----
        In general, calculation of the gradient involves calling this function
        once per design variable.  In other words, if :samp:`len(params)` is
        equal to N, then this method is called at least N times in order to
        calculate the gradient.  For cases where N is large, it is recommended
        an effort be made to avoid performing very costly operations in this
        method.

        Parameters
        ----------
        params : numpy.ndarray
            1D array containing design parameter values (one value per design
            parameter)
        """
        pass

    @abstractmethod
    def calc_fom(self, sim, params):
        """Calculate the figure of merit.

        Notes
        -----
        This function is called by the :func:`.AdjointMethod.fom` function. In
        this case, update_system(params) and sim.solve_forward() are guaranteed
        to be called before this function is executed.

        If this function is called outside of the :func:`.AdjointMethod.fom`
        function (which is not advised), it is up to the caller to ensure that
        the :func:`.emopt.FDFD.solve_forward` has been run previously.

        Parameters
        ----------
        sim : emopt.fdfd.FDFD
            Simulation object
        params : numpy.ndarray
            1D vector containing design parameter values.
        """
        pass

    @abstractmethod
    def calc_dFdx(self, sim, params):
        """Calculate the derivative of the figure of merit with respect to the
        vector containing the electric and magnetic fields.

        In order to calculate the gradient of the figure of merit, an adjoint
        simulation must first be run.  The sources in the adjoint simulation are
        given by :math:`\partial F / \partial x` where :math:`F` is the figure of
        merit and :math:`x` is a vector containing the electric and magnetic fields
        contained on a discreter grid.  Because we are differentiating with respect
        to a vector, the resulting derivative will also be a vector.

        This function must be overriden and implemented to calculate the derivative
        of the figure of merit defined in :func:`calc_fom`.

        The exact format of :math:`x` depends on the exact type of
        :class:`emopt.fdfd.FDFD` object which generated it.  Consult
        :mod:`emopt.fdfd` for details.

        See Also
        --------
        emopt.fdfd.FDFD
            Base class for simulators which generate :math:`x`
        """
        pass

    @abstractmethod
    def calc_grad_y(self, sim, params):
        """Compute the gradient of :math:`y_i` and :math:`y_i^*` with
        respect to the design parameters of the system

        This function should calculate the list of gradients/derivatives
        of :math:`y_i` and :math:`y_i^*` given by

        .. math::
            \\left[ \\frac{\\partial F}{\\partial y_1}\\nabla_\mathbf{p} y_i,
            \\frac{\\partial F}{\\partial y_1^*}\\nabla_\mathbf{p} y_i^*, \cdots,
            \\frac{\\partial F}{\\partial y_M}\\nabla_\mathbf{p} y_M,
            \\frac{\\partial F}{\\partial y_M^*}\\nabla_\mathbf{p} y_M^* \\right]

        Notes
        -----
        This function is executed in parallel on all nodes.  If execution on
        master node is desired, you can either apply the @run_on_master
        decorator or use if(NOT_PARALLEL).

        Parameters
        ----------
        sim : emoptg.fdfd.FDFD
            The FDFD object
        params : numpy.ndarray
            The array containing the current set of design parameters

        Returns
        -------
        numpy.ndarray
            The sum of the extra "non-adjoint" gradient terms
        """
        pass

    def get_update_boxes(self, sim, params):
        """Get update boxes used to speed up the updating of A.

        In order to compute the gradient, we need to calculate how A changes
        with respect to modification of the design variables.  This generally
        requires updating the material values in A.  We can speed this process
        up by only updating the part of the system which is affect by the
        modification of each design variable.

        By default, the update boxes cover the whole simulation area.  This
        method can be overriden in order to modify this behavior.

        Parameters
        ----------
        sim : FDFD
            Simulation object.  sim = self.sim
        params : numpy.array or list of floats
            List of design parameters.

        Returns
        -------
            List of tuples containing (x_min, x_max, y_min, y_max) where x_min,
            etc are the minimum and maximum x and y spatial indices.
        """
        M = sim.M
        N = sim.N
        lenp = len(params)
        return [(0, N, 0, M) for i in range(lenp)]

    def fom(self, params):
        """Run a forward simulation and calculate the figure of merit.

        Notes
        -----
        The simualtion is performed in parallel with the help of all of the
        MPI node, however the calculation of the figure of merit itself is
        currently only performed on the master node (RANK == 0)

        Parameters
        ----------
        params : numpy.ndarray
            List of design parameters of the system

        Returns
        -------
        float
            **(Master node only)** The figure of merit :math:`F(\mathbf{E},\mathbf{H} ; \mathbf{p})`
        """
        # update the system using the design parameters
        self.update_system(params)
        self.prev_params = params
        self.sim.update()

        #run the forward sim
        self.sim.solve_forward()

        # calculate the figure of merit
        return self.calc_fom(self.sim, params)

    def calc_gradient(self, sim, params):
        """Calculate the gradient of the figure of merit.

        The gradient of the figure of merit is computed by running a forward
        simulation, adjoint simulation, and then computing the derivatives of
        the system matrix :math:`A` with respect to the design parameters of
        the system, i.e. :math:`\partial A / \partial p_i`. In the most general
        case, we can compute this derivative using finite differences. This
        involves perturbing each design variable of the system by a small
        amount one at a time and updating :math:`A`.  Doing so allows us to
        approximate the derivative as

        .. math::
            \\frac{\partial A}{\partial p_i} \\approx \\frac{A(p_i + \Delta p) - A(p_i)}{\Delta p}

        So long as :math:`\Delta p` is small enough, this approximation is
        quite accurate.

        This function handles this process.

        Notes
        -----
        1. This function is called after the forward and adjoint simulations have
        been executed.

        2. Technically, a centered difference would be more accurate, however
        this whole implementation relies on mesh smoothing which allows us to
        make very small steps :math:`\Delta p` and thus in reality, the benefit
        is negligable.

        Parameters
        ----------
        sim : FDFD
            Simulation object.  sim = self.sim
        params : numpy.array or list of floats
            List of design parameters.

        Returns
        -------
        numpy.array
            **(Master node only)** Gradient of figure of merit, i.e. list of
            derivatives of fom with respect to each design variable
        """
        # get the current diagonal elements of A.
        # only these elements change when the design variables change.
        Ai = PETSc.Vec()
        Af = PETSc.Vec()
        Ai = sim.get_A_diag(Ai)

        x = sim.x.copy()
        x_adj = sim.x_adj.copy()

        step = self._step
        update_boxes = self.get_update_boxes(sim, params)
        lenp = len(params)

        grad_full = None
        if(RANK == 0):
            grad_full = np.zeros(sim.nunks, dtype=np.double)

        gradient = np.zeros(lenp)
        for i in xrange(lenp):
            if(NOT_PARALLEL):
                print i
            p0 = params[i]
            ub = update_boxes[i]

            # perturb the system
            params[i] += step
            self.update_system(params)
            self.sim.update(ub)

            # get the updated diagonal elements of A
            Af = sim.get_A_diag(Af)

            # calculate dAdp and assemble the full result on the master node
            dAdp = (Af-Ai)/step
            product = x_adj * dAdp * x
            grad_part = -2*np.real( np.sum(product[...]) )

            # send the partially computed gradient to the master node to finish
            # up the calculation
            MPI.COMM_WORLD.Gatherv(grad_part, grad_full, root=0)

            # finish calculating the gradient
            if(NOT_PARALLEL):
                gradient[i] = np.sum(grad_full)

            # revert the system to its original state
            params[i] = p0
            self.update_system(params)
            self.sim.update(ub)

        if(NOT_PARALLEL):
            return gradient

    def gradient(self, params):
        """Manage the calculation of the gradient figure of merit.

        To calculate the gradient, we update the system, run a forward and
        adjoint simulation, and then calculate the gradient using
        :func:`calc_gradient`.  Most of these operations are done in parallel
        using MPI.

        Parameters
        ----------
        params : numpy.ndarray
            List of design parameters of the system

        Returns
        -------
        numpy.ndarray
            (Master node only) The gradient of the figure of merit computed  with 
            respect to the design variables
        """
        # update system
        self.update_system(params)
        self.sim.update()

        # run the forward simulation
        if(self.prev_params is not params):
            self.sim.solve_forward()

        # get dFdx which will be used as the source for the adjoint simulation
        # dFdx is calculated on the root node and then broadcast to the other
        # nodes.
        # TODO: parallelize this operation?
        comm = MPI.COMM_WORLD

        # This should return only non-null on RANK=0
        dFdx = self.calc_dFdx(self.sim, params)
        dFdx = comm.bcast(dFdx, root=0)

        # run the adjoint source
        self.sim.set_adjoint_sources(dFdx)
        self.sim.solve_adjoint()

        if(NOT_PARALLEL):
            info_message('Calculating gradient...')

        grad_f = self.calc_gradient(self.sim, params)
        grad_y = self.calc_grad_y(self.sim, params)

        if(NOT_PARALLEL):
            return grad_f + grad_y
        else:
            return None

    def check_gradient(self, params, indices=[]):
        """Verify that the gradient is accurate.

        It is highly recommended that the accuracy of the gradients be checked
        prior to being used. If the accuracy is above ~1%, it is likely that
        there is an inconsitency between how the figure of merit and adjoint
        sources (dFdx) are being computed.

        The adjoint method gradient error is evaluated by comparing the
        gradient computed using the adjoint method to a gradient computed by
        direct finite differences.  In other words, the "correct" derivatives
        to which the adjoint method gradient is compared are given by

        .. math::
            \\frac{\partial F}{\partial p_i} \\approx \\frac{F(p_i + \Delta p) - F(p_i)}{\Delta p}

        Note that this method for calculating the gradient is not used in a
        practical setting because it requires performing N+1 simulations in
        order to compute the gradient with respect to N design variables
        (compared to only 2 simulations in the case of the adjoint method).

        Parameters
        ----------
        params : numpy.ndarray
            design parameters
        indices : list or numpy.ndarray
            list of gradient indices to check. An empty list indicates that the
            whole gradient should be verified. A subset of indices may be
            desirable for large problems.  (default = [])

        Returns
        -------
        float
            Relative error in gradient.
        """

        if(indices == []):
            indices = np.arange(0, len(params),1)

        # make sure everything is up to date
        self.update_system(params)
        self.sim.update()

        grad_am = self.gradient(params)
        grad_fd = np.zeros(len(indices))

        fom0 = self.fom(params)

        # calculate the "true" derivatives using finite differences
        if(NOT_PARALLEL):
            info_message('Checking gradient...')

        for i in range(len(indices)):
            if(NOT_PARALLEL):
                print('\tDerivative %d of %d' % (i+1, len(indices)))

            j = indices[i]
            p0 = params[j]
            params[j] += self._step
            fom1 = self.fom(params)
            if(NOT_PARALLEL):
                grad_fd[i] = (fom1-fom0)/self._step
            params[j] = p0


        if(NOT_PARALLEL):
            errors = np.abs(grad_fd - grad_am[indices]) / np.abs(grad_fd)
            error_tot = np.linalg.norm(grad_fd - grad_am[indices]) / np.linalg.norm(grad_fd)

            if(error_tot < 0.01):
                info_message('The total error in the gradient is %0.4E' % \
                             (error_tot))
            else:
                warning_message('The total error in the gradient is %0.4E '
                                'which is over 1%%' % (error_tot), \
                                'emopt.adjoint_method')

            import matplotlib.pyplot as plt
            f = plt.figure()
            ax1 = f.add_subplot(311)
            ax2 = f.add_subplot(312)
            ax3 = f.add_subplot(313)

            ax1.bar(indices, grad_fd)
            ax2.bar(indices, grad_am[indices])
            ax3.bar(indices, errors)

            ax3.set_yscale('log', nonposy='clip')

            plt.show()

            return error_tot

class AdjointMethodMO(AdjointMethod):
    """An AdjointMethod object for an ensemble of different figures of merit
    (Multi-objective adjoint method).

    In many situations, it is desirable to calculate the sensitivities of a
    structure corresponding to multiple objective functions.  A simple common
    exmaple of this a broadband figure of merit which considers the performance
    of structure at a range of different excitation frequencies/wavelengths.
    In other cases, it may be desirable to calculate a total sensitivity which
    is made up of two different figures of merit which are calculated for the
    same excitation.

    In either case, we need a way to easily handle these more complicated
    figures of merits and their gradients (i.e. the sensitivities). This class
    provides a simple interface to do just that.  By overriding calc_total_fom
    and calc_total_gradient, you can build up more complicated figures of
    merit.

    Parameters
    ----------
    ams : list of :class:`.AdjointMethod`
        A list containing *extended* AdjointMethod objects

    Attributes
    ----------
    ams : list of :class:`.AdjointMethod`
        A list containing extended AdjointMethod objects
    """
    __metaclass__ = ABCMeta

    def __init__(self, ams):
        self._ams = ams
        self._foms_current = np.zeros(len(ams))

    @property
    def ams(self):
        return self._ams

    @ams.getter
    def ams(self, new_ams):
        self._ams = new_ams

    def update_system(self, params):
        # We dont need this -- all system updates are performed by
        # AdjointMethod objects contained in self._ams
        pass

    def calc_fom(self, sim, params):
        """Calculate the figure of merit.
        """
        # this just redirects to calc_total_foms
        return self.calc_total_fom(self._foms_current)

    def calc_dFdx(self, sim, params):
        # We dont need this -- all dFdx's are performed by
        # AdjointMethod objects contained in self._ams
        pass

    def calc_grad_y(self, sim, params):
        # We dont need this -- all individual grad_y calculations are handled
        # by supplied AdjointMethod objects.
        pass

    @abstractmethod
    def calc_total_fom(self, foms):
        """Calculate the 'total' figure of merit based on a list of evaluated
        objective functions.

        The user should override this function in order to define how all of the
        individual figures of merit are combined to form a single 'total'
        figure of merit. This may be a sum of the input FOMs, a minimax of the
        FOMs, etc.  A common example is to combine figures of merit calculated
        for different wavelengths of operation.

        See Also
        --------
        :ref:`emopt.fomutils` for functions which may be useful for
        combining figures of merit

        Parameters
        ----------
        foms : list of float
            List containing individual FOMs which are used to compute the total
            figure of merit.

        Returns
        -------
        float
            The total figure of merit
        """
        pass

    @abstractmethod
    def calc_total_gradient(self, foms, grads):
        """Calculate the 'total' gradient of a figure of merit based on a list
        of evaluated objective functions.

        The user should override this function in order to define the gradient
        of the total figure of merit.

        See Also
        --------
        :ref:`emopt.fomutils` for functions which may be useful for combining
        figures of merit and their gradients.

        Parameters
        ----------
        foms : list
            List of individual foms
        grads : list
            List of individual grads

        Returns
        -------
        numpy.ndarray
            1D numpy array containing total gradient. note: the output vector
            should have the same shape as the input vectors contained in grads
        """
        pass

    def fom(self, params):
        """Calculate the total figure of merit.

        Notes
        -----
        Overrides :class:`.AdjointMethod`.fom(...)

        Parameters
        ----------
        params : numpy.ndarray
            Design parameters

        Returns
        -------
        float
            (Master node only) The total figure of merit
        """
        foms = []

        for am in self._ams:
            foms.append(am.fom(params))

        self._foms_current = foms
        if(NOT_PARALLEL):
            fom_total = self.calc_total_fom(foms)
            return fom_total
        else:
            return None

    def gradient(self, params):
        """Calculate the total gradient.

        Notes
        -----
        Overrides :class:`.AdjointMethod`.gradient(...)

        Parameters
        ----------
        params : numpy.ndarray
            Design parameters with respect to which gradient is evaluated

        Returns
        -------
        numpy.ndarray
            (Master node only) The gradient of total figure of merit.
        """
        foms = []
        grads = []

        for am in self._ams:
            foms.append( am.fom(params) )
            grads.append( am.gradient(params) )

        if(NOT_PARALLEL):
            grad_total = self.calc_total_gradient(foms, grads)
            return grad_total
        else:
            return None


    def check_gradient(self, params, indices=[]):
        """Check the gradient of an multi-objective AdjointMethod.

        Parameters
        ----------
        params : numpy.ndarray
            design parameters
        indices : list or numpy.ndarray
            list of gradient indices to check. An empty list indicates that the
            whole gradient should be verified. A subset of indices may be
            desirable for large problems.  (default = [])

        Returns
        -------
        float
            Relative error in gradient.
        """
        # we override this function so we can initially update all of the ams
        # as desired
        self._step = 1e-8
        self.sim = self._ams[0].sim
        for am in self._ams:
            am.update_system(params)
            am.sim.update()

        super(AdjointMethodMO, self).check_gradient(params, indices)

class AdjointMethodFM(AdjointMethod):
    """Define an :class:`.AdjointMethod` which simplifies the calculation of
    gradients which are a function of the materials (eps and mu).

    In certain cases, the gradient of a function of the fields, permittivity,
    and permeability may be desired.  Differentiating the function with respect
    to the permittivity and permeability shares many of the same calculations
    in common with :meth:`.AdjointMethod.calc_gradient`.  In order to maximize
    performance and simplify the implementation of material-dependent figures
    of merit, this class reimplements the :meth:`calc_gradient` function.

    Attributes
    ----------
    sim : :class:`emopt.fdfd.FDFD`
        The simulation object
    step : float (optional)
        The step size used by gradient calculation (default = False)
    """
    def __init__(self, sim, step=1e-8):
        super(AdjointMethodFM, self).__init__(sim, step)

    @abstractmethod
    def calc_dFdm(self, sim, params):
        """Calculate the derivative of F with respect to :math:`\epsilon`,
        :math:`\epsilon^*`, :math:`\mu`, and :math:`\mu^*`

        Parameters
        ----------
        sim : emopt.fdfd.FDFD
            The FDFD simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        tuple of numpy.array
            The derivatives of F with respect to spatially-dependent eps and mu
            and their complex conjugates. These derivatives should be arrays
            with dimension (M,N) and should be returned in a tuple with the
            format (dFdeps, dFdeps_conf, dFdmu, dFdmu_conj)
        """
        pass

    def calc_gradient(self, sim, params):
        """Calculate the gradient of a figure of merit which depends on the
        permittivity and permeability.

        Parameters
        ----------
        sim : FDFD
            Simulation object.  sim = self.sim
        params : numpy.array or list of floats
            List of design parameters.

        Returns
        -------
        numpy.array
            **(Master node only)** Gradient of figure of merit, i.e. list of
            derivatives of fom with respect to each design variable
        """
        # Semantically, we would not normally need to override this method,
        # however it turns out that the operations needed to compute the
        # gradient of a field-dependent function and a permittivit-dependent
        # function are very similar (both require the calculation of the
        # derivative of the materials wrt the design parameters.)  For the sake
        # of performance, we combine the two calculations here.

        w_pml_l = sim.w_pml_left
        w_pml_r = sim.w_pml_right
        w_pml_t = sim.w_pml_top
        w_pml_b = sim.w_pml_bottom
        M = sim.M
        N = sim.N

        # get the current diagonal elements of A.
        # only these elements change when the design variables change.
        Ai = PETSc.Vec()
        Af = PETSc.Vec()
        Ai = sim.get_A_diag(Ai)

        # Get the derivatives w.r.t. eps, mu
        if(NOT_PARALLEL):
            dFdeps, dFdeps_conj, dFdmu, dFdmu_conj = self.calc_dFdm(sim, params)

        x = sim.x.copy()
        x_adj = sim.x_adj.copy()

        step = self._step
        update_boxes = self.get_update_boxes(sim, params)
        lenp = len(params)

        grad_full = None
        if(RANK == 0):
            grad_full = np.zeros(sim.nunks, dtype=np.double)

        gradient = np.zeros(lenp)
        for i in xrange(lenp):
            #if(NOT_PARALLEL):
            #    print i
            p0 = params[i]
            ub = update_boxes[i]

            # perturb the system
            params[i] += step
            self.update_system(params)
            self.sim.update(ub)

            # get the updated diagonal elements of A
            Af = sim.get_A_diag(Af)

            # calculate dAdp and assemble the full result on the master node
            dAdp = (Af-Ai)/step
            product = x_adj * dAdp * x
            grad_part = -2*np.real( np.sum(product[...]) )

            # send the partially computed gradient to the master node to finish
            # up the calculation
            MPI.COMM_WORLD.Gatherv(grad_part, grad_full, root=0)

            # We also need dAdp to account for the derivative of eps and mu
            gatherer, dAdp_full = PETSc.Scatter().toZero(dAdp)
            gatherer.scatter(dAdp, dAdp_full, False, PETSc.Scatter.Mode.FORWARD)

            # finish calculating the gradient
            if(NOT_PARALLEL):
                # derivative with respect to fields
                gradient[i] = np.sum(grad_full)

                # Next we compute the derivative with respect to eps and mu. We
                # exclude the PML regions because changes to the materials in
                # the PMLs are generally not something we want to consider.
                jmin = ub[0]; jmax = ub[1]
                imin = ub[2]; imax = ub[3]
                if(jmin < w_pml_l): jmin = w_pml_l
                if(jmax > N-w_pml_r): jmax = N-w_pml_r
                if(imin < w_pml_b): imin = w_pml_b
                if(imax > M-w_pml_t): imax = M-w_pml_t

                # note that the extraction of eps and mu from A must be handled
                # slightly differently in the TE and TM cases since the signs
                # along the diagonal are swapped and eps and mu are positioned
                # in different parts
                if(isinstance(sim, fdfd.FDFD_TM)):
                    dmudp = dAdp_full[0:M*N].reshape([M,N]) * 1j
                    depsdp = dAdp_full[M*N:2*M*N].reshape([M,N]) / 1j
                elif(isinstance(sim, fdfd.FDFD_TE)):
                    depsdp = dAdp_full[0:M*N].reshape([M,N]) / 1j
                    dmudp = dAdp_full[M*N:2*M*N].reshape([M,N]) * 1j

                gradient[i] += np.real(
                               np.sum(dFdeps[imin:imax, jmin:jmax] * \
                                      depsdp[imin:imax, jmin:jmax]) + \
                               np.sum(dFdeps_conj[imin:imax, jmin:jmax] * \
                                      np.conj(depsdp[imin:imax, jmin:jmax])) + \
                               np.sum(dFdmu[imin:imax, jmin:jmax] * \
                                      dmudp[imin:imax, jmin:jmax]) + \
                               np.sum(dFdmu_conj[imin:imax, jmin:jmax] * \
                                      np.conj(dmudp[imin:imax, jmin:jmax])) \
                               )

            # revert the system to its original state
            params[i] = p0
            self.update_system(params)
            self.sim.update(ub)

        if(NOT_PARALLEL):
            return gradient

class AdjointMethodPNF(AdjointMethodFM):
    """Define an AdjointMethod object for a figure of merit which contains
    power normalization.

    A power-normalized figure of merit has the form

    .. math::
        F(\\mathbf{E}, \\mathbf{H}, \\epsilon, \\mu) = \\frac{f(\\mathbf{E},
        \\mathbf{H})} {P_\mathrm{src}(\\mathbf{E}, \\mathbf{H}, \\epsilon, \\mu)}

    where :math:`\\epsilon` and :math:`\\mu` are the permittivity and
    permeability and :math:`f(...)` is a figure of merit which depends only on
    the fields (e.g. power flowing through a plane, mode match, etc)
    """

    def __init__(self, sim, step=1e-8):
        super(AdjointMethodPNF, self).__init__(sim, step)

    @abstractmethod
    def calc_f(self, sim, params):
        """Calculate the non-power-normalized figure of merit
        :math:`f(\\mathbf{E}, \\mathbf{H})`.

        Notes
        -----
        This function is called only on the master node.

        Parameters
        ----------
        sim : emopt.fdfd.FDFD
            The FDFD simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        float
            The value of the non-power-normalized figure of merit.
        """
        pass

    @abstractmethod
    def calc_dfdx(self, sim, params):
        """Calculate the derivative of the non-power-normalized figure of merit
        with respect to the fields in the discretized grid.

        Notes
        -----
        This function is called only on the master node.

        Parameters
        ----------
        sim : emopt.fdfd.FDFD
            The FDFD simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        tuple of numpy.ndarray
            The derivative of f with respect to the fields in the form (E, H)
        """
        pass

    def calc_fom(self, sim, params):
        """Calculate the power-normalized figure of merit.

        Parameters
        ----------
        sim : emopt.fdfd.FDFD
            The FDFD simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        float
            The value of the power-normalized figure of merit.
        """
        f = self.calc_f(sim, params)
        Psrc = sim.get_source_power()

        if(NOT_PARALLEL):
            return f / Psrc
        else:
            return None

    def calc_dFdx(self, sim, params):
        """Calculate the derivative of the power-normalized figure of merit
        with respect to the field.

        Parameters
        ----------
        sim : emopt.fdfd.FDFD
            The FDFD simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        tuple of numpy.ndarray
            The derivative of F with respect to the fields in the form (E, H)
        """
        dfdx = self.calc_dfdx(sim, params)
        f = self.calc_f(sim, params)

        if(NOT_PARALLEL):
            if(isinstance(sim, fdfd.FDFD_TM)):
                dfdHz = dfdx[0]
                dfdEx = dfdx[1]
                dfdEy = dfdx[2]

                dFdHz, dFdEx, dFdEy = fomutils.power_norm_dFdx_TM(sim, f, dfdHz, \
                                                                          dfdEx, \
                                                                          dfdEy)
                return (dFdHz, dFdEx, dFdEy)
            elif(isinstance(sim, fdfd.FDFD_TE)):
                dfdEz = dfdx[0]
                dfdHx = dfdx[1]
                dfdHy = dfdx[2]

                dFdEz, dFdHx, dFdHy = fomutils.power_norm_dFdx_TE(sim, f, dfdEz, \
                                                                          dfdHx, \
                                                                          dfdHy)
                return (dFdEz, dFdHx, dFdHy)
        else:
            return None

    def calc_dFdm(self, sim, params):
        """Calculate the derivative of the power-normalized figure of merit
        with respect to the permittivity and permeability.

        Parameters
        ----------
        sim : emopt.fdfd.FDFD
            The FDFD simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        list of numpy.ndarray
            (Master node only) The derivative of F with respect to :math:`\epsilon`,
            :math:`\epsilon^*`, :math:`\mu`, and :math:`\mu^*`
        """
        # isinstance(sim, fdfd.FDFD_TM) must come before TE since TM is a
        # subclass of TE
        if(isinstance(sim, fdfd.FDFD_TM)):
            M = sim.M
            N = sim.N
            dx = sim.dx
            dy = sim.dy

            Hz = sim.get_field_interp('Hz')
            Ex = sim.get_field_interp('Ex')
            Ey = sim.get_field_interp('Ey')

            # compute the magnitudes squared of E and H -- this is all we need
            # here.
            if(NOT_PARALLEL):
                E2 = Ex * np.conj(Ex) + Ey * np.conj(Ey)
                H2 = Hz * np.conj(Hz)

        elif(isinstance(sim, fdfd.FDFD_TE)):
            M = sim.M
            N = sim.N
            dx = sim.dx
            dy = sim.dy

            Ez = sim.get_field_interp('Ez')
            Hx = sim.get_field_interp('Hx')
            Hy = sim.get_field_interp('Hy')

            # compute the magnitudes squared of E and H -- this is all we need
            # here.
            if(NOT_PARALLEL):
                E2 = Ez * np.conj(Ez)
                H2 = Hx * np.conj(Hx) + \
                     Hy * np.conj(Hy)

        if(NOT_PARALLEL):
            #y1 = eps, y2 = eps^*, y3 = mu, y4 = mu^*
            dPdy1 = -1j*0.125 * dx * dy * E2
            dPdy2 = 1j*0.125 * dx * dy * E2
            dPdy3 = -1j*0.125 * dx * dy * H2
            dPdy4 = 1j*0.125 * dx * dy * H2

            f = self.calc_f(sim, params)
            Ptot = sim.get_source_power()

            dFdy1 = -f / Ptot**2 * dPdy1
            dFdy2 = -f / Ptot**2 * dPdy2
            dFdy3 = -f / Ptot**2 * dPdy3
            dFdy4 = -f / Ptot**2 * dPdy4

            return dFdy1, dFdy2, dFdy3, dFdy4
        else:
            return None
