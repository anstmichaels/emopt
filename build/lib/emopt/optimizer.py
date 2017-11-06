"""
This :class:`.Optimizer` class provides a simple wrapper around scipy.minimize.optimize
which allows you optimize an electromagnetic structure given an arbitrary
(user-defined) set of design parameters. The :class:`.Optimizer` class
minimizes a figure of merit defined in an
:class:`emopt.adjoint_method.AdjointMethod` object and takes advantage of the
gradient computed by the supplied
:class:`emopt.adjoint_method.AdjointMethod` object.


Examples
--------
The :class:`.Optimizer` is used approximately as follows:

.. doctest::
    import emopt.fdfd
    import emopt.adjoint_method
    import emopt.optimizer

    # Setup the simulation object
    sim = ...

    # Define a custom adjoint method class and instantiate it
    am = MyAdjointMethod(...)

    # Define a callback function
    def my_callback(params, ...):
        ...

    callback_func = lambda params : my_callback(params, other_inputs)

    # Specify initial guess for the design parameters
    design_params = ....

    # Create the optimizer object
    opt = Optimizer(sim, am, design_params, callback=callback_func)

    # run the optimization
    opt.run()
"""

import fdfd # this needs to come first
from misc import info_message, warning_message, error_message, RANK, \
NOT_PARALLEL, run_on_master

import numpy as np
from math import pi
from abc import ABCMeta, abstractmethod
from petsc4py import PETSc
from mpi4py import MPI
from scipy.optimize import minimize

__author__ = "Andrew Michaels"
__license__ = "Apache License, Version 2.0"
__version__ = "0.2"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

class Optimizer(object):
    """Handles the optimization of an electromagnetic structure.

    Given a set of design variables, a figure of merit, and a gradient, any
    electromagnetic structure can be optimized regardless of the underlying
    implementations of the simulator. With these quantities defined, it is in
    theory quite easy to run an optimization.

    Currently, the optimization code
    is based on scipy.optimize.minimize, which is not parallel/MPI-compatible.
    As is such, this class manages the interface between the sequential scipy
    calls and the parallel components of EMOpt (like running simulations).

    Fully parallelizing the optimization code should be possible using petsc.
    However, parallelizing the gradient computation is quite tricky for the
    most general case of arbitrary design variables.  The process of computing
    gradients in paralle is significantly simplified when the material in each
    grid cell is an independent design variable (i.e. grayscale topology
    optimization). This type of problem, however, is not the core goal of
    EMOpt. This functionality may be added in the future.

    Parameters
    ----------
    sim : emopt.fdfd.FDFD
        Simulation object
    am : emopt.adjoint_method.AdjointMethod
        Object containing problem-specific implementation of AdjointMethod
    p0 : numpy.ndarray or list
        Initial guess for design parameters of system
    callback_func : function
        Function which accepts the current design variables as the only
        argument. This function is called after each iteration of the
        optimization.  By default, no callback function is used.
    opt_method : str
        Optimization method to use.  The recommended options are: CG, BFGS,
        L-BFGS-B, TNC, SLSQP. (default='BFGS')
    Nmax : int
        Maximum number of interations of optimization method before process is
        terminated. (default=1000)
    tol : float
        Minimum change in figure of merit below which the optimization will
        complete. (default=1e-5)
    bounds : list of tuples
        List of tuples containing two floats which specify the lower and upper
        bounds on each design variable.  This is not compatible with all
        optimization methods.  Consult the scipy.optimize.minimize
        documentation for details. (default=None)

    Attributes
    ----------
    am : emopt.adjoint_method.AdjointMethod
        The adjoint method object for calculating FOM and gradient
    p0 : numpy.ndarray or list
        The initial guess for design variables
    callback : function
        The callback function to call after each optimization iteration.
    Nmax : int
        The maximum number of iterations
    tol : float
        The minimum change in figure of merit under which optimization
        terminates
    bounds : list of 2-tuple
        The list of bounds to put on design variables in the formate (minv, maxv)

    Methods
    -------
    run(self)
        Run the optimization.
    run_sequence(self, sim, am)
        Define the sequence of figure of merit and gradient calls for the
        optimization.
    """

    class RunCommands(object):
        """Run command codes used during message passing.

        We need a way to signal the non-master nodes to perform different
        operations during the optimization.  We do this by sending integers
        from the master node to the other nodes containing a command code. The
        commands are specified using an enum-like class.

        Attributes
        ----------
        FOM : int
            Tells worker nodes to compute the figure of merit
        GRAD : int
            Tells the worker nodes to compute the gradient of the figure of
            merit
        EXIT : int
            Tells the worker nodes to finish.
        """
        FOM = 0
        GRAD = 1
        EXIT = 2

    def __init__(self, am, p0, callback_func=None, opt_method='BFGS', \
                 Nmax=1000, tol=1e-5, bounds=None):
        self.am = am

        self.p0 = p0
        self.callback = callback_func
        self.opt_method = opt_method
        self.Nmax = Nmax
        self.tol = tol
        self.bounds = bounds

        self._comm = MPI.COMM_WORLD

    def run(self):
        """Run the optimization.

        Returns
        -------
        float
            The final figure of merit
        numpy.array
            The optimized design parameters
        """
        command = None
        running = True
        params = np.zeros(self.p0.shape)
        if(RANK == 0):
            pfinal, params = self.run_sequence(self.am)
            return pfinal, params
        else:
            while(running):
                # Wait for commands from the master node
                command = self._comm.bcast(command, root=0)

                if(command == self.RunCommands.FOM):
                    params = self._comm.bcast(params, root=0)
                    self.am.fom(params)
                elif(command == self.RunCommands.GRAD):
                    params = self._comm.bcast(params, root=0)
                    self.am.gradient(params)
                elif(command == self.RunCommands.EXIT):
                    running = False


    def __fom(self, params):
        # Execute the figure of merit in parallel
        command = self.RunCommands.FOM
        self._comm.bcast(command, root=0)
        self._comm.bcast(params, root=0)
        return self.am.fom(params)

    def __gradient(self, params):
        # Execute the figure of merit in parallel
        command = self.RunCommands.GRAD
        self._comm.bcast(command, root=0)
        self._comm.bcast(params, root=0)
        return self.am.gradient(params)

    def run_sequence(self, am):
        """Sequential optimization code.

        In general, the optimization itself is run in parallel.  Instead, only
        the calculation of the figure of merit and gradient takes advantage of
        paralellism (which is where the bulk of the computational complexity
        comes in).  This function defines the sequential optimization code and
        makes calls to the parallel components.

        Notes
        -----
        Override this method for custom functionality!

        Parameters
        ----------
        am : :class:`emopt.adjoint_method.AdjointMethod`
            The adjoint method object responsible for FOM and gradient
            calculations.

        Returns
        -------
        (float, numpy.ndarray)
            The optimized figure of merit and the corresponding set of optimal
            design parameters.
        """
        self.__fom(self.p0)
        self.callback(self.p0)
        result = minimize(self.__fom, self.p0, method=self.opt_method, \
                          jac=self.__gradient, callback=self.callback, \
                          tol=self.tol, options={'maxiter':self.Nmax, \
                                                 'disp':True})

        command = self.RunCommands.EXIT
        self._comm.bcast(command, root=0)

        return result.fun, result.x
