from __future__ import absolute_import

from .. import fdfd # this needs to come first
from ..misc import info_message, warning_message, error_message, RANK, \
NOT_PARALLEL, run_on_master, COMM
from ..optimizer import Optimizer

import numpy as np
from scipy.optimize import minimize

import time

__author__ = "Andrew Michaels"
__license__ = "BSD-3"
__version__ = "2019.5.6"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

class TimedOptimizer(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_times = []
        self.grad_f_times = []
        self.fom_times = []
        self.nit = 0
        self.nfev = 0
        self.njev = 0
        self.total_time = 0

    def __fom(self, params):
        start = time.time()
        fom = Optimizer._Optimizer__fom(self, params)
        end = time.time()
        self.fom_times.append(end-start)
        return fom

    def __gradient(self, params):
        start = time.time()
        grad = Optimizer._Optimizer__gradient(self, params)
        end = time.time()
        self.grad_times.append(end-start)
        self.grad_f_times.append(self.am._grad_f_time)
        return grad

    def run_sequence(self, am):
        self.num_params = self.p0.shape[0]
        start = time.time()
        self.__fom(self.p0)
        self.callback(self.p0)
        result = minimize(self.__fom, self.p0, method=self.opt_method,
                          jac=self.__gradient, callback=self.callback,
                          tol=self.tol, bounds=self.bounds,
                          options={'maxiter':self.Nmax, \
                                   'disp':self.scipy_verbose})

        command = self.RunCommands.EXIT
        self._comm.bcast(command, root=0)
        end = time.time()

        self.total_time = end-start
        self.nit = result.nit
        self.nfev = result.nfev
        self.njev = result.njev

        return result.fun, result.x
