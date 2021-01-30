"""EMopt: Simulate and optimize electromagnetic devices with an emphasis on waveguiding
devices.
"""
__author__ = "Andrew Michaels"
__license__ = "GPL License, Version 3.0"
__version__ = "2020.10.3"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

from . import opt_def, solvers, fomutils, grid, io, misc, optimizer, geometry

__all__ = ["opt_def", "solvers", "fomutils", "grid", "io", "misc",
          "optimizer", "geometry"]
