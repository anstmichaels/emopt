"""Define an interface for accessing the grid library written in c++."""

from __future__ import division, print_function, absolute_import
from ctypes import *
import os
import numpy as np
from numpy.ctypeslib import ndpointer

__author__ = "Andrew Michaels"
__license__ = "GPL License, Version 3.0"
__version__ = "0.4"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

dir_path = os.path.dirname(os.path.realpath(__file__))

so_path = ''.join([dir_path, '/Grid.so'])
libGrid = cdll.LoadLibrary(so_path)

# some useful definitions
c_complex_2D_p = ndpointer(np.complex128, ndim=2, flags='C')
c_complex_1D_p = ndpointer(np.complex128, ndim=1, flags='C')
c_double_p = ndpointer(np.double, ndim=1, flags='C')

#####################################################################################
# Material2D configuration 
#####################################################################################
libGrid.Material2D_get_value.argtypes = [c_void_p, c_complex_1D_p, c_double, c_double]
libGrid.Material2D_get_value.restype = None
libGrid.Material2D_get_values.argtypes = [c_void_p, c_int, c_int, c_int, c_int,
                                          c_complex_1D_p]
libGrid.Material2D_get_values.restype = None

####################################################################################
# GridMaterial2D configuration
####################################################################################
libGrid.GridMaterial2D_new.argtypes = [c_int, c_int, c_complex_2D_p]
libGrid.GridMaterial2D_new.restype = c_void_p
libGrid.GridMaterial2D_delete.argtypes = [c_void_p]
libGrid.GridMaterial2D_delete.restype = None
libGrid.GridMaterial2D_set_grid.argtypes = [c_void_p, c_int, c_int, c_complex_2D_p]
libGrid.GridMaterial2D_set_grid.restype = None
libGrid.GridMaterial2D_get_M.argtypes = [c_void_p]
libGrid.GridMaterial2D_get_M.restype = c_int
libGrid.GridMaterial2D_get_N.argtypes = [c_void_p]
libGrid.GridMaterial2D_get_N.restype = c_int

####################################################################################
# StructuredMaterial2D configuration
####################################################################################
libGrid.StructuredMaterial2D_new.argtypes = [c_double, c_double, c_double, c_double]
libGrid.StructuredMaterial2D_new.restype = c_void_p
libGrid.StructuredMaterial2D_delete.argtypes = [c_void_p]
libGrid.StructuredMaterial2D_delete.restype = None
libGrid.StructuredMaterial2D_add_primitive.argtypes = [c_void_p, c_void_p]
libGrid.StructuredMaterial2D_add_primitive.restype = None

####################################################################################
# MaterialPrimitives configuration
####################################################################################
libGrid.MaterialPrimitive_set_layer.argtypes = [c_void_p, c_int]
libGrid.MaterialPrimitive_set_layer.restype = None
libGrid.MaterialPrimitive_get_layer.argtypes = [c_void_p]
libGrid.MaterialPrimitive_get_layer.restype = c_int
libGrid.MaterialPrimitive_contains_point.argtypes = [c_void_p, c_double, c_double]
libGrid.MaterialPrimitive_contains_point.restype = c_bool
libGrid.MaterialPrimitive_get_material_real.argtypes = [c_void_p, c_double, c_double]
libGrid.MaterialPrimitive_get_material_real.restype = c_double
libGrid.MaterialPrimitive_get_material_imag.argtypes = [c_void_p, c_double, c_double]
libGrid.MaterialPrimitive_get_material_imag.restype = c_double


####################################################################################
# Circle configuration
####################################################################################
libGrid.Circle_new.argtypes = [c_double, c_double, c_double]
libGrid.Circle_new.restype = c_void_p
libGrid.Circle_delete.argtypes = [c_void_p]
libGrid.Circle_delete.restype = None
libGrid.Circle_set_material.argtypes = [c_void_p, c_double, c_double]
libGrid.Circle_set_material.restype = None
libGrid.Circle_set_position.argtypes = [c_void_p, c_double, c_double]
libGrid.Circle_set_position.restype = None
libGrid.Circle_set_radius.argtypes = [c_void_p, c_double]
libGrid.Circle_set_radius.restype = None
libGrid.Circle_get_x0.argtypes = [c_void_p]
libGrid.Circle_get_x0.restype = c_double
libGrid.Circle_get_y0.argtypes = [c_void_p]
libGrid.Circle_get_y0.restype = c_double
libGrid.Circle_get_r.argtypes = [c_void_p]
libGrid.Circle_get_r.restype = c_double

####################################################################################
# Rectangle configuration
####################################################################################
libGrid.Rectangle_new.argtypes = [c_double, c_double, c_double, c_double]
libGrid.Rectangle_new.restype = c_void_p
libGrid.Rectangle_delete.argtypes = [c_void_p]
libGrid.Rectangle_delete.restype = None
libGrid.Rectangle_set_material.argtypes = [c_void_p, c_double, c_double]
libGrid.Rectangle_set_material.restype = None
libGrid.Rectangle_set_position.argtypes = [c_void_p, c_double, c_double]
libGrid.Rectangle_set_position.restype = None
libGrid.Rectangle_set_width.argtypes = [c_void_p, c_double]
libGrid.Rectangle_set_width.restype = None
libGrid.Rectangle_set_height.argtypes = [c_void_p, c_double]
libGrid.Rectangle_set_height.restype = None

####################################################################################
# Polygon configuration
####################################################################################
libGrid.Polygon_new.argtypes = []
libGrid.Polygon_new.restype = c_void_p
libGrid.Polygon_delete.argtypes = [c_void_p]
libGrid.Polygon_delete.restype = None
libGrid.Polygon_set_material.argtypes = [c_void_p, c_double, c_double]
libGrid.Polygon_set_material.restype = None
libGrid.Polygon_add_point.argtypes = [c_void_p, c_double, c_double]
libGrid.Polygon_add_point.restype = None
libGrid.Polygon_add_points.argtypes = [c_void_p, c_double_p, c_double_p, c_int]
libGrid.Polygon_add_points.restype = None
libGrid.Polygon_set_points.argtypes = [c_void_p, c_double_p, c_double_p, c_int]
libGrid.Polygon_set_points.restype = None
libGrid.Polygon_set_point.argtypes = [c_void_p, c_double, c_double, c_int]
libGrid.Polygon_set_point.restype = None

####################################################################################
# ConstantMaterial2D configuration
####################################################################################
libGrid.ConstantMaterial2D_new.argtypes = [c_double, c_double]
libGrid.ConstantMaterial2D_new.restype = c_void_p
libGrid.ConstantMaterial2D_set_material.argtypes = [c_void_p, c_double, c_double]
libGrid.ConstantMaterial2D_set_material.restype = None
libGrid.ConstantMaterial2D_get_material_real.argtypes = [c_void_p]
libGrid.ConstantMaterial2D_get_material_real.restype = None
libGrid.ConstantMaterial2D_get_material_imag.argtypes = [c_void_p]
libGrid.ConstantMaterial2D_get_material_imag.restype = None

#####################################################################################
# Material3D configuration 
#####################################################################################
libGrid.Material3D_get_value.argtypes = [c_void_p, c_complex_1D_p, c_double,
                                         c_double, c_double]
libGrid.Material3D_get_value.restype = None
libGrid.Material3D_get_values.argtypes = [c_void_p, c_complex_1D_p, c_int, c_int, c_int, c_int,
                                          c_int, c_int, c_double, c_double,
                                          c_double]
libGrid.Material3D_get_values.restype = None

####################################################################################
# ConstantMaterial3D configuration
####################################################################################
libGrid.ConstantMaterial3D_new.argtypes = [c_double, c_double]
libGrid.ConstantMaterial3D_new.restype = c_void_p
libGrid.ConstantMaterial3D_set_material.argtypes = [c_void_p, c_double, c_double]
libGrid.ConstantMaterial3D_set_material.restype = None
libGrid.ConstantMaterial3D_get_material_real.argtypes = [c_void_p]
libGrid.ConstantMaterial3D_get_material_real.restype = None
libGrid.ConstantMaterial3D_get_material_imag.argtypes = [c_void_p]
libGrid.ConstantMaterial3D_get_material_imag.restype = None

####################################################################################
# StructuredMaterial3D configuration
####################################################################################
libGrid.StructuredMaterial3D_new.argtypes = [c_double, c_double, c_double,
                                             c_double, c_double, c_double]
libGrid.StructuredMaterial3D_new.restype = c_void_p
libGrid.StructuredMaterial3D_delete.argtypes = [c_void_p]
libGrid.StructuredMaterial3D_delete.restype = None
libGrid.StructuredMaterial3D_add_primitive.argtypes = [c_void_p, c_void_p,
                                                       c_double, c_double]
libGrid.StructuredMaterial3D_add_primitive.restype = None

####################################################################################
# Misc
####################################################################################
libGrid.row_wise_A_update.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, \
                                      c_int, c_int, c_int, c_int, c_complex_1D_p]
libGrid.row_wise_A_update.restype = None


