"""Provides a ctypes interface between C++ FDTD library and python."""

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

so_path = ''.join([dir_path, '/FDTD.so'])
libFDTD = cdll.LoadLibrary(so_path)

# useful defs
c_complex_p = ndpointer(np.complex128, ndim=1, flags='C')
c_double_p = ndpointer(np.double, ndim=1, flags='C')

#######################################################
# ctypes interface definition
#######################################################
libFDTD.FDTD_new.argtypes = []
libFDTD.FDTD_new.restype = c_void_p

libFDTD.FDTD_set_wavelength.argtypes = [c_void_p, c_double]
libFDTD.FDTD_set_wavelength.restype = None

libFDTD.FDTD_set_physical_dims.argtypes = [c_void_p,
                                           c_double, c_double, c_double,
                                           c_double, c_double, c_double]
libFDTD.FDTD_set_physical_dims.restype = None

libFDTD.FDTD_set_grid_dims.argtypes = [c_void_p, c_int, c_int, c_int]
libFDTD.FDTD_set_grid_dims.restype = None

libFDTD.FDTD_set_local_grid.argtypes = [c_void_p,
                                        c_int, c_int, c_int,
                                        c_int, c_int, c_int]
libFDTD.FDTD_set_local_grid.restype = None

libFDTD.FDTD_set_dt.argtypes = [c_void_p, c_double]
libFDTD.FDTD_set_dt.restype = None

libFDTD.FDTD_set_complex_eps.argtypes = [c_void_p, c_bool]
libFDTD.FDTD_set_complex_eps.restype = None

libFDTD.FDTD_set_field_arrays.argtypes = [c_void_p,
                                          c_double_p, c_double_p, c_double_p,
                                          c_double_p, c_double_p, c_double_p]

libFDTD.FDTD_set_field_arrays.restype = None

libFDTD.FDTD_set_mat_arrays.argtypes = [c_void_p,
                                          c_complex_p, c_complex_p, c_complex_p,
                                          c_complex_p, c_complex_p, c_complex_p]

libFDTD.FDTD_set_mat_arrays.restype = None

libFDTD.FDTD_update_H.argtypes = [c_void_p, c_int, c_double]
libFDTD.FDTD_update_H.restype = None

libFDTD.FDTD_update_E.argtypes = [c_void_p, c_int, c_double]
libFDTD.FDTD_update_E.restype = None

libFDTD.FDTD_set_pml_widths.argtypes = [c_void_p, c_int, c_int,
                                                  c_int, c_int,
                                                  c_int, c_int]
libFDTD.FDTD_set_pml_widths.restype = None

libFDTD.FDTD_set_pml_properties.argtypes = [c_void_p, c_double, c_double,
                                                      c_double, c_double]
libFDTD.FDTD_set_pml_properties.restype = None

libFDTD.FDTD_build_pml.argtypes = [c_void_p]
libFDTD.FDTD_build_pml.argtypes = None

libFDTD.FDTD_set_t0_arrays.argtypes = [c_void_p,
                                       c_complex_p, c_complex_p, c_complex_p,
                                       c_complex_p, c_complex_p, c_complex_p]
libFDTD.FDTD_set_t0_arrays.restype = None

libFDTD.FDTD_set_t1_arrays.argtypes = [c_void_p,
                                       c_complex_p, c_complex_p, c_complex_p,
                                       c_complex_p, c_complex_p, c_complex_p]
libFDTD.FDTD_set_t1_arrays.restype = None

libFDTD.FDTD_calc_phase_2T.argtypes = [c_double, c_double, c_double, c_double]
libFDTD.FDTD_calc_phase_2T.restype = c_double

libFDTD.FDTD_calc_phase_3T.argtypes = [c_double, c_double, c_double,
                                       c_double, c_double, c_double]
libFDTD.FDTD_calc_phase_3T.restype = c_double

libFDTD.FDTD_calc_amplitude_2T.argtypes = [c_double, c_double, c_double,
                                           c_double, c_double]
libFDTD.FDTD_calc_amplitude_2T.restype = c_double

libFDTD.FDTD_calc_amplitude_3T.argtypes = [c_double, c_double, c_double,
                                           c_double, c_double, c_double,
                                           c_double]
libFDTD.FDTD_calc_amplitude_3T.restype = c_double

libFDTD.FDTD_capture_t0_fields.argtypes = [c_void_p]
libFDTD.FDTD_capture_t0_fields.restype = None

libFDTD.FDTD_capture_t1_fields.argtypes = [c_void_p]
libFDTD.FDTD_capture_t1_fields.restype = None

libFDTD.FDTD_calc_complex_fields_2T.argtypes = [c_void_p, c_double, c_double]
libFDTD.FDTD_calc_complex_fields_2T.restype = None

libFDTD.FDTD_calc_complex_fields_3T.argtypes = [c_void_p, c_double, c_double,
                                                c_double]
libFDTD.FDTD_calc_complex_fields_3T.restype = None

libFDTD.FDTD_add_source.argtypes = [c_void_p,
                                    c_complex_p, c_complex_p, c_complex_p,
                                    c_complex_p, c_complex_p, c_complex_p,
                                    c_int, c_int, c_int,
                                    c_int, c_int, c_int,
                                    c_bool]
libFDTD.FDTD_add_source.restype = None

libFDTD.FDTD_clear_sources.argtypes = [c_void_p]
libFDTD.FDTD_clear_sources.restype = None

libFDTD.FDTD_set_source_properties.argtypes = [c_void_p, c_double, c_double]
libFDTD.FDTD_set_source_properties.restype = None

libFDTD.FDTD_src_func_t.argtypes = [c_void_p, c_int, c_double, c_double]
libFDTD.FDTD_src_func_t.restype = c_double

libFDTD.FDTD_set_bc.argtypes = [c_void_p, c_char_p]
libFDTD.FDTD_set_bc.restype = None

libFDTD.FDTD_copy_to_ghost_comm.argtypes = [c_double_p, c_complex_p,
                                            c_int, c_int, c_int]
libFDTD.FDTD_copy_to_ghost_comm.restype = None

libFDTD.FDTD_copy_from_ghost_comm.argtypes = [c_double_p, c_complex_p,
                                              c_int, c_int, c_int]
libFDTD.FDTD_copy_from_ghost_comm.restype = None
