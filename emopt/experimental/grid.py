import numpy as np
from ..grid import Material2D, Material3D
from math import floor, ceil
import torch

__author__ = "Sean Hooten"
__copyright__ = 'Copyright 2023 Hewlett Packard Enterprise Development LP.'
__license__ = "BSD-3"
__maintainer__ = "Sean Hooten"
__status__ = "development"


class AutoDiffMaterial2D(Material2D):
    def __init__(self, dx, dy, func, v):
        self._dx = dx
        self._dy = dy
        self._func = func
        self._v = v

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, new_v):
        self._v = torch.as_tensor(new_v).squeeze()

    def get_value(self, x, y, bg=None):
        xx = torch.tensor([x*self._dx])
        yy = torch.tensor([y*self._dy])

        return (self._func(self._v, [yy, xx], bg=bg)+0j).item()

    def get_values(self, k1, k2, j1, j2, sx=0, sy=0, arr=None,
                   reshape=True, bg=None):
        Nx = k2-k1
        Ny = j2-j1

        xx = (torch.arange(k1,k2,1) + sx) * self._dx
        yy = (torch.arange(j1,j2,1) + sy) * self._dy

        if(type(arr) == type(None)):
            arr = np.zeros((Ny, Nx), dtype=np.complex128)
        else:
            arr = np.reshape(arr, [Ny, Nx])

        arr[:,:] = (self._func(self._v, [yy, xx], bg=bg)+0j).detach().cpu().numpy()

        if(not reshape):
            arr = np.ravel(arr)

        return arr

class HybridMaterial2D(Material2D):
    # WARNING: WILL RESULT IN UNDEFINED BEHAVIOR FOR MODE CALCULATIONS
    # Cross-section that includes two materials for mode definition should not be used
    # this is an unintended side effect of the way EMopt was originally written.

    def __init__(self, mats: Material2D, matf: AutoDiffMaterial2D, fdomain):
        self._mats = mats
        self._matf= matf
        self._fd = fdomain
        self._func = matf._func

    @property
    def v(self):
        return self._matf.v

    @v.setter
    def v(self, new_v):
        self._matf.v = new_v

    def get_value(self, x, y):
        mat = self._mats.get_value(x, y)
        if contains_index_2D(x,y, self._fd.k1, self._fd.k2,
                                  self._fd.j1, self._fd.j2):
            return self._matf.get_value(x, y, bg=mat)
        else:
            return mat

    def get_values(self, k1, k2, j1, j2, sx=0, sy=0, arr=None,
                   reshape=True):

        fdk1 = self._fd.k1 if self._fd.k1 > k1 else k1
        fdk2 = self._fd.k2 if self._fd.k2 < k2 else k2
        fdj1 = self._fd.j1 if self._fd.j1 > j1 else j1
        fdj2 = self._fd.j2 if self._fd.j2 < j2 else j2

        #print(k1, k2, j1, j2, i1, i2)
        #print(self._fd.k1, self._fd.k2, self._fd.j1, self._fd.j2, self._fd.i1, self._fd.i2)
        #print(fdk1, fdk2, fdj1, fdj2, fdi1, fdi2)

        #arr = self._mats.get_values(k1, k2, j1, j2,
        #                            sx=sx, sy=sy, arr=arr, reshape=reshape)
        arr = self._mats.get_values(k1, k2, j1, j2,
                                    sx=sx, sy=sy, arr=arr)

        if fdk1 <= fdk2 and fdj1 <= fdj2:
            arrf = self._matf.get_values(fdk1, fdk2, fdj1, fdj2,
                                        sx=sx, sy=sy, arr=None, reshape=reshape, bg=arr)

            indk1 = fdk1 - k1
            indk2 = fdk2 - k2
            indj1 = fdj1 - j1
            indj2 = fdj2 - j2

            indk2 = indk2 if indk2 < 0 else None
            indj2 = indj2 if indj2 < 0 else None

            arr[indj1:indj2, indk1:indk2] = arrf

        return arr

class AutoDiffMaterial3D(Material3D):
    def __init__(self, dx, dy, dz, func, v):
        self._dx = dx
        self._dy = dy
        self._dz = dz
        self._func = func
        self._v = v

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, new_v):
        self._v = torch.as_tensor(new_v).squeeze()

    def get_value(self, x, y, z, bg=None):
        xx = torch.tensor([x*self._dx])
        yy = torch.tensor([y*self._dy])
        zz = torch.tensor([z*self._dz])

        return (self._func(self._v, [zz, yy, xx], bg=bg)+0j).item()

    def get_values(self, k1, k2, j1, j2, i1, i2, sx=0, sy=0, sz=0, arr=None,
                   reshape=True, bg=None):
        Nx = k2-k1
        Ny = j2-j1
        Nz = i2-i1

        xx = (torch.arange(k1,k2,1) + sx) * self._dx
        yy = (torch.arange(j1,j2,1) + sy) * self._dy
        zz = (torch.arange(i1,i2,1) + sz) * self._dz

        if(type(arr) == type(None)):
            arr = np.zeros((Nz, Ny, Nx), dtype=np.complex128)
        else:
            arr = np.reshape(arr, [Nz, Ny, Nx])

        arr[:,:,:] = (self._func(self._v, [zz, yy, xx], bg=bg)+0j).detach().cpu().numpy()

        if(not reshape):
            arr = np.ravel(arr)

        return arr

class HybridMaterial3D(Material3D):
    # WARNING: WILL RESULT IN UNDEFINED BEHAVIOR FOR MODE CALCULATIONS
    # Cross-section that includes two materials for mode definition should not be used
    # this is an unintended side effect of the way EMopt was originally written.

    def __init__(self, mats: Material3D, matf: AutoDiffMaterial3D, fdomain):
        self._mats = mats
        self._matf= matf
        self._fd = fdomain
        self._func = matf._func

    @property
    def v(self):
        return self._matf.v

    @v.setter
    def v(self, new_v):
        self._matf.v = new_v

    def get_value(self, x, y, z):
        mat = self._mats.get_value(x, y, z)
        #if self._fd.contains_index(x, y, z):
        if contains_index(x,y,z, self._fd.k1, self._fd.k2,
                                 self._fd.j1, self._fd.j2,
                                 self._fd.i1, self._fd.i2):
            # matf overrides mats
            return self._matf.get_value(x, y, z, bg=mat)
        else:
            return mat

    def get_values(self, k1, k2, j1, j2, i1, i2, sx=0, sy=0, sz=0, arr=None,
                   reshape=True):

        fdk1 = self._fd.k1 if self._fd.k1 > k1 else k1
        fdk2 = self._fd.k2 if self._fd.k2 < k2 else k2
        fdj1 = self._fd.j1 if self._fd.j1 > j1 else j1
        fdj2 = self._fd.j2 if self._fd.j2 < j2 else j2
        fdi1 = self._fd.i1 if self._fd.i1 > i1 else i1
        fdi2 = self._fd.i2 if self._fd.i2 < i2 else i2

        #print(k1, k2, j1, j2, i1, i2)
        #print(self._fd.k1, self._fd.k2, self._fd.j1, self._fd.j2, self._fd.i1, self._fd.i2)
        #print(fdk1, fdk2, fdj1, fdj2, fdi1, fdi2)

        indk1 = fdk1 - k1
        indk2 = fdk2 - k2
        indj1 = fdj1 - j1
        indj2 = fdj2 - j2
        indi1 = fdi1 - i1
        indi2 = fdi2 - i2

        indk2 = indk2 if indk2 < 0 else None
        indj2 = indj2 if indj2 < 0 else None
        indi2 = indi2 if indi2 < 0 else None


        arr = self._mats.get_values(k1, k2, j1, j2, i1, i2,
                                    sx=sx, sy=sy, sz=sz, arr=arr, reshape=reshape)

        bg = arr[indi1:indi2, indj1:indj2, indk1:indk2]

        arrf = self._matf.get_values(fdk1, fdk2, fdj1, fdj2, fdi1, fdi2,
                                    sx=sx, sy=sy, sz=sz, arr=None, reshape=reshape, bg=bg)


        arr[indi1:indi2, indj1:indj2, indk1:indk2] = arrf

        return arr

def contains_index_2D(k, j, k1, k2, j1, j2):
    return (k+1 > k1 and k+1 <= k2) and \
           (j >= j1 and j < j2)

def contains_index(k, j, i, k1, k2, j1, j2, i1, i2):
    return (k >= k1 and k < k2) and \
           (j >= j1 and j < j2) and \
           (i+1 > i1 and i+1 <= i2)
           #(i+1 >= i1 and i+1 < i2)
