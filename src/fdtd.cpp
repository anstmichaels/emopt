#include "fdtd.hpp"
#include <math.h>
#include <algorithm>

fdtd::FDTD::FDTD() 
{
    // make sure all of our PML arrays start NULL
    _pml_Exy0 = NULL; _pml_Exy1 = NULL; _pml_Exz0 = NULL; _pml_Exz1 = NULL;
    _pml_Eyx0 = NULL; _pml_Eyx1 = NULL; _pml_Eyz0 = NULL; _pml_Eyz1 = NULL;
    _pml_Ezx0 = NULL; _pml_Ezx1 = NULL; _pml_Ezy0 = NULL; _pml_Ezy1 = NULL;
    _pml_Hxy0 = NULL; _pml_Hxy1 = NULL; _pml_Hxz0 = NULL; _pml_Hxz1 = NULL;
    _pml_Hyx0 = NULL; _pml_Hyx1 = NULL; _pml_Hyz0 = NULL; _pml_Hyz1 = NULL;
    _pml_Hzx0 = NULL; _pml_Hzx1 = NULL; _pml_Hzy0 = NULL; _pml_Hzy1 = NULL;   
}

fdtd::FDTD::~FDTD()
{
    // Clean up PML arrays
    delete[] _pml_Exy0; delete[] _pml_Exy1; delete[] _pml_Exz0; delete[] _pml_Exz1;
    delete[] _pml_Eyx0; delete[] _pml_Eyx1; delete[] _pml_Eyz0; delete[] _pml_Eyz1;
    delete[] _pml_Ezx0; delete[] _pml_Ezx1; delete[] _pml_Ezy0; delete[] _pml_Ezy1;
    delete[] _pml_Hxy0; delete[] _pml_Hxy1; delete[] _pml_Hxz0; delete[] _pml_Hxz1;
    delete[] _pml_Hyx0; delete[] _pml_Hyx1; delete[] _pml_Hyz0; delete[] _pml_Hyz1;
    delete[] _pml_Hzx0; delete[] _pml_Hzx1; delete[] _pml_Hzy0; delete[] _pml_Hzy1;
}

void fdtd::FDTD::set_physical_dims(double X, double Y, double Z,
                                         double dx, double dy, double dz)
{
    _X = X; _Y = Y; _Z = Z;
    _dx = dx; _dy = dy; _dz = dz;
}

void fdtd::FDTD::set_grid_dims(int Nx, int Ny, int Nz)
{
    _Nx = Nx;
    _Ny = Ny;
    _Nz = Nz;
}


void fdtd::FDTD::set_local_grid(int k0, int j0, int i0, int K, int J, int I)
{
    _i0 = i0; _j0 = j0; _k0 = k0;
    _I = I; _J = J; _K = K;

}


void fdtd::FDTD::set_wavelength(double wavelength)
{
    _wavelength = wavelength;
    _R = _wavelength/(2*M_PI);
}


void fdtd::FDTD::set_dt(double dt)
{
    _dt = dt;
}

void fdtd::FDTD::set_field_arrays(complex128 *Ex, complex128 *Ey, complex128 *Ez,
                                  complex128 *Hx, complex128 *Hy, complex128 *Hz)
{
    _Ex = Ex; _Ey = Ey; _Ez = Ez;
    _Hx = Hx; _Hy = Hy; _Hz = Hz;
}

void fdtd::FDTD::set_mat_arrays(complex128 *eps_x, complex128 *eps_y, complex128 *eps_z,
                                complex128 *mu_x, complex128 *mu_y, complex128 *mu_z)
{
    _eps_x = eps_x; _eps_y = eps_y; _eps_z = eps_z;
    _mu_x = mu_x; _mu_y = mu_y; _mu_z = mu_z;
}

void fdtd::FDTD::update_H(int n, double t)
{
    double odx = _R/_dx,
           ody = _R/_dy,
           odz = _R/_dz,
           pml_factor,
           b, C, pml_dist,
           sigma, kappa, alpha,
           src_t;

    int pml_xmin = _w_pml_x0, pml_xmax = _Nx-_w_pml_x1,
        pml_ymin = _w_pml_y0, pml_ymax = _Ny-_w_pml_y1,
        pml_zmin = _w_pml_z0, pml_zmax = _Nz-_w_pml_z1;

    int ind_ijk, ind_ip1jk, ind_ijp1k, ind_ijkp1, ind_global,
        ind_pml, ind_src, i0s, j0s, k0s, Is, Js, Ks;

    complex128 dExdy, dExdz, dEydx, dEydz, dEzdx, dEzdy;
    complex128 *Mx, *My, *Mz;

    // Setup the fields on the simulation boundary based on the boundary conditions
    if(_bc[0] != 'P' && _k0 + _K == _Nx){
        for(int i = 0; i < _I; i++) {
            for(int j = 0; j < _J; j++) {
                ind_ijk = (i+1)*(_J+2)*(_K+2) + (j+1)*(_K+2) + _K + 1;

                _Ey[ind_ijk] = 0.0;
                _Ez[ind_ijk] = 0.0;
            }
        }
    }

    if(_bc[1] != 'P' && _j0 + _J == _Ny){
        for(int i = 0; i < _I; i++) {
            for(int k = 0; k < _K; k++) {
                ind_ijk = (i+1)*(_J+2)*(_K+2) + (_J+1)*(_K+2) + k + 1;

                _Ex[ind_ijk] = 0.0;
                _Ez[ind_ijk] = 0.0;
            }
        }
    }

    if(_bc[2] != 'P' && _i0 + _I == _Nz){
        for(int j = 0; j < _J; j++) {
            for(int k = 0; k < _K; k++) {
                ind_ijk = (_I+1)*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;

                _Ex[ind_ijk] = 0.0;
                _Ey[ind_ijk] = 0.0;
            }
        }
    }

    for(int i = 0; i < _I; i++) {
        for(int j = 0; j < _J; j++) {
            for(int k = 0; k < _K; k++) {
                ind_ijk = (i+1)*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
                ind_ijp1k = (i+1)*(_J+2)*(_K+2) + (j+2)*(_K+2) + k + 1;
                ind_ip1jk = (i+2)*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
                ind_ijkp1 = (i+1)*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 2;

                ind_global = i*_J*_K + j*_K + k;
        
                // Update Hx
                dEzdy = ody*(_Ez[ind_ijp1k] - _Ez[ind_ijk]);
                dEydz = odz*(_Ey[ind_ip1jk] - _Ey[ind_ijk]);

                _Hx[ind_ijk] = _Hx[ind_ijk] +  _dt/_mu_x[ind_global] * 
                               (dEydz - dEzdy);

                // update Hy
                dExdz = odz*(_Ex[ind_ip1jk] - _Ex[ind_ijk]);
                dEzdx = odx * (_Ez[ind_ijkp1] - _Ez[ind_ijk]);

                _Hy[ind_ijk] = _Hy[ind_ijk] + _dt / _mu_y[ind_global] *
                               (dEzdx - dExdz);

                // update Hz
                dEydx = odx*(_Ey[ind_ijkp1] - _Ey[ind_ijk]);
                dExdy = ody * (_Ex[ind_ijp1k] - _Ex[ind_ijk]);
                
                _Hz[ind_ijk] = _Hz[ind_ijk] + _dt / _mu_z[ind_global] *
                               (dExdy - dEydx);

                // Do PML updates
                if(k + _k0 < pml_xmin) {
                    pml_dist = double(pml_xmin - k - _k0 - 0.5)/_w_pml_x0; // distance from pml edge
                    ind_pml = i*_J*(pml_xmin - _k0) +j*(pml_xmin - _k0) + k;

                    pml_factor = pml_ramp(pml_dist);

                    // compute coefficients
                    sigma = _sigma * pml_factor;
                    kappa = (_kappa-1) * pml_factor+1;
                    alpha = _alpha * (1-pml_factor);
                    b = std::exp(-_dt*(sigma/kappa + alpha));
                    C = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

                    _pml_Eyx0[ind_pml] = C * dEydx + b*_pml_Eyx0[ind_pml];
                    _pml_Ezx0[ind_pml] = C * dEzdx + b*_pml_Ezx0[ind_pml];

                    _Hz[ind_ijk] = _Hz[ind_ijk] - _dt / _mu_z[ind_global] * (_pml_Eyx0[ind_pml]-dEydx+dEydx/kappa);
                    _Hy[ind_ijk] = _Hy[ind_ijk] + _dt / _mu_y[ind_global] * (_pml_Ezx0[ind_pml]-dEzdx+dEzdx/kappa);
                
                }
                else if(k + _k0 > pml_xmax) {
                    pml_dist = double(k + _k0 + 1 + 0.5 - pml_xmax)/_w_pml_x1; // distance from pml edge
                    ind_pml = i*_J*(_k0 + _K - pml_xmax) + j*(_k0 + _K - pml_xmax) + k + _k0 - pml_xmax;

                    pml_factor = pml_ramp(pml_dist);

                    // compute coefficients
                    sigma = _sigma * pml_factor;
                    kappa = (_kappa-1) * pml_factor+1;
                    alpha = _alpha * (1-pml_factor);
                    b = std::exp(-_dt*(sigma/kappa + alpha));
                    C = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

                    _pml_Eyx1[ind_pml] = C * dEydx + b*_pml_Eyx1[ind_pml];
                    _pml_Ezx1[ind_pml] = C * dEzdx + b*_pml_Ezx1[ind_pml];

                    _Hz[ind_ijk] = _Hz[ind_ijk] - _dt / _mu_z[ind_global] * (_pml_Eyx1[ind_pml]-dEydx+dEydx/kappa);
                    _Hy[ind_ijk] = _Hy[ind_ijk] + _dt / _mu_y[ind_global] * (_pml_Ezx1[ind_pml]-dEzdx+dEzdx/kappa);
                }

                if(j + _j0 < pml_ymin) {
                    pml_dist = double(pml_ymin - j - _j0 - 0.5)/_w_pml_y0; // distance from pml edge
                    ind_pml = i*(pml_ymin - _j0)*_K +j*_K + k;

                    pml_factor = pml_ramp(pml_dist);

                    // compute coefficients
                    sigma = _sigma * pml_factor;
                    kappa = (_kappa-1) * pml_factor+1;
                    alpha = _alpha * (1-pml_factor);
                    b = std::exp(-_dt*(sigma/kappa + alpha));
                    C = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

                    _pml_Exy0[ind_pml] = C * dExdy + b*_pml_Exy0[ind_pml];
                    _pml_Ezy0[ind_pml] = C * dEzdy + b*_pml_Ezy0[ind_pml];

                    _Hz[ind_ijk] = _Hz[ind_ijk] + _dt / _mu_z[ind_global] * (_pml_Exy0[ind_pml]-dExdy+dExdy/kappa);
                    _Hx[ind_ijk] = _Hx[ind_ijk] - _dt / _mu_x[ind_global] * (_pml_Ezy0[ind_pml]-dEzdy+dEzdy/kappa);
                }
                else if(j + _j0 > pml_ymax) {
                    pml_dist = double(j + _j0 + 1 + 0.5 - pml_ymax)/_w_pml_y1; // distance from pml edge
                    ind_pml = i*(_j0 + _J - pml_ymax)*_K +(_j0 + j - pml_ymax)*_K + k;

                    pml_factor = pml_ramp(pml_dist);

                    // compute coefficients
                    sigma = _sigma * pml_factor;
                    kappa = (_kappa-1) * pml_factor+1;
                    alpha = _alpha * (1-pml_factor);
                    b = std::exp(-_dt*(sigma/kappa + alpha));
                    C = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

                    _pml_Exy1[ind_pml] = C * dExdy + b*_pml_Exy1[ind_pml];
                    _pml_Ezy1[ind_pml] = C * dEzdy + b*_pml_Ezy1[ind_pml];

                    _Hz[ind_ijk] = _Hz[ind_ijk] + _dt / _mu_z[ind_global] * (_pml_Exy1[ind_pml]-dExdy+dExdy/kappa);
                    _Hx[ind_ijk] = _Hx[ind_ijk] - _dt / _mu_x[ind_global] * (_pml_Ezy1[ind_pml]-dEzdy+dEzdy/kappa);
                }

                if(i + _i0 < pml_zmin) {
                    pml_dist = double(pml_zmin - i - _i0 - 0.5)/_w_pml_z0; // distance from pml edge
                    ind_pml = i*_J*_K +j*_K + k;

                    pml_factor = pml_ramp(pml_dist);

                    // compute coefficients
                    sigma = _sigma * pml_factor;
                    kappa = (_kappa-1) * pml_factor+1;
                    alpha = _alpha * (1-pml_factor);
                    b = std::exp(-_dt*(sigma/kappa + alpha));
                    C = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

                    _pml_Exz0[ind_pml] = C * dExdz + b*_pml_Exz0[ind_pml];
                    _pml_Eyz0[ind_pml] = C * dEydz + b*_pml_Eyz0[ind_pml];

                    _Hx[ind_ijk] = _Hx[ind_ijk] + _dt / _mu_x[ind_global] * (_pml_Eyz0[ind_pml]-dEydz+dEydz/kappa);
                    _Hy[ind_ijk] = _Hy[ind_ijk] - _dt / _mu_y[ind_global] * (_pml_Exz0[ind_pml]-dExdz+dExdz/kappa);
                }
                else if(i + _i0 > pml_zmax) {
                    pml_dist = double(i + _i0 + 1 + 0.5 - pml_zmax)/_w_pml_z1; // distance from pml edge
                    ind_pml = (_i0 + i - pml_zmax)*_J*_K +j*_K + k;

                    pml_factor = pml_ramp(pml_dist);

                    // compute coefficients
                    sigma = _sigma * pml_factor;
                    kappa = (_kappa-1) * pml_factor+1;
                    alpha = _alpha * (1-pml_factor);
                    b = std::exp(-_dt*(sigma/kappa + alpha));
                    C = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

                    _pml_Exz1[ind_pml] = C * dExdz + b*_pml_Exz1[ind_pml];
                    _pml_Eyz1[ind_pml] = C * dEydz + b*_pml_Eyz1[ind_pml];

                    _Hx[ind_ijk] = _Hx[ind_ijk] + _dt / _mu_x[ind_global] * (_pml_Eyz1[ind_pml]-dEydz+dEydz/kappa);
                    _Hy[ind_ijk] = _Hy[ind_ijk] - _dt / _mu_y[ind_global] * (_pml_Exz1[ind_pml]-dExdz+dExdz/kappa);
                }

            }
        }
    }

    // Update sources
    for(auto const& src : _sources) {
        i0s = src.i0; Is = src.I;
        j0s = src.j0; Js = src.J;
        k0s = src.k0; Ks = src.K;

        // update Mx
        Mx = src.Mx;

        for(int i = 0; i < Is; i++) {
            for(int j = 0; j < Js; j++) {
                for(int k = 0; k < Ks; k++) {
                    ind_ijk = (i+i0s+1)*(_J+2)*(_K+2) + (j+j0s+1)*(_K+2) + k + k0s + 1;
                    ind_global = (i+i0s)*_J*_K + (j+j0s)*_K + k+k0s;
                    ind_src = i*Js*Ks + j*Ks + k;
                    
                    src_t = src_func_t(n, t, Mx[ind_src].imag);
                    _Hx[ind_ijk] = _Hx[ind_ijk] + src_t * Mx[ind_src].real * _dt / _mu_x[ind_global];                   
                }
            }
        }

        // update My
        My = src.My;

        for(int i = 0; i < Is; i++) {
            for(int j = 0; j < Js; j++) {
                for(int k = 0; k < Ks; k++) {
                    ind_ijk = (i+i0s+1)*(_J+2)*(_K+2) + (j+j0s+1)*(_K+2) + k + k0s + 1;
                    ind_global = (i+i0s)*_J*_K + (j+j0s)*_K + k+k0s;
                    ind_src = i*Js*Ks + j*Ks + k;
                    
                    src_t = src_func_t(n, t, My[ind_src].imag);
                    _Hy[ind_ijk] = _Hy[ind_ijk] + src_t * My[ind_src].real * _dt / _mu_y[ind_global];                   
                }
            }
        }

        // update Mz
        Mz = src.Mz;

        for(int i = 0; i < Is; i++) {
            for(int j = 0; j < Js; j++) {
                for(int k = 0; k < Ks; k++) {
                    ind_ijk = (i+i0s+1)*(_J+2)*(_K+2) + (j+j0s+1)*(_K+2) + k + k0s + 1;
                    ind_global = (i+i0s)*_J*_K + (j+j0s)*_K + k+k0s;
                    ind_src = i*Js*Ks + j*Ks + k;
                    
                    src_t = src_func_t(n, t, Mz[ind_src].imag);
                    _Hz[ind_ijk] = _Hz[ind_ijk] + src_t * Mz[ind_src].real * _dt / _mu_z[ind_global];                   
                }
            }
        }

    }
}

void fdtd::FDTD::update_E(int n, double t)
{
    double odx = _R/_dx,
           ody = _R/_dy,
           odz = _R/_dz,
           pml_factor,
           b, C, pml_dist,
           sigma, kappa, alpha,
           src_t;

    int pml_xmin = _w_pml_x0, pml_xmax = _Nx-_w_pml_x1,
        pml_ymin = _w_pml_y0, pml_ymax = _Ny-_w_pml_y1,
        pml_zmin = _w_pml_z0, pml_zmax = _Nz-_w_pml_z1;

    int ind_ijk, ind_im1jk, ind_ijm1k, ind_ijkm1, ind_global,
        ind_pml, ind_src, i0s, j0s, k0s, Is, Js, Ks;

    int ind_ijkp1, ind_ijp1k, ind_ip1jk; // used for setting boundary values

    complex128 dHxdy, dHxdz, dHydx, dHydz, dHzdx, dHzdy;
    complex128 *Jx, *Jy, *Jz;

    // Setup the fields on the simulation boundary based on the boundary conditions
    if(_k0 == 0){
        if(_bc[0] == '0') {
            for(int i = 0; i < _I; i++) {
                for(int j = 0; j < _J; j++) {
                    ind_ijk = (i+1)*(_J+2)*(_K+2) + (j+1)*(_K+2);

                    _Hy[ind_ijk] = 0.0;
                    _Hz[ind_ijk] = 0.0;
                }
            }
        }
        else if(_bc[0] == 'E') {
            for(int i = 0; i < _I; i++) {
                for(int j = 0; j < _J; j++) {
                    ind_ijk = (i+1)*(_J+2)*(_K+2) + (j+1)*(_K+2);
                    ind_ijkp1 = (i+1)*(_J+2)*(_K+2) + (j+1)*(_K+2)+1;

                    _Hy[ind_ijk] = -1*_Hy[ind_ijkp1];
                    _Hz[ind_ijk] = -1*_Hz[ind_ijkp1];
                }
            }
        }
        else if(_bc[0] == 'H') {
            for(int i = 0; i < _I; i++) {
                for(int j = 0; j < _J; j++) {
                    ind_ijk = (i+1)*(_J+2)*(_K+2) + (j+1)*(_K+2);
                    ind_ijkp1 = (i+1)*(_J+2)*(_K+2) + (j+1)*(_K+2)+1;

                    _Hy[ind_ijk] = _Hy[ind_ijkp1];
                    _Hz[ind_ijk] = _Hz[ind_ijkp1];
                }
            }
        }
    }

    if(_j0 == 0){
        if(_bc[1] == '0') {
            for(int i = 0; i < _I; i++) {
                for(int k = 0; k < _K; k++) {
                    ind_ijk = (i+1)*(_J+2)*(_K+2) + 0*(_K+2) + k + 1;

                    _Hx[ind_ijk] = 0.0;
                    _Hz[ind_ijk] = 0.0;
                }
            }
        }
        else if(_bc[1] == 'E') {
            for(int i = 0; i < _I; i++) {
                for(int k = 0; k < _K; k++) {
                    ind_ijk = (i+1)*(_J+2)*(_K+2) + 0*(_K+2) + k + 1;
                    ind_ijp1k = (i+1)*(_J+2)*(_K+2) + 1*(_K+2) + k + 1;

                    _Hx[ind_ijk] = -1*_Hx[ind_ijp1k];
                    _Hz[ind_ijk] = -1*_Hz[ind_ijp1k];
                }
            }
        }
        else if(_bc[1] == 'H') {
            for(int i = 0; i < _I; i++) {
                for(int k = 0; k < _K; k++) {
                    ind_ijk = (i+1)*(_J+2)*(_K+2) + 0*(_K+2) + k + 1;
                    ind_ijp1k = (i+1)*(_J+2)*(_K+2) + 1*(_K+2) + k + 1;

                    _Hx[ind_ijk] = _Hx[ind_ijp1k];
                    _Hz[ind_ijk] = _Hz[ind_ijp1k];
                }
            }
        }
    }

    if(_i0 == 0){
        if(_bc[2] == '0') {
            for(int j = 0; j < _J; j++) {
                for(int k = 0; k < _K; k++) {
                    ind_ijk = 0*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;

                    _Hx[ind_ijk] = 0.0;
                    _Hy[ind_ijk] = 0.0;
                }
            }
        }
        else if(_bc[2] == 'E') {
            for(int j = 0; j < _J; j++) {
                for(int k = 0; k < _K; k++) {
                    ind_ijk = 0*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
                    ind_ip1jk = 1*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;

                    _Hx[ind_ijk] = -1*_Hx[ind_ip1jk];
                    _Hy[ind_ijk] = -1*_Hy[ind_ip1jk];
                }
            }
        }
        else if(_bc[2] == 'H') {
            for(int j = 0; j < _J; j++) {
                for(int k = 0; k < _K; k++) {
                    ind_ijk = 0*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
                    ind_ip1jk = 1*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;

                    _Hx[ind_ijk] = _Hx[ind_ip1jk];
                    _Hy[ind_ijk] = _Hy[ind_ip1jk];
                }
            }
        }
    }

    for(int i = 0; i < _I; i++) {
        for(int j = 0; j < _J; j++) {
            for(int k = 0; k < _K; k++) {
                ind_ijk = (i+1)*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
                ind_ijm1k = (i+1)*(_J+2)*(_K+2) + (j)*(_K+2) + k + 1;
                ind_im1jk = (i)*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
                ind_ijkm1 = (i+1)*(_J+2)*(_K+2) + (j+1)*(_K+2) + k;

                ind_global = i*_J*_K + j*_K + k;
        
                // Update Ex
                dHzdy = ody*(_Hz[ind_ijk] - _Hz[ind_ijm1k]);
                dHydz = odz*(_Hy[ind_ijk] - _Hy[ind_im1jk]);
                
                _Ex[ind_ijk] = _Ex[ind_ijk] +  _dt/_eps_x[ind_global] * 
                               (dHzdy - dHydz);

                // Update Ey
                dHxdz = odz*(_Hx[ind_ijk] - _Hx[ind_im1jk]);
                dHzdx = odx * (_Hz[ind_ijk] - _Hz[ind_ijkm1]);
                
                _Ey[ind_ijk] = _Ey[ind_ijk] + _dt / _eps_y[ind_global] *
                               (dHxdz - dHzdx);
                
                // Update Ez
                dHydx = odx*(_Hy[ind_ijk] - _Hy[ind_ijkm1]);
                dHxdy = ody * (_Hx[ind_ijk] - _Hx[ind_ijm1k]);

                _Ez[ind_ijk] = _Ez[ind_ijk] + _dt / _eps_z[ind_global] *
                               (dHydx - dHxdy);

                
                // Do PML updates
                if(k + _k0 < pml_xmin) {
                    pml_dist = double(pml_xmin - k - _k0)/_w_pml_x0; // distance from pml edge
                    ind_pml = i*_J*(pml_xmin-_k0) +j*(pml_xmin-_k0) + k;
                    
                    pml_factor = pml_ramp(pml_dist);

                    // compute coefficients
                    sigma = _sigma * pml_factor;
                    kappa = (_kappa-1) * pml_factor+1;
                    alpha = _alpha * (1-pml_factor);
                    b = std::exp(-_dt*(sigma/kappa + alpha));
                    C = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

                    _pml_Hyx0[ind_pml] = C * dHydx + b*_pml_Hyx0[ind_pml];
                    _pml_Hzx0[ind_pml] = C * dHzdx + b*_pml_Hzx0[ind_pml];

                    _Ez[ind_ijk] = _Ez[ind_ijk] + _dt / _eps_z[ind_global] * (_pml_Hyx0[ind_pml]-dHydx+dHydx/kappa);
                    _Ey[ind_ijk] = _Ey[ind_ijk] - _dt / _eps_y[ind_global] * (_pml_Hzx0[ind_pml]-dHzdx+dHzdx/kappa);

                }
                else if(k + _k0 > pml_xmax) {
                    pml_dist = double(k + _k0 + 1 - pml_xmax)/_w_pml_x1; // distance from pml edge
                    ind_pml = i*_J*(_k0 + _K - pml_xmax) +j*(_k0 + _K - pml_xmax) + k + _k0 - pml_xmax;
                    
                    pml_factor = pml_ramp(pml_dist);

                    // compute coefficients
                    sigma = _sigma * pml_factor;
                    kappa = (_kappa-1) * pml_factor+1;
                    alpha = _alpha * (1-pml_factor);
                    b = std::exp(-_dt*(sigma/kappa + alpha));
                    C = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

                    _pml_Hyx1[ind_pml] = C * dHydx + b*_pml_Hyx1[ind_pml];
                    _pml_Hzx1[ind_pml] = C * dHzdx + b*_pml_Hzx1[ind_pml];

                    _Ez[ind_ijk] = _Ez[ind_ijk] + _dt / _eps_z[ind_global] * (_pml_Hyx1[ind_pml]-dHydx+dHydx/kappa);
                    _Ey[ind_ijk] = _Ey[ind_ijk] - _dt / _eps_y[ind_global] * (_pml_Hzx1[ind_pml]-dHzdx+dHzdx/kappa);
                }

                if(j + _j0 < pml_ymin) {
                    pml_dist = double(pml_ymin - j - _j0)/_w_pml_y0; // distance from pml edge
                    ind_pml = i*(pml_ymin - _j0)*_K +j*_K + k;

                    pml_factor = pml_ramp(pml_dist);

                    // compute coefficients
                    sigma = _sigma * pml_factor;
                    kappa = (_kappa-1) * pml_factor+1;
                    alpha = _alpha * (1-pml_factor);
                    b = std::exp(-_dt*(sigma/kappa + alpha));
                    C = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

                    _pml_Hxy0[ind_pml] = C * dHxdy + b*_pml_Hxy0[ind_pml];
                    _pml_Hzy0[ind_pml] = C * dHzdy + b*_pml_Hzy0[ind_pml];

                    _Ez[ind_ijk] = _Ez[ind_ijk] - _dt / _eps_z[ind_global] * (_pml_Hxy0[ind_pml]-dHxdy+dHxdy/kappa);
                    _Ex[ind_ijk] = _Ex[ind_ijk] + _dt / _eps_x[ind_global] * (_pml_Hzy0[ind_pml]-dHzdy+dHzdy/kappa);
                }
                else if(j + _j0 > pml_ymax) {
                    pml_dist = double(j + _j0 + 1 - pml_ymax)/_w_pml_y1; // distance from pml edge
                    ind_pml = i*(_j0 + _J - pml_ymax)*_K +(_j0 + j - pml_ymax)*_K + k;

                    pml_factor = pml_ramp(pml_dist);

                    // compute coefficients
                    sigma = _sigma * pml_factor;
                    kappa = (_kappa-1) * pml_factor+1;
                    alpha = _alpha * (1-pml_factor);
                    b = std::exp(-_dt*(sigma/kappa + alpha));
                    C = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);     

                    _pml_Hxy1[ind_pml] = C * dHxdy + b*_pml_Hxy1[ind_pml];
                    _pml_Hzy1[ind_pml] = C * dHzdy + b*_pml_Hzy1[ind_pml];

                    _Ez[ind_ijk] = _Ez[ind_ijk] - _dt / _eps_z[ind_global] * (_pml_Hxy1[ind_pml]-dHxdy+dHxdy/kappa);
                    _Ex[ind_ijk] = _Ex[ind_ijk] + _dt / _eps_x[ind_global] * (_pml_Hzy1[ind_pml]-dHzdy+dHzdy/kappa);
                }

                if(i + _i0 < pml_zmin) {
                    pml_dist = double(pml_zmin - i - _i0)/_w_pml_z0; // distance from pml edge
                    ind_pml = i*_J*_K +j*_K + k;

                    pml_factor = pml_ramp(pml_dist);

                    // compute coefficients
                    sigma = _sigma * pml_factor;
                    kappa = (_kappa-1) * pml_factor+1;
                    alpha = _alpha * (1-pml_factor);
                    b = std::exp(-_dt*(sigma/kappa + alpha));
                    C = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);     

                    _pml_Hxz0[ind_pml] = C * dHxdz + b*_pml_Hxz0[ind_pml];
                    _pml_Hyz0[ind_pml] = C * dHydz + b*_pml_Hyz0[ind_pml];

                    _Ex[ind_ijk] = _Ex[ind_ijk] - _dt / _eps_x[ind_global] * (_pml_Hyz0[ind_pml]-dHydz+dHydz/kappa);
                    _Ey[ind_ijk] = _Ey[ind_ijk] + _dt / _eps_y[ind_global] * (_pml_Hxz0[ind_pml]-dHxdz+dHxdz/kappa);       
                }
                else if(i + _i0 > pml_zmax) {
                    pml_dist = double(i + _i0 + 1 - pml_zmax)/_w_pml_z1; // distance from pml edge
                    ind_pml = (_i0 + i - pml_zmax)*_J*_K +j*_K + k;

                    pml_factor = pml_ramp(pml_dist);

                    // compute coefficients
                    sigma = _sigma * pml_factor;
                    kappa = (_kappa-1) * pml_factor+1;
                    alpha = _alpha * (1-pml_factor);
                    b = std::exp(-_dt*(sigma/kappa + alpha));
                    C = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

                    _pml_Hxz1[ind_pml] = C * dHxdz + b*_pml_Hxz1[ind_pml];
                    _pml_Hyz1[ind_pml] = C * dHydz + b*_pml_Hyz1[ind_pml];

                    _Ex[ind_ijk] = _Ex[ind_ijk] - _dt / _eps_x[ind_global] * (_pml_Hyz1[ind_pml]-dHydz+dHydz/kappa);
                    _Ey[ind_ijk] = _Ey[ind_ijk] + _dt / _eps_y[ind_global] * (_pml_Hxz1[ind_pml]-dHxdz+dHxdz/kappa);
                }
            }
        }
    }

    // Update sources
    for(auto const& src : _sources) {
        i0s = src.i0; Is = src.I;
        j0s = src.j0; Js = src.J;
        k0s = src.k0; Ks = src.K;

        // update Jx
        Jx = src.Jx;

        for(int i = 0; i < Is; i++) {
            for(int j = 0; j < Js; j++) {
                for(int k = 0; k < Ks; k++) {
                    ind_ijk = (i+i0s+1)*(_J+2)*(_K+2) + (j+j0s+1)*(_K+2) + k + k0s + 1;
                    ind_global = (i+i0s)*_J*_K + (j+j0s)*_K + k+k0s;
                    ind_src = i*Js*Ks + j*Ks + k;
                    
                    src_t = src_func_t(n, t, Jx[ind_src].imag);
                    _Ex[ind_ijk] = _Ex[ind_ijk] - src_t * Jx[ind_src].real * _dt / _eps_x[ind_global];                   
                }
            }
        }

        // update Jy
        Jy = src.Jy;

        for(int i = 0; i < Is; i++) {
            for(int j = 0; j < Js; j++) {
                for(int k = 0; k < Ks; k++) {
                    ind_ijk = (i+i0s+1)*(_J+2)*(_K+2) + (j+j0s+1)*(_K+2) + k + k0s + 1;
                    ind_global = (i+i0s)*_J*_K + (j+j0s)*_K + k+k0s;
                    ind_src = i*Js*Ks + j*Ks + k;
                    
                    src_t = src_func_t(n, t, Jy[ind_src].imag);
                    _Ey[ind_ijk] = _Ey[ind_ijk] - src_t * Jy[ind_src].real * _dt / _eps_y[ind_global];                   
                }
            }
        }

        // update Jz
        Jz = src.Jz;

        for(int i = 0; i < Is; i++) {
            for(int j = 0; j < Js; j++) {
                for(int k = 0; k < Ks; k++) {
                    ind_ijk = (i+i0s+1)*(_J+2)*(_K+2) + (j+j0s+1)*(_K+2) + k + k0s + 1;
                    ind_global = (i+i0s)*_J*_K + (j+j0s)*_K + k+k0s;
                    ind_src = i*Js*Ks + j*Ks + k;
                    
                    src_t = src_func_t(n, t, Jz[ind_src].imag);
                    _Ez[ind_ijk] = _Ez[ind_ijk] - src_t * Jz[ind_src].real * _dt / _eps_z[ind_global];                   
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// PML Management
///////////////////////////////////////////////////////////////////////////

double fdtd::FDTD::pml_ramp(double pml_dist)
{
    if(_pow == 1.0) return pml_dist;
    else if(_pow == 2.0) return pow2(pml_dist);
    else if(_pow == 3.0) return pow3(pml_dist);
    else return std::pow(pml_dist, _pow);
}

void fdtd::FDTD::set_pml_widths(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax)
{
    _w_pml_x0 = xmin; _w_pml_x1 = xmax;
    _w_pml_y0 = ymin; _w_pml_y1 = ymax;
    _w_pml_z0 = zmin; _w_pml_z1 = zmax;
}

void fdtd::FDTD::set_pml_properties(double sigma, double alpha, double kappa, double pow)
{
    _sigma = sigma;
    _alpha = alpha;
    _kappa = kappa;
    _pow   = pow;
}

void fdtd::FDTD::build_pml()
{
    int N,
        xmin = _w_pml_x0, xmax = _Nx-_w_pml_x1,
        ymin = _w_pml_y0, ymax = _Ny-_w_pml_y1,
        zmin = _w_pml_z0, zmax = _Nz-_w_pml_z1;

    // touches xmin boudary
    if(_k0 < xmin) {
        N = _I * _J * (xmin - _k0);

        // Clean up old arrays and allocate new ones
        delete [] _pml_Eyx0; _pml_Eyx0 = NULL;
        delete [] _pml_Ezx0; _pml_Ezx0 = NULL;
        _pml_Eyx0 = new complex128[N];
        _pml_Ezx0 = new complex128[N];

        delete [] _pml_Hyx0; _pml_Hyx0 = NULL;
        delete [] _pml_Hzx0; _pml_Hzx0 = NULL;
        _pml_Hyx0 = new complex128[N];
        _pml_Hzx0 = new complex128[N];
    }

    // touches xmax boundary
    if(_k0 +_K > xmax) {
        N = _I * _J * (_k0  + _K - xmax);

        // Clean up old arrays and allocate new ones
        delete [] _pml_Eyx1; _pml_Eyx1 = NULL;
        delete [] _pml_Ezx1; _pml_Ezx1 = NULL;
        _pml_Eyx1 = new complex128[N];
        _pml_Ezx1 = new complex128[N];

        delete [] _pml_Hyx1; _pml_Hyx1 = NULL;
        delete [] _pml_Hzx1; _pml_Hzx1 = NULL;
        _pml_Hyx1 = new complex128[N];
        _pml_Hzx1 = new complex128[N];
    }

    // touches ymin boundary
    if(_j0 < ymin) {
        N = _I * _K * (ymin - _j0);

        delete [] _pml_Exy0; _pml_Exy0 = NULL;
        delete [] _pml_Ezy0; _pml_Ezy0 = NULL;
        _pml_Exy0 = new complex128[N];
        _pml_Ezy0 = new complex128[N];

        delete [] _pml_Hxy0; _pml_Hxy0 = NULL;
        delete [] _pml_Hzy0; _pml_Hzy0 = NULL;
        _pml_Hxy0 = new complex128[N];
        _pml_Hzy0 = new complex128[N];
    }

    // touches ymax boundary
    if(_j0 + _J > ymax) {
        N = _I * _K * (_j0 + _J - ymax);

        delete [] _pml_Exy1; _pml_Exy1 = NULL;
        delete [] _pml_Ezy1; _pml_Ezy1 = NULL;
        _pml_Exy1 = new complex128[N];
        _pml_Ezy1 = new complex128[N];

        delete [] _pml_Hxy1; _pml_Hxy1 = NULL;
        delete [] _pml_Hzy1; _pml_Hzy1 = NULL;
        _pml_Hxy1 = new complex128[N];
        _pml_Hzy1 = new complex128[N];
    }

    // touches zmin boundary
    if(_i0 < zmin) {
        N = _J * _K * (zmin - _i0);

        delete [] _pml_Exz0; _pml_Exz0 = NULL;
        delete [] _pml_Eyz0; _pml_Eyz0 = NULL;
        _pml_Exz0 = new complex128[N];
        _pml_Eyz0 = new complex128[N];

        delete [] _pml_Hxz0; _pml_Hxz0 = NULL;
        delete [] _pml_Hyz0; _pml_Hyz0 = NULL;
        _pml_Hxz0 = new complex128[N];
        _pml_Hyz0 = new complex128[N];
    }

    // touches zmax boundary
    if(_i0 + _I > zmax) {
        N = _J * _K * (_i0 + _I - zmax);

        delete [] _pml_Hxz1; _pml_Hxz1 = NULL;
        delete [] _pml_Hyz1; _pml_Hyz1 = NULL;
        _pml_Exz1 = new complex128[N];
        _pml_Eyz1 = new complex128[N];

        delete [] _pml_Hxz1; _pml_Hxz1 = NULL;
        delete [] _pml_Hyz1; _pml_Hyz1 = NULL;
        _pml_Hxz1 = new complex128[N];
        _pml_Hyz1 = new complex128[N];
    }
}

void fdtd::FDTD::reset_pml()
{
    int N,
        xmin = _w_pml_x0, xmax = _Nx-_w_pml_x1,
        ymin = _w_pml_y0, ymax = _Ny-_w_pml_y1,
        zmin = _w_pml_z0, zmax = _Nz-_w_pml_z1;

    // touches xmin boudary
    if(_k0 < xmin) {
        N = _I * _J * (xmin - _k0);
        std::fill(_pml_Eyx0, _pml_Eyx0 + N, 0);
        std::fill(_pml_Ezx0, _pml_Ezx0 + N, 0);
        std::fill(_pml_Hyx0, _pml_Hyx0 + N, 0);
        std::fill(_pml_Hzx0, _pml_Hzx0 + N, 0);
    }

    // touches xmax boundary
    if(_k0 +_K > xmax) {
        N = _I * _J * (_k0  + _K - xmax);

        std::fill(_pml_Eyx1, _pml_Eyx1 + N, 0);
        std::fill(_pml_Ezx1, _pml_Ezx1 + N, 0);
        std::fill(_pml_Hyx1, _pml_Hyx1 + N, 0);
        std::fill(_pml_Hzx1, _pml_Hzx1 + N, 0);
    }

    // touches ymin boundary
    if(_j0 < ymin) {
        N = _I * _K * (ymin - _j0);

        std::fill(_pml_Exy0, _pml_Exy0 + N, 0);
        std::fill(_pml_Ezy0, _pml_Ezy0 + N, 0);
        std::fill(_pml_Hxy0, _pml_Hxy0 + N, 0);
        std::fill(_pml_Hzy0, _pml_Hzy0 + N, 0);
    }

    // touches ymax boundary
    if(_j0 + _J > ymax) {
        N = _I * _K * (_j0 + _J - ymax);

        std::fill(_pml_Exy1, _pml_Exy1 + N, 0);
        std::fill(_pml_Ezy1, _pml_Ezy1 + N, 0);
        std::fill(_pml_Hxy1, _pml_Hxy1 + N, 0);
        std::fill(_pml_Hzy1, _pml_Hzy1 + N, 0);
    }

    // touches zmin boundary
    if(_i0 < zmin) {
        N = _J * _K * (zmin - _i0);

        std::fill(_pml_Exz0, _pml_Exz0 + N, 0);
        std::fill(_pml_Eyz0, _pml_Eyz0 + N, 0);
        std::fill(_pml_Hxz0, _pml_Hxz0 + N, 0);
        std::fill(_pml_Hyz0, _pml_Hyz0 + N, 0);
    }

    // touches zmax boundary
    if(_i0 + _I > zmax) {
        N = _J * _K * (_i0 + _I - zmax);

        std::fill(_pml_Exz1, _pml_Exz1 + N, 0);
        std::fill(_pml_Eyz1, _pml_Eyz1 + N, 0);
        std::fill(_pml_Hxz1, _pml_Hxz1 + N, 0);
        std::fill(_pml_Hyz1, _pml_Hyz1 + N, 0);
    }
}

///////////////////////////////////////////////////////////////////////////
// Amp/Phase Calculation management Management
///////////////////////////////////////////////////////////////////////////
void fdtd::FDTD::set_aux_arrays(complex128 *Ex_aux, complex128 *Ey_aux, complex128 *Ez_aux,
                                complex128 *Hx_aux, complex128 *Hy_aux, complex128 *Hz_aux)
{
    _Ex_aux = Ex_aux; _Ey_aux = Ey_aux; _Ez_aux = Ez_aux;
    _Hx_aux = Hx_aux; _Hy_aux = Hy_aux; _Hz_aux = Hz_aux;
}

void fdtd::FDTD::capture_t0_fields()
{
    int ind_local, ind_global;

    for(int i = 0; i < _I; i++) {
        for(int j = 0; j < _J; j++) {
            for(int k = 0; k < _K; k++) {
                ind_local = (i+1)*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
                ind_global = i*_J*_K + j*_K + k;

                // Copy the fields at the current time to the auxillary arrays
                _Ex_aux[ind_global] = _Ex[ind_local];
                _Ey_aux[ind_global] = _Ey[ind_local];
                _Ez_aux[ind_global] = _Ez[ind_local];

                _Hx_aux[ind_global] = _Hx[ind_local];
                _Hy_aux[ind_global] = _Hy[ind_local];
                _Hz_aux[ind_global] = _Hz[ind_local];
            }
        }
    }

}

void fdtd::FDTD::calc_complex_fields(double t0, double t1)
{
    double f0, f1, phi, A, t0H, t1H;
    int ind_local, ind_global;

    t0H = t0 - 0.5*_dt;
    t1H = t1 - 0.5*_dt;

    for(int i = 0; i < _I; i++) {
        for(int j = 0; j < _J; j++) {
            for(int k = 0; k < _K; k++) {
                ind_local = (i+1)*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
                ind_global = i*_J*_K + j*_K + k;
                
                // Compute amplitude and phase for Ex
                // Note: we are careful to assume exp(-i*w*t) time dependence
                f0 = _Ex_aux[ind_global].real;
                f1 = _Ex[ind_local].real;
                phi = calc_phase(t0, t1, f0, f1);
                A = calc_amplitude(t0, t1, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ex_aux[ind_global].real = A*cos(phi);
                _Ex_aux[ind_global].imag = -A*sin(phi); 

                // Ey
                f0 = _Ey_aux[ind_global].real;
                f1 = _Ey[ind_local].real;
                phi = calc_phase(t0, t1, f0, f1);
                A = calc_amplitude(t0, t1, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ey_aux[ind_global].real = A*cos(phi);
                _Ey_aux[ind_global].imag = -A*sin(phi); 

                // Ez
                f0 = _Ez_aux[ind_global].real;
                f1 = _Ez[ind_local].real;
                phi = calc_phase(t0, t1, f0, f1);
                A = calc_amplitude(t0, t1, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ez_aux[ind_global].real = A*cos(phi);
                _Ez_aux[ind_global].imag = -A*sin(phi); 

                // Hx
                f0 = _Hx_aux[ind_global].real;
                f1 = _Hx[ind_local].real;
                phi = calc_phase(t0H, t1H, f0, f1);
                A = calc_amplitude(t0H, t1H, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hx_aux[ind_global].real = A*cos(phi);
                _Hx_aux[ind_global].imag = -A*sin(phi); 

                // Hy
                f0 = _Hy_aux[ind_global].real;
                f1 = _Hy[ind_local].real;
                phi = calc_phase(t0H, t1H, f0, f1);
                A = calc_amplitude(t0H, t1H, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hy_aux[ind_global].real = A*cos(phi);
                _Hy_aux[ind_global].imag = -A*sin(phi); 

                // Hz
                f0 = _Hz_aux[ind_global].real;
                f1 = _Hz[ind_local].real;
                phi = calc_phase(t0H, t1H, f0, f1);
                A = calc_amplitude(t0H, t1H, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hz_aux[ind_global].real = A*cos(phi);
                _Hz_aux[ind_global].imag = -A*sin(phi); 
            }
        }
    }

}

inline double fdtd::calc_phase(double t0, double t1, double f0, double f1)
{
    if(f0 == 0.0 and f1 == 0) {
        return 0.0;
    }
    else {
        return atan((f1*sin(t0)-f0*sin(t1))/(f0*cos(t1)-f1*cos(t0)));
    }
}

inline double fdtd::calc_amplitude(double t0, double t1, double f0, double f1, double phase)
{
    if(f0*f0 > f1*f1) {
        return f1 / (sin(t1)*cos(phase) + cos(t1)*sin(phase));
    }
    else {
        return f0 / (sin(t0)*cos(phase) + cos(t0)*sin(phase));
    }
}

///////////////////////////////////////////////////////////////////////////
// Source management
///////////////////////////////////////////////////////////////////////////
void fdtd::FDTD::add_source(complex128 *Jx, complex128 *Jy, complex128 *Jz,
                            complex128 *Mx, complex128 *My, complex128 *Mz,
                            int i0, int j0, int k0, int I, int J, int K,
                            bool calc_phase)
{
    int ind=0;
    double real, imag;
    SourceArray src = {Jx, Jy, Jz, Mx, My, Mz, i0, j0, k0, I, J, K};

    // these source arrays may *actually* be compelx-valued. In the time
    // domain, complex values correspond to temporal phase shifts. We need
    // to convert the complex value to an amplitude and phase. Fortunately,
    // we can use the memory that is already allocated for these values.
    // Specifically, we use src_array.real = amplitude and
    // src_array.imag = phase
    //
    // Important note: EMopt assumes the time dependence is exp(-i*omega*t).
    // In order to account for this minus sign, we need to invert the sign
    // of the calculated phase.
    if(calc_phase) {

    for(int i = 0; i < I; i++) {
        for(int j = 0; j < J; j++) {
            for(int k = 0; k < K; k++) {
                ind = i*J*K + j*K + k;

                
                // Jx
                real = Jx[ind].real;
                imag = Jx[ind].imag;

                Jx[ind].real = sqrt(real*real + imag*imag);
                if(imag == 0 && real == 0) Jx[ind].imag = 0.0;
                else Jx[ind].imag = -1*atan2(imag, real);

                // Jy
                real = Jy[ind].real;
                imag = Jy[ind].imag;

                Jy[ind].real = sqrt(real*real + imag*imag);
                if(imag == 0 && real == 0) Jy[ind].imag = 0.0;
                else Jy[ind].imag = -1*atan2(imag, real);

                // Jz
                real = Jz[ind].real;
                imag = Jz[ind].imag;

                Jz[ind].real = sqrt(real*real + imag*imag);
                if(imag == 0 && real == 0) Jz[ind].imag = 0.0;
                else Jz[ind].imag = -1*atan2(imag, real);

                // Mx
                real = Mx[ind].real;
                imag = Mx[ind].imag;

                Mx[ind].real = sqrt(real*real + imag*imag);
                if(imag == 0 && real == 0) Mx[ind].imag = 0.0;
                else Mx[ind].imag = -1*atan2(imag, real);

                // My
                real = My[ind].real;
                imag = My[ind].imag;

                My[ind].real = sqrt(real*real + imag*imag);
                if(imag == 0 && real == 0) My[ind].imag = 0.0;
                else My[ind].imag = -1*atan2(imag, real);

                // Mz
                real = Mz[ind].real;
                imag = Mz[ind].imag;

                Mz[ind].real = sqrt(real*real + imag*imag);
                if(imag == 0 && real == 0) Mz[ind].imag = 0.0;
                else Mz[ind].imag = -1*atan2(imag, real);
                
            }
        }
    }
    }

    _sources.push_back(src);
}

void fdtd::FDTD::clear_sources()
{
    _sources.clear();
}

void fdtd::FDTD::set_source_properties(double src_T, double src_min)
{
    _src_T = src_T;
    _src_min = src_min;
    _src_k = 6.0 / src_T; // rate of src turn on
    _src_n0 = 1.0 / _src_k * log((1.0-src_min)/src_min); // src delay
}

inline double fdtd::FDTD::src_func_t(int n, double t, double phase)
{
    return sin(t + phase) / (1.0 + exp(-_src_k*(n-_src_n0)));
}

///////////////////////////////////////////////////////////////////////////
// Boundary Conditions
///////////////////////////////////////////////////////////////////////////
void fdtd::FDTD::set_bc(char* newbc)
{
    for(int i = 0; i < 3; i++){
        _bc[i] = newbc[i];
    }
}

///////////////////////////////////////////////////////////////////////////
// ctypes interface
///////////////////////////////////////////////////////////////////////////

fdtd::FDTD* FDTD_new()
{
    return new fdtd::FDTD();
}

void FDTD_set_wavelength(fdtd::FDTD* fdtd, double wavelength)
{
    fdtd->set_wavelength(wavelength);
}

void FDTD_set_physical_dims(fdtd::FDTD* fdtd, 
                            double X, double Y, double Z,
                            double dx, double dy, double dz)
{
    fdtd->set_physical_dims(X, Y, Z, dx, dy, dz);
}

void FDTD_set_grid_dims(fdtd::FDTD* fdtd, int Nx, int Ny, int Nz)
{
    fdtd->set_grid_dims(Nx, Ny, Nz);
}

void FDTD_set_local_grid(fdtd::FDTD* fdtd, 
                         int k0, int j0, int i0,
                         int K, int J, int I)
{
    fdtd->set_local_grid(k0, j0, i0, K, J, I);
}


void FDTD_set_dt(fdtd::FDTD* fdtd, double dt)
{
    fdtd->set_dt(dt);
}

void FDTD_set_field_arrays(fdtd::FDTD* fdtd,
                           complex128 *Ex, complex128 *Ey, complex128 *Ez,
                           complex128 *Hx, complex128 *Hy, complex128 *Hz)
{
    fdtd->set_field_arrays(Ex, Ey, Ez, Hx, Hy, Hz);
}

void FDTD_set_mat_arrays(fdtd::FDTD* fdtd,
                         complex128 *eps_x, complex128 *eps_y, complex128 *eps_z,
                         complex128 *mu_x, complex128 *mu_y, complex128 *mu_z)
{
    fdtd->set_mat_arrays(eps_x, eps_y, eps_z, mu_x, mu_y, mu_z);
}

void FDTD_update_H(fdtd::FDTD* fdtd, int n, double t)
{
    fdtd->update_H(n, t);
}

void FDTD_update_E(fdtd::FDTD* fdtd, int n, double t)
{
    fdtd->update_E(n, t);
}

void FDTD_set_pml_widths(fdtd::FDTD* fdtd, int xmin, int xmax,
                                           int ymin, int ymax,
                                           int zmin, int zmax)
{
    fdtd->set_pml_widths(xmin, xmax, ymin, ymax, zmin, zmax);
}

void FDTD_set_pml_properties(fdtd::FDTD* fdtd, double sigma, double alpha,
                                               double kappa, double pow)
{
    fdtd->set_pml_properties(sigma, alpha, kappa, pow);
}

void FDTD_build_pml(fdtd::FDTD* fdtd)
{
    fdtd->build_pml();
}

void FDTD_reset_pml(fdtd::FDTD* fdtd)
{
    fdtd->reset_pml();
}

void FDTD_set_aux_arrays(fdtd::FDTD* fdtd,
                         complex128 *Ex_aux, complex128 *Ey_aux, complex128 *Ez_aux,
                         complex128 *Hx_aux, complex128 *Hy_aux, complex128 *Hz_aux)
{
    fdtd->set_aux_arrays(Ex_aux, Ey_aux, Ez_aux, Hx_aux, Hy_aux, Hz_aux);
}


double FDTD_calc_phase(double t0, double t1, double f0, double f1)
{
    return fdtd::calc_phase(t0, t1, f0, f1);
}

double FDTD_calc_amplitude(double t0, double t1, double f0, double f1, double phase)
{
    return fdtd::calc_amplitude(t0, t1, f0, f1, phase);
}

void FDTD_capture_t0_fields(fdtd::FDTD* fdtd)
{
    fdtd->capture_t0_fields();
}


void FDTD_calc_complex_fields(fdtd::FDTD* fdtd, double t0, double t1)
{
    fdtd->calc_complex_fields(t0, t1);
}

void FDTD_add_source(fdtd::FDTD* fdtd,
                     complex128 *Jx, complex128 *Jy, complex128 *Jz,
                     complex128 *Mx, complex128 *My, complex128 *Mz,
                     int i0, int j0, int k0, int I, int J, int K, bool calc_phase)
{
    fdtd->add_source(Jx, Jy, Jz, Mx, My, Mz, i0, j0, k0, I, J, K, calc_phase);
}

void FDTD_clear_sources(fdtd::FDTD* fdtd)
{
    fdtd->clear_sources();
}

void FDTD_set_source_properties(fdtd::FDTD* fdtd, double src_T, double src_min)
{
    fdtd->set_source_properties(src_T, src_min);
}

void FDTD_set_bc(fdtd::FDTD* fdtd, char* newbc)
{
    fdtd->set_bc(newbc);
}
