#include <iostream>
#include <memory>
#include <vector>

#ifndef __FDTD_HPP__
#define __FDTD_HPP__

#define pow2(x) x*x
#define pow3(x) x*x*x

typedef struct struct_complex128 {
    double real, imag;


    /* Some important notes:
     *
     * 1) The FDTD algorithm works purely with real numbers,
     *    so the overload arithmetic operators reflect this.
     */
    struct_complex128 operator+(const struct_complex128& val) {
        struct_complex128 output;
        output.real = real + val.real;
        output.imag = 0;
        return output;
    }

    struct_complex128 operator-(const struct_complex128& val) {
        struct_complex128 output;
        output.real = real - val.real;
        output.imag = 0;
        return output;
    }

    struct_complex128 operator*(const struct_complex128& val) {
        struct_complex128 output;
        output.real = real*val.real;
        output.imag =  0;
        return output;
    }

    struct_complex128 operator/(double val1) {
        struct_complex128 output;
        output.real = real/val1;
        output.imag = 0;
        return output;
    }

    friend struct_complex128 operator/(double val1, const struct_complex128& val2) {
        struct_complex128 output;

        output.real = val1 / val2.real;
        output.imag = 0;
        return output;
    }

    friend struct_complex128 operator*(double val1, const struct_complex128& val2) {
        struct_complex128 output;
        output.real = val1*val2.real;
        output.imag = 0;
        return output;
    }

    friend struct_complex128 operator+(double val1, const struct_complex128& val2) {
        struct_complex128 output;
        output.real = val1 + val2.real;
        output.imag = 0;
        return output;
    }

    struct_complex128& operator=(double val) {
        real = val;
        return *this;
    }

} complex128;


namespace fdtd {

    double calc_phase(double t0, double t1, double f0, double f1);
    double calc_amplitude(double t0, double t1, double f0, double f1, double phase);

    typedef struct struct_SourceArray {
        complex128 *Jx, *Jy, *Jz, *Mx, *My, *Mz;
        int i0, j0, k0, I, J, K;
    } SourceArray;

    class FDTD {
        private:
            int _Nx, _Ny, _Nz;
            int _I, _J, _K, _i0, _j0, _k0;

            double _X, _Y, _Z, _dx, _dy, _dz;
            double _wavelength;
            double _R, _dt;
            double _src_T, _src_min, _src_k, _src_n0;

            // Field and source arrays
            complex128 *_Ex, *_Ey, *_Ez,
                       *_Hx, *_Hy, *_Hz,
                       *_eps_x, *_eps_y, *_eps_z,
                       *_mu_x, *_mu_y, *_mu_z,
                       *_Ex_aux, *_Ey_aux, *_Ez_aux,
                       *_Hx_aux, *_Hy_aux, *_Hz_aux;

            // PML parameters
            int _w_pml_x0, _w_pml_x1,
                _w_pml_y0, _w_pml_y1,
                _w_pml_z0, _w_pml_z1;

            double _sigma, _alpha, _kappa, _pow;

            char _bc[3];

            // PML arrays -- because convolutions
            // Not ever processor will need all of different PML layers.
            // For example, a processor which touches the xmin boundary of the
            // simulation only needs to store pml values corresponding to derivatives
            // along the x direction.
            complex128 *_pml_Exy0, *_pml_Exy1, *_pml_Exz0, *_pml_Exz1,
                       *_pml_Eyx0, *_pml_Eyx1, *_pml_Eyz0, *_pml_Eyz1,
                       *_pml_Ezx0, *_pml_Ezx1, *_pml_Ezy0, *_pml_Ezy1,
                       *_pml_Hxy0, *_pml_Hxy1, *_pml_Hxz0, *_pml_Hxz1,
                       *_pml_Hyx0, *_pml_Hyx1, *_pml_Hyz0, *_pml_Hyz1,
                       *_pml_Hzx0, *_pml_Hzx1, *_pml_Hzy0, *_pml_Hzy1;

            std::vector<SourceArray> _sources;

            double pml_ramp(double distance);

        public:
            
            FDTD();
            ~FDTD();

            void set_physical_dims(double X, double Y, double Z,
                                         double dx, double dy, double dz);
            void set_grid_dims(int Nx, int Ny, int Nz);
            void set_local_grid(int k0, int j0, int i0, int K, int J, int I);
            void set_wavelength(double wavelength);
            void set_dt(double dt);
            void set_field_arrays(complex128 *Ex, complex128 *Ey, complex128 *Ez,
                                  complex128 *Hx, complex128 *Hy, complex128 *Hz);

            void set_mat_arrays(complex128 *eps_x, complex128 *eps_y, complex128 *eps_z,
                                complex128 *mu_x, complex128 *mu_y, complex128 *mu_z);

            void update_H(int n, double t);
            void update_E(int n, double t);

            // PML configuration
            void set_pml_widths(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);

            /**
             * Note: if pow==1.0, pow==2.0, or pow==3.0 exactly, then the pml ramp is
             * computed efficiently. Otherwise, the std::pow function is used.
             */
            void set_pml_properties(double sigma, double alpha, double kappa, double pow);

            /**
             * This must be called after set_local_grid(...) and set_pml_widths(...)
             */
            void build_pml();

            /**
             * Reset the PML arrays to zero. This must be called AFTER build_pml(...)
             */
            void reset_pml();

            // manage auxilary fields + amp/phase calculation
            void set_aux_arrays(complex128 *Ex_aux, complex128 *Ey_aux, complex128 *Ez_aux,
                                complex128 *Hx_aux, complex128 *Hy_aux, complex128 *Hz_aux);

            void capture_t0_fields();
            void calc_complex_fields(double t0, double t1);

            // Manage source arrays
            /*
             * i0, j0, and k0 define the lower corner of the array in the local index
             * space.
             */
            void add_source(complex128 *Jx, complex128 *Jy, complex128 *Jz,
                            complex128 *Mx, complex128 *My, complex128 *Mz,
                            int i0, int j0, int k0, int I, int J, int K,
                            bool calc_phase);
            void clear_sources();

            void set_source_properties(double src_T, double src_min);
            double src_func_t(int n, double t, double phase);

            /* Set the boundary conditions
             */
            void set_bc(char* newbc);
    };

};

extern "C" {
        fdtd::FDTD* FDTD_new();


        void FDTD_set_wavelength(fdtd::FDTD* fdtd, double wavelength);
        void FDTD_set_physical_dims(fdtd::FDTD* fdtd, 
                                    double X, double Y, double Z,
                                    double dx, double dy, double dz);
        void FDTD_set_grid_dims(fdtd::FDTD* fdtd, int Nx, int Ny, int Nz);
        void FDTD_set_local_grid(fdtd::FDTD* fdtd, 
                                 int k0, int j0, int i0,
                                 int K, int J, int I);
        void FDTD_set_dt(fdtd::FDTD* fdtd, double dt);
        void FDTD_set_field_arrays(fdtd::FDTD* fdtd,
                                   complex128 *Ex, complex128 *Ey, complex128 *Ez,
                                   complex128 *Hx, complex128 *Hy, complex128 *Hz);

        void FDTD_set_mat_arrays(fdtd::FDTD* fdtd,
                                 complex128 *eps_x, complex128 *eps_y, complex128 *eps_z,
                                 complex128 *mu_x, complex128 *mu_y, complex128 *mu_z);

        void FDTD_update_H(fdtd::FDTD* fdtd, int n, double t);
        void FDTD_update_E(fdtd::FDTD* fdtd, int n, double t);

        // Pml management
        void FDTD_set_pml_widths(fdtd::FDTD* fdtd, int xmin, int xmax,
                                                   int ymin, int ymax,
                                                   int zmin, int zmax);
        void FDTD_set_pml_properties(fdtd::FDTD* fdtd, double sigma, double alpha,
                                                       double kappa, double pow);
        void FDTD_build_pml(fdtd::FDTD* fdtd);
        void FDTD_reset_pml(fdtd::FDTD* fdtd);

        // auxillary array management
        void FDTD_set_aux_arrays(fdtd::FDTD* fdtd,
                                 complex128 *Ex_aux, complex128 *Ey_aux, complex128 *Ez_aux,
                                 complex128 *Hx_aux, complex128 *Hy_aux, complex128 *Hz_aux);
        double FDTD_calc_phase(double t0, double t1, double f0, double f1);
        double FDTD_calc_amplitude(double t0, double t1, double f0, double f1, double phase);

        void FDTD_capture_t0_fields(fdtd::FDTD* fdtd);
        void FDTD_calc_complex_fields(fdtd::FDTD* fdtd, double t0, double t1);

        // Source management
        void FDTD_add_source(fdtd::FDTD* fdtd,
                             complex128 *Jx, complex128 *Jy, complex128 *Jz,
                             complex128 *Mx, complex128 *My, complex128 *Mz,
                             int i0, int j0, int k0, int I, int J, int K,
                             bool calc_phase);
        void FDTD_clear_sources(fdtd::FDTD* fdtd);

        void FDTD_set_source_properties(fdtd::FDTD* fdtd, double src_T, double src_min);

        // boundary conditions
        void FDTD_set_bc(fdtd::FDTD* fdtd, char* newbc);

};

#endif
