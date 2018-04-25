#include "Grid_ctypes.hpp"
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>

/////////////////////////////////////////////////////////////////////////////////////
// Material2D
/////////////////////////////////////////////////////////////////////////////////////

void Material2D_get_value(Material2D* mat, complex64* val, double x, double y) { 
    std::complex<double> value = mat->get_value(x,y);

    val[0].real = std::real(value);
    val[0].imag = std::imag(value);
}

void Material2D_get_values(Material2D* mat, int k1, int k2, int j1, int j2,
                         complex64* arr)
{
    std::complex<double> val;
    int Ny = j2-j1,
        Nx = k2-k1;

	ArrayXcd grid(Nx*Ny);
    mat->get_values(grid, k1, k2, j1, j2);

    for(int i = 0; i < Nx*Ny; i++) {
        val = grid(i);
        arr[i].real = std::real(val);
        arr[i].imag = std::imag(val);
    }
}

///////////////////////////////////////////////////////////////////////////////////// 
// Grid Material2D
/////////////////////////////////////////////////////////////////////////////////////

GridMaterial2D* GridMaterial2D_new(int M, int N, complex64* arr) {
	
	ArrayXXcd grid(N,M);
	complex64 val;

	for(int y = 0; y < N; y++) {
		for(int x = 0; x < M; x++) {
			val = arr[y*M+x];
			grid(y,x) = std::complex<double>(val.real, val.imag);
		}
	}

	return new GridMaterial2D(M, N, grid); 
}

void GridMaterial2D_delete(GridMaterial2D* mat) {
	delete mat;
}

void GridMaterial2D_set_grid(GridMaterial2D* mat, int M, int N, complex64* arr)
{
	ArrayXXcd grid(N,M);
	complex64 val;

	for(int y = 0; y < N; y++) {
		for(int x = 0; x < M; x++) {
			val = arr[y*M+x];
			grid(y,x) = std::complex<double>(val.real, val.imag);
		}
	}

	mat->set_grid(M, N, grid);
}

int GridMaterial2D_get_M(GridMaterial2D* mat) {
	return mat->get_M();
}

int GridMaterial2D_get_N(GridMaterial2D* mat) {
	return mat->get_N();
}

/////////////////////////////////////////////////////////////////////////////////////
// Structured Material2D
/////////////////////////////////////////////////////////////////////////////////////

StructuredMaterial2D* StructuredMaterial2D_new(double w, double h, double dx, double dy)
{
	return new StructuredMaterial2D(w,h,dx,dy);
}


void StructuredMaterial2D_delete(StructuredMaterial2D* sm)
{
	delete sm;
}


void StructuredMaterial2D_add_primitive(StructuredMaterial2D* sm, 
                                      MaterialPrimitive* prim)
{
	sm->add_primitive(prim);
}

/////////////////////////////////////////////////////////////////////////////////////
// MaterialPrimitives
/////////////////////////////////////////////////////////////////////////////////////

void MaterialPrimitive_set_layer(MaterialPrimitive* prim, int layer)
{
	prim->set_layer(layer);
}

int MaterialPrimitive_get_layer(MaterialPrimitive* prim)
{
	return prim->get_layer();
}

bool MaterialPrimitive_contains_point(MaterialPrimitive* prim, double x, double y)
{
	return prim->contains_point(x,y);
}

double MaterialPrimitive_get_material_real(MaterialPrimitive* prim, 
                                           double x, double y)
{
	return std::real(prim->get_material(x,y));
}

double MaterialPrimitive_get_material_imag(MaterialPrimitive* prim,
                                           double x, double y)
{
	return std::imag(prim->get_material(x,y));
}

/////////////////////////////////////////////////////////////////////////////////////
// Circle Primitives
/////////////////////////////////////////////////////////////////////////////////////

Circle* Circle_new(double x0, double y0, double r) {
	return new Circle(x0, y0, r);
}

void Circle_delete(Circle* c) {
	delete c;
}

void Circle_set_material(Circle* c, double real, double imag)
{
	c->set_material(std::complex<double>(real, imag));
}

void Circle_set_position(Circle* c, double x0, double y0)
{
	c->set_position(x0, y0);
}

void Circle_set_radius(Circle* c, double r)
{
	c->set_radius(r);
}

double Circle_get_x0(Circle* c)
{
	return c->get_x0();
}

double Circle_get_y0(Circle* c)
{
	return c->get_y0();
}

double Circle_get_r(Circle* c)
{
	return c->get_r();
}


/////////////////////////////////////////////////////////////////////////////////////
//Rectangle Primitives
/////////////////////////////////////////////////////////////////////////////////////

Rectangle* Rectangle_new(double x0, double y0, double xspan, double yspan)
{
	return new Rectangle(x0, y0, xspan, yspan);
}

void Rectangle_delete(Rectangle* r) {
	delete r;
}

void Rectangle_set_material(Rectangle* r, double real, double imag)
{
	r->set_material(std::complex<double>(real, imag));
}

void Rectangle_set_position(Rectangle* r, double x0, double y0)
{
	r->set_position(x0, y0);
}

void Rectangle_set_width(Rectangle* r, double width)
{
	r->set_width(width);
}

void Rectangle_set_height(Rectangle* r, double height)
{
	r->set_height(height);
}

/////////////////////////////////////////////////////////////////////////////////////
// Polygon Primitives
/////////////////////////////////////////////////////////////////////////////////////

Polygon* Polygon_new()
{
	return new Polygon();
}

void Polygon_delete(Polygon* poly)
{
	delete poly;
}

void Polygon_add_point(Polygon* poly, double x, double y)
{
	poly->add_point(x,y);
}

void Polygon_add_points(Polygon* poly, double* x, double* y, int n)
{
	poly->add_points(x,y,n);
}


void Polygon_set_point(Polygon* poly, double x, double y, int index)
{
	poly->set_point(x, y, index);
}

void Polygon_set_points(Polygon* poly, double* x, double* y, int n)
{
	poly->set_points(x,y,n);
}

void Polygon_set_material(Polygon* poly, double real, double imag)
{
	poly->set_material(std::complex<double>(real, imag));
}

/////////////////////////////////////////////////////////////////////////////////////
// ConstantMaterial2D
/////////////////////////////////////////////////////////////////////////////////////
ConstantMaterial2D* ConstantMaterial2D_new(double real, double imag)
{
    return new ConstantMaterial2D(std::complex<double>(real, imag));
}

void ConstantMaterial2D_set_material(ConstantMaterial2D* cm, double real, double imag)
{
    cm->set_material(std::complex<double>(real, imag));
}

double ConstantMaterial2D_get_material_real(ConstantMaterial2D* cm)
{
    return std::real(cm->get_material());
}

double ConstantMaterial2D_get_material_imag(ConstantMaterial2D* cm)
{
    return std::imag(cm->get_material());
}

/////////////////////////////////////////////////////////////////////////////////////
// Material3D
/////////////////////////////////////////////////////////////////////////////////////

void Material3D_get_value(Material3D* mat, complex64* val, double x, double y, double z) { 
	std::complex<double> value = mat->get_value(x,y,z);

    val[0].real = std::real(value);
    val[0].imag = std::imag(value);
}

void Material3D_get_values(Material3D* mat, complex64* arr, int k1, int k2, 
                                                            int j1, int j2,
                                                            int i1, int i2,
                                                            double sx, double sy,
                                                            double sz)
{
    std::complex<double> val;
    int Ny = j2-j1,
        Nx = k2-k1,
        Nz = i2-i1;

	ArrayXcd grid(Nx*Ny*Nz);
    mat->get_values(grid, k1, k2, j1, j2, i1, i2, sx, sy, sz);

    for(int i = 0; i < Nx*Ny*Nz; i++) {
        val = grid(i);
        arr[i].real = std::real(val);
        arr[i].imag = std::imag(val);
    }
}

/////////////////////////////////////////////////////////////////////////////////////
// ConstantMaterial3D
/////////////////////////////////////////////////////////////////////////////////////
ConstantMaterial3D* ConstantMaterial3D_new(double real, double imag)
{
    return new ConstantMaterial3D(std::complex<double>(real, imag));
}

void ConstantMaterial3D_set_material(ConstantMaterial3D* cm, double real, double imag)
{
    cm->set_material(std::complex<double>(real, imag));
}

double ConstantMaterial3D_get_material_real(ConstantMaterial3D* cm)
{
    return std::real(cm->get_material());
}

double ConstantMaterial3D_get_material_imag(ConstantMaterial3D* cm)
{
    return std::imag(cm->get_material());
}

/////////////////////////////////////////////////////////////////////////////////////
// Structured3DMaterial
/////////////////////////////////////////////////////////////////////////////////////
StructuredMaterial3D* StructuredMaterial3D_new(double X, double Y, double Z,
                                               double dx, double dy, double dz)
{
    return new StructuredMaterial3D(X, Y, Z, dx, dy ,dz);
}

void StructuredMaterial3D_delete(StructuredMaterial3D* sm)
{
    delete sm;
}

void StructuredMaterial3D_add_primitive(StructuredMaterial3D* sm, 
                                        MaterialPrimitive* prim, 
                                        double z1, double z2)
{
  sm->add_primitive(prim, z1, z2);  
}

/////////////////////////////////////////////////////////////////////////////////////
// MISC
/////////////////////////////////////////////////////////////////////////////////////
void row_wise_A_update(Material2D* eps, Material2D* mu, int ib, int ie, int M, int N,
                       int x1, int x2, int y1, int y2, complex64* vdiag)
{
    int x = 0,
        y = 0,
        j = 0;

    std::complex<double> value;
    std::complex<double> I(0.0, 1.0);

    for(int i=ib; i < ie; i++) {
        j = i-ib;
        if(i < M*N) {
            y = i/N;
            x = i - y*N;

            if(x >= x1 && x < x2 && y >= y1 && y < y2) {
                value = I*eps->get_value(x,y);
                vdiag[j].real = std::real(value);
                vdiag[j].imag = std::imag(value);
            }
        }
        else if(i < 2*M*N) {
            y = (i-M*N)/N;
            x = (i-M*N) - y*N;

            if(x >= x1 && x < x2 && y >= y1 && y < y2) {
                value = -I*mu->get_value(double(x),double(y)+0.5);
                vdiag[j].real = std::real(value);
                vdiag[j].imag = std::imag(value);
            }
        }
        else {
            y = (i-2*M*N)/N;
            x = (i-2*M*N) - y*N;

            if(x >= x1 && x < x2 && y >= y1 && y < y2) {
                value = -I*mu->get_value(double(x)-0.5,double(y));
                vdiag[j].real = std::real(value);
                vdiag[j].imag = std::imag(value);
            }
        }
    }
}
