#include "Grid_ctypes.hpp"
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>

//----------------- Material ---------------------------

double Material_get_value_real(Material* mat, double x, double y) { 
	return std::real(mat->get_value(x,y)); 
}

double Material_get_value_imag(Material* mat, double x, double y) { 
	return std::imag(mat->get_value(x,y)); 
} 


void Material_get_values(Material* mat, int m1, int m2, int n1, int n2, complex64* arr)
{
    std::complex<double> val;
    int N = n2-n1;

    for(int i = m1; i < m2; i++) {
        for(int j = n1; j < n2; j++) {
            val = mat->get_value(double(j), double(i));
            arr[(i-m1)*N + j-n1].real = std::real(val);
            arr[(i-m1)*N + j-n1].imag = std::imag(val);
        }
    }
}

//----------------- Grid Material ---------------------------

GridMaterial* GridMaterial_new(int M, int N, complex64* arr) {
	
	ArrayXXcd grid(N,M);
	complex64 val;

	for(int y = 0; y < N; y++) {
		for(int x = 0; x < M; x++) {
			val = arr[y*M+x];
			grid(y,x) = std::complex<double>(val.real, val.imag);
		}
	}

	return new GridMaterial(M, N, grid); 
}

void GridMaterial_delete(GridMaterial* mat) {
	delete mat;
}

void GridMaterial_set_grid(GridMaterial* mat, int M, int N, complex64* arr)
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

int GridMaterial_get_M(GridMaterial* mat) {
	return mat->get_M();
}

int GridMaterial_get_N(GridMaterial* mat) {
	return mat->get_N();
}

//----------------- Structured Material ---------------------------

StructuredMaterial* StructuredMaterial_new(double w, double h, double dx, double dy)
{
	return new StructuredMaterial(w,h,dx,dy);
}


void StructuredMaterial_delete(StructuredMaterial* sm)
{
	delete sm;
}


void StructuredMaterial_add_primitive(StructuredMaterial* sm, MaterialPrimitive* prim)
{
	sm->add_primitive(prim);
}

//---------------------- MaterialPrimitives --------------------------

void MaterialPrimitive_set_layer(MaterialPrimitive* prim, int layer)
{
	prim->set_layer(layer);
}

int MaterialPrimitive_get_layer(MaterialPrimitive* prim)
{
	return prim->get_layer();
}

bool MaterialPrimitive_contains_point(MaterialPrimitive* prim, double x, double y) {
	return prim->contains_point(x,y);
}

double MaterialPrimitive_get_material_real(MaterialPrimitive* prim, double x, double y) {
	return std::real(prim->get_material(x,y));
}

double MaterialPrimitive_get_material_imag(MaterialPrimitive* prim, double x, double y) {
	return std::imag(prim->get_material(x,y));
}

//---------------------- Circle Primitives --------------------------

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


//---------------------- Rectangle Primitives --------------------------

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

//---------------------- Polygon Primitives --------------------------

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

//-------------------------------- MISC --------------------------------
void row_wise_A_update(Material* eps, Material* mu, int ib, int ie, int M, int N, int x1, int x2, int y1, int y2, complex64* vdiag)
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
