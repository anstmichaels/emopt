#include "Grid.hpp"

#ifndef __GRID_CTYPES_HPP__
#define __GRID_CTYPES_HPP__

using namespace Grid;

// This acts as an interface to the numpy.complex64 data type
typedef struct struct_complex64 {
	double real,
		  imag;
} complex64;

extern "C" {

    ////////////////////////////////////////////////////////////////////////////////
	// Material
	////////////////////////////////////////////////////////////////////////////////
	double Material2D_get_value_real(Material2D* mat, double x, double y);
	double Material2D_get_value_imag(Material2D* mat, double x, double y);

    void Material2D_get_values(Material2D* mat, int k1, int k2, int j1, int j2, complex64* arr);

	////////////////////////////////////////////////////////////////////////////////
	// GridMaterial2D
	////////////////////////////////////////////////////////////////////////////////
	GridMaterial2D* GridMaterial2D_new(int M, int N, complex64* arr); 
	void GridMaterial2D_delete(GridMaterial2D* mat);
	void GridMaterial2D_set_grid(GridMaterial2D* mat, int M, int N, complex64* arr);
	int GridMaterial2D_get_M(GridMaterial2D* mat);
	int GridMaterial2D_get_N(GridMaterial2D* mat);

	////////////////////////////////////////////////////////////////////////////////
	// StructuredMaterial
	////////////////////////////////////////////////////////////////////////////////
	StructuredMaterial2D* StructuredMaterial2D_new(double w, double h, double dx, double dy);
	void StructuredMaterial2D_delete(StructuredMaterial2D* sm);
	void StructuredMaterial2D_add_primitive(StructuredMaterial2D* sm, MaterialPrimitive* prim);

    ////////////////////////////////////////////////////////////////////////////////
	// MaterialPrimitive
	////////////////////////////////////////////////////////////////////////////////
	void MaterialPrimitive_set_layer(MaterialPrimitive* prim, int layer);
	int MaterialPrimitive_get_layer(MaterialPrimitive* prim);
	bool MaterialPrimitive_contains_point(MaterialPrimitive* prim, double x, double y);
	double MaterialPrimitive_get_material_real(MaterialPrimitive* prim, double x, double y);
	double MaterialPrimitive_get_material_imag(MaterialPrimitive* prim, double x, double y);

	////////////////////////////////////////////////////////////////////////////////
	// Circle
	////////////////////////////////////////////////////////////////////////////////
	Circle* Circle_new(double x0, double y0, double r);
	void Circle_delete(Circle* c);
	void Circle_set_material(Circle* c, double real, double imag);
	void Circle_set_position(Circle* c, double x0, double y0);
	void Circle_set_radius(Circle* c, double r);
	double Circle_get_x0(Circle* c);
	double Circle_get_y0(Circle* c);
	double Circle_get_r(Circle* c);

	////////////////////////////////////////////////////////////////////////////////
	// Rectangle
	////////////////////////////////////////////////////////////////////////////////
	Rectangle* Rectangle_new(double x0, double y0, double xspan, double yspan);
	void Rectangle_delete(Rectangle* r);
	void Rectangle_set_material(Rectangle* r, double real, double imag);
	void Rectangle_set_position(Rectangle* r, double x0, double y0);
	void Rectangle_set_width(Rectangle* r, double width);
	void Rectangle_set_height(Rectangle* r, double height);

	
	////////////////////////////////////////////////////////////////////////////////
	// Polygon
	////////////////////////////////////////////////////////////////////////////////
	Polygon* Polygon_new();
	void Polygon_delete(Polygon* poly);
	void Polygon_add_point(Polygon* poly, double x, double y);
	void Polygon_add_points(Polygon* poly, double* x, double* y, int n);
	void Polygon_set_point(Polygon* poly, double x, double y, int index);
	void Polygon_set_points(Polygon* poly, double* x, double* y, int n);
	void Polygon_set_material(Polygon* poly, double real, double imag);

	////////////////////////////////////////////////////////////////////////////////
	// ConstantMaterial
	////////////////////////////////////////////////////////////////////////////////
    ConstantMaterial2D* ConstantMaterial2D_new(double real, double imag);
    void ConstantMaterial2D_set_material(ConstantMaterial2D* cm, double real, double imag);
	double ConstantMaterial2D_get_material_real(ConstantMaterial2D* cm);
	double ConstantMaterial2D_get_material_imag(ConstantMaterial2D* cm);

	////////////////////////////////////////////////////////////////////////////////
	// Structured3DMaterial
	////////////////////////////////////////////////////////////////////////////////
    Structured3DMaterial* Structured3DMaterial_new(double X, double Y, double Z, double dx, double dy, double dz);
	void Structured3DMaterial_delete(Structured3DMaterial* sm);
	void Structured3DMaterial_add_primitive(Structured3DMaterial* sm, MaterialPrimitive* prim, double z1, double z2);

    ////////////////////////////////////////////////////////////////////////////////
    // Misc
    ////////////////////////////////////////////////////////////////////////////////
    void row_wise_A_update(Material2D* eps, Material2D* mu, int ib, int ie, int M, int N, int x1, int x2, int y1, int y2, complex64* vdiag);
}

#endif
