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

    //-----------------------------------------------------------
	//----------------------- Material---------------------------
	//-----------------------------------------------------------
	double Material_get_value_real(Material* mat, double x, double y);
	double Material_get_value_imag(Material* mat, double x, double y);

    void Material_get_values(Material* mat, int m1, int m2, int n1, int n2, complex64* arr);

	//----------------------------------------------------------------
	//----------------------- GridMaterial ---------------------------
	//----------------------------------------------------------------
	GridMaterial* GridMaterial_new(int M, int N, complex64* arr); 
	void GridMaterial_delete(GridMaterial* mat);
	void GridMaterial_set_grid(GridMaterial* mat, int M, int N, complex64* arr);
	int GridMaterial_get_M(GridMaterial* mat);
	int GridMaterial_get_N(GridMaterial* mat);

	//----------------------------------------------------------------------
	//----------------------- StructuredMaterial ---------------------------
	//----------------------------------------------------------------------
	StructuredMaterial* StructuredMaterial_new(double w, double h, double dx, double dy);
	void StructuredMaterial_delete(StructuredMaterial* sm);
	void StructuredMaterial_add_primitive(StructuredMaterial* sm, MaterialPrimitive* prim);

    //---------------------------------------------------------------------
	//----------------------- MaterialPrimitive ---------------------------
	//---------------------------------------------------------------------
	void MaterialPrimitive_set_layer(MaterialPrimitive* prim, int layer);
	int MaterialPrimitive_get_layer(MaterialPrimitive* prim);
	bool MaterialPrimitive_contains_point(MaterialPrimitive* prim, double x, double y);
	double MaterialPrimitive_get_material_real(MaterialPrimitive* prim, double x, double y);
	double MaterialPrimitive_get_material_imag(MaterialPrimitive* prim, double x, double y);

	//----------------------------------------------------------
	//----------------------- Circle ---------------------------
	//----------------------------------------------------------
	Circle* Circle_new(double x0, double y0, double r);
	void Circle_delete(Circle* c);
	void Circle_set_material(Circle* c, double real, double imag);
	void Circle_set_position(Circle* c, double x0, double y0);
	void Circle_set_radius(Circle* c, double r);
	double Circle_get_x0(Circle* c);
	double Circle_get_y0(Circle* c);
	double Circle_get_r(Circle* c);

	//-------------------------------------------------------------
	//----------------------- Rectangle ---------------------------
	//-------------------------------------------------------------
	Rectangle* Rectangle_new(double x0, double y0, double xspan, double yspan);
	void Rectangle_delete(Rectangle* r);
	void Rectangle_set_material(Rectangle* r, double real, double imag);
	void Rectangle_set_position(Rectangle* r, double x0, double y0);
	void Rectangle_set_width(Rectangle* r, double width);
	void Rectangle_set_height(Rectangle* r, double height);

	
	//-----------------------------------------------------------
	//----------------------- Polygon ---------------------------
	//-----------------------------------------------------------
	Polygon* Polygon_new();
	void Polygon_delete(Polygon* poly);
	void Polygon_add_point(Polygon* poly, double x, double y);
	void Polygon_add_points(Polygon* poly, double* x, double* y, int n);
	void Polygon_set_point(Polygon* poly, double x, double y, int index);
	void Polygon_set_points(Polygon* poly, double* x, double* y, int n);
	void Polygon_set_material(Polygon* poly, double real, double imag);

    //------------------------ Misc -----------------------------
    void row_wise_A_update(Material* eps, Material* mu, int ib, int ie, int M, int N, int x1, int x2, int y1, int y2, complex64* vdiag);
}

#endif
