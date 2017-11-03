#include "Grid.hpp"
#include <iostream>
#include <climits>
#include <ctime>
#include <exception>
#include <omp.h>

using namespace Grid;

/**************************************** Materials ****************************************/
//------------------------------ Grid Material ------------------------------------/
GridMaterial::GridMaterial(int M, int N, ArrayXXcd grid) : _M(M), _N(N), _grid(grid) {}

std::complex<double> GridMaterial::get_value(int x, int y)
{
	if(x > 0 && x < _M && y > 0 && y < _N) 
		return _grid(y,x);
	else
		return 1.0;
}


void GridMaterial::get_values(ArrayXXcd& grid, int m1, int m2, int n1, int n2)
{
    for(int i = m1; i < m2; i++) {
        for(int j = n1; j < n2; j++) {
            grid(i-m1,j-n1) = _grid(i,j);
        }
    }
}

void GridMaterial::set_grid(int M, int N, ArrayXXcd grid)
{
	_M = M;
	_N = N;
	_grid = grid;
}

int GridMaterial::get_M() { return _M; }
int GridMaterial::get_N() { return _N; }


//------------------------------ MaterialPrimitives ------------------------------------/
GridCell::GridCell()
{
}
		
void GridCell::set_vertices(double xmin, double xmax, double ymin, double ymax)
{
	double area;
	_verts.clear();

	Polygon_2D new_poly;
	boost::geometry::append(new_poly, Point_2D(xmin, ymin));
	boost::geometry::append(new_poly, Point_2D(xmin, ymax));
	boost::geometry::append(new_poly, Point_2D(xmax, ymax));
	boost::geometry::append(new_poly, Point_2D(xmax, ymin));
	boost::geometry::append(new_poly, Point_2D(xmin, ymin));
    boost::geometry::correct(new_poly);

    boost::geometry::assign(_original, new_poly);

    _verts.push_back(new_poly);
	area = boost::geometry::area(new_poly);
	_area = fabs(area);
	_max_area = _area;
}

double GridCell::intersect(const Polygon_2D poly)
{
	double area = 0.0,
		   intersected_area;

	_diffs.clear();
	
	std::list<Polygon_2D>::const_iterator i;

    // Do the difference
	for(i = _verts.begin(); i != _verts.end(); ++i) {
		boost::geometry::difference((*i), poly, _diffs);
	}
	_verts.clear();
	
	for(i = _diffs.begin(); i != _diffs.end(); i++) {
		_verts.push_back(*i);
		area += fabs(boost::geometry::area(*i));
	}


	intersected_area = _area - area;
    //if(intersected_area < 0) {
    //    std::cout << intersected_area << std::endl;
    //    std::cout << "Size: " << _verts.size() << std::endl;
    //    std::cout << "Poly 1: " << boost::geometry::dsv(_verts.front()) << std::endl;
    //    std::cout << "Poly original: " << boost::geometry::dsv(_original) << std::endl;
    //    std::cout << "Poly intersecting: " << boost::geometry::dsv(poly) << std::endl;
    //}
	_area = area;
	return intersected_area/_max_area;

}

double GridCell::get_area()
{
	return _area;
}

double GridCell::get_max_area()
{
	return _max_area;
}


double GridCell::get_area_ratio()
{
	return _area/_max_area;
}		

//------------------------------ MaterialPrimitives ------------------------------------/
MaterialPrimitive::MaterialPrimitive()
{
	_layer = 1;
}

int MaterialPrimitive::get_layer() const { return _layer; }
void MaterialPrimitive::set_layer(int layer) { _layer = layer; }


bool MaterialPrimitive::operator<(const MaterialPrimitive& rhs)
{
	return _layer < rhs.get_layer();
}

Circle::Circle(double x0, double y0, double r) : _x0(x0), _y0(y0), _r(r) {}
Circle::~Circle() {}

bool Circle::contains_point(double x, double y)
{
	double dx = x - _x0,
		   dy = y - _y0;
	return dx*dx + dy*dy < _r*_r;
}


bool Circle::bbox_contains_point(double x, double y)
{
	double xmin = _x0-_r,
		   xmax = _x0+_r,
		   ymin = _y0-_r,
		   ymax = _y0+_r;

	return (x > xmin) && (x < xmax) && (y > ymin) && (y < ymax);
}

std::complex<double> Circle::get_material(double x, double y)
{
	return _mat;
}

// TODO: Implement this properly
double Circle::get_cell_overlap(GridCell& cell)
{
	return 0.0;
}	

void Circle::set_material(std::complex<double> mat)
{
	_mat = mat;
}

void Circle::set_position(double x0, double y0)
{
	_x0 = x0;
	_y0 = y0;
}

void Circle::set_radius(double r)
{
	_r = r;
}

double Circle::get_x0()
{
	return _x0;
}

double Circle::get_y0()
{
	return _y0;
}

double Circle::get_r()
{
	return _r;
}

Rectangle::Rectangle(double x0, double y0, double width, double height) : 
		_x0(x0), _y0(y0), _width(width), _height(height)
{
    // Points must be defined clockwise and the polygon must be closed
    boost::geometry::append(_poly_rep, Point_2D(x0-width/2.0, y0-height/2.0));
    boost::geometry::append(_poly_rep, Point_2D(x0-width/2.0, y0+height/2.0));
    boost::geometry::append(_poly_rep, Point_2D(x0+width/2.0, y0+height/2.0));
    boost::geometry::append(_poly_rep, Point_2D(x0+width/2.0, y0-height/2.0));
    boost::geometry::append(_poly_rep, Point_2D(x0-width/2.0, y0-height/2.0));
}

Rectangle::~Rectangle()
{
}

bool Rectangle::contains_point(double x, double y)
{
	double hwidth = _width/2,
		   hheight = _height/2;

	return (x > _x0-hwidth) && (x < _x0+hwidth) && (y > _y0-hheight) && (y < _y0+hheight);
}


bool Rectangle::bbox_contains_point(double x, double y)
{
	return contains_point(x,y);
}

std::complex<double> Rectangle::get_material(double x, double y)
{
	return _mat;
}

double Rectangle::get_cell_overlap(GridCell& cell)
{
	return cell.intersect(_poly_rep);
}

void Rectangle::set_material(std::complex<double> mat)
{
	_mat = mat;
}

void Rectangle::set_width(double w)
{
	_width = w;


    std::vector<Point_2D>& outer = _poly_rep.outer();
    outer[0].x(_x0-_width/2.0);
    outer[1].x(_x0-_width/2.0);
    outer[2].x(_x0+_width/2.0);
    outer[3].x(_x0+_width/2.0);
    outer[4].x(_x0-_width/2.0);
}

void Rectangle::set_height(double h)
{
	_height = h;

    std::vector<Point_2D>& outer = _poly_rep.outer();
    outer[0].y(_y0-_height/2.0);
    outer[1].y(_y0+_height/2.0);
    outer[2].y(_y0+_height/2.0);
    outer[3].y(_y0-_height/2.0);
    outer[4].y(_y0-_height/2.0);
}

void Rectangle::set_position(double x0, double y0)
{
	_x0 = x0;
	_y0 = y0;

    std::vector<Point_2D>& outer = _poly_rep.outer();
    outer[0].x(_x0-_width/2.0);
    outer[1].x(_x0-_width/2.0);
    outer[2].x(_x0+_width/2.0);
    outer[3].x(_x0+_width/2.0);
    outer[4].x(_x0-_width/2.0);

    outer[0].y(_y0-_height/2.0);
    outer[1].y(_y0+_height/2.0);
    outer[2].y(_y0+_height/2.0);
    outer[3].y(_y0-_height/2.0);
    outer[4].y(_y0-_height/2.0);

}


//------------------------------ Polygon ------------------------------------/

Polygon::Polygon()
{
}

Polygon::Polygon(double* x, double* y, int n)
{
	set_points(x, y, n);
}

Polygon::~Polygon()
{
	_verts.clear();
}

void Polygon::add_point(double x, double y)
{
	boost::geometry::append(_verts, boost::geometry::make<Point_2D>(x,y));

    // update the bounding box
    boost::geometry::envelope(_verts, _bbox);

    // correct the geometry
    boost::geometry::correct(_verts);
}

/**
 * NOTE: Currently a copy of the input points is made.  This will be slowish.
 */
void Polygon::add_points(double* x, double* y, int n)
{
    for(int i = 0; i < n; i++) {
        boost::geometry::append(_verts, boost::geometry::make<Point_2D>(x[i], y[i]));
    }

    // update the bounding box
    boost::geometry::envelope(_verts, _bbox);

    // correct the geometry
    boost::geometry::correct(_verts);
}

void Polygon::set_point(double x, double y, int index)
{
	Point_2D& p = _verts.outer()[index];
    p.x(x);
    p.y(y);

    // update the bounding box
    boost::geometry::envelope(_verts, _bbox);

    // assume the geometry is correct.

}

void Polygon::set_points(double* x, double* y, int n)
{
	_verts.clear();
	add_points(x,y,n);

    // update the bounding box
    boost::geometry::envelope(_verts, _bbox);

    // correct the geometry
    boost::geometry::correct(_verts);
}

bool Polygon::contains_point(double x, double y)
{
    Point_2D p(x, y);
	bool inside = boost::geometry::within(p, _verts);

	return inside;
}

bool Polygon::bbox_contains_point(double x, double y)
{
    Point_2D p(x,y);
    bool inside = boost::geometry::within(p, _bbox);

	return inside;
}

std::complex<double> Polygon::get_material(double x, double y)
{
	//if(contains_point(x,y))
	return _mat;
}

double Polygon::get_cell_overlap(GridCell& cell)
{
    
	return cell.intersect(_verts);
}

void Polygon::set_material(std::complex<double> mat)
{
	_mat = mat;
}

//------------------------------ Structured Material ------------------------------------/
StructuredMaterial::StructuredMaterial(double w, double h, double dx, double dy) :
	_w(w), _h(h), _dx(dx), _dy(dy)
{}

StructuredMaterial::~StructuredMaterial() {}

/* It is important to the material averaging algorithm that primitives be stored in an 
 * ordered list according to their layer.  Lower layers are stored first (have priority).
 * This means that once you have added a primitive to a list, you cannot change its
 * layer!
 */
void StructuredMaterial::add_primitive(MaterialPrimitive* prim)
{
	std::list<MaterialPrimitive*>::iterator it, insert_pos = _primitives.end();

	if(_primitives.size() == 0) {
		_primitives.push_back(prim);
	}
	else {
		for(it = _primitives.begin(); it != _primitives.end(); it++) {
			if( prim->get_layer() < (*it)->get_layer() ) {
				insert_pos = it; 
				break;
			}
		}

		_primitives.insert(it, prim);
	}

}


std::complex<double> StructuredMaterial::get_value(int x, int y) {
    return get_value(double(x), double(y));
}

void StructuredMaterial::get_values(ArrayXXcd& grid, int m1, int m2, int n1, int n2)
{
    for(int i = m1; i < m2; i++) {
        for(int j = n1; j < n2; j++) {
            grid(i-m1,j-n1) = get_value(j, i);
        }
    }
}

// This attempts to compute a reasonable average of the materials in a given Yee cell
// Note that there are a few situations where this average will not quite be what they
// should be.  In particular, if three or more materials intersect a cell, this 
// average will begin to deviate from the "correct" average
std::complex<double> StructuredMaterial::get_value(double x, double y)
{
	std::complex<double> val = 0.0;
	std::list<MaterialPrimitive*>::iterator it = _primitives.begin();
	MaterialPrimitive* prim;
	GridCell cell;
	
	double xd = x*_dx, //+ _dx/2.0,
		   yd = y*_dy; //+ _dy/2.0;

	double xmin = xd - _dx/2.0,
		   xmax = xd + _dx/2.0,
		   ymin = yd - _dy/2.0,
		   ymax = yd + _dy/2.0,
		   overlap = 1.0;

    bool contains_p1,
         contains_p2,
         contains_p3,
         contains_p4;

	cell.set_vertices(xmin,xmax,ymin,ymax);
	
	if(_primitives.size() == 0) {
		std::cerr << "Error: StructuredMaterial list is empty." << std::endl;
		return 0.0;
	}

	//std::cout << "------------------------" << std::endl;
	while(it != _primitives.end()) {
		prim = (*it);
        
        // These values are used twice, so we recompute them
        contains_p1 = prim->contains_point(xmin,ymin);
        contains_p2 = prim->contains_point(xmax,ymin);
        contains_p3 = prim->contains_point(xmax,ymax);
        contains_p4 = prim->contains_point(xmin,ymax);
		
		if(contains_p1 && contains_p2 &&
		   contains_p3 && contains_p4 &&
		   cell.get_area_ratio() == 1.0) 
		{
				return prim->get_material(xd,yd);
		}
		else if(contains_p1 || contains_p2 ||
		        contains_p3 || contains_p4) 
		{
			overlap = prim->get_cell_overlap(cell);

			val += overlap * prim->get_material(xd,yd);
		}
		it++;

		if(cell.get_area_ratio() == 0) {
			break;
		}

	}

	// assume background has index of 1.0
	if(cell.get_area_ratio() > 0) {
		val += cell.get_area_ratio()*1.0;
	}

	return val;
}
