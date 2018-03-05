#include "Grid.hpp"
#include <iostream>
#include <climits>
#include <ctime>
#include <exception>
#include <omp.h>

using namespace Grid;

/**************************************** Materials ****************************************/
//------------------------------ Grid Material ------------------------------------/
GridMaterial2D::GridMaterial2D(int M, int N, ArrayXXcd grid) : _M(M), _N(N), _grid(grid) {}

std::complex<double> GridMaterial2D::get_value(int x, int y)
{
	if(x > 0 && x < _M && y > 0 && y < _N) 
		return _grid(y,x);
	else
		return 1.0;
}


void GridMaterial2D::get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2)
{
    int N = k2 - k1;

    for(int i = j1; i < j2; i++) {
        for(int j = k1; j < k2; j++) {
            grid((i-j1)*N + j-k1) = _grid(i,j);
        }
    }
}

void GridMaterial2D::set_grid(int M, int N, ArrayXXcd grid)
{
	_M = M;
	_N = N;
	_grid = grid;
}

int GridMaterial2D::get_M() { return _M; }
int GridMaterial2D::get_N() { return _N; }


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
StructuredMaterial2D::StructuredMaterial2D(double w, double h, double dx, double dy) :
	_w(w), _h(h), _dx(dx), _dy(dy)
{}

StructuredMaterial2D::~StructuredMaterial2D() {}

/* It is important to the material averaging algorithm that primitives be stored in an 
 * ordered list according to their layer.  Lower layers are stored first (have priority).
 * This means that once you have added a primitive to a list, you cannot change its
 * layer!
 */
void StructuredMaterial2D::add_primitive(MaterialPrimitive* prim)
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


void StructuredMaterial2D::add_primitives(std::list<MaterialPrimitive*> primitives)
{
    std::list<MaterialPrimitive*>::iterator it;
    for(it = primitives.begin(); it != primitives.end(); it++) {
        add_primitive(*it);
    }
}


std::complex<double> StructuredMaterial2D::get_value(int x, int y) {
    return get_value(double(x), double(y));
}

void StructuredMaterial2D::get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2)
{
    int N = k2 - k1;

    for(int i = j1; i < j2; i++) {
        for(int j = k1; j < k2; j++) {
            grid((i-j1)*N+j-k1) = get_value(j, i);
        }
    }
}

// This attempts to compute a reasonable average of the materials in a given Yee cell
// Note that there are a few situations where this average will not quite be what they
// should be.  In particular, if three or more materials intersect a cell, this 
// average will begin to deviate from the "correct" average
std::complex<double> StructuredMaterial2D::get_value(double x, double y)
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


std::list<MaterialPrimitive*> StructuredMaterial2D::get_primitives()
{
    return _primitives;
}

////////////////////////////////////////////////////////////////////////////////////
// Constant Material
////////////////////////////////////////////////////////////////////////////////////
ConstantMaterial2D::ConstantMaterial2D(std::complex<double> value)
{
    _value = value;
}

std::complex<double> ConstantMaterial2D::get_value(int x, int y)
{
    return _value;
}

void ConstantMaterial2D::get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2)
{
    int N = k2 - k1;

    for(int i = j1; i < j2; i++) {
        for(int j = k1; j < k2; j++) {
            grid((i-j1)*N + j-k1) = _value;
        }
    }
}

void ConstantMaterial2D::set_material(std::complex<double> val)
{
    _value = val;
}

std::complex<double> ConstantMaterial2D::get_material()
{
    return _value;
}

////////////////////////////////////////////////////////////////////////////////////
// Structured 3D Material
////////////////////////////////////////////////////////////////////////////////////

Structured3DMaterial::Structured3DMaterial(double X, double Y, double Z,
                                           double dx, double dy, double dz) :
                                           _X(X), _Y(Y), _Z(Z), 
                                           _dx(dx), _dy(dy), _dz(dz)
{}

// We allocate memory -- Need to free it!
Structured3DMaterial::~Structured3DMaterial()
{
	for(auto it = _layers.begin(); it != _layers.end(); it++) {
        delete (*it);
    }
}

void Structured3DMaterial::add_primitive(MaterialPrimitive* prim, double z1, double z2)
{
    // Dummy variables
    StructuredMaterial2D* layer;
    double znew[2] = {z1, z2},
           z = 0;

    // Get access to relevant lists
    auto itl = _layers.begin();
    auto itz = _zs.begin();
    
    std::list<StructuredMaterial2D*>::iterator itl_ins;
    std::list<double>::iterator itz_ins;

    // Make sure the layer has a thickness
    if(z1 == z2) {
        std::cout << "Warning in Structured3DMaterial: Provided layer has no \
                      thickness. It will be ignored." << std :: endl;

        return;
    }
    else if(z2 < z1) {
        std::cout << "Warning in Structured3DMaterial: Provided layer has negative \
                      thickness. It will be ignored." << std :: endl;

        return;
    }

    // If this is the first addition, things are simple
    if(itz == _zs.end()) {
        _zs.push_back(z1);
        _zs.push_back(z2);
        
        layer = new StructuredMaterial2D(_X, _Y, _dx, _dy);
        layer->add_primitive(prim);
        _layers.push_back(layer);

        return;
    }

    // now we insert the beginning and end point of the layer one at a time, breaking
    // up or inserting new layers as necessary
    for(int i = 0; i < 2; i++) {
        z = znew[i];

        itz = _zs.begin();
        itl = _layers.begin();
        itz_ins = _zs.end();
        itl_ins = _layers.end();

        // figure out where the point is going to go
        while(itz != _zs.end()) {
            if(z >= *itz) {
                itz_ins = itz;
                itl_ins = itl;
            }
            itz++;
            if(itl != _layers.end())
                itl++;
        }

        // Three cases to consider: (1) point below stack (2) point above stack (3)
        // point in stack
        if(itz_ins == _zs.end()) {
            layer = new StructuredMaterial2D(_X, _Y, _dx, _dy);
            _layers.push_front(layer);
            _zs.push_front(z);
        }
        else if(itz_ins == --_zs.end()) {
            layer = new StructuredMaterial2D(_X, _Y, _dx, _dy);
            _layers.push_back(layer);
            _zs.push_back(z);
        }
        else {
            // make sure the point to insert is not already in the stack
            if(z != *itz_ins) {
                layer = new StructuredMaterial2D(_X, _Y, _dx, _dy);
                layer->add_primitives( (*itl_ins)->get_primitives() );
                _layers.insert(itl_ins, layer);
                _zs.insert(++itz_ins, z);
            }
        }
    }

    // Finally, insert the supplied MaterialPrimitve into the desired locations
    itz = _zs.begin();
    itl = _layers.begin();

    // figure out where the point is going to go
    while(itl != _layers.end()) {
        z = (*itz);
        if(z >= z1 && z <= z2) {
            (*itl)->add_primitive(prim);
        }
        itz++;
        itl++;
    }

    // aaannnddd we're done!
}

std::complex<double> Structured3DMaterial::get_value(int k, int j, int i)
{
    return get_value(double(k), double(j), double(i));
}

std::complex<double> Structured3DMaterial::get_value(double k, double j, double i)
{

}

// Note that this takes a 1D array!
void Structured3DMaterial::get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2, int i1, int i2)
{
    int index = 0,
        Nx = k2-k1,
        Ny = j2-j1;

    for(int i = i1; i < i2; i++) {
        for(int j = j1; j < j2; j++) {
            for(int k = k1; k < k2; k++) {
                index = (i-i1)*Nx*Ny + (j-j1)*Nx + (k-k1);
                grid(index) = get_value(k, j, i);
            }
        }
    }
}
