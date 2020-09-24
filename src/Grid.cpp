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

std::complex<double> GridMaterial2D::get_value(double x, double y)
{
    int xx = int(x),
        yy = int(y);

	if(xx > 0 && xx < _M && yy > 0 && yy < _N) 
		return _grid(yy,xx);
	else
		return 1.0;
}


void GridMaterial2D::get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2, double sx, double sy)
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
		   intersected_area,
           geo_area;

	_diffs.clear();
	
	std::vector<Polygon_2D>::const_iterator i;

    // Do the difference
	for(i = _verts.begin(); i != _verts.end(); ++i) {
		boost::geometry::difference((*i), poly, _diffs);
	}
	_verts.clear();
	
	for(i = _diffs.begin(); i != _diffs.end(); i++) {
		_verts.push_back(*i);
        geo_area = boost::geometry::area(*i);
		area += fabs(geo_area);
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


////////////////////////////////////////////////////////////////////////////
// Polygon
////////////////////////////////////////////////////////////////////////////

Polygon::Polygon()
{
}

Polygon::Polygon(double* x, double* y, int n)
{
	set_points(x, y, n);
}

Polygon::Polygon(Polygon_2D verts)
{
    _verts = verts;
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

Polygon_2D Polygon::get_vertices()
{
    return _verts;
}

std::vector<Polygon*> Polygon::add(Polygon& p2)
{
    std::vector<Polygon_2D> result;
    std::vector<Polygon*> output;
    
    boost::geometry::union_(_verts, p2.get_vertices(), result);

    for(auto& p: result)
    {
        // extract the outer boundary -- this should be the only boundary in a union
        Polygon_2D new_verts;
        Polygon* wrapped_poly;

        new_verts.clear();
        boost::geometry::assign(new_verts, p.outer());
        wrapped_poly = new Polygon(new_verts);
        wrapped_poly->set_material(_mat);

        output.push_back(wrapped_poly);
    }
    
    
    return output;
}

std::vector<Polygon*> Polygon::subtract(Polygon& p2)
{
    std::vector<Polygon_2D> result;
    std::vector<Polygon*> output;
    
    boost::geometry::difference(_verts, p2.get_vertices(), result);

    for(auto& p: result)
    {
        // extract the outer boundary first
        Polygon_2D new_verts;
        Polygon* wrapped_poly;

        new_verts.clear();
        boost::geometry::assign(new_verts, p.outer());
        wrapped_poly = new Polygon(new_verts);
        wrapped_poly->set_material(_mat);

        output.push_back(wrapped_poly);

        for(auto& inner: p.inners())
        {
            new_verts.clear();
            boost::geometry::assign(new_verts, inner);
            wrapped_poly = new Polygon(new_verts);
            wrapped_poly->set_material(1.0);
            output.push_back(wrapped_poly);
        }
    }
    
    
    return output;
}

std::vector<Polygon*> Polygon::intersect(Polygon& p2)
{
    std::vector<Polygon_2D> result;
    std::vector<Polygon*> output;
    
    boost::geometry::intersection(_verts, p2.get_vertices(), result);

    for(auto& p: result)
    {
        // extract the outer boundary -- this should be the only boundary in a union
        Polygon_2D new_verts;
        Polygon* wrapped_poly;

        new_verts.clear();
        boost::geometry::assign(new_verts, p.outer());
        wrapped_poly = new Polygon(new_verts);
        wrapped_poly->set_material(_mat);

        output.push_back(wrapped_poly);
    }
    
    return output;
}

/////////////////////////////////////////////////////////////////////////////////////
// StructuredMaterial2D
/////////////////////////////////////////////////////////////////////////////////////
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


void StructuredMaterial2D::get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2, double sx, double sy)
{
    int N = k2 - k1;

    for(int j = j1; j < j2; j++) {
        for(int k = k1; k < k2; k++) {
            grid((j-j1)*N+k-k1) = get_value(k+sx, j+sy);
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
// ConstantMaterial2D
////////////////////////////////////////////////////////////////////////////////////
ConstantMaterial2D::ConstantMaterial2D(std::complex<double> value)
{
    _value = value;
}

std::complex<double> ConstantMaterial2D::get_value(double x, double y)
{
    return _value;
}

void ConstantMaterial2D::get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2, double sx, double sy)
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
// ConstantMaterial3D
////////////////////////////////////////////////////////////////////////////////////
ConstantMaterial3D::ConstantMaterial3D(std::complex<double> value)
{
    _value = value;
}

std::complex<double> ConstantMaterial3D::get_value(double k, double j, double i)
{
    return _value;
}

void ConstantMaterial3D::get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2,
                                    int i1, int i2, double sx, double sy, double sz)
{
    int N = k2 - k1,
        M = j2 - j1;

    for(int i = i1; i < i2; i++) {
        for(int j = j1; j < j2; j++) {
            for(int k = k1; k < k2; k++) {
                grid((i-i1)*N*M + (j-j1)*N + k-k1) = _value;
            }
        }
    }
}

void ConstantMaterial3D::set_material(std::complex<double> val)
{
    _value = val;
}

std::complex<double> ConstantMaterial3D::get_material()
{
    return _value;
}

////////////////////////////////////////////////////////////////////////////////////
// StructuredMaterial3D
////////////////////////////////////////////////////////////////////////////////////

StructuredMaterial3D::StructuredMaterial3D(double X, double Y, double Z,
                                           double dx, double dy, double dz) :
                                           _X(X), _Y(Y), _Z(Z), 
                                           _dx(dx), _dy(dy), _dz(dz)
{
    _background = 1.0;
    _use_cache = true;
    _cache_active = false;
}

// We allocate memory -- Need to free it!
StructuredMaterial3D::~StructuredMaterial3D()
{
	for(auto it = _layers.begin(); it != _layers.end(); it++) {
        delete (*it);
    }
}

void StructuredMaterial3D::add_primitive(MaterialPrimitive* prim, double z1, double z2)
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
        else if(itz_ins == --_zs.end() && z != *itz_ins) {
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
        if(z >= z1 && z < z2) {
            (*itl)->add_primitive(prim);
        }
        itz++;
        itl++;
    }

    //for(itl = _layers.begin(); itl != _layers.end(); itl++) {
    //    std::cout << std::endl;
    //    std::list<MaterialPrimitive*> prims = (*itl)->get_primitives();

    //    for(auto ip = prims.begin(); ip != prims.end(); ip++)
    //        std::cout << *ip << std::endl;
    //}

    // aaannnddd we're done!
}

std::complex<double> StructuredMaterial3D::get_value(double k, double j, double i)
{
    double zmin = (i-0.5) * _dz,
           zmax = (i+0.5) * _dz;

    std::complex<double> value = 0.0,
                         mat_val;

    std::list<double>::iterator itz = _zs.begin(),
                                itz_next;
    auto itl = _layers.begin();
    auto itcv = _cached_values.begin();
    auto itcf = _cached_flags.begin(); 

    bool cached = false;
    int jc = 0, kc = 0;

    // Check if i is below the stack
    if(zmax <= *itz) {
        return _background;
    }

    if(zmax > *itz && zmin < *itz) {
        value = (*itz - zmin) / _dz * 1.0;
        zmin = *itz;
    }

    while(itl != _layers.end())
    {

        itz_next = std::next(itz);
        if(zmin >= *itz && zmax <= *itz_next)
        {
            if(_use_cache and _cache_active) {
                jc = int(j) - _cache_j0;
                kc = int(k) - _cache_k0;

                cached = (*itcf)(jc, kc);
                if(cached) {
                    mat_val = (*itcv)(jc, kc);
                }
                else {
                    mat_val = (*itl)->get_value(k, j);
                    (*itcv)(jc, kc) = mat_val;
                    (*itcf)(jc, kc) = true;
                }
            }
            else {
                mat_val = (*itl)->get_value(k, j);
            }
      

            value += (zmax - zmin) / _dz * mat_val;

            return value;
        }
        else if(zmin >= *itz && zmin < *itz_next && zmax > *itz_next)
        {
            if(_use_cache and _cache_active) {
                jc = int(j) - _cache_j0;
                kc = int(k) - _cache_k0;

                cached = (*itcf)(jc, kc);
                if(cached) {
                    mat_val = (*itcv)(jc, kc);
                }
                else {
                    mat_val = (*itl)->get_value(k, j);
                    (*itcv)(jc, kc) = mat_val;
                    (*itcf)(jc, kc) = true;
                }
            }
            else {
                mat_val = (*itl)->get_value(k, j);
            }

            value += (*itz_next - zmin) / _dz * mat_val;
            zmin = *itz_next;
        }

        itl++;
        itz++;
        itcv++;
        itcf++;
    }

    value += (zmax - zmin) / _dz * 1.0;
    return value;
}

// Note that this takes a 1D array!
void StructuredMaterial3D::get_values(ArrayXcd& grid, int k1, int k2, 
                                                      int j1, int j2, 
                                                      int i1, int i2, 
                                                      double sx, double sy, double sz)
{
    int index = 0,
        Nx = k2-k1,
        Ny = j2-j1;

    int Nl, Nc;

    // if caching is enabled, setup all of the cache arrays
    // We need two arrays per slab: one to keep track of which values are already cached
    // and one to actually store the cached values
    if(_use_cache) {
        Nl = _layers.size();
        Nc = _cached_values.size();
        if(Nc != Nl) {
            _cached_values.resize(Nl);
            _cached_flags.resize(Nl);
        }

        for(auto ic = _cached_values.begin(); ic != _cached_values.end(); ic++) {
            (*ic).setZero(j2-j1, k2-k1);
        }
        for(auto flag = _cached_flags.begin(); flag != _cached_flags.end(); flag++) {
            (*flag).setZero(j2-j1, k2-k1);
        }

        _cache_j0 = int(j1+sy);
        _cache_k0 = int(k1+sx);
        _cache_J = j2-j1;
        _cache_K = k2-k1;
        _cache_active = true;

    }

    for(int i = i1; i < i2; i++) {
        for(int j = j1; j < j2; j++) {
            for(int k = k1; k < k2; k++) {
                index = (i-i1)*Nx*Ny + (j-j1)*Nx + (k-k1);
                grid(index) = get_value(k+sx, j+sy, i+sz);
            }
        }
    }

    _cache_active = false;
}
