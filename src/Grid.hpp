#include <Eigen/Core>
#include <Eigen/Dense>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include <cmath>
#include <complex>
#include <string>
#include <vector>
#include <list>

#ifndef __GRID_HPP__
#define __GRID_HPP__

// Specify the precision used for computational geometry subroutines
// In many situations, double is sufficient.  However, cases may be
// encountered where quad-precision is required (this is why CGAL 
// supports arbitrary precision).  To be safe, we generally use quad-
// precision.
#define GFLOAT __float128

typedef boost::geometry::model::d2::point_xy<GFLOAT> Point_2D;
typedef boost::geometry::model::polygon<Point_2D> Polygon_2D;
typedef boost::geometry::model::box<Point_2D> BBox;
using namespace Eigen;

namespace Grid {

/* Material class which provides the foundation for defining the system materials/structure.
 *
 * A Material must satisfy perform one function: given a spatial index, a complex 
 * material value is returned.  This is accomplished by extending the Material class and
 * implementing the <get_value> function.
 */
class Material2D {
	
		public:
			/* Query the material value at a point in real space.
			 * @x The x index of the query
			 * @y The y index of the query
			 *
			 * The structure of the electromagnetic system being solved is ultimately defined
			 * in terms of spatially-dependent materials. The material is defined on a 
			 * spatial grid which is directly compatible with finite differences.
			 * See <GridMaterial> and <StructuredMaterial> for specific implementations.
			 *
			 * @return the complex material at position (x,y).
			 */
			virtual std::complex<double> get_value(double x, double y) = 0;

            /* Get a block of values.
             */
            virtual void get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2) = 0;
			virtual ~Material2D() {};
};

/* Simple array-based implementation of a <Material>
 *
 * A GridMaterial is a material which is explicitly defined on a grid (2D array).  The complex
 * material is explicitly defined at each point in a NxM grid.
 */
class GridMaterial2D : public Material2D {
	
		private:

			int _M, _N;
			ArrayXXcd _grid;

		public:
			/* Create a new GridMaterial2D.
			 * @M the number of columns in the array (system width)
			 * @N the number of rows in the array (system height)
			 * @grid an array containing complex material values defined at each point in space
			 */
			GridMaterial2D(int M, int N, ArrayXXcd grid);
			
			/* GridMaterial2D destructor
			 */
			~GridMaterial2D(){};

			/* Get the material value at index (x,y).
			 * @x the x index (column) of the material value
			 * @y the y index (row) of the material value
			 *
			 * @return the complex material value at (x,y)
			 */
			std::complex<double> get_value(double x, double y);
            void get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2);

			/* Assign a new grid as the <Material>
			 * @M the number of columns in the array (width)
			 * @N the number of rows in the array (height)
			 * @grid an array containing complex material values defined at each point in space
			 */
			void set_grid(int M, int N, ArrayXXcd grid);

			/* Get the number of grid columns (width).
			 * @return the number of grid columns (material width).
			 */
			int get_M();

			/* Get the number of grid rows (height)
			 * @return the number of grid rows (material height)
			 */
			int get_N();

};

/* A polygon which defines the boundary of an entire Yee cell or an arbitrary portion of a Yee cell.
 *
 * Although we are using a discrete grid of Yee cells, it is possible to continuously vary 
 * the material constants within this Yee cell.  We can take advantage of this fact by
 * smoothing our grid in such a way that our matrix A changes continously with perturbations to
 * the system geometry.
 *
 * In order to make this smoothing accurate, it is convenient to maintain a true geometric 
 * representation of a Yee cells.  This allows us to compute the exact overlap between a 
 * polygon which defines the material structure of the system and a given Yee cell.
 */
class GridCell {
	private:
        const static int NPOLY = 4;
		std::vector<Polygon_2D> _verts;
		std::vector<Polygon_2D> _diffs;

        Polygon_2D _original;

		double _area,
			   _max_area;

	public:
		GridCell();
		
		void set_vertices(double xmin, double xmax, double ymin, double ymax);
		double intersect(const Polygon_2D poly);
		double get_area();
		double get_max_area();
		double get_area_ratio();		
};

/* A shape or other (piecewise) continuous structure which defines the Material in a 
 * region of real space.
 *
 * A MaterialPrimitive is a shape such as a rectangle, circle, or polygon which defines the 
 * complex material in a region of real-space (i.e. space defined by doubleing point coordinates,
 * not integer indices corresponding to an array). See <Circle>, <Rectangle>, and <Polygon>
 * for implementation examples.
 *
 * In order to build up complex Materials, many MaterialPrimitives can combined using
 * <StructuredMaterial>.  In order to facilitate complicated arrangements, each 
 * MaterialPrimitive is assigned a "layer" using <set_layer> which is used to determine
 * which material value should be used if multiple MaterialPrimitives overlap at a point
 * in space. The lower the layer, the higher priority of the MaterialPrimitive.
 */
class MaterialPrimitive {
	private:
		int _layer;

	public:
		//- Constructor: defined to avoid compiler warnings
		MaterialPrimitive();

		//- Destructor: defined to avoid compiler warnings
		virtual ~MaterialPrimitive(){};

		/* Check if a given point lies within this MaterialPrimitive.
		 * @x the real space x coordinate
		 * @y the real space y coordinate
		 *
		 * The implementing class must define when a point (x,y) is within the material primitve.
		 *
		 * @return true if (x,y) falls within the material primitive. False otherwise.
		 */
		virtual bool contains_point(double x, double y) = 0;
		
		/* Check if a given point lies within the boudning box of this MaterialPrimitive.
		 * @x the real space x coordinate
		 * @y the real space y coordinate
		 *
		 * The implementing class must define when a point (x,y) is within the material primitve's
		 * bounding box.
		 *
		 * @return true if (x,y) falls within the material primitive's bounding box. False otherwise.
		 */
		virtual bool bbox_contains_point(double x, double y) = 0;

		/* get the complex material value of the MaterialPrimitive at position (x,y).
		 * @x the real space x coordinate
		 * @y the real space y coordinate
		 *
		 * Warning: some inheriting classes may assume that <contains_point> was called prior.
		 *
		 * @return the complex material value at real space position (x,y)
		 */
		virtual std::complex<double> get_material(double x, double y) = 0;
		
		virtual double get_cell_overlap(GridCell& cell) = 0;

		/* Set the layer of the primitive.
		 * 
		 * If multiple MaterialPrimitives overlap a certain region in space, the material of 
		 * the MaterialPrimitive with the lowest layer should be used.
		 */
		void set_layer(int layer);

		/* Get the layer.
		 * @return the layer.
		 */
		int get_layer() const;

		bool operator<(const MaterialPrimitive& rhs);
		
};

/* A Circle primitive.
 *
 * A circle is specified by the location of its center and its radius.  
 */
class Circle : public MaterialPrimitive {
	
	private:
		double _x0,
			   _y0,
			   _r;

		std::complex<double> _mat;

	public:
		/* Constructor
		 * @x0 the x position of the circle's center (real space)
		 * @y0 the y position of the circle's center (real space)
		 * @r the radius of the circle
		 */
		Circle(double x0, double y0, double r);

		//- Destructor
		~Circle();

		/* Determine whether a point in real space is contained within the Circle
		 * @x the x coordinate (real space)
		 * @y the y coordinate (real space)
		 * @return true if the point (x,y) is contained within the circle. False otherwise.
		 */
		bool contains_point(double x, double y);
		
		bool bbox_contains_point(double x, double y);

		/* Get the circle's material value.
		 * 
		 * Note: This does not check if (x,y) is contained in the circle.  Use <contains_point>
		 * first if that functionality is needed.
		 *
		 * @return the complex material value of the circle.
		 */
		std::complex<double> get_material(double x, double y);
		double get_cell_overlap(GridCell& cell);

		/* Set the complex material value of the Circle.
		 * @mat the complex material value
		 */
		void set_material(std::complex<double> mat);

		void set_position(double x0, double y0);
		void set_radius(double r);

		double get_x0();
		double get_y0();
		double get_r();
};

/* A Rectangle primitive.
 * 
 * A Rectangle is defined by the position of its center and its width and height.
 */
class Rectangle : public MaterialPrimitive {
	
	private:
		double _x0,
			   _y0,
			   _width,
			   _height;

        Polygon_2D _poly_rep;

		std::complex<double> _mat;

	public:
		/* Constructor
		 * @x0 the real space x position of the center of the rectangle 
		 * @y0 the real space y position of the center of the rectangle
		 * @width the width of the rectangle
		 * @height the height of the rectangle
		 */
		Rectangle(double x0, double y0, double width, double height);

		//- Destructor
		~Rectangle();

		/* Determine whether a point in real space is contained within the Rectangle 
		 * @x the x coordinate (real space)
		 * @y the y coordinate (real space)
		 * @return true if the point (x,y) is contained within the Rectangle. False otherwise.
		 */	
		bool contains_point(double x, double y);

		bool bbox_contains_point(double x, double y);
		
		/* Get the Rectangle's material value.
		 * 
		 * Note: This does not check if (x,y) is contained in the Rectangle.  Use 
		 * <contains_point> first if that functionality is needed.
		 *
		 * @return the complex material value of the Rectangle.
		 */
		std::complex<double> get_material(double x, double y);
		double get_cell_overlap(GridCell& cell);

		/* Set the complex material value of the Rectangle.
		 * @mat the complex material value
		 */
		void set_material(std::complex<double> mat);

		void set_width(double w);
		void set_height(double h);
		void set_position(double x0, double y0);
};

/* A solid Polygon primitive.
 *
 * A Polygon is defined by a list of points specified in clockwise or counterclockwise order.
 * Both concave and convex polygons are supported. The Polygon is intended to be a flexible
 * primitive that can handle both simple and complicated geometry.
 */
class Polygon : public MaterialPrimitive {
	private:
		std::complex<double> _mat;

		Polygon_2D _verts;
        BBox _bbox;
        

	public:
		//- Default Constructor
		Polygon();

		/* Constructor
		 * @x list of x positions of polygon vertices
		 * @y list of y positions of polygon vertices
		 * @n number of elements in x and y
		 */
		Polygon(double* x, double* y, int n);

		//- Destructor
		~Polygon();
		
		/* Add a single vertice to the Polygon
		 * @x the x position of the point to be added
		 * @y the y position of the point to be added
		 *
		 * Points are added to the end of an internally-stored list of points.  In order for 
		 * functions such as <contains_point> to work properly, points must be added in 
		 * either clockwise or counterclockwise order.
		 */
		void add_point(double x, double y);

		/* Add a single vertice to the Polygon
		 * @x the x coordinates of the vertices to be added
		 * @x the y coordinates of the vertices to be added
		 * @n the number of points to be added
		 *
		 * The list of vertices are appended to the end of an internally maintained list.
		 * In order for functions such as <contains_point> to work properly, points must be 
		 * added in either clockwise or counterclockwise order.
		 */
		void add_points(double* x, double* y, int n);

		/* Modify an existing point in the polygon
		 * @x the new x coordinate of the point
		 * @y the new y coordinate of the point
		 * @index the index of the polygon to be modified
		 */
		void set_point(double x, double y, int index);

		/* Set the points of the Polygon
		 * @x the x coordinates of the points
		 * @y the y coordinates of the points
		 * @n the number of points
		 *
		 * The current list of vertices is cleared and the supplied points are added.
		 * In order for functions such as <contains_point> to work properly, points must be 
		 * provided in either clockwise or counterclockwise order.
		 */	
		void set_points(double* x, double* y, int n);
		
		/* Determine whether a point in real space is contained within the Polygon 
		 * @x the x coordinate (real space)
		 * @y the y coordinate (real space)
		 * @return true if the point (x,y) is contained within the Polygon. False otherwise.
		 */
		bool contains_point(double x, double y);

		bool bbox_contains_point(double x, double y);

		/* Get the Polygon's material value.
		 * 
		 * Note: This does not check if (x,y) is contained in the Polygon.  Use 
		 * <contains_point> first if that functionality is needed.
		 *
		 * @return the complex material value of the Polygon.
		 */
		std::complex<double> get_material(double x, double y);

		double get_cell_overlap(GridCell& cell);

		/* Set the complex material value of the Polygon.
		 * @mat the complex material value
		 */
		void set_material(std::complex<double> mat);

};

/* A flexible <Material> which consists of layerd <MaterialPrimitives>.
 *
 * A StructuredMaterial consists of one or more MaterialPrimitives defined by the user 
 * which are arranged within the simulation region.  
 */
class StructuredMaterial2D : public Material2D {
	private:
		std::list<MaterialPrimitive*> _primitives;

        std::complex<double> _value;

		double _w,
			   _h,
			   _dx,
			   _dy;
	public:

		/* Constructor
		 * @w the width of the simulation region
		 * @h the height of the simulation region
		 * @dx the horizontal grid spacing of the simulation region
		 * @dy the vertical grid spacing of the simulation region
		 *
		 * The width, height, and grid spacing must be the same as those supplied when creating
		 * the corresponding FDFD object.  This is essential to mapping from real space to 
		 * array indexing when constructing the system matrix.
		 */
		StructuredMaterial2D(double w, double h, double dx, double dy);

		//- Destructor
		~StructuredMaterial2D();
		
		/* Add a primitive object to the Material.
		 * @prim the primitive to add.
		 *
		 * References to primitives are stored in an internal vector.  Working with references
		 * are advantageous as it allows the user to modify the geometry with minimal fuss
		 * between simulations.  This, however, necessitates that the corresponding 
		 * <MaterialPrimitive> objects not go out of scope while the StructuredMaterial is
		 * still in use.
		 */
		void add_primitive(MaterialPrimitive* prim);
        void add_primitives(std::list<MaterialPrimitive*> primitives);

		/* Get the complex material value at an indexed position.
		 * @x the x index (column) of the material value
		 * @y the y index (row) of the material value
		 * @return the complex material value at (x,y).  If no MaterialPrimitive exists at (x,y), 1.0 is returned.
		 */
		std::complex<double> get_value(double x, double y);

        void get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2);

        /* Get the list of primitives belonging to this StructuredMaterial
         * @return The std::list<MaterialPrimitive*> containing the constituent
         * MaterialPrimitives
         */
        std::list<MaterialPrimitive*> get_primitives();

};

/* A 2D material distribution defined by a single constant value.
 *
 * Use this for uniform materials.
 */
class ConstantMaterial2D : public Material2D {
        private:
            std::complex<double> _value;
	
		public:
            ConstantMaterial2D(std::complex<double> value);

			/* Query the material value at a point in real space.
             *
             * This will always return the same value
             *
			 * @x The x index of the query
			 * @y The y index of the query
			 * @return the complex material
			 */
			std::complex<double> get_value(double x, double y);

            /* Get a block of values.
             *
             * This just fills the provided array with a single value
             */
            void get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2);

            /* Set the complex material value.
             * @val the complex material value
             */
            void set_material(std::complex<double> val);

            /* Get the complex material value.
             *
             * This function is redundant.
             *
             * @return the complex material value.
             */
            std::complex<double> get_material();
}; // ConstantMaterial2D

/* Material class which provides the foundation for defining the system materials/structure.
 *
 * A Material must satisfy perform one function: given a spatial index, a complex 
 * material value is returned.  This is accomplished by extending the Material class and
 * implementing the <get_value> function.
 */
class Material3D {
	
		public:
			/* Query the material value at a point in real space.
			 * @x The x index of the query
			 * @y The y index of the query
			 *
			 * The structure of the electromagnetic system being solved is ultimately defined
			 * in terms of spatially-dependent materials. The material is defined on a 
			 * spatial grid which is directly compatible with finite differences.
			 * See <GridMaterial> and <StructuredMaterial> for specific implementations.
			 *
			 * @return the complex material at position (x,y).
			 */
			virtual std::complex<double> get_value(double k, double j, double i) = 0;

            /* Get a block of values.
             */
            virtual void get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2, 
                                    int i1, int i2, double sx, double sy, double sz) = 0;
			virtual ~Material3D() {};
};

/* A 3D material distribution defined by a single constant value.
 *
 * Use this for uniform materials.
 */
class ConstantMaterial3D : public Material3D {
        private:
            std::complex<double> _value;
	
		public:
            ConstantMaterial3D(std::complex<double> value);

			/* Query the material value at a point in real space.
             *
             * This will always return the same value
             *
			 * @x The x index of the query
			 * @y The y index of the query
			 * @return the complex material
			 */
			std::complex<double> get_value(double k, double j, double i);

            /* Get a block of values.
             *
             * This just fills the provided array with a single value
             */
            void get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2,
                            int i1, int i2, double sx, double sy, double sz);

            /* Set the complex material value.
             * @val the complex material value
             */
            void set_material(std::complex<double> val);

            /* Get the complex material value.
             *
             * This function is redundant.
             *
             * @return the complex material value.
             */
            std::complex<double> get_material();
}; // ConstantMaterial3D

/* Define a 3D planar stack structure.
 *
 * This class is essentially an extension of the structured3DMaterial to three
 * dimensions. It is built up of StructuredMaterials which have a defined position
 * and thickness in the z-direction. This allows the used to define 3D grid-smoothed
 * structures that have a slab-like construction (which is most common in the micro-
 * and nanoscale worlds).
 */
class StructuredMaterial3D : public Material3D {
	private:
		std::list<MaterialPrimitive*> _primitives;
        std::list<StructuredMaterial2D*> _layers;
        std::list<double> _zs;

		double _X,
			   _Y,
			   _Z,
			   _dx,
			   _dy,
               _dz,
               _background;

	public:

		/* Constructor
		 * @X the width of the simulation region in x
		 * @Y the width of the simulation region in y
		 * @Z the width of the simulation region in z
		 * @dx the x grid spacing of the simulation region
		 * @dy the y grid spacing of the simulation region
		 * @dz the z grid spacing of the simulation region
		 *
		 * The width, height, and grid spacing must be the same as those supplied when creating
		 * the corresponding FDFD object.  This is essential to mapping from real space to 
		 * array indexing when constructing the system matrix.
		 */
		StructuredMaterial3D(double X, double Y, double Z, double dx, double dy, double dz);

		//- Destructor
		~StructuredMaterial3D();
		
		/* Add a primitive object to the Material.
		 * @prim the primitive to add.
		 * @z1 the lower z bound of the primitive
		 * @z2 the upper z bound of the primitive
		 *
		 * References to primitives are stored in an internal vector.  Working with references
		 * are advantageous as it allows the user to modify the geometry with minimal fuss
		 * between simulations.  This, however, necessitates that the corresponding 
		 * <MaterialPrimitive> objects not go out of scope while the StructuredMaterial is
		 * still in use.
		 */
		void add_primitive(MaterialPrimitive* prim, double z1, double z2);

		/* Get the complex material value at an indexed position.
		 * @x the x index (column) of the material value
		 * @y the y index (row) of the material value
		 * @return the complex material value at (x,y).  If no MaterialPrimitive exists at (x,y), 1.0 is returned.
		 */
		std::complex<double> get_value(double k, double j, double i);

        void get_values(ArrayXcd& grid, int k1, int k2, 
                                        int j1, int j2, 
                                        int i1, int i2, 
                                        double sx=0, double sy=0, double sz=0);

};

}; // grid namespace

#endif
