"""Demonstration of how shapes in EMopt can be combined using various boolean operations.

To run this script:

    $ python3 geometry_fun.py
"""
import emopt
import matplotlib.pyplot as plt

#####################################################################################
# Example 1: Adding shapes (union)
#####################################################################################
r = emopt.geometry.Rectangle(0, 0, 10, 1.0, mat_val=2.0)
c = emopt.geometry.Circle(0, 0, 1.5, mat_val=2.0)
union = r.add(c)

# Adding two shapes generates a list of polygons.
# There should be one polygon in this union
print(f'Number of polygons in union = {len(union)}')

# Show the result
poly = union[0]

plt.plot(r.xs, r.ys, 'b.-')
plt.plot(c.xs, c.ys, 'c.-')
plt.plot(poly.xs, poly.ys, 'r.--')
plt.axis('equal')
plt.show()

# Generate a grid and display it
X = 12
Y = 4
dx = dy = 0.05

# Need to move the polygon since the grid origin is in the bottom left
poly.translate(X/2, Y/2)

grid = emopt.grid.StructuredMaterial2D(X, Y, dx, dy)
grid.add_primitive(poly)
domain = emopt.misc.DomainCoordinates(0,X,0,Y,0,0,dx,dy,1)

plt.imshow(grid.get_values_in(domain).real, extent=[0,X,0,Y])
plt.show()

#####################################################################################
# Example 2: Subtracting shapes
#####################################################################################
c1 = emopt.geometry.Circle(0, 0, 2.0, mat_val=2.0)
c2 = emopt.geometry.Circle(1.0, 0, 1.5, mat_val=2.0)
diff = c1.subtract(c2)

# Subtracting these two circles should produce only 1 polygon
print(f'Number of polygons in difference = {len(diff)}')

# Show the result
plt.plot(c1.xs, c1.ys, 'b.-')
plt.plot(c2.xs, c2.ys, 'c.-')
plt.plot(diff[0].xs, diff[0].ys, 'r.--')
plt.axis('equal')
plt.show()

# Generate a grid and display it
X = 5
Y = 5
dx = dy = 0.05
grid = emopt.grid.StructuredMaterial2D(X, Y, dx, dy)

# Need to move the polygon since the grid origin is in the bottom left
diff[0].translate(X/2, Y/2)

grid.add_primitives(diff)
domain = emopt.misc.DomainCoordinates(0,X,0,Y,0,0,dx,dy,1)

plt.imshow(grid.get_values_in(domain).real, extent=[0,X,0,Y])
plt.show()

#####################################################################################
# Example 3: Intersecting shapes
#####################################################################################
c = emopt.geometry.Circle(0, 0, 2.0, mat_val=2.0)
r = emopt.geometry.Rectangle(0, 0, 5.0, 1.5, mat_val=2.0)
inters = c.intersect(r)

# Subtracting these two circles should produce only 1 polygon
print(f'Number of polygons in union = {len(inters)}')

# Show the result
plt.plot(c.xs, c.ys, 'b.-')
plt.plot(r.xs, r.ys, 'c.-')
plt.plot(inters[0].xs, inters[0].ys, 'r.--')
plt.axis('equal')
plt.show()

# Generate a grid and display it
X = 5
Y = 5
dx = dy = 0.05
grid = emopt.grid.StructuredMaterial2D(X, Y, dx, dy)

# Need to move the polygon since the grid origin is in the bottom left
inters[0].translate(X/2, Y/2)

grid.add_primitives(inters)
domain = emopt.misc.DomainCoordinates(0,X,0,Y,0,0,dx,dy,1)

plt.imshow(grid.get_values_in(domain).real, extent=[0,X,0,Y])
plt.show()
