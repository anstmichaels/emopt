"""Demonstration of how shapes in EMopt can be combined using a boolean union operation.

To run this script:

    $ python3 unions.py
"""
import emopt
import matplotlib.pyplot as plt

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
