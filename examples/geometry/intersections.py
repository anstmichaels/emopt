"""Demonstration of how shapes in EMopt can be combined using the intersection boolean
operation.

To run this script:

    $ python3 intersections.py
"""
import emopt
import matplotlib.pyplot as plt

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
