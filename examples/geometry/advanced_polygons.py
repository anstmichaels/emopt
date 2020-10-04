"""This example demonstrates how to perform more advanced operations on polygons which are
useful for optimizations.

To run this example:

    $ python3 advanced_polygons.py
"""
import emopt
import matplotlib.pyplot as plt

# Create rectangles and then form a union
rh1 = emopt.geometry.Rectangle(0, 0, 10, 0.5, mat_val=2.0)
rh2 = emopt.geometry.Rectangle(0, 0, 5, 1.5, mat_val=2.0)

rv1 = emopt.geometry.Rectangle(0, 0, 10, 0.5, mat_val=2.0)
rv1.rotate(90.0)

rv2 = emopt.geometry.Rectangle(0, 0, 5, 1.5, mat_val=2.0)
rv2.rotate(90.0)

p = rh1.add(rh2)[0]
p = p.add(rv1)[0]
p = p.add(rv2)[0]

# Next, we will round the corners of the polygon
p.fillet(0.2, selection=(-4, 4, -3, 3))

plt.plot(p.xs, p.ys, 'b.-')
plt.axis('equal')
plt.show()

# Generate a grid and display it
X = 9.5
Y = 9.5
dx = dy = 0.05

# Need to move the polygon since the grid origin is in the bottom left
p.translate(X/2, Y/2)

grid = emopt.grid.StructuredMaterial2D(X, Y, dx, dy)
grid.add_primitive(p)
domain = emopt.misc.DomainCoordinates(0,X,0,Y,0,0,dx,dy,1)

plt.imshow(grid.get_values_in(domain).real, extent=[0,X,0,Y])
plt.show()
