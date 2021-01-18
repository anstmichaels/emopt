"""Demonstrate how to setup and modify a ParameterizedPolygon.

To run this example:
    $ python3 parameterized_polygon.py
"""
import emopt
import matplotlib.pyplot as plt

# First example: Select and parameterize the top and bottom parts of a circle. Based on
# these selections, the polygon is modified
circ = emopt.geometry.Circle(0, 0, 1.0)
poly = emopt.geometry.ParameterizedPolygon(circ.xs, circ.ys)

# Parameterize by defining a bounding box in the form (xmin, xmax, ymin, ymax). Points in this
# box will be selected and the x and y coordinates will become parameters
poly.parameterize((-2, 2, 0.1, 2), True, True, 'top')
poly.parameterize((-2, 2, -2, -0.5), True, True, 'bot')

# Plot the parameterization
f = plt.figure()
ax = f.add_subplot(111)
ax.plot(poly.xs, poly.ys, 'c.-', markersize=4)
ax.plot(poly.xs[poly.xparam['top']], poly.ys[poly.xparam['top']],
        'r.', markersize=12, alpha=0.25)
ax.plot(poly.xs[poly.yparam['bot']], poly.ys[poly.yparam['bot']],
        'b.', markersize=12, alpha=0.25)
plt.axis('equal')
plt.show()

# Update the points using the parameterization
xs = poly.xs
ys = poly.ys

xs[poly.xparam['top']] *= 1.25
ys[poly.yparam['top']] *= 1.25
xs[poly.xparam['bot']] *= 0.75
ys[poly.yparam['bot']] *= 0.75
poly.set_points(xs, ys)

f = plt.figure()
ax = f.add_subplot(111)
ax.plot(poly.xs, poly.ys, 'c.-', markersize=4)
ax.plot(poly.xs[poly.xparam['top']], poly.ys[poly.xparam['top']],
        'r.', markersize=12, alpha=0.25)
ax.plot(poly.xs[poly.yparam['bot']], poly.ys[poly.yparam['bot']],
        'b.', markersize=12, alpha=0.25)
plt.axis('equal')
plt.show()
