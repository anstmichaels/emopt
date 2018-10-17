from __future__ import division, print_function, absolute_import
import emopt
import numpy as np
import matplotlib.pyplot as plt

polygons = emopt.io.load_gds_txt('example_layout.txt')

p1 = polygons[1][0]; p2 = polygons[2][0]

p1.material_value = 1.444**2
p1.layer = 2

p2.material_value = 3.45**2
p2.layer = 1

eps = emopt.grid.StructuredMaterial2D(5, 3, 0.02, 0.02)
eps.add_primitives([p2,p1])

Nx = int(5/0.02)
Ny = int(3/0.02)
eps_grid = eps.get_values(0,Nx,0,Ny)

f = plt.figure()
ax = f.add_subplot(111)
ax.imshow(eps_grid.real, extent=[0,5,0,3], cmap='Blues')
plt.show()


