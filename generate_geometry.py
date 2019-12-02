import os
import sys
import numpy as np
import random as rd
import math
from matplotlib import pyplot as plt

# define the geometry here
# 0 represents tissue, 0.5 represents skull, 1 2 3 ... n represents the number of the vessel
x_max = 60	# unit: mm
z_max = 30 	# unit: mm
spatial_resolution = 0.01	# unit: mm
h_space = 0.25	# unit: mm
geometry= np.zeros([int(x_max/spatial_resolution), int(z_max/spatial_resolution)])
vessel_radius = 0.05 	# unit：mm, note the vessel_radius here should be consistent with that in configuration
skull_thickness = 1

# generate the basic part of gemotry
geometry_part = np.zeros([int(2*h_space/spatial_resolution), int(z_max/spatial_resolution)])
vessel_col_num = math.floor( (x_max)/h_space -1) 	# number of vessels for each row along the x-axis 
vessel_row_num = math.floor( (z_max-skull_thickness)/h_space -1)	# number of vessels for each column along the z-axis 
print(vessel_col_num,vessel_row_num)


[X, Z] = np.mgrid[0:2*h_space/spatial_resolution, 0:2*h_space/spatial_resolution]
mask = (X- 1*h_space/spatial_resolution)**2 + (Z- (1*h_space)/spatial_resolution)**2 <= (vessel_radius/spatial_resolution)**2
X_index = np.reshape(X[mask], (np.sum(mask),1)).astype('int')
Z_index = np.reshape(Z[mask], (np.sum(mask),1)).astype('int')
print(X_index)
print(Z_index)
for i in range(vessel_row_num):
	for j in range(vessel_col_num):
		print(i,j)
		geometry[int(j*h_space/spatial_resolution)+X_index,int(skull_thickness/spatial_resolution+i*h_space/spatial_resolution)+Z_index] = (i+1-1)*vessel_col_num + (j+1)
geometry[:, 0: np.int(skull_thickness/spatial_resolution)] = 0.5
config = {'x_max':x_max, 'z_max':z_max, 'spatial_resolution':spatial_resolution, 'h_space':h_space, 'vessel_radius':vessel_radius, 'skull_thickness':skull_thickness, 'geometry':geometry}
np.save('output/config_skull_2.npy',config)

plt.imshow(geometry[:, 0:int(30/spatial_resolution)].swapaxes(0,1),cmap = plt.cm.hot)
plt.colorbar()
plt.show()
