import openpnm as op
import porespy as ps
import numpy as np
import imageio as io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import sys

sampleName = 'Carbonate'
fileName = 'Carbonate1.raw'
resolution = 2.85e-6 #For berea : 5.345e-6 / For carbonate: 2.85e-6
accuracy = 'high' # high or standard

timeZero = time.time()

im = np.fromfile(fileName, dtype='bool', sep="").reshape(400,400,400)

im = ~np.array(im, dtype=bool)
plt.imshow(im[3,:,:],cmap='gray')
plt.show()

porosity = ps.metrics.porosity(im)
print('porosity = ', porosity)


#---------------------------------
"""
snow = ps.filters.snow_partitioning(im)

net = ps.networks.regions_to_network(regions=snow.regions, accuracy='high',voxel_size=resolution)

pn = op.io.network_from_porespy(net)
"""
#---------------------------------



#-----------------------
snow_output = ps.networks.snow2(im,
                   voxel_size=resolution,
                   boundary_width=3,
                   accuracy=accuracy,
                   legacy='yes', #different from no
                   parallelization=None)
pn = op.io.network_from_porespy(snow_output.network)
#---------------------------------

print(pn)

prj = pn.project

print('Elapsed time during network extraction (min)', (time.time()-timeZero)/60)



ws = op.Workspace()

#op.io._vtk.project_to_vtk(pn.project, filename=sampleName)
ws.save_project(prj, filename=sampleName + accuracy)

