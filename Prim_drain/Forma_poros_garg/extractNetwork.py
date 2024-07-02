import openpnm as op
import porespy as ps
import numpy as np
import imageio as io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import sys

sampleName = 'Berea_high'
fileName = 'Berea.raw'
resolution = 5.345e-6

timeZero = time.time()

im = np.fromfile(fileName, dtype='bool', sep="").reshape(400,400,400)
im = im[:100,:100,:100]

im = ~np.array(im, dtype=bool)
plt.imshow(im[3,:,:],cmap='gray')
plt.show()

porosity = ps.metrics.porosity(im)
print('porosity = ', porosity)


snow_output = ps.networks.snow2(im,
                   voxel_size=resolution,
                   boundary_width=3,
                   accuracy='high', # high
                   legacy='yes', #different from 'no'
                   parallelization=None)

pn = op.io.network_from_porespy(snow_output.network)
prj = pn.project

print('Elapsed time during network extraction (min)', (time.time()-timeZero)/60)

print(prj)

ws = op.Workspace()

op.io._vtk.project_to_vtk(pn.project, filename=sampleName)
ws.save_project(prj, filename=sampleName)

