import openpnm as op
import porespy as ps
import numpy as np
import imageio as io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

timeZero = time.time()

im = np.fromfile('Berea.raw', dtype='bool', sep="").reshape(400,400,400)

im = ~np.array(im, dtype=bool)
plt.imshow(im[3,:,:],cmap='gray')
plt.show()

porosity = ps.metrics.porosity(im)
print(porosity)

snow_output = ps.networks.snow(im,
                   voxel_size=5.435e-6)


prj = op.io.PoreSpy.import_data(snow_output)
pn = prj.network
print('L = ', ((5.435e-6*400)**3*(1.0-porosity))/pn['pore.surface_area'].sum())
print('Elapsed time during network extraction (min)', (time.time()-timeZero)/60)

vol = (5.435e-6*400)**3
porosity1 = pn['pore.region_volume'].sum()/vol


print('Number of pores before trimming: ', pn.Np)
h = pn.check_network_health()
op.topotools.trim(network=pn, pores=h['trim_pores'])
print('Number of pores after trimming: ', pn.Np)

porosity2 = pn['pore.region_volume'].sum()/vol

print('porosity before and after trim', porosity1, porosity2)

pn.project.save_project(filename='Berea3')

