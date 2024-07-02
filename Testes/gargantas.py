# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:21:16 2023

@author: cesar
"""

import porespy as ps
from porespy.tools import get_tqdm, make_contiguous, extend_slice
import matplotlib.pyplot as plt
import numpy as np
from edt import edt
from skimage.morphology import disk, ball
import scipy.ndimage as spim
import openpnm as op

np.random.seed(13)

"""
#Grafica 2D
im = ps.generators.overlapping_spheres([100, 100], r=7, porosity=0.7)
snow = ps.filters.snow_partitioning(im)
"""


#Grafica 3D
im = ps.generators.overlapping_spheres([100, 100, 100], r=7, porosity=0.7)
#plt.imshow(im, origin='lower', interpolation='none'); #Grafica 2D

snow = ps.filters.snow_partitioning(im)
#plt.imshow(snow.regions/im, origin='lower', interpolation='none'); #Grafica 2D


#Tratamiento paso por paso para definir topologia de poros y gargantas
regions = snow.regions

im = make_contiguous(regions)
struc_elem = disk if im.ndim == 2 else ball
phases = (im > 0).astype(int)
dt = np.zeros_like(phases, dtype="float32")  # since edt returns float32
for i in np.unique(phases[phases.nonzero()]):
    dt += edt(phases == i)



#Encontrar los recuadros donde estan cada poro
slices = spim.find_objects(im)

#Creando algunos vectores
t_perimeter = []
t_conns = []


#Pretratamiento para gargantas. 
#Recordar que el poro 0 tiene cubos con el numero 1, y asi en adelante.
i = 128 + 1  #Cambiar el numero izquerdo. El +1 es para localizar las casillas
pore = i - 1
s = extend_slice(slices[pore], im.shape)
sub_im = im[s]
sub_dt = dt[s]
pore_im = sub_im == i #boolean del poro analizado
im_w_throats = spim.binary_dilation(input=pore_im, structure=struc_elem(1))
im_w_throats = im_w_throats*sub_im
Pn = np.unique(im_w_throats)[1:] - 1



p2 = 127 + 1 #Cambiar el numero izquerdo. El +1 es para localizar las casillas
b_neig = sub_im == p2 #boolean del poro vecino
b_throat = im_w_throats == p2 # boolean de la garganta
b_neig = (b_neig == True) & (b_throat == False)#retirando los voxel garganta del poro vecino
b_vox =  pore_im | b_neig | b_throat #boolean unido de los dos poros

# set the colors of each object
colors = np.empty(b_vox.shape, dtype=object)
colors[pore_im] = 'red'
colors[b_neig] = 'blue'
colors[b_throat] = 'green'

# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(b_vox, facecolors=colors, edgecolor='k')
plt.show()


#Plot solid only
b_solid = sub_im == 0 #boolean del solido
color_s = np.empty(b_solid.shape, dtype=object)
color_s[b_solid] = 'red'
ax2 = plt.figure().add_subplot(projection='3d')
ax2.voxels(b_solid, facecolors=color_s, edgecolor='k')
plt.show()

for j in Pn:
    t_conns.append([pore, j])
    vx = np.where(im_w_throats == (j + 1))
    t_perimeter.append(np.sum(sub_dt[vx] < 2))

"""
#Ahora, para las gargantas vecinas, solo se analiza si el poro es menor que el poro vecino
for j in Pn:
    if j > pore:
        t_conns.append([pore, j])
        vx = np.where(im_w_throats == (j + 1))
        t_perimeter.append(np.sum(sub_dt[vx] < 2))
"""        


"""
#No llenar nada aqui


#Codigo para convertir en red de poros y gargantas
net = ps.networks.regions_to_network(regions=snow.regions)

#Para exportar red de porespy
pn = op.io.network_from_porespy(net)
prj = pn.project
ws = op.Workspace()
ws.save_project(prj, filename='red_poros')
"""
"""
#Codigo para llamar a una red y mostrarla 

ws = op.Workspace()

testName = 'red_poros.pnm'

proj = ws.load_project(filename=testName)

pn = proj.network

print(pn)

op.io._vtk.project_to_vtk(pn.project, filename='Paraview_net')
"""