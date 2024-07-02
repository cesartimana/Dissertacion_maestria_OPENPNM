#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:21:55 2023

@author: cesar
"""

import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
from edt import edt
from porespy.tools import get_tqdm, make_contiguous, extend_slice
import scipy.ndimage as spim
from skimage.morphology import disk, ball
import imageio
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
np.random.seed(13)


#3D image

im = ps.generators.overlapping_spheres([100, 100, 3], r=7, porosity=0.7)
snow = ps.filters.snow_partitioning(im)
regions = snow.regions
#imageio.volsave("regions.tif", np.array(regions.astype("uint8")))



#--------------------------------------
#Calculando gargantas com metodo regions_to_network ppaso a paso


im = make_contiguous(regions)
struc_elem = disk if im.ndim == 2 else ball #for im_w_throats
phases = (im > 0).astype(int)
dt = np.zeros_like(phases, dtype="float32")
for i in np.unique(phases[phases.nonzero()]):
    dt += edt(phases == i)


slices = spim.find_objects(im)

Ps = np.arange(1, np.amax(im)+1)
Np = np.size(Ps)
p_area_surf = np.zeros((Np, ), dtype=int)

t_conns = []
t_perimeter = []
t_coords = []

#Previa de preparaçao de poros
i = 8
pore = i - 1
s = extend_slice(slices[pore], im.shape)
sub_im = im[s]
sub_dt = dt[s]
pore_im = sub_im == i
poro_is = pore_im*sub_im
padded_mask = np.pad(pore_im, pad_width=1, mode='constant')
pore_dt = edt(padded_mask)

s_offset = np.array([i.start for i in s])
                   
#To have true on pore and surroundings (throats and solid)
im_w_throats = spim.binary_dilation(input=pore_im, structure=struc_elem(1))
#This makes all voxels of neighbor pores that are not throat = 0
im_w_throats = im_w_throats*sub_im
#Choose all pore labels. Eliminate the zero (solid)
Pn = np.unique(im_w_throats)[1:] - 1
tho_im = im_w_throats > i #im_w_throats !=0,


p_area_surf[pore] = np.sum(pore_dt == 1) #surface area


for j in Pn:
            if j > pore:
                t_conns.append([pore, j])
                vx = np.where(im_w_throats == (j + 1))
                # The following is overwritten if accuracy is set to 'high'
                t_perimeter.append(np.sum(sub_dt[vx] < 2))
                # The following is overwritten if accuracy is set to 'high'
                #p_area_surf[pore] -= np.size(vx[0])
                t_inds = tuple([i+j for i, j in zip(vx, s_offset)])
                temp = np.where(dt[t_inds] == np.amax(dt[t_inds]))[0][0]
                t_coords.append(tuple([t_inds[k][temp] for k in range(im.ndim)]))

#-------------------------------------------







"""
#-----------------------------------------
im = make_contiguous(regions)
slices = spim.find_objects(im)
#nieghbor pores
p1 = 98 
p2 = 111
s = extend_slice(slices[p1], im.shape)
im = im[s] #Note that here I only create a matrix for p1 and p2. Normaly is sub_im = ...

vis = np.zeros(im.shape) + p1*(im == (p1+1)) + p2*(im == (p2+1)) 
imageio.volsave("vis.tif", np.array(vis.astype("uint8")))



#To plot a voxel graph
b_p1 = vis == p1
b_p2 = vis == p2
b_vis = vis > 0 #boolean vis

colors = np.empty(b_vis.shape, dtype=object)
colors[b_p1] = 'red'
colors[b_p2] = 'blue'

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(b_vis, facecolors=colors, edgecolor='k')

plt.show()
#----------------------------------------
"""

#Applying regions to network
pn_s = ps.networks.regions_to_network(regions=regions, accuracy='standard') #remember code regoins = snow.regions
pn_h = ps.networks.regions_to_network(regions=regions, accuracy='high')

area_std = pn_s['pore.surface_area']
area_high = pn_h['pore.surface_area']

dif = area_high - area_std
perc = np.round(dif / area_std * 100, 2)

print(len([x for x in dif if x>=0 ]))
print(len([x for x in dif if x<0 ]))

print(perc)


"""
print(pn['pore.surface_area'])

per = pn['throat.perimeter']
t_con = pn['throat.conns']
for i in range(len(per)):
    if per[i]<3:
        print('throat: ', i, '/ perimeter voxels: ',  per[i], '/ pores: ', t_con[i])
#print(net2)


"""
