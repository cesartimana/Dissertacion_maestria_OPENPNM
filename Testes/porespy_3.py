#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:55:29 2023

@author: cesar
"""

import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
from edt import edt

np.random.seed(13)
im = ps.generators.overlapping_spheres([100, 100], r=7, porosity=0.7)
plt.imshow(im, origin='lower', interpolation='none');

#print(ps.networks.regions_to_network.__doc__)

snow = ps.filters.snow_partitioning(im)
plt.imshow(snow.regions/im, origin='lower', interpolation='none');

plt.show()

net1 = ps.networks.regions_to_network(regions=snow.regions)

"""
for item in net1.keys():
    print(item)
"""
    
snow2 = ps.filters.snow_partitioning(~im)
plt.imshow(snow2.regions, origin='lower', interpolation='none');
plt.show()

ws = snow.regions + (snow2.regions + snow.regions.max())*~im
plt.imshow(ws, origin='lower', interpolation='none');

net2 = ps.networks.regions_to_network(regions=ws, phases=im.astype(int)+1)
"""
for item in net2.keys():
    print(item)

"""    
    
print(net1['pore.phase'])
print(net2['pore.phase'])