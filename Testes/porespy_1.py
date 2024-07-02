#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:39:53 2023

@author: cesar
"""

#https://porespy.org/examples/filters/tutorials/snow_partitioning.html


import numpy as np
import porespy as ps
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation
ps.visualization.set_mpl_style()
np.random.seed(1)

im = ps.generators.overlapping_spheres([500, 500], r=10, porosity=0.5)
fig, ax = plt.subplots()
ax.imshow(im, origin='lower');
plt.plot()

snow_out = ps.filters.snow_partitioning(im, r_max=4, sigma=0.4)
print(snow_out)

fig, ax = plt.subplots(2, 2, figsize=[8, 8])
ax[0, 0].imshow(snow_out.im, origin='lower')
ax[0, 1].imshow(snow_out.dt, origin='lower')
dt_peak = snow_out.dt.copy()
peaks_dilated = binary_dilation(snow_out.peaks > 0)
dt_peak[peaks_dilated > 0] = np.nan
ax[1, 0].imshow(dt_peak, origin='lower')
ax[1, 1].imshow(ps.tools.randomize_colors(snow_out.regions), origin='lower')
ax[0, 0].set_title("Binary image");
ax[0, 1].set_title("Distance transform");
ax[1, 0].set_title("Distance transform peaks");
ax[1, 1].set_title("Segmentation");