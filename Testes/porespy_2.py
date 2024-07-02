#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:43:19 2023

@author: cesar
"""

import matplotlib.pyplot as plt
import numpy as np
import porespy as ps
import imageio

np.random.seed(10)
im = ps.generators.blobs(shape=[50,50,50])
snow = ps.filters.snow_partitioning(im)
regions = snow.regions
fig, ax = plt.subplots(1, 1, figsize=[6, 6])
ax.imshow(regions[:,:,20], origin='lower', interpolation='none')
ax.axis(False);

imageio.volsave("regions.tif", np.array(regions.astype("uint8")))