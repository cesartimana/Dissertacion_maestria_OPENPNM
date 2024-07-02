#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:37:17 2023

@author: cesar
"""

import porespy as ps
import numpy as np
import openpnm as op
np.random.seed(13)

resolution = 5e-7

im = ps.generators.overlapping_spheres([100, 100, 100], r=7, porosity=0.7)

snow_output = ps.networks.snow2(im,
                   voxel_size=resolution,
                   boundary_width=3,
                   accuracy='high', # high
                   legacy='yes', #'no' is default
                   parallelization=None)

pn = op.io.network_from_porespy(snow_output.network)

print(pn)
"""
A = pn['throat.cross_sectional_area']
print(A[1:10])
"""