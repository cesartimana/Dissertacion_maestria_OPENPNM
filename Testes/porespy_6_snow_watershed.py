#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:11:36 2023

@author: cesar
"""

import logging
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

logger = logging.getLogger(__name__)

#Analisign snow function snow2, whoo does watershed + regions to network


#2D image

im = ps.generators.overlapping_spheres([100, 100], r=7, porosity=0.7)
#plt.imshow(im, origin='lower', interpolation='none'); #2D graphic

phases = im.astype(int)

#Extrae numeros relacionados a las fases en la imagen. Por ahora es 1 y 0
vals = np.unique(phases) 
vals = vals[vals > 0]


regions = None

for i in vals:
        logger.info(f"Processing phase {i}...")
        phase = phases == i
        pk = None 
        snow = ps.filters.snow_partitioning(im=phase,peaks=pk)
        # ant_reg = snow.regions # las siguientes lineas no cambian en nada a regions
        if regions is None:
            regions = np.zeros_like(snow.regions, dtype=int)
        # Note: Using snow.regions > 0 here instead of phase is needed to
        # handle a bug in snow_partitioning, see issue #169 and #430
        regions += snow.regions + regions.max()*(snow.regions > 0)

