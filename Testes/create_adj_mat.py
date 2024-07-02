#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:48:52 2023

@author: cesar
"""

import numpy as np
import openpnm as op
from scipy.sparse import csgraph as csg
np.random.seed(5)
pn = op.network.Cubic(shape=[3, 3, 1])
weights = np.random.rand(pn.num_throats(), ) < 0.5
am = pn.create_adjacency_matrix(weights=weights, fmt='csr',triu=False, drop_zeros=True)
clusters = csg.connected_components(am, directed=False)[1]
print(clusters)
print(pn['throat.conns'])
print(am)