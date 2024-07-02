#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:22:05 2023

@author: cesar
"""

import numpy as np
import openpnm as op
op.visualization.set_mpl_style()
np.random.seed(10)
np.set_printoptions(precision=5)

pn = op.network.Cubic(shape=[3, 3, 3], spacing=1e-6)
pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
pn.regenerate_models()

phase = op.phase.Water(network=pn)
phase['pore.viscosity']=1.0
phase.add_model_collection(op.models.collections.physics.basic)
phase.regenerate_models()

inlet = pn.pores('left')
outlet = pn.pores('right')
flow = op.algorithms.StokesFlow(network=pn, phase=phase)
flow.set_value_BC(pores=inlet, values=1)
flow.set_value_BC(pores=outlet, values=0)
flow.run()
phase.update(flow.soln)

Qi = flow.rate(pores=inlet, mode='group')[0] #Without [0], it shows an array of 1 element. We dont want array format
Qf = flow.rate(pores=outlet, mode='single')# for flow in each pore. 'group' is for total flow