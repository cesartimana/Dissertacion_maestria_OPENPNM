# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar

"""

import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import math as m

#importing functions for other file
import sys
sys.path.insert(1, '/home/cesar/OpenPNM_files/_funcs')
import _conductance_funcs as _cf
import _invasion_funcs as _if
#end

#np.random.seed(13)

#Flowrate function
def Rate_calc(network, ph, inlet, outlet, conductance):
    St_p = op.algorithms.StokesFlow(network=network, phase=ph)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=inlet, values=1) #P_inlet = 1
    St_p.set_value_BC(pores=outlet, values=0) #P_outlet = 1
    St_p.run()
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val
ws = op.Workspace()
Networks = np.arange(10)
Geometric = np.arange(10)

for net_label in Networks:
    pn = op.network.Cubic(shape=[20, 20, 20], spacing=1e-5, connectivity = 26)
    prj = pn.project
    T_trim = op.topotools.reduce_coordination(pn, z = 4)
    op.topotools.trim(pn, throats = T_trim)
    f = op.models.collections.geometry.spheres_and_cylinders
    pn.add_model_collection(f)
    pn.regenerate_models()
    Np = pn.Np
    Nt = pn.Nt
    for geo_label in Geometric:
    #Creating G, beta
        beta_min = 5 * np.pi / 180
        beta_rep = np.random.rand(Nt) * (44.9 * np.pi / 180 - 3 * beta_min / 2) + beta_min #angulo repetido
        beta_dif = np.pi / 2 - 2 * beta_rep
        beta = np.sort(np.vstack((beta_rep, beta_rep, beta_dif)).T, axis = 1)
        sum_cot = 1 / np.tan(beta_dif) + 2 / np.tan(beta_rep)
        G = 1 / ( 4 * sum_cot )
        A = pn['throat.cross_sectional_area']
        pn['throat.shape_factor'] = G
        pn['throat.half_corner_angle'] = beta
        pn['throat.prism_diameter'] = 2 * (A / sum_cot)**0.5
        ws.save_project(prj, filename='Net_' + str(int(net_label)) + '_geo_' + str(int(geo_label)) )
