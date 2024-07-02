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
import _algorithm_class as _alg
import _conductance_funcs as _cf
import _invasion_funcs as _if
#end

np.random.seed(13)

resolution = 5.345e-6

#Flowrate function
def Rate_calc(network, ph, inlet, outlet, conductance):
    St_p = op.algorithms.StokesFlow(network=network, phase=ph)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=inlet, values=1) #P_inlet = 1
    St_p.set_value_BC(pores=outlet, values=0) #P_outlet = 1
    St_p.run()
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val


#Reading netwrork data
ws = op.Workspace()
testName_h = 'Berea_G_0.pnm' # 'Berea_G_0.pnm' or 'Berea.pnm'
proj_h = ws.load_project(filename=testName_h)
pn = proj_h.network
Np = pn.Np
Nt = pn.Nt

#Properties extractend from Valvatne - Blunt (2004) Table 1
tension = 30e-3 #N/m
visc = 1e-3 #Pa/s

#Setting type for pore diameter and K_array
label_d_type = 'equivalent'
print('Using the pore ' + label_d_type + ' diameter')
print('----------------------------------------')

#Calculating conduit length
#---------start--------------

#D usado en conductance, pero requiero antes para setear L_min
pn['pore.diameter'] = pn['pore.' + label_d_type +'_diameter'] #Check line 46
pn['throat.diameter'] = pn['throat.equivalent_diameter']
pn['throat.spacing'] = pn['throat.total_length']

#Setting minimum length
#D usado en conductance, pero requiero antes para setear L_min

#VERSION 1 , SOLO RESOLUTION
L_min = resolution


L_1 = _cf.conduit_length_spheres_cylinders_improved(pn,
                                        pore_diameter = 'pore.diameter',
                                        throat_diameter = 'throat.diameter',
                                        throat_spacing = 'throat.spacing',
                                        L_min = L_min,
                                        check_boundary = True)

#VERSION 2, COMPARANDO ENTRE LA LONGITUD DE CONDUITE Y RESOLUTION
L_min_tl = 0.05 *  pn['throat.spacing']
L_min = np.maximum(L_min_tl , np.ones_like(L_min_tl) * resolution )
L_2 = _cf.conduit_length_spheres_cylinders_improved(pn,
                                        pore_diameter = 'pore.diameter',
                                        throat_diameter = 'throat.diameter',
                                        throat_spacing = 'throat.spacing',
                                        L_min = L_min,
                                        check_boundary = True)

#Identify boundary
mask_tBC = pn['throat.boundary']
print(np.sum( (L_1 != L_2)[~mask_tBC] ))
print(L_1[~mask_tBC])
print(np.abs( (L_1 - L_2)[~mask_tBC] ) )
value = np.abs( (L_1 - L_2)[~mask_tBC] )
row = np.sum(~mask_tBC)
value_array = []
perc = []
for i in range(row):
    for j in range(3):
        if value[i,j] != 0:
            value_array.append(value[i,j])
            perc.append(value[i,j] / resolution)
value_array = np.array(value_array)
perc = np.array(perc)
print(value_array)
print(perc)
print(np.mean(perc))
