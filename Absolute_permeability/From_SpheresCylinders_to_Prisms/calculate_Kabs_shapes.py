# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar
USANDO RED CON POROS ESFERICOS Y GARGANTAS COMO PRISMAS

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
visc = 1e-3 #Pa/s , este valor no importa. K ignora esto

#Setting type for pore diameter and K_array
label_d_type = 'equivalent'
print('SPHERES AND PRISMS')
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
#L_min_tl = 0.05 *  pn['throat.spacing']
L_min = resolution
#L_min = np.maximum(L_min_tl , np.ones_like(L_min_tl) * resolution )

L = _cf.conduit_length_spheres_cylinders_improved(pn,
                                        pore_diameter = 'pore.diameter',
                                        throat_diameter = 'throat.diameter',
                                        throat_spacing = 'throat.spacing',
                                        L_min = L_min,
                                        check_boundary = True)

pn['throat.conduit_lengths'] = L
#---------end--------------


HSF_SC = _cf.HydraulicSizeFactors_SpheresCylinders_improved(pn, check_boundary = True)
pn['throat.hydraulic_size_factors'] = HSF_SC

#Modificating cylindrical throats with prisms
conns = pn['throat.conns']
mask_tBC = pn['throat.boundary']
mask_pBC = np.isin(conns, pn.pores('boundary'))
#---------start--------------
A = pn['throat.cross_sectional_area']
G = pn['throat.shape_factor']
beta = np.sort(pn['throat.half_corner_angle'], axis = 1)
beta_dif = np.pi/2 - 2 * beta[:,1]
pol = np.array([-0.17997611,  0.57966346, -0.46275726,  1.10633925])
F = np.polyval(pol, beta_dif)
HSF_t_prism= 0.6 * F * A ** 2 * G / L[:, 1]
print(np.mean( G[~mask_tBC]))
print(3**0.5 / 36)
HSF_t_prism[mask_tBC] = np.inf
pn['throat.hydraulic_size_factors'][:, 1] = HSF_t_prism
#---------end--------------





#Calculating hydraulic conductance
phase = op.phase.Phase(network=pn)
phase['pore.viscosity'] = visc
f = op.models.physics.hydraulic_conductance.generic_hydraulic
phase.add_model(propname = 'throat.hydraulic_conductance', model = f)


#Defining boundary conditions
for axis in ['x', 'y', 'z' ]:
    inlet_pores = pn['pore.'+axis+'min']
    outlet_pores = pn['pore.' + axis + 'max']

    Q = Rate_calc(pn, phase, inlet_pores, outlet_pores, 'throat.hydraulic_conductance')
    L = 400 * resolution
    A = L**2
    K = Q[0] * L * visc / (A) #Q[0] * L * mu / (A * Delta_P), Delta_P = 1
    #print(K)
    print(f'The value of K is: {K/0.9869233e-12*1000:.2f} mD')


