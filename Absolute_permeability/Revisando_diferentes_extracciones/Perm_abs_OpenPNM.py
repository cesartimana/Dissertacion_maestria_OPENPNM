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
testName_h = 'Berea_V7.pnm' # 'Berea_G_0.pnm' or 'Berea.pnm', 'Berea_V1.pnm'
proj_h = ws.load_project(filename=testName_h)
pn = proj_h.network

#Eliminating isolated pores
h = op.utils.check_network_health(pn)
op.topotools.trim(network=pn, pores=h['disconnected_pores'])

#Setting boundaries
boundary_t = pn.find_neighbor_throats(pores=pn['pore.boundary'])
pn['pore.internal'] = ~pn['pore.boundary']
pn['throat.boundary'] = False
pn['throat.boundary'][boundary_t] = True
pn['throat.internal'] = ~pn['throat.boundary']

Np = pn.Np
Nt = pn.Nt

#Defining boundary conditions
axis = 'z'
inlet_pores = pn['pore.'+axis+'min']
index_inlet = pn.pores(axis + 'min')
outlet_pores = pn['pore.' + axis + 'max']
index_outlet = pn.pores(axis + 'max')
pn['pore.BC'] = inlet_pores | outlet_pores

BC_t = pn.find_neighbor_throats(pores=pn['pore.BC'])
pn['throat.BC'] = False
pn['throat.BC'][BC_t] = True

#Properties extractend from Valvatne - Blunt (2004) Table 1
visc = 1e-3 #Pa/s  #Esto se simplifica al calcular K, asi que no depende
#D usado en conductance, pero requiero antes para setear L_min
pn['pore.diameter'] = pn['pore.extended_diameter'] #Change between extended and equivalent
pn['throat.diameter'] = pn['throat.equivalent_diameter']
D1, Dt, D2 = pn.get_conduit_data('diameter').T
conns = pn['throat.conns']

#conduit_length
pn['throat.spacing'] = pn['throat.total_length']

#Modifying diameters to not have problems
#---------start--------------
#diameter of boundary pores and throats
mask_tBC = pn['throat.boundary']
mask_pBC = np.isin(conns, pn.pores('boundary'))
for i in range(Nt):
    if mask_tBC[i]:
        D = D1[i]*~mask_pBC[i, 0] + D2[i]*~mask_pBC[i, 1]
        pn['throat.spacing'][i] = 2*D * 1.05
        pn['pore.diameter'][conns[i, 0]] = D
        pn['pore.diameter'][conns[i, 1]] = D
        pn['throat.diameter'][i] = 0.99*D  #choose arbitrary
#Solving problems with conduits with Dt > = Dp
D1, Dt, D2 = pn.get_conduit_data('diameter').T
mask = (D1 <= Dt) | (D2 <= Dt)
print(f'Number of internal throats with Dt > = Dp: {np.sum(mask):.2f}' )
Dt[mask] = np.minimum(D1, D2)[mask] * 0.99 #choose arbitrary
pn['throat.diameter'] = Dt
#Solving problems with , supposely, a pore inscribed into another,
#equivalent = 3
L_ctc =pn['throat.spacing']
D1, Dt, D2 = pn.get_conduit_data('diameter').T
mask_L = np.abs(D1/2 - D2/2) > L_ctc
print(f'Number of conduits with a pore suposely inside another:   {np.sum(mask_L):.2f} ')
pn['throat.spacing'][mask_L] = (np.abs(D1/2 - D2/2)*1.001)[mask_L]
#---------end--------------
print( np.mean( pn['throat.spacing'][pn['throat.boundary']] ) )
print( np.mean( pn['throat.diameter'][pn['throat.boundary']] ) )
print( np.mean( pn['pore.diameter'][pn['pore.boundary']] ) )
#Calculationg hydraulic conductance
#---------start--------------

f1 = op.models.geometry.hydraulic_size_factors.spheres_and_cylinders
pn.add_model(propname = 'throat.hydraulic_size_factors',
              model = f1)

#Setting BC high for just one pore
for i in range(Nt):
    if mask_tBC[i]:
        pn['throat.hydraulic_size_factors'][i, 0] = np.inf
        pn['throat.hydraulic_size_factors'][i, 1] = np.inf

#Calculating hydraulic conductance
phase = op.phase.Phase(network=pn)
phase['pore.viscosity'] = visc
print(phase['pore.viscosity'])
f2 = op.models.physics.hydraulic_conductance.generic_hydraulic
phase.add_model(propname = 'throat.hydraulic_conductance',
              model = f2)

#print(phase['throat.hydraulic_conductance'][mask_tBC][50:100])
Q = Rate_calc(pn, phase, inlet_pores, outlet_pores, 'throat.hydraulic_conductance')
L = 400 * resolution
A = L**2
K = Q[0] * L * visc / (A) #Q[0] * L * mu / (A * Delta_P), Delta_P = 1
print(K)
print(f'The value of K is: {K/0.98e-12*1000:.2f} mD')
