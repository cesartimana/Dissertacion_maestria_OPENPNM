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

#Setting type for pore diameter and K_array
label_d_type = 'inscribed'
K_array = []
factor_L_array = np.array([0.01,0.05,0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97,0.98, 0.99])

#Defining boundary conditions
axis = 'x'
inlet_pores = pn['pore.'+axis+'min']
index_inlet = pn.pores(axis + 'min')
outlet_pores = pn['pore.' + axis + 'max']
index_outlet = pn.pores(axis + 'max')
pn['pore.BC'] = inlet_pores | outlet_pores

BC_t = pn.find_neighbor_throats(pores=pn['pore.BC'])
pn['throat.BC'] = False
pn['throat.BC'][BC_t] = True

#Properties extractend from Valvatne - Blunt (2004) Table 1
tension = 30e-3 #N/m
visc = 1e-3 #Pa/s

#Calculating conduit length
#---------start--------------

#D usado en conductance, pero requiero antes para setear L_min
pn['pore.diameter'] = pn['pore.' + label_d_type +'_diameter'] #Check line 46
pn['throat.diameter'] = pn['throat.equivalent_diameter']
pn['throat.spacing'] = pn['throat.total_length']
D1, Dt, D2 = pn.get_conduit_data('diameter').T
conns = pn['throat.conns']

#Modifying diameters to not have problems
#---------start--------------
#diameter of boundary pores and throats
mask_tBC = pn['throat.boundary']
mask_pBC = np.isin(conns, pn.pores('boundary'))
for i in range(Nt):
    if mask_tBC[i]:
        D = D1[i]*~mask_pBC[i, 0] + D2[i]*~mask_pBC[i, 1]
        pn['pore.diameter'][conns[i, 0]] = D
        pn['pore.diameter'][conns[i, 1]] = D
        pn['throat.diameter'][i] = 0.95*D  #choose arbitrary
        #pn['throat.spacing'] = D
#---------end--------------

#Setting minimum length
#D usado en conductance, pero requiero antes para setear L_min

L_min = resolution

L = _cf.conduit_length_spheres_cylinders(pn,
                                        pore_diameter = 'pore.diameter',
                                        throat_diameter = 'throat.diameter',
                                        throat_spacing = 'throat.spacing',
                                        L_min = L_min)
L1 = L[:, 0]
Lt = L[:, 1]
L2 = L[:, 2]

D1, Dt, D2 = pn.get_conduit_data('diameter').T

#Modificando comprimento de conduite del poro interno asociado a los boundary pores
#Notese que esta feo esto. Hay que hacerlo mas bonito
#---------start--------------
factor_L = 0.9
for factor_L in factor_L_array:

    for i in range(Nt):
        if mask_tBC[i]:
            D = D1[i]*~mask_pBC[i, 0] + D2[i]*~mask_pBC[i, 1]
            L2[i] = factor_L * (D/2)
    #---------end--------------

    #Calculating hydraulic conductance
    #---------start--------------



    # Fi is the integral of (1/A^2) dx, x = [0, Li]
    a = 4 / (D1**3 * np.pi**2)
    b = 2 * D1 * L1 / (D1**2 - 4 * L1**2) + np.arctanh(2 * L1 / D1)
    F1 = a * b
    a = 4 / (D2**3 * np.pi**2)
    b = 2 * D2 * L2 / (D2**2 - 4 * L2**2) + np.arctanh(2 * L2 / D2)
    F2 = a * b
    Ft = Lt / (np.pi / 4 * Dt**2)**2

    # I is the integral of (y^2 + z^2) dA, divided by A^2
    I1 = I2 = It = 1 / (2 * np.pi)

    # S is 1 / (16 * pi^2 * I * F)
    S1 = 1 / (16 * np.pi**2 * I1 * F1)
    St = 1 / (16 * np.pi**2 * It * Ft)
    S2 = 1 / (16 * np.pi**2 * I2 * F2)

    #Set size factors of boundary elements to np.inf
    for i in range(Nt):
        if mask_tBC[i]:
            S1[i] = np.inf
            St[i] = np.inf

    pn['throat.hydraulic_size_factors'] = np.vstack([S1, St, S2]).T

    #--------------end----------------

    #Calculating hydraulic conductance
    phase = op.phase.Phase(network=pn)
    phase['pore.viscosity'] = visc
    f = op.models.physics.hydraulic_conductance.generic_hydraulic
    phase.add_model(propname = 'throat.hydraulic_conductance',
                model = f)


    Q = Rate_calc(pn, phase, inlet_pores, outlet_pores, 'throat.hydraulic_conductance')
    L = 400 * resolution
    A = L**2
    K = Q[0] * L * visc / (A) #Q[0] * L * mu / (A * Delta_P), Delta_P = 1
    #print(K)
    #print(f'The value of K is: {K/0.98e-12*1000:.2f} mD')
    K_array.append(K/0.98e-12*1000)
K_array = np.array(K_array)
output  = np.vstack((factor_L_array,  K_array)).T
np.savetxt('K_'+ label_d_type + '.txt', output, fmt=' %.5e '+' %.5e ', header=' factor_L// K')
np.save('K_' + label_d_type, output)
