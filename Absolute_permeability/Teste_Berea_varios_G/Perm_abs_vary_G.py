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

from tqdm import tqdm
import pandas as pd


#Flowrate function
def Rate_calc(network, ph, inlet, outlet, conductance):
    St_p = op.algorithms.StokesFlow(network=network, phase=ph)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=inlet, values=1) #P_inlet = 1
    St_p.set_value_BC(pores=outlet, values=0) #P_outlet = 1
    St_p.run()
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val

resolution = 5.345e-6

f = op.models.physics.hydraulic_conductance.generic_hydraulic
visc = 1e-3 #Pa/s
L_sample = 400 * resolution
A_sample = L_sample**2

#Creating workspace
ws = op.Workspace()

Networks = np.array([0, 10, 25, 50, 100])
Kxyz = np.zeros((5,3))

#To save data to plot
i = 0

#Reading netwrork data
for net_label in Networks:
    testName = 'Berea_G_'+ str(int(net_label)) +'.pnm'
    proj = ws.load_project(filename=testName)
    pn = proj.network
    Np = pn.Np
    Nt = pn.Nt
    print('Working with Network '+str(int(net_label)))

    #Previous data for prism hidraylic factor
    A_t = pn['throat.cross_sectional_area']
    G_t = pn['throat.shape_factor']
    beta_dif = np.pi/2 - 2 * pn['throat.half_corner_angle'][:,1]
    pol = np.array([-0.17997611,  0.57966346, -0.46275726,  1.10633925])
    factor = np.polyval(pol, beta_dif)
    G_min = np.min(G_t[pn['throat.internal']])
    perc = np.sum(G_t == G_min) / np.sum(pn['throat.internal'])
    print(f'G_min is: {G_min:.5f}, representing {perc*100:.2f}% of the internal throats')

    #To save data to plot
    j = 0

    #Working with the 3 axis
    for axis in ['x', 'y', 'z']:
        print('Axis '+ axis )

        #Boundary conditions
        inlet_pores = pn['pore.'+axis+'min']
        outlet_pores = pn['pore.' + axis + 'max']
        pn['pore.BC'] = inlet_pores | outlet_pores
        BC_t = pn.find_neighbor_throats(pores=pn['pore.BC'])
        pn['throat.BC'] = False
        pn['throat.BC'][BC_t] = True

        #Setting minimum length (Se sabe que es igual apra todas las redes)
        #D usado en conductance, pero requiero antes para setear L_min
        D1, Dt, D2 = pn.get_conduit_data('equivalent_diameter').T
        L_min = resolution
        L = _cf.conduit_length_spheres_cylinders(pn,
                                                pore_diameter = 'pore.equivalent_diameter',
                                                throat_diameter = 'throat.equivalent_diameter',
                                                throat_spacing = 'throat.total_length',
                                                L_min = L_min)
        L1 = L[:, 0]
        Lt = L[:, 1]
        L2 = L[:, 2]
        #Calculating hydraulic conductance
        #---------start--------------
        #print(np.any(L1 >= D1/2))
        #print(np.any(L2 >= D2/2))

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
        #St = 1 / (16 * np.pi**2 * It * Ft)
        S2 = 1 / (16 * np.pi**2 * I2 * F2)

        St = 0.6 * factor * A_t ** 2 * G_t / (Lt)

        pn['throat.hydraulic_size_factors'] = np.vstack([S1, St, S2]).T

        #Calculating hydraulic conductance
        phase = op.phase.Phase(network=pn)
        phase['pore.viscosity'] = visc
        phase.add_model(propname = 'throat.hydraulic_conductance',
                    model = f)
        phase.regenerate_models()
        #Setting a high conductance on BC throats
        phase['throat.hydraulic_conductance'][pn['throat.BC']] = np.max(phase['throat.hydraulic_conductance']) * 1000

        Q = Rate_calc(pn, phase, inlet_pores, outlet_pores, 'throat.hydraulic_conductance')
        K = Q[0] * L_sample * visc / (A_sample) #Q[0] * L * mu / (A * Delta_P), Delta_P = 1
        print(K)
        print(f'The value of K is: {K/0.98e-12*1000:.2f} mD')

        #Saving data
        Kxyz[i, j] = K/0.98e-12*1000
        j += 1
    i += 1

print(Kxyz)
fig, ax = plt.subplots(1, figsize = (6,6))
ax.plot(Networks / 10,Kxyz[:,0] , 'ok',label='Kx')
ax.plot(Networks / 10,Kxyz[:,1] , 'ob',label='Ky')
ax.plot(Networks / 10,Kxyz[:,2] , 'or',label='Kz')
ax.set_xlabel(r'$\%$ of internal throats with $G_p = G_{min}$', fontsize = 14)
ax.set_ylabel(r'$K_{abs}$, in mD', fontsize = 14)
ax.set_xlim([0, 10])
ax.set_xticks(np.arange(11))
ax.grid(True)
ax.legend()
plt.show()
