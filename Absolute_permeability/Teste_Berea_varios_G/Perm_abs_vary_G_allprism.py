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

L_min = resolution

#To save data to plot
i = 0

#Reading netwrork data
for net_label in Networks:
    testName = 'Berea_G_'+ str(int(net_label)) +'.pnm'
    proj = ws.load_project(filename=testName)
    pn = proj.network
    Np = pn.Np
    Nt = pn.Nt

    #Setting phase
    #Add phase properties
    phase = op.phase.Phase(network=pn)
    phase['pore.viscosity'] = visc
    phase['throat.viscosity'] = visc

    #Calculation conduit length
    pn['throat.conduit_lengths'] = _cf.conduit_length_tubes(pn,
                                pore_length = "pore.equivalent_diameter",
                                throat_spacing = "throat.total_length",
                                L_min = resolution,
                                check_boundary = True)

    #Testing Formulas for mono and two-phase
    label_conductance = 'throat.conductance'

    #Using formula for monophase
    #---------start----------
    g = _cf.conductance_triangle_OnePhase(
        phase,
        correction = True)

    phase[label_conductance] = g

    print('Working with Network '+str(int(net_label)))

    #To save data to plot
    j = 0

    #Working with the 3 axis
    for axis in ['x', 'y', 'z']:
        print('Axis '+ axis )

        #Boundary conditions
        inlet_pores = pn['pore.'+axis+'min']
        outlet_pores = pn['pore.' + axis + 'max']




        #Setting minimum length (Se sabe que es igual apra todas las redes)
        #D usado en conductance, pero requiero antes para setear L_min
        D1, Dt, D2 = pn.get_conduit_data('equivalent_diameter').T



        Q = Rate_calc(pn, phase, inlet_pores, outlet_pores, label_conductance)
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
