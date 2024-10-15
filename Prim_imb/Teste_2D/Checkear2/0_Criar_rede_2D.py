# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar

En este archivo  se calculara todo lo referente a la invasion por Drenaje primario
Se investiga como calcula la presion capilar de entrada  de las gargantas y el proceso de invasion
El algoritmo toma commo referencia el codigo de Invasion_Percolation
El fluido a usar sera el no mojante. El angulo de contacto estará aqui.
El angulo de contacto puede sr un objeto de tamaño Nx1 o Nx2. Siendo N el numero de poros o gargantas
Si es Nx2, la primera fila tiene al menor de los valores y es el receiding contact angle

La presión capilar sera guardada en el objeto de simulacion de drenaje primario porque puede cambiar con el caso.




"""

#Preambulo

import openpnm as op
import porespy as ps
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import copy
from tqdm import tqdm
#importing functions for other file
import sys
sys.path.insert(1, '/home/cesar/OpenPNM_files/mestrado/_funcs')
import _algorithm_class as _alg
import _conductance_funcs as _cf
import _invasion_funcs as _if
#end
np.random.seed(13)

ws = op.Workspace()
n_side = 10 #pores per side
pn = op.network.Demo(shape=[n_side , n_side , 1], spacing=1e-4)
#pn = op.network.Demo(shape=[15, 15, 1], spacing=1e-4)
Np = pn.Np
Nt = pn.Nt

P12 = pn['throat.conns']
C1 = pn['pore.coords'][pn['throat.conns'][:, 0]]
C2 = pn['pore.coords'][pn['throat.conns'][:, 1]]
L = np.sqrt(np.sum((C1 - C2)**2, axis=1))
print(len(L))
print(Nt)

D = np.inf*np.ones([Np, ], dtype=float)
np.minimum.at(D, P12[:, 0], L)
#Ej: Si P12[5] = 50. Se compara D[50] con L[5]
np.minimum.at(D, P12[:, 1], L)


print(pn['throat.diameter'][50])
print(pn['pore.diameter'][ pn['throat.conns'][50,:] ])

raise Exception('')

#Set boundary pores
pn['pore.inlet'] = pn['pore.front']
pn['pore.outlet'] = pn['pore.back']
pn['pore.boundary'] = pn['pore.inlet'] | pn['pore.outlet']

for throat_type in ['inlet', 'outlet', 'boundary']:
    boundary_t = pn.find_neighbor_throats(pores=pn[f'pore.{throat_type}'])
    pn[f'throat.{throat_type}'] = False
    pn[f'throat.{throat_type}'][boundary_t] = True
pn['pore.internal'] = ~pn['pore.boundary']
pn['throat.internal'] = ~pn['throat.boundary']

#Assuming diameter as cross-sectional diameter of a prism, assign beta and calculate G
elements = ['pore', 'throat']
for item in elements:
    G = np.random.rand(len( pn[f'{item}.diameter'])) * 3 ** 0.5/36
    pn[f'{item}.shape_factor'] = G
    pn[f'{item}.half_corner_angle'] = op.teste.geometry.half_angle_isosceles(G)

#For the boundary throats that will be kept, copy the props of the internal throats
#That is in order to share the phase distribution
for t in range(Nt):
    if pn['throat.boundary'][t]:
        conns = pn['throat.conns'][t]
        mask = pn['pore.internal'][conns]
        if np.any(mask):
            pi = conns[mask]
            pb = conns[~mask]
            G = pn['pore.shape_factor'][pi]
            beta = pn['pore.half_corner_angle'][pi,:]
            pn['pore.shape_factor'][pb] = G
            pn['throat.shape_factor'][t] = G
            pn['pore.half_corner_angle'][pb,:] = beta
            pn['throat.half_corner_angle'][t,:] = beta

#Checking if throats have smaller diameter than connected pores
D1, Dt, D2 = pn.get_conduit_data('diameter').T
print( np.any( Dt > D1) or np.any( Dt > D2) ) #must be false

#setting a path to be invaded first
#path = np.arange((n_side - 1 ) * (n_side - 6) , (n_side - 1 ) * (n_side - 5)) #front-back
path = np.array([27,28,29,30,31,32,33,34,35, #main branch
                 40,41,42,43,44, #second branch
                 124, #connection second branch
                 122, #connectrion third branch
                 131,141,133,143,37,38,55,56#third branch
                 ])

#modifying throat diameter (The cross-sectional area change if d  change. Howver. Pc does not depends on that
pn['throat.diameter'] = pn['throat.diameter'] * 0.4 #reducing all diameters
pn['throat.diameter'][path] = np.minimum(D1[path], D2[path]) * 0.999 #set path with bigger diameter, but less than connected pores
pn['throat.diameter'][122] = np.minimum(D1[122], D2[122]) * 0.5 #Connection of third branch with a lower diameter than other throats from path
pn['throat.diameter'][124] = pn['throat.diameter'][27] * 0.99

#After setting diameters, modify cross sectional areas
for item in elements:
    R = pn[f'{item}.diameter'] / 2
    beta = pn[f'{item}.half_corner_angle']
    pn[f'{item}.cross_sectional_area'] = R ** 2 * np.sum(1 / np.tan(beta), axis = 1)
    pn[f'{item}.perimeter'] = R ** 2 * np.sum(1 / np.tan(beta), axis = 1)

#Removing throats between inlet pores
t_removed = np.concatenate((np.arange(90, 180, 10) ,  np.arange(99, 189, 10) ))
op.topotools.trim(pn, throats = t_removed)

#Saving
ws.save_project(pn.project, filename='Rede_2D_10x10')


#ABSOLUTE PERMEABILITY

#Flowrate function
def Rate_calc(network, ph, inlet, outlet, conductance):
    St_p = op.algorithms.StokesFlow(network=network, phase=ph)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=inlet, values=1)
    St_p.set_value_BC(pores=outlet, values=0)
    St_p.run()
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val

#Calculation conduit length
L = _cf.conduit_length_tubes(pn,
                             pore_length = "pore.diameter",
                             throat_spacing = "throat.spacing",
                             L_min = 1e-7,
                             check_boundary = True)

pn['throat.conduit_lengths'] = L
#---------end--------------

#Calculating hydraulic conductance
visc = 1e-3
phase = op.phase.Phase(network=pn)
phase['pore.viscosity'] = visc
g = _cf.conductance_triangle_OnePhase(phase, correction = True, check_boundary = True)
phase['throat.hydraulic_conductance'] = g

inlet_pores = pn['pore.inlet']
outlet_pores = pn['pore.outlet']

Q = Rate_calc(pn, phase, inlet_pores, outlet_pores, 'throat.hydraulic_conductance')
L_net = 1e-4 * 7
A_net = 1e-4 * 7 * np.mean(pn['pore.diameter'])
K = Q[0] * L_net * visc / A_net #Q[0] * L * mu / (A * Delta_P), Delta_P = 1
print(f'The value of K (not real for a 2D network) is: {K/0.9869233e-12*1000:.2f} mD')
