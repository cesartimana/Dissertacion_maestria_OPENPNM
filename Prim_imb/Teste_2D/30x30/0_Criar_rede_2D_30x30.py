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
plt.rcParams.update({'font.size': 22})
#end
np.random.seed(13)

ws = op.Workspace()
n_side = 30 #pores per side
pn = op.network.Demo(shape=[n_side , n_side , 1], spacing=1e-4)
#pn = op.network.Demo(shape=[15, 15, 1], spacing=1e-4)
Np = pn.Np
Nt = pn.Nt

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

#Removing throats between inlet pores
t_removed = np.where(np.all(pn['pore.boundary'][pn['throat.conns']], axis = 1))[0]
op.topotools.trim(pn, throats = t_removed)

Np = pn.Np
Nt = pn.Nt

#Plotting network
#plotting
fig1, ax1 = plt.subplots(figsize = (12,12))
#Boundary elements
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('boundary'), ax=ax1, c = 'r', linewidth = 6, zorder = 0)
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=250, ax = ax1, zorder = 0)
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('internal'), ax=ax1, c = 'gray', linewidth = 6, zorder = 0)
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('internal'), c='gray', s=250, ax = ax1, zorder = 0)

plt.show()

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

#After setting diameters, modify cross sectional areas
for item in elements:
    R = pn[f'{item}.diameter'] / 2
    beta = pn[f'{item}.half_corner_angle']
    pn[f'{item}.cross_sectional_area'] = R ** 2 * np.sum(1 / np.tan(beta), axis = 1)
    pn[f'{item}.perimeter'] = R ** 2 * np.sum(1 / np.tan(beta), axis = 1)

#Saving
ws.save_project(pn.project, filename='Rede_2D_30x30')

"""
#Plotting properties

def cum_hist(G):
    Gsort = np.sort(G)
    N = len(G)
    order = np.arange(N) + 1
    return Gsort, order/(N+1)

Dt, y_Dt = cum_hist(pn['throat.diameter'])
Dp, y_Dp = cum_hist(pn['pore.diameter'])
fig1, ax1 = plt.subplots(figsize = (7,7))
ax1.plot(Dt * 1e5,  y_Dt,'b', linewidth = 3, label = 'throats')
ax1.plot(Dp * 1e5,  y_Dp,'r', linewidth = 3, label = 'pores')
xticks = np.arange(0, 8, 1)
ax1.set_xticks(xticks)
ax1.set_xlim([xticks[0], xticks[-1]])
ax1.set_xlabel(r'$x$ ($\mu m$x$10$)')
ax1.set_ylabel('$\mathbb{P}(D < x)$')
ax1.set_ylim([0, 1])
ax1.legend()
plt.tight_layout()

Gt, y_Gt = cum_hist(pn['throat.shape_factor'])
fig2, ax2 = plt.subplots(figsize = (7,7))
ax2.plot(Gt,  y_Dt,'b', linewidth = 3, label = 'throats')
ax2.plot(np.ones_like(Gt) * 3 ** 0.5/36 ,  y_Dt,'k--', linewidth = 2, label = '$x = \sqrt{3}/36$')
xticks = np.arange(0, 0.06, 0.01)
ax2.set_xticks(xticks)
ax2.set_xlim([xticks[0], xticks[-1]])
ax2.set_xlabel('$x$')# (x $10 ^{-5}$)')
ax2.set_ylabel('$\mathbb{P}(G_t < x)$')
ax2.set_ylim([0, 1])
ax2.legend()
plt.tight_layout()

beta1, y_beta1 = cum_hist(pn['throat.half_corner_angle'][:,0])
beta2, y_beta2 = cum_hist(pn['throat.half_corner_angle'][:,1])
beta3, y_beta3 = cum_hist(pn['throat.half_corner_angle'][:,2])
fig3, ax3 = plt.subplots(figsize = (7,7))
ax3.plot(beta1,  y_beta1,'b', linewidth = 3, label = r'$\beta_1$')
ax3.plot(beta2,  y_beta2,'r', linewidth = 3, label = r'$\beta_2$')
ax3.plot(beta3,  y_beta3,'g', linewidth = 3, label = r'$\beta_3$')
ax3.plot(np.ones_like(Gt) * np.pi / 2 ,  y_Dt,'k--', linewidth = 2, label = r'$x = \pi / 2$')
xticks = np.arange(0, 2.1, 0.5)
ax3.set_xticks(xticks)
ax3.set_xlim([xticks[0], 1.7])
ax3.set_xlabel('$x$ (rad)')# (x $10 ^{-5}$)')
ax3.set_ylabel(r'$\mathbb{P}(\beta < x)$')
ax3.set_ylim([0, 1])
ax3.legend( loc = 'lower center', bbox_to_anchor=(0.7, 0.02) )
plt.tight_layout()

plt.show()
"""

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
