# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar

"""

import openpnm as op
import numpy as np
import matplotlib.pyplot as plt

#importing functions for other file
import sys
sys.path.insert(1, '/home/cesar/OpenPNM_files/_funcs')
import _algorithm_class as _alg
import _conductance_funcs as _cf
import _invasion_funcs as _if
#end

import matplotlib
matplotlib.rcParams.update({'font.size': 18})


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

#Setting empty arrays
labels = [416, 400, 300, 200, 100, 0]
Kx = []
Ky = []
Kz = []
Gmin = []

for label in labels:

    #Reading netwrork data
    ws = op.Workspace()
    testName_h = 'Berea_G_' + str(label) + '.pnm' # 'Berea_G_0.pnm' or 'Berea.pnm'
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
    L_min_tl = 0.05 *  pn['throat.spacing']
    L_min = resolution
    L_min = np.maximum(L_min_tl , np.ones_like(L_min_tl) * resolution )

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
    HSF_t_prism[mask_tBC] = np.inf
    pn['throat.hydraulic_size_factors'][:, 1] = HSF_t_prism
    #---------end--------------

    #Calculating hydraulic conductance
    phase = op.phase.Phase(network=pn)
    phase['pore.viscosity'] = visc
    f = op.models.physics.hydraulic_conductance.generic_hydraulic
    phase.add_model(propname = 'throat.hydraulic_conductance', model = f)

    Gmin.append(np.min(G[~mask_tBC]))

    #Defining boundary conditions
    for axis in ['x', 'y', 'z' ]:
        inlet_pores = pn['pore.'+axis+'min']
        outlet_pores = pn['pore.' + axis + 'max']

        Q = Rate_calc(pn, phase, inlet_pores, outlet_pores, 'throat.hydraulic_conductance')
        L = 400 * resolution
        A = L**2
        K = Q[0] * L * visc / (A) /0.9869233e-12*1000 #Q[0] * L * mu / (A * Delta_P), Delta_P = 1

        #print(K)
        #print(f'The value of K is: {K/0.98e-12*1000:.2f} mD')

        if label == 416:
            print(K)
        if axis == 'x':
            Kx.append(K)
        elif axis == 'y':
            Ky.append(K)
        else:
            Kz.append(K)

print(Gmin)

#Plotting
#Previous calculations for internal throat G
#0 is the final label, an allows to plot distribution
G_int = np.sort(G[~mask_tBC])
dist_cum = np.arange(len(G_int)) / (len(G_int) + 1)
one_array = np.ones_like(G_int)

fig1, ax1 = plt.subplots(1, figsize = (7,7))
ax1.plot(G_int, dist_cum ,'k', linewidth = 3, label = 'Berea data')
ax1.plot(one_array * 3**0.5 / 36, dist_cum , 'r--', linewidth = 2, label = r'$G = \sqrt{3}/36$')
ax1.plot(G_int, one_array * 0.416 , 'b--', linewidth = 2, label = 'prob. = 41.6%')
ax1.set_xlabel('G')
ax1.set_ylabel('cumulative probability')
ax1.set_xlim([0,0.05])
ax1.set_ylim([0,1])
ax1.legend()

#Transforming to numpy arrays
dist_G_label = np.array(labels) / 10
Kx = np.array(Kx)
Ky = np.array(Ky)
Kz = np.array(Kz)
Gmin = np.array(Gmin)
one_array = np.ones_like(Kx)
y_array = np.arange(len(Kx)) * 1200

fig2, ax2 = plt.subplots(1, figsize = (8,7))
ax2.plot(dist_G_label, Kx ,'ko', markersize = 10, label = r'$K_x$')
ax2.plot(dist_G_label, Ky ,'ro', markersize = 10, label = r'$K_y$')
ax2.plot(dist_G_label, Kz ,'bo', markersize = 10, label = r'$K_z$')
ax2.plot(one_array * 41.6, y_array ,'k--', linewidth = 1.5, label = 'prob. = 41.6%')
ax2.set_xlabel(r'probability($G = G^*$)')
ax2.set_ylabel('K')
ax2.set_xlim([0,45])
ax2.set_ylim([0,1200])
ax2.legend()

fig3, ax3 = plt.subplots(1, figsize = (8,7))
ax3.plot(Gmin, Kx ,'ko', markersize = 10, label = r'$K_x$')
ax3.plot(Gmin, Ky ,'ro', markersize = 10, label = r'$K_y$')
ax3.plot(Gmin, Kz ,'bo', markersize = 10, label = r'$K_z$')
ax3.plot(one_array * 3 ** 0.5 / 36, y_array ,'k--', linewidth = 1.5, label = r'$G^* = \sqrt{3}/36$')
ax3.set_xlabel(r'$G^*$')
ax3.set_ylabel('K')
ax3.set_xlim([0,0.05])
ax3.set_ylim([0,1200])
ax3.legend()
plt.show()
