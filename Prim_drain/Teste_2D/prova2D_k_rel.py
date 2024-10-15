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
from tqdm import tqdm
#importing functions for other file
import sys
sys.path.insert(1, '/home/cesar/OpenPNM_files/mestrado/_funcs')
import _algorithm_class as _alg
import _conductance_funcs as _cf
import _invasion_funcs as _if
#end
np.random.seed(13)

#Flowrate function
def Rate_calc(network, ph, inlet, outlet, conductance):
    St_p = op.algorithms.StokesFlow(network=network, phase=ph)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=inlet, values=1)
    St_p.set_value_BC(pores=outlet, values=0)
    St_p.run()
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val
#Reading pnm arquive/ network
ws = op.Workspace()
testName ='Rede_2D_10x10.pnm'
proj = ws.load_project(filename=testName)

pn = proj.network

#SOLO PARA RED CREADA, ACTUALIZANDO NUMERO DE GARGANTAS
Np = pn.Np
Nt = pn.Nt

#Properties extracted from Valvatne - Blunt (2004) Table 1
tension = 30e-3 #N/m
water_visc = 1.05e-3 #Pa/s
oil_visc = 1.39e-3 #Pa/s
theta_r = np.pi / 180 * 0 #Respecto al agua, en sexagecimal

#Add phase properties
#OIL
oil = op.phase.Phase(network=pn, name='oil')
oil['pore.surface_tension'] = tension
oil['throat.surface_tension'] = tension
oil['pore.viscosity'] = oil_visc
oil['throat.viscosity'] = oil_visc

#WATER
water = op.phase.Phase(network=pn,name='water')
water['pore.surface_tension'] = tension
water['throat.surface_tension'] = tension
water['pore.viscosity'] = water_visc
water['throat.viscosity'] = water_visc
water['pore.contact_angle'] = theta_r
water['throat.contact_angle'] = theta_r

#Simulating primary Drainage
pdr = _alg.Primary_Drainage(network=pn, phase=water)
pdr.set_inlet_BC(pores=pn['pore.inlet'])
pdr.set_outlet_BC(pores=pn['pore.outlet'])
pdr.run(throat_diameter = 'throat.diameter')

#Post processing
pmax_drn = 18000
#p_vals =  np.array([0, pmax_drn])
p_vals = np.unique(pdr['throat.invasion_pressure'][(pdr['throat.invasion_pressure'] < pmax_drn) & pn['throat.internal'] ] ) + 0.2
p_vals = np.append(p_vals, pmax_drn)

#Obtaining phase distribution and clusters for each stage of invasion according to p_vals
results_pdr = pdr.postprocessing2(mode = 'pressure', inv_vals = p_vals)
#Ya revise que a pc = 18000 y theta = 0, todos los clujsters de agua son 0, y de oleo diferente de 0

elements = ['pore', 'throat']
locations = ['center', 'corner']

"""
#COMPROBANDO QUE LOS RESULTADOS DEL POSTPROCESSING CONCUERDAN CON LO QUE DEBERIA SALIR
#------------START--------------
i = 2
p_inv = results_pdr[f'status_{i}']['invasion_pressure']
print(p_inv)
print(results_pdr[f'status_{i}']['cluster_info']['index_list'])
inv_pattern = pdr['throat.invasion_pressure'] <= p_inv
inv_pattern2 = ~results_pdr[f'status_{i}']['invasion_info']['throat.center']
inv_pattern3 = results_pdr[f'status_{i}']['cluster_info']['pore.center'] == 1
ax = op.visualization.plot_coordinates(network=pn, pores=pn.pores('front'), c='r', s=50)
ax = op.visualization.plot_coordinates(network=pn, pores=pn.pores('front', mode='not'), c='grey', ax=ax)
ax = op.visualization.plot_coordinates(network=pn, pores = inv_pattern3, c='g', s=100, ax=ax)
op.visualization.plot_connections(network=pn, throats=inv_pattern, ax=ax, c = 'blue', linewidth = 5)
op.visualization.plot_connections(network=pn, throats=inv_pattern2, ax=ax, c = 'yellow', linewidth = 2)

plt.show()
#------------END--------------

raise Exception('Hola')
"""

#Volume from pores
#Assuming all volume is on the pores
V_sph = sum(pn['pore.volume'][pn['pore.internal']])

#Calculation conduit length
L = _cf.conduit_length_tubes(pn,
                             pore_length = "pore.diameter",
                             throat_spacing = "throat.spacing",
                             L_min = 1e-5,
                             check_boundary = True)

sat = []

"""
#Modifying boundary elements
#invade inlet elements (pore, throat) of a boundary conduit with the center phase of the internal pore
for i in tqdm(range(len(p_vals)), desc = 'Adapting phase on boundary conduits'):
    status_str = 'status_' + str(i+1)
    for t in range(Nt):
        if pn['throat.boundary'][t]:
            conns = pn['throat.conns'][t]
            for p in conns:
                if pn['pore.boundary'][p]:
                    p_b = p
                else:
                    p_int = p
            status = results_pdr[status_str]['invasion_info']['pore.center'][p_int]
            index = results_pdr[status_str]['cluster_info']['pore.center'][p_int]
            #Modifying boundary elements, center phase (I ignore corners, assuming thats OK. Not sure)
            results_pdr[status_str]['invasion_info']['pore.center'][p_b] = status
            results_pdr[status_str]['invasion_info']['throat.center'][t] = status
            results_pdr[status_str]['cluster_info']['pore.center'][p_b] = index
            results_pdr[status_str]['cluster_info']['throat.center'][t] = index
"""

#Calculating flow rate for each phase: wp and nwp
for wp in [True,False]:
    rel_perm = []
    if wp:
        phase = water
    else:
        phase = oil

    #Calculating conductance on each element
    for item in elements:
        status_center = np.ones_like(pn[f'{item}.shape_factor'], dtype = bool)
        status_corner = np.ones_like(pn[f'{item}.half_corner_angle'], dtype = bool)
        theta_corner = np.tile(water[f'{item}.contact_angle'],(3,1)).T #Funciona con un solo valor de theta
        bi_corner = np.zeros_like(pn[f'{item}.half_corner_angle'])
        viscosity = phase[f'{item}.viscosity'][0]
        g, _, _ = _cf.conductance(pn, status_center, status_corner, theta_corner, bi_corner, viscosity, item = item, correction = True)
        if item == 'pore':
            gp = g
        else:
            gt = g

    #Calculating conduit conductance
    sph_g_L = _cf.conduit_conductance_2phases(network = pn,
                                            pore_g_ce = gp,
                                            throat_g_ce = gt,
                                            conduit_length = L,
                                            check_boundary = True)
    label_conductance = 'throat.conductance'
    phase[label_conductance] = sph_g_L

    #Calculating single phase flow
    sph_flow = Rate_calc(pn, phase, pn['pore.inlet'] , pn['pore.outlet'], label_conductance)

    #MULTIPHASE
    for i in tqdm(range(len(p_vals)), desc = 'Working with ' + phase.name):
        print('----')
        print(i)
        print(phase.name)
        print(p_vals[i])
        status_str = 'status_' + str(i+1)

        #Extracting data
        cluster_ce = results_pdr[status_str]['cluster_info']['pore.center']
        cluster_co = results_pdr[status_str]['cluster_info']['pore.corner']
        phase_ce = results_pdr[status_str]['invasion_info']['pore.center'] == wp #True if wp is present
        phase_co = results_pdr[status_str]['invasion_info']['pore.corner'] == wp
        q_mph = []

        #Calculating saturation
        interface = (np.tile(phase_ce, (3,1)).T != phase_co)
        theta_corner = np.tile(water['pore.contact_angle'],(3,1)).T #Funciona con un solo valor de theta
        beta = pn['pore.half_corner_angle']
        R = water['pore.surface_tension'][0]/ p_vals[i]
        bi_corner = _if.interfacial_corner_distance(R, theta_corner, beta, int_cond = interface)
        viscosity = phase['pore.viscosity'][0]
        _, _, pore_ratio = _cf.conductance(pn, phase_ce, phase_co, theta_corner, bi_corner, viscosity, item = 'pore')
        V_mph = np.sum((pn['pore.volume'] * pore_ratio)[pn['pore.internal']])
        if wp:
            sat.append(V_mph / V_sph)

        #Determining continuous clusters for the phase
        continuity_list = _if.identify_continuous_cluster(cluster_ce , cluster_co, pn['pore.inlet'], pn['pore.outlet'])
        cluster_list_ce = np.unique(cluster_ce[phase_ce])
        cluster_list_co = np.unique(cluster_co[phase_co])
        cluster_list = np.union1d(cluster_list_ce, cluster_list_co)
        phase_clusters = np.intersect1d(cluster_list, continuity_list)

        #For each cluster of the same phase
        for n in phase_clusters:
            #Calculating conductance on each element
            for item in elements:
                BC = pn[f'{item}.boundary']
                status_center = (results_pdr[status_str]['cluster_info'][f'{item}.center'] == n) | BC
                status_corner = (results_pdr[status_str]['cluster_info'][f'{item}.corner'] == n) | np.tile(BC, (3,1)).T
                interface = (np.tile(status_center, (3,1)).T != status_corner)
                theta_corner = np.tile(water[f'{item}.contact_angle'],(3,1)).T #Funciona con un solo valor de theta
                beta = pn[f'{item}.half_corner_angle']
                R = water[f'{item}.surface_tension'][0]/ p_vals[i]
                bi_corner = _if.interfacial_corner_distance(R, theta_corner, beta, int_cond = interface)
                viscosity = phase[f'{item}.viscosity'][0]
                if item == 'pore':
                    gp_ce, gp_co, _ = _cf.conductance(pn, status_center, status_corner, theta_corner, bi_corner, viscosity, item = item, correction = True)
                else:
                    gt_ce, gt_co,_ = _cf.conductance(pn, status_center, status_corner, theta_corner, bi_corner, viscosity, item = item, correction = True)

            #Calculating conduit conductance
            g_L_mph = _cf.conduit_conductance_2phases(network = pn,
                                        pore_g_ce = gp_ce,
                                        throat_g_ce = gt_ce,
                                        conduit_length = L,
                                        pore_g_co = gp_co,
                                        throat_g_co = gt_co,
                                        corner = True,
                                        check_boundary = True)
            label_conductance = 'throat.conduit_conductance'
            phase[label_conductance] = g_L_mph

            #Calculating multiphase rate flow. Append in q_mph for each cluster
            # Conductance cannot be zero. If that, matrix A become singular (empty rows) and can not be used
            q_mph.append(Rate_calc(pn, phase, pn['pore.inlet'] , pn['pore.outlet'], label_conductance))

        #Calculating k_r only if there are continuous clusters. Otherwise, k_r = 0
        if len(phase_clusters) > 0:
            mph_flow = np.sum(q_mph)
            rel_perm.append(np.squeeze(mph_flow/sph_flow))
        else:
            rel_perm.append(0)

    #Saving data
    output = np.vstack((p_vals,  sat, rel_perm)).T
    np.save('K_' + phase.name, output) #If wp, then 0. Otherwise 1
    np.savetxt('K_'+ phase.name + '.txt', output, fmt=' %.5e '+' %.5e '+' %.5e ', header=' p// sat_w // kr')
