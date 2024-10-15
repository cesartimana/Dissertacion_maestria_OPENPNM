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
Np = pn.Np
Nt = pn.Nt

#Properties extracted from Valvatne - Blunt (2004) Table 1
tension = 30e-3 #N/m
water_visc = 1.05e-3 #Pa/s
oil_visc = 1.39e-3 #Pa/s
theta_r = np.pi / 180 * 0  #Respecto al agua, en sexagecimal
theta_a = np.pi / 180 * 60  #Respecto al agua, en sexagecimal

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
p_vals =  np.array([0, pmax_drn])

#Obtaining phase distribution and clusters for each stage of invasion according to p_vals
results_pdr = pdr.postprocessing2(mode = 'pressure', inv_vals = p_vals)

"""
#COMPROBANDO QUE LOS RESULTADOS DEL POSTPROCESSING CONCUERDAN CON LO QUE DEBERIA SALIR
#------------START--------------
i = 1
p_inv = results_pdr[f'status_{i}']['invasion_pressure']
print(p_inv)
print(results_pdr[f'status_{i}']['cluster_info']['index_list'])
inv_pattern = pdr['throat.invasion_pressure'] <= p_inv
inv_pattern2 = ~results_pdr[f'status_{i}']['invasion_info']['throat.center']
inv_pattern3 = results_pdr[f'status_{i}']['cluster_info']['pore.center'] == 2
ax = op.visualization.plot_coordinates(network=pn, pores=pn.pores('front'), c='r', s=50)
ax = op.visualization.plot_coordinates(network=pn, pores=pn.pores('front', mode='not'), c='grey', ax=ax)
ax = op.visualization.plot_coordinates(network=pn, pores = inv_pattern3, c='g', s=100, ax=ax)
op.visualization.plot_connections(network=pn, throats=inv_pattern, ax=ax, c = 'blue', linewidth = 5)
op.visualization.plot_connections(network=pn, throats=inv_pattern2, ax=ax, c = 'yellow', linewidth = 2)

plt.show()
#------------END--------------

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

#-------------------------------------

#Strings used
elements = ['pore', 'throat']
locations = ['center', 'corner']
displacement_type = ['snapoff' , 'MTM_displacement']

#-------------------------------------

#------------------------------------------
#Read imbibition results
#Obtener datos de la ultima invasion registrada
with open('imbibition_process_normal.pkl', 'rb') as fp:
    info_imbibition = pickle.load(fp)

#Picking last info and transform in int
last_s_inv = int(list(info_imbibition)[-1][7:])
print(last_s_inv)
status_str = 'status ' + str(last_s_inv)

#How to extract info
print(info_imbibition[status_str].keys())
copy_invasion_info = copy.deepcopy(info_imbibition[status_str]['invasion_info'])
copy_cluster_info = copy.deepcopy(info_imbibition[status_str]['cluster_info'])
p_inv = info_imbibition[status_str]['invasion pressure']

#Dictionary used to save results
flow_results = {}

#------------------------------------------

#Calcular datos de bi al inicio de la imbibicion.
#LAMENTABLEMENTE DEBI SACAR ESE DATO DESDE EL DRENAJE, NO DESDE STATUS 1
#------------------------------------------

for item in elements:
    cluster_ce = copy.deepcopy(info_imbibition['status 1']['cluster_info'][f'{item}.center'])
    cluster_co = copy.deepcopy(info_imbibition['status 1']['cluster_info'][f'{item}.corner'])
    phase_ce = copy.deepcopy(info_imbibition['status 1']['invasion_info'][f'{item}.center'])
    phase_co = copy.deepcopy(info_imbibition['status 1']['invasion_info'][f'{item}.corner'])
    interface = (np.tile(phase_ce, (3,1)).T != phase_co)
    theta_corner = np.tile(water[f'{item}.contact_angle'],(3,1)).T #Funciona con un solo valor de theta
    beta = pn[f'{item}.half_corner_angle']
    R = water[f'{item}.surface_tension'][0]/ pmax_drn
    bi_corner = _if.interfacial_corner_distance(R, theta_corner, beta, int_cond = interface)
    flow_results[f'{item}.bi_drainage'] = bi_corner
    flow_results[f'{item}.pc_theta_a'] = pmax_drn * np.cos(theta_a + beta) / np.cos(theta_corner + beta)

#------------------------------------------

#STARTING CALCULATION
sat = []
V_t = sum(pn['pore.volume'][pn['pore.internal']])
status_array = np.arange(1, last_s_inv, 1, dtype = int)
flow_results['status_index'] = np.array(status_array)

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
    flow_results[phase.name+ '_flow'] = sph_flow

#MULTIPHASE
for s_inv in tqdm(status_array, desc = 'Calculating flow at a specific equilibrium status '):
    status_str = 'status ' + str(s_inv)
    flow_results[status_str] = {}
    pc = info_imbibition[status_str]['invasion pressure']
    print('----------------')
    print(status_str)
    print('pc = ' + str(pc))
    print('----------------')
    for wp in [True,False]:
        rel_perm = []
        if wp:
            phase = water
            #invadiendo elementos de frontera con agua
            for t in range(Nt):
                if pn['throat.boundary'][t]:
                    conns = pn['throat.conns'][t]
                    for p in conns:
                        if pn['pore.boundary'][p]:
                            p_b = p
                        else:
                            p_int = p
                    status = True
                    index = 0
                    #Modifying boundary elements, center and corner phases
                    info_imbibition[status_str]['invasion_info']['pore.center'][p_b] = status
                    info_imbibition[status_str]['invasion_info']['pore.corner'][p_b] = status
                    info_imbibition[status_str]['invasion_info']['throat.center'][t] = status
                    info_imbibition[status_str]['invasion_info']['throat.corner'][t] = status
                    info_imbibition[status_str]['cluster_info']['pore.center'][p_b] = index
                    info_imbibition[status_str]['cluster_info']['pore.corner'][p_b] = index
                    info_imbibition[status_str]['cluster_info']['throat.center'][t] = index
                    info_imbibition[status_str]['cluster_info']['throat.corner'][t] = index

        else:
            phase = oil
            #invadiendo elementos de frontera de acuerdo al internal pore
            for t in range(Nt):
                if pn['throat.boundary'][t]:
                    conns = pn['throat.conns'][t]
                    for p in conns:
                        if pn['pore.boundary'][p]:
                            p_b = p
                        else:
                            p_int = p
                    status = info_imbibition[status_str]['invasion_info']['pore.center'][p_int]
                    index = info_imbibition[status_str]['cluster_info']['pore.center'][p_int]
                    #Modifying boundary elements, center and corner phases
                    info_imbibition[status_str]['invasion_info']['pore.center'][p_b] = status
                    info_imbibition[status_str]['invasion_info']['pore.corner'][p_b] = status
                    info_imbibition[status_str]['invasion_info']['throat.center'][t] = status
                    info_imbibition[status_str]['invasion_info']['throat.corner'][t] = status
                    info_imbibition[status_str]['cluster_info']['pore.center'][p_b] = index
                    info_imbibition[status_str]['cluster_info']['pore.corner'][p_b] = index
                    info_imbibition[status_str]['cluster_info']['throat.center'][t] = index
                    info_imbibition[status_str]['cluster_info']['throat.corner'][t] = index
        q_mph = []
        viscosity = phase['pore.viscosity'][0]
        for item in elements:
            #Extracting data
            cluster_ce = copy.deepcopy(info_imbibition[status_str]['cluster_info'][f'{item}.center'])
            cluster_co = copy.deepcopy(info_imbibition[status_str]['cluster_info'][f'{item}.corner'])
            phase_ce = copy.deepcopy(info_imbibition[status_str]['invasion_info'][f'{item}.center']) == wp #True if wp is present
            phase_co = copy.deepcopy(info_imbibition[status_str]['invasion_info'][f'{item}.corner']) == wp
            mask_bi = flow_results[f'{item}.pc_theta_a'] < pc #True if bi is fixed (theta_hinging)

            #Obtaining continuous clusters just looking pores
            if item == 'pore':
                continuity_list = _if.identify_continuous_cluster(cluster_ce , cluster_co, pn['pore.inlet'], pn['pore.outlet'])
                ph_clusters_ce = np.unique(cluster_ce[phase_ce])
                ph_clusters_co = np.unique(cluster_co[phase_co])
                cluster_list = np.union1d(ph_clusters_ce, ph_clusters_co)
                phase_clusters = np.intersect1d(cluster_list, continuity_list)
                print(phase_clusters)

            #Calculating saturation

            #Setting pressure for normal and trapped clusters
            N = len(phase_ce)
            pc_array = np.ones(N) * pc
            trapped = info_imbibition[status_str]['trapped_clusters']
            pc_trapped = info_imbibition[status_str]['pc_trapped']
            for i in range(N):
                if cluster_ce[i] in trapped:
                    pos = np.where(cluster_ce[i] == trapped)[0]
                    pc_array[i] = pc_trapped[pos]
            #Calculating other geometric properties
            pc_array = np.tile(pc_array, (3,1)).T
            interface = (np.tile(phase_ce, (3,1)).T != phase_co)
            beta = pn[f'{item}.half_corner_angle']
            theta_corner = np.arccos(pc_array / pmax_drn * np.cos(theta_r + beta) ) - beta
            theta_corner[~mask_bi] = theta_a
            R = tension / pc_array
            bi_corner = _if.interfacial_corner_distance(R, theta_corner, beta, int_cond = interface)
            A_center, A_corner = _cf.calc_area_twophase(pn, phase_ce, phase_co, theta_corner, bi_corner, item = item)
            flow_results[status_str][f'{item}.bi'] = bi_corner
            flow_results[status_str][f'{item}.theta'] = theta_corner
            flow_results[status_str][f'{item}.center_area_{(phase.name)}'] = A_center
            flow_results[status_str][f'{item}.corner_area_{(phase.name)}'] = A_corner
            if item == 'pore' and wp:
                #Water sat
                ratio_sat = (A_center + np.sum(A_corner, axis = 1)) / pn['pore.cross_sectional_area']
                V_w = np.sum((pn['pore.volume'] * ratio_sat)[pn['pore.internal']])
                sat.append(V_w / V_t)
        #For each cluster of the same phase
        for n in phase_clusters:
            conductance_dict = {}
            for item in elements:
                g_ce = _cf.conductance_center(area_center = flow_results[status_str][f'{item}.center_area_{(phase.name)}'],
                                              shape_factor = pn[f'{item}.shape_factor'],
                                              viscosity = viscosity,
                                              beta = pn[f'{item}.half_corner_angle'],
                                              correction = True)
                g_co = _cf.conductance_corner(area_corner = flow_results[status_str][f'{item}.corner_area_{(phase.name)}'],
                                              beta = pn[f'{item}.half_corner_angle'],
                                              bi = flow_results[status_str][f'{item}.bi'],
                                              theta = flow_results[status_str][f'{item}.theta'],
                                              viscosity = viscosity)
                conductance_dict[f'{item}.center'] = g_ce
                conductance_dict[f'{item}.corner'] = g_co

            #Calculating conduit conductance
            g_L_mph = _cf.conduit_conductance_2phases(network = pn,
                                        pore_g_ce = conductance_dict['pore.center'],
                                        throat_g_ce = conductance_dict['throat.center'],
                                        conduit_length = L,
                                        pore_g_co = conductance_dict['pore.corner'],
                                        throat_g_co = conductance_dict['throat.corner'],
                                        corner = True,
                                        check_boundary = True)
            label_conductance = 'throat.conduit_conductance'
            phase[label_conductance] = g_L_mph

            #Calculating multiphase rate flow. Append in q_mph for each cluster
            # Conductance cannot be zero. If that, matrix A become singular (empty rows) and can not be used
            q_mph.append(Rate_calc(pn, phase, pn['pore.inlet'] , pn['pore.outlet'], label_conductance))

        #Calculating k_r only if there are continuous clusters. Otherwise, k_r = 0
        sph_flow = flow_results[phase.name+ '_flow']
        if len(phase_clusters) > 0:
            mph_flow = np.sum(q_mph)
            flow_results[status_str][f'{phase.name}_flow'] = mph_flow
            flow_results[status_str][f'{phase.name}_krel'] = np.squeeze(mph_flow/sph_flow)
        else:
            flow_results[status_str][f'{phase.name}_flow'] = 0
            flow_results[status_str][f'{phase.name}_krel'] = 0

krel_water = []
krel_oil = []
pc_array = []
for s_inv in status_array:
    status_str = 'status ' + str(s_inv)
    krel_water.append(flow_results[status_str]['water_krel'])
    krel_oil.append(flow_results[status_str]['oil_krel'])
    pc_array.append(info_imbibition[status_str]['invasion pressure'])

krel_water = np.array(krel_water)
krel_oil = np.array(krel_oil)
pc_array = np.array(pc_array)

#Saving data
output = np.vstack((sat, krel_water, krel_oil, pc_array)).T
np.save('results_K_Pc', output) #If wp, then 0. Otherwise 1
np.savetxt('results_K_Pc.txt', output, fmt=' %.5e '+' %.5e '+' %.5e ' + ' %.5e ', header=' sat // krel_w// krel_o // pc')
