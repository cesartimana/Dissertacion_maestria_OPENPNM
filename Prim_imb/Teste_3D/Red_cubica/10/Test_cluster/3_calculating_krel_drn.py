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
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import copy
from Properties import *
#importing functions for other file
import sys
sys.path.insert(1, '/home/cbeteta/_funcs')
import _drainage_class as _drain
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
ws.settings.default_solver = 'ScipySpsolve'
proj = ws.load_project(filename=testName)

pn = proj.network

#SOLO PARA RED CREADA, ACTUALIZANDO NUMERO DE GARGANTAS
Np = pn.Np
Nt = pn.Nt

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

#Calculation conduit length
L = pn['throat.conduit_lengths']

#Volume from pores
#Assuming all volume is on the pores
V_t = sum(pn['pore.volume'][pn['pore.internal']])

with open(f'drainage_process_{theta_r_sexag}_{p_kPa}kPa.pkl', 'rb') as fp:
    results_pdr = pickle.load(fp)
last_s_inv = int(list(results_pdr)[-1][7:])
status_array = np.arange(0, last_s_inv + 1, 1, dtype = int)
progressbar1 = tqdm(status_array)
progressbar2 = tqdm(status_array)
sat = []
p_vals = []
elements = ['pore', 'throat']
locations = ['center', 'corner']
flow_results = {}
geometric_data = {}

#NEW: CALCULATING GEOMETRIC DATA
for s_inv in progressbar1:
    status_str = 'status ' + str(s_inv)
    progressbar1.set_description("Calculating geometric data for %s" % status_str)
    geometric_data[status_str] = {}
    pc = results_pdr[status_str]['invasion pressure']
    trapped_list = results_pdr[status_str]['trapped_clusters']
    pc_trapped_list = results_pdr[status_str]['pc_trapped']
    for item in elements:
        N = len(pn[f'{item}.boundary'])
        status_info = {}
        for loc in locations:
            status_info[f'cluster_{loc}'] = copy.deepcopy(results_pdr[status_str]['cluster_info'][f'{item}.{loc}'])
            status_info[f'phase_{loc}'] = copy.deepcopy(results_pdr[status_str]['invasion_info'][f'{item}.{loc}'])
        #Obtaining some geometric data for all elements
        beta = pn[f'{item}.half_corner_angle']
        A_cross_section =  pn[f'{item}.cross_sectional_area']
        interface = (np.tile(status_info['phase_center'], (3,1)).T != status_info['phase_corner'])
        #Calculating pc
        pc_array = np.ones(N) * pc
        for i in range(N):
            #Obtaining pc analyzing if the wp is trapped.
            bool_trap = False
            posible_cluster = status_info['cluster_corner'][i,:]
            wp_cluster = np.unique(posible_cluster[posible_cluster < 1])
            if len(wp_cluster) > 0:
                if wp_cluster in trapped_list:
                    pos = np.where(wp_cluster == trapped_list)[0]
                    bool_trap = True
            if bool_trap:
                pc_array[i] = pc_trapped_list[pos]
            else:
                pc_array[i] = pc
        #Calcualting some geometric data for all elements (less cost than do one by one)
        pc_array = np.tile(pc_array, (3,1)).T
        r = tension / pc_array
        bi_array = _if.interfacial_corner_distance(r, theta_r, beta, int_cond = interface)
        #Obtaining info for the outer AM (if it doesn exist, values are zero)
        A_corner_array = np.zeros_like(bi_array, dtype = float)
        A_center_array = np.copy(A_cross_section)
        ratio_wp_array = np.ones_like(A_cross_section, dtype= float)
        #Realizar lista de elementos para ignorar, que son los elementos del conduite de frontera
        #Para PDR, estos elementos siempre tendrán una unica fase
        for i in range(N):
            #Revisar si este elemento NO se puede ignorar
            if pn[f'{item}.internal'][i]:
                #Solo si hay oleo en el medio, se calcula corner area
                if ~status_info['phase_center'][i]:
                    #Center with the nwp. Check if the corners have wp
                    ratio_wp_array[i] = 0
                    for j in range(3):
                        if status_info['phase_corner'][i,j]:
                            A_corner = _if.corner_area(beta = beta[i,j],
                                                    theta = theta_r,
                                                    bi = bi_array[i,j])
                            A_corner_array[i,j] = A_corner
                            A_center_array[i] -= A_corner
                            ratio_wp_array[i] += A_corner / A_cross_section[i]
        #Saving data
        if np.any(A_corner_array < 0) or np.any(A_center_array < 0):
            revisar = np.where(A_center_array < 0)[0]
            print(trapped_list)
            print(status_str)
            print(f'pressure: {pc}')
            print(item)
            print(revisar)
            print(status_info['cluster_corner'][revisar,:])
            print(status_info['cluster_center'][revisar])
            print(pn[f'{item}.boundary'][revisar])
            print(pc_array[revisar,0])
            raise Exception('Problemas con Area que no deberian pasar')
        geometric_data[status_str][f'{item}.corner_area'] = np.copy(A_corner_array)
        geometric_data[status_str][f'{item}.center_area'] = np.copy(A_center_array)
        geometric_data[status_str][f'{item}.b_inner'] = np.copy(bi_array)
        geometric_data[status_str][f'{item}.wetting_phase_fraction'] = np.copy(ratio_wp_array)
        #Calculating saturation
        if item == 'pore':
            V_w = np.sum((pn['pore.volume'] * ratio_wp_array)[pn['pore.internal']])
            geometric_data[status_str]['wetting_phase_saturation'] = V_w / V_t
            sat.append(V_w / V_t)

#Calculating flow rate for each phase: wp and nwp
for wp in [True,False]:
    rel_perm = []
    if wp:
        phase = water
    else:
        phase = oil

    #Calculating conductance on each element
    for item in elements:
        viscosity = phase[f'{item}.viscosity'][0]
        A = pn[f'{item}.cross_sectional_area']
        G = pn[f'{item}.shape_factor']
        beta = pn[f'{item}.half_corner_angle']
        g = _cf.conductance_center(area_center = A,
                                   shape_factor = G,
                                   viscosity = viscosity,
                                   beta = beta,
                                   correction = True)
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
    flow_results[f'throat.{phase.name}_conductance'] = sph_g_L
    label_conductance = 'throat.conductance'
    phase[label_conductance] = sph_g_L

    #Calculating single phase flow
    sph_flow = Rate_calc(pn, phase, pn['pore.inlet'] , pn['pore.outlet'] , label_conductance)
    flow_results[phase.name+ '_flow'] = sph_flow

#MULTIPHASE
for s_inv in progressbar2:
    status_str = 'status ' + str(s_inv)
    flow_results[status_str] = {}
    pc = results_pdr[status_str]['invasion pressure']
    p_vals.append(pc)
    progressbar2.set_description(f"Calculating flow for {status_str}, pc = {str(pc)}")
    for wp in [True,False]:
        #ANTES DE INVADIR LOS ELEMENTOS DE ENTRADA. FALTA SOLUCIONAR EL TEMA DE AGUA
        rel_perm = []
        q_mph = []
        status_info = {}
        for item in elements:
            #Extracting data
            for loc in locations:
                status_info[f'{item}.cluster_{loc}'] = copy.deepcopy(results_pdr[status_str]['cluster_info'][f'{item}.{loc}'])
                status_info[f'{item}.phase_{loc}'] = copy.deepcopy(results_pdr[status_str]['invasion_info'][f'{item}.{loc}']) == wp
        if wp:
            phase = water
            viscosity = phase['pore.viscosity'][0]
            #invadiendo elementos del conduite de frontera con agua
            status = True
            for t in range(Nt):
                if pn['throat.boundary'][t]:
                    conns = pn['throat.conns'][t]
                    mask_pb = pn['pore.boundary'][conns]
                    pb = conns[mask_pb][0]
                    pi = conns[~mask_pb][0]
                    mask_wp_co = status_info['pore.phase_corner'][pi]
                    if np.any(mask_wp_co):
                        index = np.unique((status_info['pore.cluster_corner'][pi,:])[mask_wp_co])
                        for loc in locations:
                                status_info[f'pore.phase_{loc}'][pb] = status
                                status_info[f'pore.cluster_{loc}'][pb] = index
                                status_info[f'throat.phase_{loc}'][t] = status
                                status_info[f'throat.cluster_{loc}'][t] = index
        else:
            phase = oil
            viscosity = phase['pore.viscosity'][0]
            status = False
            for t in range(Nt):
                if pn['throat.boundary'][t] and not status_info['throat.phase_center'][t]:
                    conns = pn['throat.conns'][t]
                    index = results_pdr[status_str]['cluster_info']['pore.center'][conns[0]]
                    for p in conns:
                        status_info['pore.phase_corner'][p] = status
                        status_info['pore.cluster_corner'][p] = index
                    status_info['throat.phase_corner'][t] = status
                    status_info['throat.cluster_corner'][t] = index
        #Obtaining continuous clusters just looking pores (
        continuity_list = _if.identify_continuous_cluster(cluster_pore_center = status_info['pore.cluster_center'],
                                                            cluster_pore_corner = status_info['pore.cluster_corner'],
                                                            inlet_pores = pn['pore.inlet'],
                                                            outlet_pores = pn['pore.outlet'])
        cluster_list = []
        for loc in locations:
            cluster_list = np.concatenate((cluster_list , np.unique(status_info[f'pore.cluster_{loc}'][status_info[f'pore.phase_{loc}']])))
        cluster_list = np.unique(cluster_list)
        phase_clusters = np.intersect1d(cluster_list, continuity_list)
        #For each cluster of the same phase
        for n in phase_clusters:
            conductance_dict = {}
            for item in elements:
                N = len(pn[f'{item}.boundary'])
                A_ce = geometric_data[status_str][f'{item}.center_area']
                g_center = _cf.conductance_center(area_center = A_ce,
                                                             shape_factor = pn[f'{item}.shape_factor'],
                                                             viscosity = viscosity,
                                                             beta = pn[f'{item}.half_corner_angle'],
                                                             correction = True)
                g_center[ status_info[f'{item}.cluster_center'] != n] = 1e-35
                g_corner = np.ones((N,3), dtype = float) * 1e-35
                for i in range(N):
                    for j in range(3):
                        A_co = geometric_data[status_str][f'{item}.corner_area'][i,j]
                        if status_info[f'{item}.cluster_corner'][i,j] == n and A_co > 0:
                            g_corner[i,j] = _cf.conductance_corner(area_corner = A_co,
                                                                    beta = pn[f'{item}.half_corner_angle'][i,j],
                                                                    bi = geometric_data[status_str][f'{item}.b_inner'][i,j],
                                                                    theta = theta_r,
                                                                    viscosity = viscosity)
                g_corner = np.sum(g_corner,axis = 1)
                conductance_dict[f'{item}.center'] = g_center
                conductance_dict[f'{item}.corner'] = g_corner
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

            sph_g_L = flow_results[f'throat.{phase.name}_conductance']

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
for s_inv in status_array:
    status_str = 'status ' + str(s_inv)
    krel_water.append(flow_results[status_str]['water_krel'])
    krel_oil.append(flow_results[status_str]['oil_krel'])

sat = np.array(sat)
krel_water = np.array(krel_water)
krel_oil = np.array(krel_oil)
pc_array = np.array(p_vals)

#Saving data
output = np.vstack((sat, krel_water, krel_oil, pc_array)).T
np.save(f'results_K_Pc_PDR_{theta_r_sexag}_{p_kPa}kPa', output) #If wp, then 0. Otherwise 1
np.savetxt(f'results_K_Pc_PDR_{theta_r_sexag}_{p_kPa}kPa.txt', output, fmt=' %.5e '+' %.5e '+' %.5e ' + ' %.5e ', header=' sat // krel_w// krel_o // pc')
