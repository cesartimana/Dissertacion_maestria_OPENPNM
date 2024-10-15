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
from Properties import *
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
ws.settings.default_solver = 'ScipySpsolve'

proj = ws.load_project(filename=testName)

pn = proj.network
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


#Simulating primary Drainage
pdr = _alg.Primary_Drainage(network=pn, phase=water)
pdr.set_inlet_BC(pores=pn['pore.inlet'])
pdr.set_outlet_BC(pores=pn['pore.outlet'])
pdr.run(throat_diameter = 'throat.diameter')

#Post processing
p_vals =  np.array([0, pmax_drn])

#Obtaining phase distribution and clusters for each stage of invasion according to p_vals
results_pdr = pdr.postprocessing2(mode = 'pressure', inv_vals = p_vals)

#Strings used
elements = ['pore', 'throat']
locations = ['center', 'corner', 'layer']

#Inverting inlet and outlet elements
for item in elements:
    a = np.copy(pn[f'{item}.inlet'])
    pn[f'{item}.inlet'] = np.copy(pn[f'{item}.outlet'])
    pn[f'{item}.outlet'] = np.copy(a)

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

#------------------------------------------
#Read imbibition results
#Obtener datos de la ultima invasion registrada
with open(f'imbibition_process_{theta_r_sexag}_{theta_a_sexag}_{p_kPa}kPa.pkl', 'rb') as fp:
    info_imbibition = pickle.load(fp)

#Picking last info and transform in int
last_s_inv = int(list(info_imbibition)[-1][7:])
print(last_s_inv)
status_str = 'status ' + str(last_s_inv)

#How to extract info
#print(info_imbibition[status_str].keys())
#print(info_imbibition[status_str]['invasion_info'].keys())
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
    beta = pn[f'{item}.half_corner_angle']
    bi_corner = info_imbibition['status 0']['corner_info'][f'{item}.bi']
    flow_results[f'{item}.bi_drainage'] = bi_corner
    pc_array = info_imbibition['status 0']['pc_max_array'][f'{item}']
    pc_array = np.tile(pc_array, (3,1)).T
    flow_results[f'{item}.pc_theta_a'] = pc_array * np.cos(theta_a + beta) / np.cos(theta_r + beta)

#------------------------------------------

#STARTING CALCULATION
V_t = sum(pn['pore.volume'][pn['pore.internal']])
status_array = np.arange(45, last_s_inv + 1, 1, dtype = int)
flow_results['status_index'] = np.array(status_array)

#NEW: CALCULATING GEOMETRIC DATA
geometric_data = {}
sat = []
#theta_in, b_in, theta_o, b_o, A_ce, A_la, A_co. ratio_wp
#Asumo que solo hay un unico cluster de agua.
progressbar1 = tqdm(status_array)
for s_inv in progressbar1:
    status_str = 'status ' + str(s_inv)
    progressbar1.set_description("Calculating geometric data for %s" % status_str)
    geometric_data[status_str] = {}
    #Obtaining pressure and trapped info
    p_inv = info_imbibition[status_str]['invasion pressure']
    trapped_list = info_imbibition[status_str]['trapped_clusters']
    pc_trapped_list = info_imbibition[status_str]['pc_trapped']
    for item in elements:
        N = len(pn[f'{item}.boundary'])
        status_info = {}
        for loc in locations:
            status_info[f'cluster_{loc}'] = copy.deepcopy(info_imbibition[status_str]['cluster_info'][f'{item}.{loc}'])
            status_info[f'phase_{loc}'] = copy.deepcopy(info_imbibition[status_str]['invasion_info'][f'{item}.{loc}'])
        #Obtaining some geometric data for all elements
        beta = pn[f'{item}.half_corner_angle']
        A_cross_section =  pn[f'{item}.cross_sectional_area']
        interface = status_info['phase_layer'] != status_info['phase_corner']
        mask_LC = info_imbibition[status_str]['invasion_info'][f'{item}.broken_layer']
        #Calculating pc
        pc_array = np.ones(N) * p_inv
        for i in range(N):
            #Obtaining pc analyzing if the nwp is trapped.
            bool_trap = False
            if status_info['cluster_center'][i] in trapped_list:
                pos = np.where(status_info['cluster_center'][i] == trapped_list)[0]
                bool_trap = True
            else:
                for loc in ['corner', 'layer']:
                    elem_clusters = np.unique(status_info[f'cluster_{loc}'][i,:])
                    mask = np.isin(elem_clusters, trapped_list)
                    if np.any(mask):
                        bool_trap = True
                        index = elem_clusters[mask]
                        pos = np.where(trapped_list == index)[0]
            if bool_trap:
                pc_array[i] = pc_trapped_list[pos]
            else:
                pc_array[i] = p_inv
        #Calcualting some geometric data for all elements (less cost than do one by one)
        pc_array = np.tile(pc_array, (3,1)).T
        r = tension / pc_array
        mask_bi = flow_results[f'{item}.pc_theta_a'] < pc_array #True if bi is fixed (theta_hinging)
        #Tengo que calcular max pc para algunas esquinas porque puede que fueron atrapadas
        pmax_drn = info_imbibition['status 0']['pc_max_array'][f'{item}']
        pmax_drn = np.tile(pmax_drn, (3,1)).T
        theta_i_array = np.arccos( pc_array / pmax_drn * np.cos(theta_r + beta) ) - beta
        theta_i_array[~mask_bi] = theta_a
        bi_array = _if.interfacial_corner_distance(r, theta_i_array, beta, int_cond = interface)
        #Obtaining info for the outer AM (if it doesn exist, values are zero)
        theta_o_array = np.zeros_like(theta_i_array, dtype = float)
        bo_array = np.zeros_like(bi_array, dtype = float)
        A_corner_array = np.zeros_like(bi_array, dtype = float)
        A_layer_array = np.zeros_like(bi_array, dtype = float)
        A_center_array = np.copy(A_cross_section)
        ratio_wp_array = np.ones_like(A_cross_section, dtype= float)
        for i in range(N):
            if pn[f'{item}.internal'][i]:
                #Obtaining geometric info
                b_inner = bi_array[i,:]
                theta_inner = theta_i_array[i,:]
                beta_corner = beta[i,:]
                if status_info['phase_center'][i]:
                    #Center with wp. Analize if we have a not collapsed nwp layer
                    for j in range(3):
                        if ~status_info['phase_layer'][i,j] and ~mask_LC[i,j]:
                            #Outer interface. Calculate bo, theta_o and areas
                            A_corner = _if.corner_area(beta = beta[i,j],
                                                    theta = theta_i_array[i,j],
                                                    bi = bi_array[i,j])
                            b_outer = _if.interfacial_corner_distance(r[i,j], theta_a, beta[i,j], outer_AM = True)
                            A_corner_outer = _if.corner_area(beta = beta[i,j],
                                                    theta = np.pi - theta_a,
                                                    bi = b_outer)
                            theta_o_array[i,j] = theta_a
                            bo_array[i,j] = b_outer
                            A_corner_array[i,j] = A_corner
                            A_layer_array[i,j] = A_corner_outer - A_corner
                            A_center_array[i] -= A_corner_outer
                            ratio_wp_array[i] -= (A_corner_outer - A_corner) / A_cross_section[i]
                else:
                    #Center with the nwp. Check if the corners have wp
                    ratio_wp_array[i] = 0
                    for j in range(3):
                        if status_info['phase_corner'][i,j]:
                            A_corner = _if.corner_area(beta = beta[i,j],
                                                    theta = theta_i_array[i,j],
                                                    bi = bi_array[i,j])
                            A_corner_array[i,j] = A_corner
                            A_center_array[i] -= A_corner
                            ratio_wp_array[i] += A_corner / A_cross_section[i]
        #Saving data
        if np.any(A_corner_array < 0) or np.any(A_center_array < 0) or np.any(A_layer_array < 0):
            revisar = np.where(A_center_array < 0)[0]
            print('Area negativo')
            print(status_str)
            print(item)
            print(revisar)
            raise Exception('Revisar area negativo')
        geometric_data[status_str][f'{item}.corner_area'] = np.copy(A_corner_array)
        geometric_data[status_str][f'{item}.layer_area'] = np.copy(A_layer_array)
        geometric_data[status_str][f'{item}.center_area'] = np.copy(A_center_array)
        geometric_data[status_str][f'{item}.theta_inner'] = np.copy(theta_i_array)
        geometric_data[status_str][f'{item}.theta_outer'] = np.copy(theta_o_array)
        geometric_data[status_str][f'{item}.b_inner'] = np.copy(bi_array)
        geometric_data[status_str][f'{item}.b_outer'] = np.copy(bo_array)
        geometric_data[status_str][f'{item}.wetting_phase_fraction'] = np.copy(ratio_wp_array)
        geometric_data[status_str][f'{item}.pc'] = np.copy(pc_array)
        #Calculating saturation
        if item == 'pore':
            V_w = np.sum((pn['pore.volume'] * ratio_wp_array)[pn['pore.internal']])
            geometric_data[status_str]['wetting_phase_saturation'] = V_w / V_t
            sat.append(V_w / V_t)
raise Exception('o')
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
    flow_results[phase.name+ '_conductance'] = sph_g_L

#MULTIPHASE
progressbar2 = tqdm(status_array)
for s_inv in progressbar2:
    status_str = 'status ' + str(s_inv)
    flow_results[status_str] = {}
    pc = info_imbibition[status_str]['invasion pressure']
    progressbar2.set_description(f"Calculating flow for {status_str}, pc = {str(pc)}")
    for wp in [True,False]:
        rel_perm = []
        q_mph = []
        status_info = {}
        for item in elements:
            #Extracting data
            for loc in locations:
                status_info[f'{item}.cluster_{loc}'] = copy.deepcopy(info_imbibition[status_str]['cluster_info'][f'{item}.{loc}'])
                status_info[f'{item}.phase_{loc}'] = copy.deepcopy(info_imbibition[status_str]['invasion_info'][f'{item}.{loc}']) == wp
        if wp:
            phase = water
            #invadiendo elementos de frontera con agua
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
            status = False
            #Hacer que los boundary elements tengan la misma distribucion de fases que el internal pore de su mismo conduit
            for t in range(Nt):
                if pn['throat.boundary'][t]:
                    conns = pn['throat.conns'][t]
                    mask_pb = pn['pore.boundary'][conns]
                    pb = conns[mask_pb][0]
                    pi = conns[~mask_pb][0]
                    mask_nwp_la = status_info['pore.phase_layer'][pi]
                    mask_nwp_ce = status_info['pore.phase_center'][pi]
                    if np.any(mask_nwp_la) or mask_nwp_ce:
                        if mask_nwp_ce:
                            index = status_info['pore.cluster_center'][pi]
                        else:
                            index = np.unique((status_info['pore.cluster_layer'][pi,:])[mask_nwp_la])
                        for loc in locations:
                                status_info[f'pore.phase_{loc}'][pb] = status
                                status_info[f'pore.cluster_{loc}'][pb] = index
                                status_info[f'throat.phase_{loc}'][t] = status
                                status_info[f'throat.cluster_{loc}'][t] = index
        viscosity = phase['pore.viscosity'][0]


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
                g_layer = np.ones((N,3), dtype = float) * 1e-35
                for i in range(N):
                    for j in range(3):
                        A_co = geometric_data[status_str][f'{item}.corner_area'][i,j]
                        A_l = geometric_data[status_str][f'{item}.layer_area'][i,j]
                        if status_info[f'{item}.cluster_corner'][i,j] == n and A_co > 0:
                            g_corner[i,j] = _cf.conductance_corner(area_corner = A_co,
                                                                    beta = pn[f'{item}.half_corner_angle'][i,j],
                                                                    bi = geometric_data[status_str][f'{item}.b_inner'][i,j],
                                                                    theta = geometric_data[status_str][f'{item}.theta_inner'][i,j],
                                                                    viscosity = viscosity)
                        if status_info[f'{item}.cluster_layer'][i,j] == n and A_l > 0:
                            g_layer[i,j] = _cf.conductance_layer(theta_in = geometric_data[status_str][f'{item}.theta_inner'][i,j],
                                                                    theta_o = geometric_data[status_str][f'{item}.theta_outer'][i,j],
                                                                    beta = pn[f'{item}.half_corner_angle'][i,j],
                                                                    b_in = geometric_data[status_str][f'{item}.b_inner'][i,j],
                                                                    b_o = geometric_data[status_str][f'{item}.b_outer'][i,j],
                                                                    viscosity = viscosity)
                g_corner = np.sum(g_corner,axis = 1)
                g_layer = np.sum(g_layer,axis = 1)
                conductance_dict[f'{item}.center'] = g_center
                conductance_dict[f'{item}.corner'] = g_corner
                conductance_dict[f'{item}.layer'] = g_layer

            #Calculating conduit conductance
            g_L_mph = _cf.conduit_conductance_2phases(network = pn,
                                        pore_g_ce = conductance_dict['pore.center'],
                                        throat_g_ce = conductance_dict['throat.center'],
                                        conduit_length = L,
                                        pore_g_co = conductance_dict['pore.corner'],
                                        throat_g_co = conductance_dict['throat.corner'],
                                        pore_g_la = conductance_dict['pore.layer'],
                                        throat_g_la = conductance_dict['throat.layer'],
                                        corner = True,
                                        layer = True,
                                        check_boundary = True)
            label_conductance = 'throat.conduit_conductance'
            phase[label_conductance] = g_L_mph
            sph_g_L = flow_results[phase.name+ '_conductance']

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

sat = np.array(sat)
krel_water = np.array(krel_water)
krel_oil = np.array(krel_oil)
pc_array = np.array(pc_array)

#Saving data
output = np.vstack((sat, krel_water, krel_oil, pc_array)).T
np.save(f'results_K_Pc_imb_{theta_r_sexag}_{theta_a_sexag}_{p_kPa}kPa', output) #If wp, then 0. Otherwise 1
np.savetxt(f'results_K_Pc_imb{theta_r_sexag}_{theta_a_sexag}_{p_kPa}kPa.txt', output, fmt=' %.5e '+' %.5e '+' %.5e ' + ' %.5e ', header=' sat // krel_w// krel_o // pc')
