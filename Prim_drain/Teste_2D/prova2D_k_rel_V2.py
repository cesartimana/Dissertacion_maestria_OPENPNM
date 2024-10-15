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
import copy
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

sat = []

#Volume from pores
#Assuming all volume is on the pores
V_t = sum(pn['pore.volume'][pn['pore.internal']])

#Calculation conduit length
L = _cf.conduit_length_tubes(pn,
                             pore_length = "pore.diameter",
                             throat_spacing = "throat.spacing",
                             L_min = 1e-5,
                             check_boundary = True)

last_s_inv = int(list(results_pdr)[-1][7:])
status_array = np.arange(1, last_s_inv + 1, 1, dtype = int)
progressbar1 = tqdm(status_array)
progressbar2 = tqdm(status_array)

flow_results = {}
geometric_data = {}

#NEW: CALCULATING GEOMETRIC DATA
for s_inv in progressbar1:
    status_str = 'status_' + str(s_inv)
    progressbar1.set_description("Calculating geometric data for %s" % status_str)
    geometric_data[status_str] = {}
    pc = p_vals[int(s_inv-1)]
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
        r = tension / pc
        bi_array = _if.interfacial_corner_distance(r, theta_r, beta, int_cond = interface)
        #Obtaining info for the outer AM (if it doesn exist, values are zero)
        A_corner_array = np.zeros_like(bi_array, dtype = float)
        A_center_array = np.copy(A_cross_section)
        ratio_wp_array = np.ones_like(A_cross_section, dtype= float)
        #Realizar lista de elementos para ignorar, que son los elementos del conduite de frontera
        #Para PDR, estos elementos siempre tendrán una unica fase
        if item == 'pore':
            ignore_list = pn['throat.conns'][ pn['throat.boundary'] ]
        else:
            ignore_list = np.where(pn['throat.boundary'])[0]
        for i in range(N):
            #Revisar si este elemento NO se puede ignorar
            if i not in ignore_list:
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
            print(status_str)
            print(item)
            print(revisar)
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
    flow_results[f'throat.{phase.name}_conductance'] = sph_g_L
    label_conductance = 'throat.conductance'
    phase[label_conductance] = sph_g_L

    #Calculating single phase flow
    sph_flow = Rate_calc(pn, phase, pn['pore.inlet'] , pn['pore.outlet'] , label_conductance)
    flow_results[phase.name+ '_flow'] = sph_flow

#MULTIPHASE
for s_inv in progressbar2:
    status_str = 'status_' + str(s_inv)
    flow_results[status_str] = {}
    pc = p_vals[int(s_inv-1)]
    progressbar2.set_description(f"Calculating flow for {status_str}, pc = {str(pc)}")
    for wp in [False,True]:
        rel_perm = []
        if wp:
            phase = water
            #invadiendo elementos del conduite de frontera con agua
            status = True
            index = 0
            for t in range(Nt):
                if pn['throat.boundary'][t]:
                    conns = pn['throat.conns'][t]
                    index = results_pdr[status_str]['cluster_info']['pore.center'][conns[0]]
                    for p in conns:
                        for loc in locations:
                            results_pdr[status_str]['invasion_info'][f'pore.{loc}'][p] = status
                            results_pdr[status_str]['cluster_info'][f'pore.{loc}'][p] = index
                    for loc in locations:
                            results_pdr[status_str]['invasion_info'][f'throat.{loc}'][t] = status
                            results_pdr[status_str]['cluster_info'][f'throat.{loc}'][t] = index
                    #SOLO PARA PDR: invadir todas las esquinas del internal pore con agua
        else:
            phase = oil
            status = False
            for t in range(Nt):
                if pn['throat.boundary'][t]:
                    conns = pn['throat.conns'][t]
                    index = results_pdr[status_str]['cluster_info']['pore.center'][conns[0]]
                    for p in conns:
                        results_pdr[status_str]['invasion_info']['pore.corner'][p] = status
                        results_pdr[status_str]['cluster_info']['pore.corner'][p] = index
                    results_pdr[status_str]['invasion_info']['throat.corner'][t] = status
                    results_pdr[status_str]['cluster_info']['throat.corner'][t] = index
        q_mph = []
        viscosity = phase['pore.viscosity'][0]
        status_info = {}
        for item in elements:
            #Extracting data
            for loc in locations:
                status_info[f'{item}.cluster_{loc}'] = copy.deepcopy(results_pdr[status_str]['cluster_info'][f'{item}.{loc}'])
                status_info[f'{item}.phase_{loc}'] = copy.deepcopy(results_pdr[status_str]['invasion_info'][f'{item}.{loc}']) == wp
            #Obtaining continuous clusters just looking pores
            if item == 'pore':
                continuity_list = _if.identify_continuous_cluster(cluster_pore_center = status_info[f'{item}.cluster_center'],
                                                                  cluster_pore_corner = status_info[f'{item}.cluster_corner'],
                                                                  inlet_pores = pn['pore.inlet'],
                                                                  outlet_pores = pn['pore.outlet'])
                cluster_list = []
                for loc in locations:
                    cluster_list = np.concatenate((cluster_list , np.unique(status_info[f'{item}.cluster_{loc}'][status_info[f'{item}.phase_{loc}']])))
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
pc_array = p_vals
for s_inv in status_array:
    status_str = 'status_' + str(s_inv)
    krel_water.append(flow_results[status_str]['water_krel'])
    krel_oil.append(flow_results[status_str]['oil_krel'])

sat = np.array(sat)
krel_water = np.array(krel_water)
krel_oil = np.array(krel_oil)
pc_array = np.array(pc_array)

#Saving data
output = np.vstack((sat, krel_water, krel_oil, pc_array)).T
np.save('results_K_Pc_PDR', output) #If wp, then 0. Otherwise 1
np.savetxt('results_K_Pc_PDR.txt', output, fmt=' %.5e '+' %.5e '+' %.5e ' + ' %.5e ', header=' sat // krel_w// krel_o // pc')
