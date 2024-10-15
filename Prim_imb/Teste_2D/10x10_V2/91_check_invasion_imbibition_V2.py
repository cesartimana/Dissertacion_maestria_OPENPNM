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
from Properties import *
#importing functions for other file
import sys
sys.path.insert(1, '/home/cesar/OpenPNM_files/mestrado/_funcs')
import _algorithm_class as _alg
import _conductance_funcs as _cf
import _invasion_funcs as _if
#end
np.random.seed(13)

#Reading pnm arquive/ network
ws = op.Workspace()
proj = ws.load_project(filename=testName)

pn = proj.network

#SOLO PARA RED CREADA, ACTUALIZANDO NUMERO DE GARGANTAS
Np = pn.Np
Nt = pn.Nt

#FOR IMBIBITION
with open(f'imbibition_process_{theta_r_sexag}_{theta_a_sexag}_{p_kPa}kPa.pkl', 'rb') as fp:
    results_pdr = pickle.load(fp)
last_s_str = list(results_pdr)[-1]
print(last_s_str)
status_index =  (list(results_pdr)[-1])[7:]
print(status_index)
check_cluster = results_pdr['status ' + str(status_index)]
p_inv = check_cluster['invasion pressure']
trapped = check_cluster['trapped_clusters']
nontrapped = check_cluster['nontrapped_clusters']
print(f'trapped clusters: {trapped}')
print(f'nontrapped clusters: {nontrapped}')

elements = ['pore', 'throat']
locations = ['center', 'corner', 'layer']

#Hacer que los boundary elements tengan la misma distribucion de fases que el internal pore de su mismo conduit
for t in range(Nt):
    if pn['throat.boundary'][t]:
        conns = pn['throat.conns'][t]
        for p in conns:
            if pn['pore.boundary'][p]:
                p_b = p
            else:
                p_int = p
        #Modifying boundary elements, center, layer and corner phases
        for loc in locations:
            status = check_cluster['invasion_info'][f'pore.{loc}'][p_int]
            index = check_cluster['cluster_info'][f'pore.{loc}'][p_int]
            for item in elements:
                if item == 'pore':
                    n = p_b
                else:
                    n = t
                check_cluster['invasion_info'][f'{item}.{loc}'][n] = status
                check_cluster['cluster_info'][f'{item}.{loc}'][n] = index
cont_list = _if.identify_continuous_cluster(cluster_pore_center = check_cluster['cluster_info']['pore.center'],
                                            cluster_pore_corner = check_cluster['cluster_info']['pore.corner'],
                                            inlet_pores = pn['pore.inlet'],
                                            outlet_pores = pn['pore.outlet'],
                                            cluster_pore_layer = check_cluster['cluster_info']['pore.layer'],
                                            layer = True)
cont_list = np.delete(cont_list, 0)

#Set size pore and throats
sp = 150
st = 5

#plotting
fig1, ax1 = plt.subplots(figsize = (8,8))

#Boundary elements
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('boundary'), ax=ax1, c = 'r', linewidth = st, zorder = 0)
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=sp, ax = ax1, zorder = 1)

#sabiendo que tipo de invasion
elements = ['pore', 'throat']
for item in elements:
    #Agua en el centro
    mask_wp_center = check_cluster['invasion_info'][f'{item}.center'] & pn[f'{item}.internal']
    #revisar si hay alguna camada con oleo
    mask_nwp_layer = np.any(~check_cluster['invasion_info'][f'{item}.layer'], axis = 1) & pn[f'{item}.internal']
    #Nwp en el centro
    mask_nwp_center = ~check_cluster['invasion_info'][f'{item}.center'] & pn[f'{item}.internal']
    #Alguna Wp en las esquinas
    mask_wp_corner = np.any(check_cluster['invasion_info'][f'{item}.corner'], axis = 1) & pn[f'{item}.internal']

    #Setting the mask for plottitng
    mask_sat_1 = mask_wp_center & ~mask_nwp_layer
    mask_wp_w_layer = mask_wp_center & mask_nwp_layer
    mask_nwp_w_corner = mask_nwp_center & mask_wp_corner
    mask_sat_0 = mask_nwp_center & ~mask_wp_corner

    #Plotting
    if item == 'pore':
        _ = op.visualization.plot_coordinates(network=pn, pores= mask_sat_1, c='darkblue', s=sp, ax = ax1, zorder = 1)
        _ = op.visualization.plot_coordinates(network=pn, pores= mask_wp_w_layer, c='blue', s=sp, ax = ax1, zorder = 1)
        _ = op.visualization.plot_coordinates(network=pn, pores= mask_nwp_w_corner, c='lightblue', s=sp, ax = ax1, zorder = 1)
        _ = op.visualization.plot_coordinates(network=pn, pores= mask_sat_0, c='gainsboro', s=sp, ax = ax1, zorder = 1)
    else:
        _ = op.visualization.plot_connections(network=pn, throats=mask_sat_1, ax=ax1, c = 'darkblue', linewidth = st, zorder = 0)
        _ = op.visualization.plot_connections(network=pn, throats=mask_wp_w_layer, ax=ax1, c = 'blue', linewidth = st, zorder = 0)
        _ = op.visualization.plot_connections(network=pn, throats=mask_nwp_w_corner, ax=ax1, c = 'lightblue', linewidth = st, zorder = 0)
        _ = op.visualization.plot_connections(network=pn, throats=mask_sat_0, ax=ax1, c = 'gainsboro', linewidth = st, zorder = 0)

#ax1.set_title(f'Invasion for status {status_index}, pc = {round(p_inv,1)} Pa', fontsize = 20)
plt.tight_layout()


"""
#throat corner
mask_co_all_wp_throat = np.all(check_cluster['invasion_info']['throat.corner'], axis = 1) & pn['throat.internal']
mask_co_all_nwp_throat = np.all(~check_cluster['invasion_info']['throat.corner'], axis = 1) & pn['throat.internal']
mask_co_mixed_throat = ~mask_co_all_wp_throat &  ~mask_co_all_nwp_throat & pn['throat.internal']

_ = op.visualization.plot_connections(network=pn, throats=mask_co_all_wp_throat, ax=ax1, c = 'b', linewidth = 15, zorder = 1)
_ = op.visualization.plot_connections(network=pn, throats=mask_co_mixed_throat, ax=ax1, c = 'skyblue', linewidth = 15, zorder = 1)
_ = op.visualization.plot_connections(network=pn, throats=mask_co_all_nwp_throat, ax=ax1, c = 'gainsboro', linewidth = 15, zorder = 1)

#pore corner
mask_co_all_wp_pore = np.all(check_cluster['invasion_info']['pore.corner'], axis = 1) & pn['pore.internal']
mask_co_all_nwp_pore = np.all(~check_cluster['invasion_info']['pore.corner'], axis = 1) & pn['pore.internal']
mask_co_mixed_pore = ~mask_co_all_wp_pore &  ~mask_co_all_nwp_pore & pn['pore.internal']

_ = op.visualization.plot_coordinates(network=pn, pores=mask_co_all_wp_pore , c='b', s=1200, ax = ax1, zorder = 8)
_ = op.visualization.plot_coordinates(network=pn, pores=mask_co_mixed_pore , c='skyblue', s=1200, ax = ax1, zorder = 4)
_ = op.visualization.plot_coordinates(network=pn, pores=mask_co_all_nwp_pore , c='gainsboro', s=1200, ax = ax1, zorder = 4)

#throat layer
mask_la_all_wp_throat = np.all(check_cluster['invasion_info']['throat.layer'], axis = 1) & pn['throat.internal']
mask_la_all_nwp_throat = np.all(~check_cluster['invasion_info']['throat.layer'], axis = 1) & pn['throat.internal']
mask_la_mixed_throat = ~mask_la_all_wp_throat &  ~mask_la_all_nwp_throat & pn['throat.internal']

_ = op.visualization.plot_connections(network=pn, throats=mask_la_all_wp_throat, ax=ax1, c = 'b', linewidth = 10, zorder = 2)
_ = op.visualization.plot_connections(network=pn, throats=mask_la_mixed_throat, ax=ax1, c = 'skyblue', linewidth = 10, zorder = 2)
_ = op.visualization.plot_connections(network=pn, throats=mask_la_all_nwp_throat, ax=ax1, c = 'gainsboro', linewidth = 10, zorder = 6)

#pore layer
mask_la_all_wp_pore = np.all(check_cluster['invasion_info']['pore.layer'], axis = 1) & pn['pore.internal']
mask_la_all_nwp_pore = np.all(~check_cluster['invasion_info']['pore.layer'], axis = 1) & pn['pore.internal']
mask_la_mixed_pore = ~mask_la_all_wp_pore &  ~mask_la_all_nwp_pore & pn['pore.internal']

_ = op.visualization.plot_coordinates(network=pn, pores=mask_la_all_wp_pore , c='b', s=700, ax = ax1, zorder = 5)
_ = op.visualization.plot_coordinates(network=pn, pores=mask_la_mixed_pore , c='skyblue', s=700, ax = ax1, zorder = 9)
_ = op.visualization.plot_coordinates(network=pn, pores=mask_la_all_nwp_pore , c='gainsboro', s=700, ax = ax1, zorder = 5)

#throat center
mask_center_wp_throat = check_cluster['invasion_info']['throat.center'] & pn['throat.internal']
mask_center_nwp_throat = ~check_cluster['invasion_info']['throat.center'] & pn['throat.internal']

_ = op.visualization.plot_connections(network=pn, throats=mask_center_wp_throat, ax=ax1, c = 'b', linewidth = 5, zorder = 3)
_ = op.visualization.plot_connections(network=pn, throats=mask_center_nwp_throat, ax=ax1, c = 'gainsboro', linewidth = 5, zorder = 7)

#pore center
mask_center_wp_pore = check_cluster['invasion_info']['pore.center'] & pn['pore.internal']
mask_center_nwp_pore = ~check_cluster['invasion_info']['pore.center'] & pn['pore.internal']

_ = op.visualization.plot_coordinates(network=pn, pores=mask_center_wp_pore , c='b', s=200, ax = ax1, zorder = 25)
_ = op.visualization.plot_coordinates(network=pn, pores=mask_center_nwp_pore , c='gainsboro', s=200, ax = ax1, zorder = 25)

ax1.set_title(f'Invasion for status {status_index}, pc = {round(p_inv,1)} Pa', fontsize = 20)
plt.tight_layout()
"""



color_list = [ 'orange', 'gold', 'lime', 'cyan', 'gainsboro', 'skyblue', 'blue', 'blueviolet', 'violet', 'pink', 'wheat', 'olive', 'teal', 'hotpink']

def plot_color_clusters(cl_index, color, ax):
    for item in elements:
        #Cluster center
        mask_cluster_center = (check_cluster['cluster_info'][f'{item}.center'] == cl_index) & pn[f'{item}.internal']
        #Cluster layer
        mask_cluster_layer = np.any(check_cluster['cluster_info'][f'{item}.layer'] == cl_index, axis = 1) & pn[f'{item}.internal']
        #Cluster corner
        mask_cluster_corner = np.any(check_cluster['cluster_info'][f'{item}.corner'] == cl_index, axis = 1) & pn[f'{item}.internal']

        mask_any_cluster = mask_cluster_center | mask_cluster_layer | mask_cluster_corner

        if item == 'pore':
            _ = op.visualization.plot_coordinates(network=pn, pores= mask_any_cluster, c=color, s=sp, ax = ax, zorder = 1)
        else:
            _ = op.visualization.plot_connections(network=pn, throats=mask_any_cluster, ax=ax, c = color, linewidth = st, zorder = 0)
    return

#Prepare non-wetting phase and wp list
cluster_list = np.array(check_cluster['cluster_info']['index_list'])
print(cluster_list)
nwp_list = cluster_list[cluster_list >= 1]
wp_list = cluster_list[cluster_list < 1]
print(f'non wetting clusters: {nwp_list}')
print(f'wetting clusters: {wp_list}')
#Plot nwp clusters
fig2, ax2 = plt.subplots(figsize = (8,8))
#Boundary elements
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('boundary'), ax=ax2, c = 'r', linewidth = st, zorder = 0)
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=sp, ax = ax2, zorder = 0)
for i in range(len(nwp_list)):
    if i < len(color_list):
        plot_color_clusters(cl_index = nwp_list[i], color = color_list[i], ax = ax2)
    else:
        plot_color_clusters(cl_index = nwp_list[i], color = str(round(i / len(nwp_list),2)), ax = ax2)
ax2.set_title('Non-wetting phase clusters', fontsize = 20)
plt.tight_layout()

#Plot wetting phase clusters
fig3, ax3 = plt.subplots(figsize = (8,8))

for i in range(len(wp_list)):
    if i < len(color_list):
        plot_color_clusters(cl_index = wp_list[i], color = color_list[i], ax = ax3)
    else:
        plot_color_clusters(cl_index = wp_list[i], color = str(round(i / len(wp_list),2)), ax = ax3)
ax3.set_title('Wetting phase clusters', fontsize = 20)
plt.tight_layout()

#Plot specific cluster_index
cluster_index = 13
fig4, ax4 = plt.subplots(figsize = (8,8))
#Boundary elements
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('boundary'), ax=ax4, c = 'r', linewidth = st, zorder = 0)
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=sp, ax = ax4, zorder = 0)
plot_color_clusters(cl_index = cluster_index, color = 'b', ax = ax4)
ax4.set_title(f'Cluster {cluster_index}', fontsize = 20)

plt.tight_layout()


plt.show()
