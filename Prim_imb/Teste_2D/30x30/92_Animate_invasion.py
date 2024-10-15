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
import matplotlib.animation as animation
from tqdm import tqdm
import pickle
import copy
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
Nt = pn.Nt

elements = ['pore', 'throat']
locations = ['center', 'corner']

fig1, ax1 = plt.subplots(figsize = (8,8))

"""
#SI NO HAY LAYERS PLOTAR ASI
#Obtener datos de la ultima invasion registrada
with open('imbibition_process_altering_BC.pkl', 'rb') as fp: #with open('imbibition_process_altering_BC.pkl', 'rb') as fp:
    info_imbibition = pickle.load(fp)
#Recordar que al cambiar la entrada, tengo que cambiar inlet por outlet
last_s_inv = int(list(info_imbibition)[-1][7:]) #Picking last info and transform in int

def update(num):
    status_str = f'status {num}'
    p_inv = info_imbibition[status_str]['invasion pressure']
    check_cluster = copy.deepcopy(info_imbibition[status_str])
    #invadiendo elementos de frontera de acuerdo al internal pore
    for t in range(Nt):
        if pn['throat.outlet'][t]: #outlet si dice altering. Inlet si no
            conns = pn['throat.conns'][t]
            for p in conns:
                if pn['pore.outlet'][p]: #outlet si cambie la entrada. Inlet si no
                    p_b = p
                else:
                    p_int = p
            status = check_cluster['invasion_info']['pore.center'][p_int]
            index = check_cluster['cluster_info']['pore.center'][p_int]
            #Modifying boundary elements, center, layer and corner phases
            for item in elements:
                if item == 'pore':
                    n = p_b
                else:
                    n = t
                loc = 'center'
                check_cluster['invasion_info'][f'{item}.{loc}'][n] = status
                check_cluster['cluster_info'][f'{item}.{loc}'][n] = index
    cont_list = _if.identify_continuous_cluster(check_cluster['cluster_info']['pore.center'] , check_cluster['cluster_info']['pore.corner'], pn['pore.inlet'], pn['pore.outlet'])
    cont_list = np.delete(cont_list, 0)
    ax1.clear()
    #Boundary elements
    _ = op.visualization.plot_connections(network=pn, throats=pn.throats('boundary'), ax=ax1, c = 'r', linewidth = 9, zorder = 2)
    _ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=500, ax = ax1, zorder = 8)
    #WP corner
    _ = op.visualization.plot_connections(network=pn, throats=pn.throats('internal'), ax=ax1, c = 'b', linewidth = 9, zorder = 0)
    _ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('internal') , c='b', s=500, ax = ax1, zorder = 3)
    #Specific NWP cluster
    pattern_cont_t = np.isin(check_cluster['cluster_info']['throat.center'], cont_list) &  pn['throat.internal']
    pattern_cont_p = np.isin(check_cluster['cluster_info']['pore.center'], cont_list) &  pn['pore.internal']
    _ = op.visualization.plot_connections(network=pn, throats=pattern_cont_t, ax=ax1, c = 'yellow', linewidth = 5, zorder = 4)
    _ = op.visualization.plot_coordinates(network=pn, pores=pattern_cont_p, c='yellow', s=250, ax = ax1, zorder = 6)
    #NWP center
    pattern_t = ~check_cluster['invasion_info']['throat.center'] & pn['throat.internal'] & ~pattern_cont_t
    pattern_p = ~check_cluster['invasion_info']['pore.center'] & pn['pore.internal']  & ~pattern_cont_p
    _ = op.visualization.plot_connections(network=pn, throats=pattern_t, ax=ax1, c = 'w', linewidth = 5, zorder = 4)
    _ = op.visualization.plot_coordinates(network=pn, pores=pattern_p, c='w', s=250, ax = ax1, zorder = 6)

    ax1.set_title(f'Status {num}, pressure = {int(p_inv)} Pa', fontsize = 25)
    fig1.tight_layout()
    return 0
"""


#SI HAY LAYERS ACTIVAR ESTO
#Obtener datos de la ultima invasion registrada
with open('imbibition_process_layer_altering_BC.pkl', 'rb') as fp: #with open('imbibition_process_altering_BC.pkl', 'rb') as fp:
    info_imbibition = pickle.load(fp)
locations = ['center', 'corner', 'layer']
#Recordar que al cambiar la entrada, tengo que cambiar inlet por outlet
last_s_inv = int(list(info_imbibition)[-1][7:]) #Picking last info and transform in int

def update_layer(num):
    status_str = f'status {num}'
    p_inv = info_imbibition[status_str]['invasion pressure']
    check_cluster = copy.deepcopy(info_imbibition[status_str])
    print(check_cluster['trapped_clusters'])
    print(check_cluster['cluster_info']['index_list'])
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
    ax1.clear()
    #Boundary elements
    _ = op.visualization.plot_connections(network=pn, throats=pn.throats('boundary'), ax=ax1, c = 'r', linewidth = 11, zorder = 0)
    _ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=600, ax = ax1, zorder = 0)
    #WP corner
    _ = op.visualization.plot_connections(network=pn, throats=pn.throats('internal'), ax=ax1, c = 'b', linewidth = 11, zorder = 1)
    _ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('internal') , c='b', s=600, ax = ax1, zorder = 1)
    #phase layer: WP, NWP, and NWP simple spanning cluster
    #Simple spanning
    pattern_cont_t_la = np.any(np.isin(check_cluster['cluster_info']['throat.layer'], cont_list), axis = 1) &  pn['throat.internal']
    pattern_cont_p_la = np.any(np.isin(check_cluster['cluster_info']['pore.layer'], cont_list), axis = 1) &  pn['pore.internal']
    _ = op.visualization.plot_connections(network=pn, throats=pattern_cont_t_la, ax=ax1, c = 'yellow', linewidth = 8, zorder = 2)
    _ = op.visualization.plot_coordinates(network=pn, pores=pattern_cont_p_la, c='yellow', s=400, ax = ax1, zorder = 3)
    #NWP
    pattern_t_la = np.any(~check_cluster['invasion_info']['throat.layer'], axis= 1) & pn['throat.internal'] & ~pattern_cont_t_la
    pattern_p_la = np.any(~check_cluster['invasion_info']['pore.layer'], axis = 1) & pn['pore.internal']  & ~pattern_cont_p_la
    _ = op.visualization.plot_connections(network=pn, throats=pattern_t_la, ax=ax1, c = 'w', linewidth = 8, zorder = 2)
    _ = op.visualization.plot_coordinates(network=pn, pores=pattern_p_la, c='w', s=400, ax = ax1, zorder = 3)
    #WP is not necessary
    #phase center:  WP, NWP, and NWP simple spanning cluster
    #Specific NWP cluster center
    pattern_cont_t_ce = np.isin(check_cluster['cluster_info']['throat.center'], cont_list) &  pn['throat.internal']
    pattern_cont_p_ce = np.isin(check_cluster['cluster_info']['pore.center'], cont_list) &  pn['pore.internal']
    _ = op.visualization.plot_connections(network=pn, throats=pattern_cont_t_ce, ax=ax1, c = 'yellow', linewidth = 5, zorder = 4)
    _ = op.visualization.plot_coordinates(network=pn, pores=pattern_cont_p_ce, c='yellow', s=200, ax = ax1, zorder = 5)
    #NWP center
    pattern_t_ce = ~check_cluster['invasion_info']['throat.center'] & pn['throat.internal'] & ~pattern_cont_t_ce
    pattern_p_ce = ~check_cluster['invasion_info']['pore.center'] & pn['pore.internal']  & ~pattern_cont_p_ce
    _ = op.visualization.plot_connections(network=pn, throats=pattern_t_ce, ax=ax1, c = 'w', linewidth = 5, zorder = 4)
    _ = op.visualization.plot_coordinates(network=pn, pores=pattern_p_ce, c='w', s=200, ax = ax1, zorder = 5)
    #WP center
    pattern_t_ce_wp = check_cluster['invasion_info']['throat.center'] & pn['throat.internal']
    pattern_p_ce_wp = check_cluster['invasion_info']['pore.center'] & pn['pore.internal']
    _ = op.visualization.plot_connections(network=pn, throats=pattern_t_ce_wp, ax=ax1, c = 'b', linewidth = 5, zorder = 4)
    _ = op.visualization.plot_coordinates(network=pn, pores=pattern_p_ce_wp, c='b', s=200, ax = ax1, zorder = 5)
    #Cluster
    #cluster_index = 23
    #cluster_t_ce = (check_cluster['cluster_info']['throat.center'] == cluster_index)
    #cluster_p_ce = (check_cluster['cluster_info']['pore.center'] == cluster_index)
    #_ = op.visualization.plot_connections(network=pn, throats=cluster_t_ce, ax=ax1, c = 'g', linewidth = 5, zorder = 6)
    #_ = op.visualization.plot_coordinates(network=pn, pores=cluster_p_ce, c='g', s=200, ax = ax1, zorder = 7)
    #ax1.set_title(f'Status {num}, pressure = {int(p_inv)} Pa', fontsize = 25)
    ax1.set_title(f'Estado {num}, $\Delta$ p = {int(p_inv)} Pa', fontsize = 25)
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    fig1.tight_layout()
    return 0

frames = np.arange(1, 202, 20) #last_s_inv es mucho. A partir de 201 solo hay ruptura de camadas
np.append(frames, last_s_inv)

ani = animation.FuncAnimation(fig1, update_layer, frames=frames)
ani.save('animation_drawing.gif', writer='imagemagick', fps=0.75)

#FFwriter = animation.FFMpegWriter(fps=1, extra_args=['-vcodec', 'libx264'])
#ani.save('animation.mp4', writer = FFwriter)
