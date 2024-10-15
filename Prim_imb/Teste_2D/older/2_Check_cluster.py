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
testName ='Rede_2D_10x10.pnm'
proj = ws.load_project(filename=testName)

pn = proj.network
Np = pn.Np
Nt = pn.Nt

#Obtener datos de la ultima invasion registrada
elements = ['pore', 'throat']
locations = ['center', 'corner', 'layer']

"""

#REVISAR SOLO SI UN CLUSTER DE NWP FUE DIVIDIDO. INFO RELACIONADA A NWP
with open('status_11_divided_cluster_.pkl', 'rb') as fp:
    check_cluster = pickle.load(fp)

print(check_cluster.keys())

for item in elements:
    for loc in locations:
        print(f'Non wetting phase in the {item} {loc}')
        print(np.any( check_cluster['invasion_info'][f'{item}.{loc}'] ))

fig1, ax1 = plt.subplots(figsize = (8,8))
#Boundary elements
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=200, ax = ax1, zorder = 1)
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('all'), ax=ax1, c = 'gray', linewidth = 4, zorder = 0)
#NWP layer
_ = op.visualization.plot_coordinates(network=pn, pores=np.any(check_cluster['invasion_info']['pore.layer'], axis=1)  , c='b', s=200, ax = ax1, zorder = 2)
_ = op.visualization.plot_connections(network=pn, throats=np.any(check_cluster['invasion_info']['throat.layer'], axis=1)  , ax=ax1, c = 'b', linewidth = 7, zorder = 2)
#NWP center
_ = op.visualization.plot_coordinates(network=pn, pores=check_cluster['invasion_info']['pore.center']  , c='y', s=120, ax = ax1, zorder = 3)
_ = op.visualization.plot_connections(network=pn, throats=check_cluster['invasion_info']['throat.center'] , ax=ax1, c = 'y', linewidth = 4, zorder = 3)
#Element invaded( yo se que elemento fue)
_ = op.visualization.plot_connections(network=pn, throats=78 , ax=ax1, c = 'g', linewidth = 4, zorder = 4)
ax1.set_title('NON WETTING PHASE', fontsize = 22)
plt.tight_layout()


fig2, ax2 = plt.subplots(figsize = (8,8))
#Boundary elements
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=200, ax = ax2, zorder = 1)
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('all'), ax=ax2, c = 'gray', linewidth = 4, zorder = 0)
#NWP layer
_ = op.visualization.plot_coordinates(network=pn, pores=np.any(check_cluster['cluster_info']['pore.layer']  == 1, axis=1)  , c='b', s=200, ax = ax2, zorder = 2)
_ = op.visualization.plot_connections(network=pn, throats=np.any(check_cluster['cluster_info']['throat.layer']  == 1, axis=1)  , ax=ax2, c = 'b', linewidth = 7, zorder = 2)
#NWP center
_ = op.visualization.plot_coordinates(network=pn, pores=check_cluster['cluster_info']['pore.center'] == 1  , c='y', s=120, ax = ax2, zorder = 3)
_ = op.visualization.plot_connections(network=pn, throats=check_cluster['cluster_info']['throat.center'] == 1 , ax=ax2, c = 'y', linewidth = 4, zorder = 3)
ax2.set_title('CLUSTER 1', fontsize = 22)
plt.tight_layout()

fig3, ax3 = plt.subplots(figsize = (8,8))
#Boundary elements
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=200, ax = ax3, zorder = 1)
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('all'), ax=ax3, c = 'gray', linewidth = 4, zorder = 0)
#NWP layer
_ = op.visualization.plot_coordinates(network=pn, pores=np.any(check_cluster['cluster_info']['pore.layer']  == 2, axis=1)  , c='b', s=200, ax = ax3, zorder = 2)
_ = op.visualization.plot_connections(network=pn, throats=np.any(check_cluster['cluster_info']['throat.layer']  == 2, axis=1)  , ax=ax3, c = 'b', linewidth = 7, zorder = 2)
#NWP center
_ = op.visualization.plot_coordinates(network=pn, pores=check_cluster['cluster_info']['pore.center'] == 2  , c='y', s=120, ax = ax3, zorder = 3)
_ = op.visualization.plot_connections(network=pn, throats=check_cluster['cluster_info']['throat.center'] == 2 , ax=ax3, c = 'y', linewidth = 4, zorder = 3)
ax3.set_title('CLUSTER 2', fontsize = 22)
plt.tight_layout()
print(np.where(np.any(check_cluster['cluster_info']['pore.layer']  == 2, axis=1)))
plt.show()

"""

"""

#TODOS LOS POSIBLES CLUSTERS DE UN ESTADO ESPECIFICO

#REVISAR SOLO SI UN CLUSTER DE NWP FUE DIVIDIDO. INFO RELACIONADA A NWP
with open('status_11_all_after.pkl', 'rb') as fp:
    check_cluster = pickle.load(fp)

print(check_cluster.keys())

for item in elements:
    for loc in locations:
        print(f'Non wetting phase in the {item} {loc}')
        print(np.any( ~check_cluster['invasion_info'][f'{item}.{loc}'] ))

print(check_cluster['cluster_info']['index_list'])
fig1, ax1 = plt.subplots(figsize = (8,8))
#Boundary elements
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=200, ax = ax1, zorder = 1)
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('all'), ax=ax1, c = 'gray', linewidth = 4, zorder = 0)
#NWP layer
_ = op.visualization.plot_coordinates(network=pn, pores=np.any(~check_cluster['invasion_info']['pore.layer'], axis=1)  , c='b', s=200, ax = ax1, zorder = 2)
_ = op.visualization.plot_connections(network=pn, throats=np.any(~check_cluster['invasion_info']['throat.layer'], axis=1)  , ax=ax1, c = 'b', linewidth = 7, zorder = 2)
#NWP center
_ = op.visualization.plot_coordinates(network=pn, pores=~check_cluster['invasion_info']['pore.center']  , c='y', s=120, ax = ax1, zorder = 3)
_ = op.visualization.plot_connections(network=pn, throats=~check_cluster['invasion_info']['throat.center'] , ax=ax1, c = 'y', linewidth = 4, zorder = 3)
#Element invaded( yo se que elemento fue)
#_ = op.visualization.plot_connections(network=pn, throats=78 , ax=ax1, c = 'g', linewidth = 4, zorder = 4)
ax1.set_title('NON WETTING PHASE', fontsize = 22)
plt.tight_layout()

fig2, ax2 = plt.subplots(figsize = (8,8))
#Boundary elements
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=200, ax = ax2, zorder = 1)
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('all'), ax=ax2, c = 'gray', linewidth = 4, zorder = 0)
#NWP layer
_ = op.visualization.plot_coordinates(network=pn, pores=np.any(check_cluster['cluster_info']['pore.layer']  == 1, axis=1)  , c='b', s=200, ax = ax2, zorder = 2)
_ = op.visualization.plot_connections(network=pn, throats=np.any(check_cluster['cluster_info']['throat.layer']  == 1, axis=1)  , ax=ax2, c = 'b', linewidth = 7, zorder = 2)
#NWP center
_ = op.visualization.plot_coordinates(network=pn, pores=check_cluster['cluster_info']['pore.center'] == 1  , c='y', s=120, ax = ax2, zorder = 3)
_ = op.visualization.plot_connections(network=pn, throats=check_cluster['cluster_info']['throat.center'] == 1 , ax=ax2, c = 'y', linewidth = 4, zorder = 3)
ax2.set_title('CLUSTER 1', fontsize = 22)
plt.tight_layout()

fig3, ax3 = plt.subplots(figsize = (8,8))
#Boundary elements
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=200, ax = ax3, zorder = 1)
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('all'), ax=ax3, c = 'gray', linewidth = 4, zorder = 0)
#NWP layer
_ = op.visualization.plot_coordinates(network=pn, pores=np.any(check_cluster['cluster_info']['pore.layer']  == 2, axis=1)  , c='b', s=200, ax = ax3, zorder = 2)
_ = op.visualization.plot_connections(network=pn, throats=np.any(check_cluster['cluster_info']['throat.layer']  == 2, axis=1)  , ax=ax3, c = 'b', linewidth = 7, zorder = 2)
#NWP center
_ = op.visualization.plot_coordinates(network=pn, pores=check_cluster['cluster_info']['pore.center'] == 2 , c='y', s=120, ax = ax3, zorder = 3)
_ = op.visualization.plot_connections(network=pn, throats=check_cluster['cluster_info']['throat.center'] == 2 , ax=ax3, c = 'y', linewidth = 4, zorder = 3)
ax3.set_title('CLUSTER 2', fontsize = 22)
plt.tight_layout()

plt.show()
"""

#LEER EL RESULTADO DE INVASION COMPLETA

#REVISAR SOLO SI UN CLUSTER DE NWP FUE DIVIDIDO. INFO RELACIONADA A NWP
with open('imbibition_process_layer_altering_BC.pkl', 'rb') as fp:
    results = pickle.load(fp)

status_str = 'status 64'
check_cluster = copy.deepcopy(results[status_str])
print(check_cluster.keys())

for item in elements:
    for loc in locations:
        print(f'Non wetting phase in the {item} {loc}')
        print(np.any( ~check_cluster['invasion_info'][f'{item}.{loc}'] ))

print(check_cluster['cluster_info']['index_list'])
fig1, ax1 = plt.subplots(figsize = (8,8))
#Boundary elements
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=200, ax = ax1, zorder = 1)
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('all'), ax=ax1, c = 'gray', linewidth = 4, zorder = 0)
#NWP layer
_ = op.visualization.plot_coordinates(network=pn, pores=np.any(~check_cluster['invasion_info']['pore.layer'], axis=1)  , c='b', s=200, ax = ax1, zorder = 2)
_ = op.visualization.plot_connections(network=pn, throats=np.any(~check_cluster['invasion_info']['throat.layer'], axis=1)  , ax=ax1, c = 'b', linewidth = 7, zorder = 2)
#NWP center
_ = op.visualization.plot_coordinates(network=pn, pores=~check_cluster['invasion_info']['pore.center']  , c='y', s=120, ax = ax1, zorder = 3)
_ = op.visualization.plot_connections(network=pn, throats=~check_cluster['invasion_info']['throat.center'] , ax=ax1, c = 'y', linewidth = 4, zorder = 3)
#Element invaded( yo se que elemento fue)
#_ = op.visualization.plot_connections(network=pn, throats=78 , ax=ax1, c = 'g', linewidth = 4, zorder = 4)
ax1.set_title('Non-wetting phase presence, '+ status_str, fontsize = 22)
plt.tight_layout()

fig2, ax2 = plt.subplots(figsize = (8,8))
#Boundary elements
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=200, ax = ax2, zorder = 1)
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('all'), ax=ax2, c = 'gray', linewidth = 4, zorder = 0)
#NWP layer
_ = op.visualization.plot_coordinates(network=pn, pores=np.any(check_cluster['cluster_info']['pore.layer']  == 1, axis=1)  , c='b', s=200, ax = ax2, zorder = 2)
_ = op.visualization.plot_connections(network=pn, throats=np.any(check_cluster['cluster_info']['throat.layer']  == 1, axis=1)  , ax=ax2, c = 'b', linewidth = 7, zorder = 2)
#NWP center
_ = op.visualization.plot_coordinates(network=pn, pores=check_cluster['cluster_info']['pore.center'] == 1  , c='y', s=120, ax = ax2, zorder = 3)
_ = op.visualization.plot_connections(network=pn, throats=check_cluster['cluster_info']['throat.center'] == 1 , ax=ax2, c = 'y', linewidth = 4, zorder = 3)
ax2.set_title('Cluster 1', fontsize = 22)
plt.tight_layout()

fig3, ax3 = plt.subplots(figsize = (8,8))
#Boundary elements
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=200, ax = ax3, zorder = 1)
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('all'), ax=ax3, c = 'gray', linewidth = 4, zorder = 0)
#NWP layer
_ = op.visualization.plot_coordinates(network=pn, pores=np.any(check_cluster['cluster_info']['pore.layer']  == 23, axis=1)  , c='b', s=200, ax = ax3, zorder = 2)
_ = op.visualization.plot_connections(network=pn, throats=np.any(check_cluster['cluster_info']['throat.layer']  == 23, axis=1)  , ax=ax3, c = 'b', linewidth = 7, zorder = 2)
#NWP center
_ = op.visualization.plot_coordinates(network=pn, pores=check_cluster['cluster_info']['pore.center'] == 23 , c='y', s=120, ax = ax3, zorder = 3)
_ = op.visualization.plot_connections(network=pn, throats=check_cluster['cluster_info']['throat.center'] == 23 , ax=ax3, c = 'y', linewidth = 4, zorder = 3)
ax3.set_title('Cluster 23', fontsize = 22)
plt.tight_layout()

fig4, ax4 = plt.subplots(figsize = (8,8))
##Boundary elements
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('all'), c='gray', s=500, ax = ax4, zorder = 0)
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=500, ax = ax4, zorder = 1)
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('all'), ax=ax4, c = 'gray', linewidth = 9, zorder = 0)

plt.show()
