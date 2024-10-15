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

#SOLO PARA RED CREADA, ACTUALIZANDO NUMERO DE GARGANTAS
Np = pn.Np
Nt = pn.Nt

#Properties extracted from Valvatne - Blunt (2004) Table 1
tension = 30e-3 #N/m
water_visc = 1.05e-3 #Pa/s
oil_visc = 1.39e-3 #Pa/s
theta_r = np.pi / 180 * 80 #Respecto al agua, en sexagecimal

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
pdr.set_inlet_BC(pores = pn['pore.inlet'])
pdr.set_outlet_BC(pores = pn['pore.outlet'])
pdr.run(throat_diameter = 'throat.diameter')

elements = ['pore', 'throat']
locations = ['center', 'corner']

#inlet throats:  [ 0 , 9, 18, 27, 36, 45, 54, 63, 72, 81]
#inlet pores: [ 8 ,17, 26, 35, 44, 53, 62, 71, 80, 89]

#check = np.vstack((pdr['throat.invasion_sequence'][np.argsort( pdr['throat.invasion_sequence'] ) ] , pn['throat.diameter'][np.argsort( pdr['throat.invasion_sequence'] ) ] , pdr['throat.entry_pressure'][np.argsort( pdr['throat.invasion_sequence'] ) ] ) )
#print(check.T)

#Post processing

pmax_drn = 2800 #1.98055e+03 #for theta_r = 0
p_vals =  np.array([0, pmax_drn])

#Obtaining phase distribution and clusters for each stage of invasion according to p_vals
results_pdr = pdr.postprocessing2(mode = 'pressure', inv_vals = p_vals)



status_index = 2
check_cluster = results_pdr['status_' + str(status_index)]


#V_s = np.sum( pn['pore.volume'][ pn['pore.internal'] ] )
#V_m = np.sum( pn['pore.volume'][ pn['pore.internal'] & check_cluster['invasion_info']['pore.center']] )
#print(V_m / V_s)

#plotting
fig1, ax1 = plt.subplots(figsize = (8,8))

#Boundary elements
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('boundary'), ax=ax1, c = 'r', linewidth = 11, zorder = 0)
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=600, ax = ax1, zorder = 0)

#throat corner
mask_co_all_wp_throat = np.all(check_cluster['invasion_info']['throat.corner'], axis = 1) & pn['throat.internal']
mask_co_all_nwp_throat = np.all(~check_cluster['invasion_info']['throat.corner'], axis = 1) & pn['throat.internal']
mask_co_mixed_throat = ~mask_co_all_wp_throat &  ~mask_co_all_nwp_throat & pn['throat.internal']

_ = op.visualization.plot_connections(network=pn, throats=mask_co_all_wp_throat, ax=ax1, c = 'b', linewidth = 11, zorder = 1)
_ = op.visualization.plot_connections(network=pn, throats=mask_co_mixed_throat, ax=ax1, c = 'skyblue', linewidth = 11, zorder = 1)
_ = op.visualization.plot_connections(network=pn, throats=mask_co_all_nwp_throat, ax=ax1, c = 'gainsboro', linewidth = 11, zorder = 1)

#pore corner
mask_co_all_wp_pore = np.all(check_cluster['invasion_info']['pore.corner'], axis = 1) & pn['pore.internal']
mask_co_all_nwp_pore = np.all(~check_cluster['invasion_info']['pore.corner'], axis = 1) & pn['pore.internal']
mask_co_mixed_pore = ~mask_co_all_wp_pore &  ~mask_co_all_nwp_pore & pn['pore.internal']

_ = op.visualization.plot_coordinates(network=pn, pores=mask_co_all_wp_pore , c='b', s=600, ax = ax1, zorder = 2)
_ = op.visualization.plot_coordinates(network=pn, pores=mask_co_mixed_pore , c='skyblue', s=600, ax = ax1, zorder = 2)
_ = op.visualization.plot_coordinates(network=pn, pores=mask_co_all_nwp_pore , c='gainsboro', s=600, ax = ax1, zorder = 2)

#throat center
mask_center_wp_throat = check_cluster['invasion_info']['throat.center'] & pn['throat.internal']
mask_center_nwp_throat = ~check_cluster['invasion_info']['throat.center'] & pn['throat.internal']

_ = op.visualization.plot_connections(network=pn, throats=mask_center_wp_throat, ax=ax1, c = 'b', linewidth = 5, zorder = 3)
_ = op.visualization.plot_connections(network=pn, throats=mask_center_nwp_throat, ax=ax1, c = 'gainsboro', linewidth = 5, zorder = 3)

#pore center
mask_center_wp_pore = check_cluster['invasion_info']['pore.center'] & pn['pore.internal']
mask_center_nwp_pore = ~check_cluster['invasion_info']['pore.center'] & pn['pore.internal']

_ = op.visualization.plot_coordinates(network=pn, pores=mask_center_wp_pore , c='b', s=200, ax = ax1, zorder = 4)
_ = op.visualization.plot_coordinates(network=pn, pores=mask_center_nwp_pore , c='gainsboro', s=200, ax = ax1, zorder = 4)

color_list = [ 'orange', 'gold', 'lime', 'cyan', 'gainsboro', 'skyblue', 'blue', 'blueviolet', 'violet', 'pink', 'wheat', 'olive', 'teal', 'hotpink']

def plot_color_clusters(cl_index, color, ax):
    #throat corner
    mask_co_cluster_throat = np.any(check_cluster['cluster_info']['throat.corner'] == cl_index, axis = 1)
    _ = op.visualization.plot_connections(network=pn, throats=mask_co_cluster_throat, ax=ax, c = color, linewidth = 11, zorder = 1)

    #pore corner
    mask_co_cluster_pore = np.any(check_cluster['cluster_info']['pore.corner'] == cl_index, axis = 1)
    _ = op.visualization.plot_coordinates(network=pn, pores=mask_co_cluster_pore , c=color, s=600, ax = ax, zorder = 2)

    #throat center
    mask_center_cluster_throat = (check_cluster['cluster_info']['throat.center'] == cl_index)
    _ = op.visualization.plot_connections(network=pn, throats=mask_center_cluster_throat, ax=ax, c = color, linewidth = 5, zorder = 3)

    #pore center
    mask_center_cluster_pore = (check_cluster['cluster_info']['pore.center'] == cl_index)
    _ = op.visualization.plot_coordinates(network=pn, pores=mask_center_cluster_pore , c=color, s=200, ax = ax, zorder = 4)
    return


#Prepare non-wetting phase and wp list
cluster_list = check_cluster['cluster_info']['index_list']
nwp_list = cluster_list[cluster_list >= 1]
wp_list = cluster_list[cluster_list < 1]
print(nwp_list)
print(wp_list)
#Plot nwp clusters
fig2, ax2 = plt.subplots(figsize = (8,8))
#Boundary elements
_ = op.visualization.plot_connections(network=pn, throats=pn.throats('boundary'), ax=ax2, c = 'r', linewidth = 11, zorder = 0)
_ = op.visualization.plot_coordinates(network=pn, pores=pn.pores('boundary'), c='r', s=600, ax = ax2, zorder = 0)
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
plt.show()
