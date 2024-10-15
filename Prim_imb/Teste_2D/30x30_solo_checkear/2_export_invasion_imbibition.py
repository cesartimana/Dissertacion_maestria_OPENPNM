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

if theta_a < theta_r or ( (np.pi - theta_a) < theta_r):
    raise Exception('contact angle data is not compatible with the deffinition of wetting and non-wetting phases')

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


with open(f'drainage_process_{theta_r_sexag}_{p_kPa}kPa.pkl', 'rb') as fp:
    results_pdr = pickle.load(fp)

#last_invasion_stage dict
last_s_str = list(results_pdr)[-1]
drainage_info = results_pdr[last_s_str]
drainage_info['pcmax_info'] = {}
trapped_list = drainage_info['trapped_clusters']
pc_trapped_list = drainage_info['pc_trapped']

#Check pc value
p_inv = drainage_info['invasion pressure']
if p_inv > pmax_drn:
    raise Exception('The maximum capillary pressure exceed the capillary pressure reported at the final drainage stage')

"""
IMBIBITION ALGORITHM

CONDITIONS
-----------
If it is the wetting phase, non-trapped are connected with the inlet pores
If it is the non-wetting phase, non-trapped are connected with the outlet pores

1 Read network properties, phase properties, boundary conditions and final results of the Primary Drainage event
FINAL RESULTS ON p_vals, invasion_info, cluster_info

2 Calculate bi for all corners with AM
"""
#--------------start---------------
elements = ['pore', 'throat']
locations = ['center', 'corner', 'layer']
displacement_type = ['snapoff' , 'MTM_displacement', 'max_LC']

#Inverting inlet and outlet elements
for item in elements:
    a = np.copy(pn[f'{item}.inlet'])
    pn[f'{item}.inlet'] = np.copy(pn[f'{item}.outlet'])
    pn[f'{item}.outlet'] = np.copy(a)

#Inserting layers with respect of the center phase
for item in elements:
    drainage_info['invasion_info'][f'{item}.layer'] = np.tile(np.copy(drainage_info['invasion_info'][f'{item}.center']), (3,1)).T
    drainage_info['cluster_info'][f'{item}.layer'] = np.tile(np.copy(drainage_info['cluster_info'][f'{item}.center']), (3,1)).T


#--------------end---------------


"""
3 Determine non-trapped clusters
Se ubican los clusters en la red que estan relacionados a cada fase y se revisa cuales de ellos estan vinculado a los poros BC
If it is the wetting phase, non-trapped are connected with the inlet pores
If it is the non-wetting phase, non-trapped are connected with the outlet pores
"""
#--------------start---------------


#DADO QUE LUEGO INVADIMOS LOS POROS DE ENTRADA, SE DECIDE CALCULAR LOS CLUSTERS LUEGO DE ELLO
#--------------end---------------

#3.5 Copiar informacion de entrada e invadir los poros de entrada con wp
#Invadir los poros y gargantas de salida con nwp si es que el conduite lo tiene. Asi se evita trapped clusters donde no hay
#Hecho. Para chequear los cambios imprimir al inicio y al final:

copy_invasion_info = copy.deepcopy(drainage_info['invasion_info'])
copy_cluster_info = copy.deepcopy(drainage_info['cluster_info'])

#Adding condition of broken layers:
for item in elements:
    copy_invasion_info[f'{item}.broken_layer'] = np.zeros_like(pn[f'{item}.half_corner_angle'], dtype = bool)

#Invading new inlet elements
#Invading inlet pores with water
for t in range(Nt):
    if pn['throat.inlet'][t]:
        conns = pn['throat.conns'][t]
        for p in conns:
            if pn['pore.boundary'][p]:
                p_b = p
            else:
                p_int = p
        bool_new_index = False
        if np.any(copy_invasion_info['pore.corner'][p_int,:]):
            index = ( copy_cluster_info['pore.corner'][p_int,:] ) [ copy_invasion_info['pore.corner'][p_int,:] ]
            index = np.unique(index)[0]
        else:
            index = np.min(copy_cluster_info['index_list']) - 1
            bool_new_index = True
        for loc in locations:
            copy_invasion_info[f'pore.{loc}'][p_b] = True
            copy_invasion_info[f'throat.{loc}'][t] = True
            copy_cluster_info[f'pore.{loc}'][p_b] = index
            copy_cluster_info[f'throat.{loc}'][t] = index
        if bool_new_index:
            copy_cluster_info = _if.update_cluster_index(cluster_dict = copy_cluster_info)

#Extracting geometric data from the final drainage stage

trapped_list = drainage_info['trapped_clusters']
pc_trapped_list = drainage_info['pc_trapped']
corner_info = {}

for item in elements:
    #Reading data
    N = len(pn[f'{item}.diameter'])
    beta = pn[f'{item}.half_corner_angle']
    mask_wp_ce = copy_invasion_info[f'{item}.center']
    mask_wp_co = copy_invasion_info[f'{item}.corner']
    #Previous calculation for the trapped cluster information
    mask_trapped = np.isin(copy_cluster_info[f'{item}.corner'], trapped_list)
    co_cluster_index = copy_cluster_info[f'{item}.corner']
    pc_array = np.ones(N) * pmax_drn
    mask_AM = (np.tile(mask_wp_ce, (3,1)).T != mask_wp_co)
    #Setting the trapped pressure if necessary
    for i in range(N):
        #Obtaining pc analyzing if the wp is trapped.
        if np.any(mask_trapped[i,:]):
            index = co_cluster_index[i,:][ mask_trapped[i,:] ]
            index = np.unique(index)[0]
            pos = np.where(index == trapped_list)[0]
            pc_array[i] = pc_trapped_list[pos]
    drainage_info['pcmax_info'][f'{item}'] = pc_array
    pc_array = np.tile(pc_array, (3,1)).T
    r = tension / pc_array
    bi_array = _if.interfacial_corner_distance(r, theta_r, beta, int_cond = mask_AM)
    #Saving data
    corner_info[f'{item}.mask_wp'] = mask_wp_co
    corner_info[f'{item}.mask_AM'] = mask_AM
    corner_info[f'{item}.bi'] = bi_array

#CALCULANDO CLUSTERS AHORA SI

nontrapped, trapped = _if.obtain_nontrapped_clusters(copy_invasion_info,
                               copy_cluster_info,
                               connect_wp_pores = pn['pore.inlet'],
                               connect_nwp_pores = pn['pore.outlet'],
                               layer = True,
                               obtain_trapped = True)
#Trapped antes y despues de la invasion del elementos de fronteras SON IGUALES
if ~np.all(np.sort(trapped_list) == trapped):
    raise Exception('revisar lista de trapped')
#4 Invadir los poros de entrada, seteando

"""
5 Calculate pressure for all possible displacements
Trabajar por grupos: poros y gargantas
Poros:
Todos los poros con fase no mojante no atrapada en el centro pueden ser invadidos por snap-off
Si un elemento está conectado a minimo 1 garganta con fase mojante no atrapada en el medio , PB es factible
Para PB, se considera el numero de gargantas con fase no mojante, independiente de si están invadidas o no
Garganta
Todas las gargantas con fase no mojante no atrapada en el centro pueden ser invadidos por snap-off
Si un elemento está conectado a minimo 1 poro con fase mojante no atrapada en el medio, PLD es factible

Fases atrapadas no serán desplazadas

NO HAY PRESIONES CON NAN
NO TENGO PRESIONES MAYPRES A P_DRN
"""
#--------------start---------------

#Looking for neighbor throats for each pore
im = pn.create_incidence_matrix(fmt='csr') #Chequear gargantas unidas a poros.
idx=im.indices #indices de las gargantas
indptr=im.indptr #indica ubicacion inicial y final de los indices de gargantas en idx
#Para tener el indice de las gargantas del poro p,  usar idx[ indptr[p]: indptr[p+1]]

#Looking for pores for each throat
conns = pn['throat.conns']

#Pore pressures
#Checking the number of connected throats with the non-wetting phase at the center.

#ESTO SE PUEDE COLOCAR ANTES. POR EL MOMENTO DEJAR AQUI
for item in elements:
    corner_info[f'{item}.mask_layer'] =_if.nwp_in_layers(beta = pn[f'{item}.half_corner_angle'],
                                                         theta_r = theta_r,
                                                         theta_a = theta_a)
    corner_info[f'{item}.pressure_LC'] = _if.pressure_LC(beta = pn[f'{item}.half_corner_angle'],
                bi = corner_info[f'{item}.bi'],
                sigma = tension,
                theta_a = theta_a,
                mask = corner_info[f'{item}.mask_layer'])
    #print(f'Have all {item}s at least one layer?')
    #print(np.all(np.any(corner_info[f'{item}.mask_layer'], axis = 1)))
    #print(f'Number of {item}s with at least one layer?')
    #print(np.sum(np.any(corner_info[f'{item}.mask_layer'], axis = 1)))
    #SI NO HAY INTERFACE, PC = -INF
    #print('minimum and maximum pc LC')
    #print( np.min((corner_info[f'{item}.pressure_LC'])[ corner_info[f'{item}.mask_layer'] & corner_info[f'{item}.mask_AM'] ])  )
    #print( np.max(corner_info[f'{item}.pressure_LC'])  )

#6 Estableciendo secuencia de invasion = 1

s_inv = 0

#7 Calcular el siguiente elemento a invadir.
#7.1 calcular la mayor presión y la ubicación. Se escoge la primera ubicación. Preferencia 1 de poro sobre garganta y 2 de snapoff sobre MTM.

#Creating inf arrays. Negative because these values are not gonna be used
p_PB = -np.inf * np.ones(Np)
p_SP = -np.inf * np.ones(Np)
p_PLD = -np.inf * np.ones(Nt)
p_ST = -np.inf * np.ones(Nt)

pc_dict = {}
#Creando el dictionario para quiebra de layers
for item in elements:
    pc_dict[f'{item}.layer_collapse'] = np.ones_like( pn[f'{item}.half_corner_angle'] ) * -np.inf
    pc_dict[f'{item}.max_LC'] = pc_dict[f'{item}.layer_collapse'][:, 0]

def update_pressure():
    pc_dict['pore.snapoff'] = -np.inf * np.ones(Np)
    pc_dict['pore.MTM_displacement'] = -np.inf * np.ones(Np)
    pc_dict['throat.snapoff'] = -np.inf * np.ones(Nt)
    pc_dict['throat.MTM_displacement'] = -np.inf * np.ones(Nt)

    #Las dos siguientes tienen forma de IDX
    mask_t_NW = (~copy_invasion_info['throat.center'])[idx] #NEW
    mask_t_W_nontrap = np.isin( (copy_cluster_info['throat.center'])[idx] , nontrapped[nontrapped<1]) #NEW

    #Calculating for each pore
    for p in range(Np):
        if (copy_cluster_info['pore.center'][p] in nontrapped[nontrapped >=1]) and (not pn['pore.boundary'][p]):
            #snap off (if at least one corner has non trapped wetting phase)
            if np.any(np.isin(copy_cluster_info['pore.corner'][p,:], nontrapped[nontrapped < 1])):
                pc_dict['pore.snapoff'][p] = _if.pressure_snapoff1(beta = pn['pore.half_corner_angle'][p],
                                interface = corner_info['pore.mask_AM'][p],
                                sigma = tension,
                                theta_r = theta_r,
                                theta_a = theta_a,
                                d = pn['pore.diameter'][p],
                                pc_max = drainage_info['pcmax_info']['pore'][p])
            #pore body filling (if at least one throat has non trapped wetting phase at the center)
            if np.any( mask_t_W_nontrap[indptr[p]: indptr[p+1]] ):
                pc_dict['pore.MTM_displacement'][p]  = _if.pressure_PB1(n_inv_t = np.sum(mask_t_NW[indptr[p]: indptr[p+1]]) ,
                            sigma = tension,
                            theta_a = theta_a,
                            d = pn['pore.diameter'][p],
                            perm = perm_data)

    #Throat pressures
    #Look if connected pores has a non trapped wetting phase on center
    mask_wp_conn = np.isin( copy_cluster_info['pore.center'][conns],  nontrapped[nontrapped < 1]) #NEW
    number_wp_conn = np.any(mask_wp_conn, axis = 1) #NEW

    for t in range(Nt):
        if (copy_cluster_info['throat.center'][t] in  nontrapped[nontrapped >=1]) and (not pn['throat.boundary'][t]):
            #snap off (if at least one corner has non trapped wetting phase)
            if np.any(np.isin(copy_cluster_info['throat.corner'][t,:], nontrapped[nontrapped < 1])):
                pc_dict['throat.snapoff'][t] = _if.pressure_snapoff1(beta = pn['throat.half_corner_angle'][t],
                                interface = corner_info['throat.mask_AM'][t],
                                sigma = tension,
                                theta_r =theta_r,
                                theta_a = theta_a,
                                d = pn['throat.diameter'][t],
                                pc_max = drainage_info['pcmax_info']['throat'][t])

            #piston like displacement.
            if number_wp_conn[t]:
                pc_dict['throat.MTM_displacement'][t], _ = _if.pressure_PLD(beta = pn['throat.half_corner_angle'][t],
                            interface = corner_info['throat.mask_AM'] [t],
                            b = corner_info['throat.bi'][t],
                            sigma = tension,
                            theta_a = theta_a,
                            A = pn['throat.cross_sectional_area'][t],
                            G = pn['throat.shape_factor'][t],
                            d = pn['throat.diameter'][t],
                            p_max = drainage_info['pcmax_info']['throat'][t],
                            theta_r = theta_r)

    return
update_pressure()

p_next = -np.inf
for item in elements:
    for displacement in displacement_type:
        p_disp = max(pc_dict[item + '.' + displacement])
        p_next = max(p_next, p_disp)

print(p_next)

#pos = np.where(pc_array == p_next)[0]

#lineas simples para saber si hay esquinas con agua en todos los poros. Es cierto.
for item in elements:
    print('At least one corner has water on all ' + item + 's?')
    print(np.all (np.any(drainage_info['cluster_info'][item + '.corner'] == 0) ) )
    print('--------')

#insert cluster list

#variables to save:
info_imbibition = {}
p_inv = pmax_drn

#SAVING STATUS 0
#Preparing corner dict + mask_LC

#Saving
info_imbibition['status 0'] = {'invasion sequence' : 0,
                               'invasion pressure' : pmax_drn,
                               'nontrapped_clusters': np.copy(nontrapped),
                               'trapped_clusters': np.copy(trapped_list),
                               'pc_trapped': np.copy(pc_trapped_list),
                               'invasion_info': copy.deepcopy(copy_invasion_info),
                               'cluster_info': copy.deepcopy(copy_cluster_info),
                               'corner_info': copy.deepcopy(corner_info),
                               'pc_max_array': copy.deepcopy(drainage_info['pcmax_info'])}
while s_inv < 10000:
    if p_next < p_inv:
        #New quasi-static state
        s_inv +=1
        status_str = 'status ' + str(s_inv)
        previous_status = 'status ' + str(s_inv - 1)
        p_inv = p_next
        info_imbibition[status_str] = {'invasion sequence' : s_inv,
                                    'invasion pressure' : p_inv}
    else:
        #Maintain actual status
        previous_status = 'status ' + str(s_inv)
    trapped = np.copy(info_imbibition[previous_status]['trapped_clusters'])
    nontrapped = np.copy(info_imbibition[previous_status]['nontrapped_clusters'])
    pc_trapped = np.copy(info_imbibition[previous_status]['pc_trapped'])
    for item in elements:
        for displacement in displacement_type:
            #FORZANDO PNEXT PARA AVERIGUAR FUNCIONAMIENTO DE CODIGO
            p_list = pc_dict[item + '.' + displacement]
            pos = np.where(p_list == p_next)[0]
            if len(pos) > 0:
                pos = pos[0]
                #Creando var bool trapped
                bool_merge_wp = False
                print('---------')
                print('status ' + str(s_inv))
                print('pressure: ' +  str(p_inv))
                print('index: ' + str(pos))
                print(item)
                print(displacement)
                print('---------')
                #IF LC, SIMPLE TREATMENT
                if displacement == 'max_LC':
                    #layer collpase. It still exist a thin layer so cluster index and phase info are not altered
                    mask_LC = pc_dict[f'{item}.layer_collapse'][pos,:] == p_next
                    for i in range(3):
                        if mask_LC[i]:
                            #Update broken layer info
                            copy_invasion_info[item + '.broken_layer'][pos, i] = True
                            #Getting the old index
                            old_index = copy_cluster_info[item + '.layer'][pos, i] #IS IT INFMPORTANT?
                            #Setting pc_LC as -inf
                            pc_dict[f'{item}.layer_collapse'][pos, i] =  -np.inf
                else:
                    print('antes ce-la-co')
                    print(copy_invasion_info[item + '.center'][pos])
                    print(copy_invasion_info[item + '.layer'][pos, :])
                    print(copy_invasion_info[item + '.corner'][pos, :])
                    #CLASSIC DISPLACEMENT
                    #Invading element with the wp
                    copy_invasion_info[item + '.center'][pos] = True
                    #Get old cluster index for the center
                    old_index = copy_cluster_info[item + '.center'][pos]
                    mask_LF = corner_info[f'{item}.mask_layer'][pos,:]
                    mask_AM = corner_info[f'{item}.mask_AM'][pos,:]
                    print('mask LF: ' + str(mask_LF))
                    #We have layer formation. Now we analyse layer collapse
                    pc_LC = corner_info[f'{item}.pressure_LC'][pos,:]
                    print('pressure LC: ' + str(pc_LC))
                    for i in range(3):
                        mask_confirm_LF = False
                        if mask_AM[i]:
                            beta_i = pn[f'{item}.half_corner_angle'][pos,i]
                            R = pn[f'{item}.diameter'][pos] / 2
                            if i < 2:
                                bi_beta_max = corner_info[f'{item}.bi'][pos,2]
                                beta_max = pn[f'{item}.half_corner_angle'][pos,2]
                            else:
                                bi_beta_max = corner_info[f'{item}.bi'][pos,1]
                                beta_max = pn[f'{item}.half_corner_angle'][pos,1]
                            if mask_LF[i]:
                                #Checking if its geometrically possible, after an MTM disp, a layer
                                mask_confirm_LF = _if.verify_LF(beta = beta_i,
                                                            beta_max = beta_max,
                                                            bi_beta_max = bi_beta_max,
                                                            R_inscribed = R,
                                                            sigma = tension,
                                                            theta_a = theta_a,
                                                            pressure = p_inv)
                        if (pc_LC[i] < p_inv) and mask_confirm_LF:
                            #Confirmed LF. Updating some pc_dicts
                            pc_dict[f'{item}.layer_collapse'][pos, i] = pc_LC[i]
                        else:
                            #No LF or it collapse instantly. LF ignored
                            mask_LF[i] = False
                            copy_invasion_info[item + '.layer'][pos, i] = True
                            copy_invasion_info[item + '.corner'][pos, i] = True
                    index_co = copy_cluster_info[item + '.corner'][pos,:]
                    print('despues ce-la-co')
                    print(copy_invasion_info[item + '.center'][pos])
                    print(copy_invasion_info[item + '.layer'][pos, :])
                    print(copy_invasion_info[item + '.corner'][pos, :])
                    if displacement == 'snapoff':
                        wp_index = np.unique(index_co[index_co < 1])[0]
                    else:
                        #MTM_disp
                        wp_item_index_list = []
                        if item == 'pore':
                            #PB
                            neighbors = idx[ indptr[pos]: indptr[pos+1]]
                            item_cn = 'throat'
                        else:
                            #PLD
                            neighbors = conns[pos,:]
                            item_cn = 'pore'
                        #Check_wp clusters
                        #First, neighbor center. Then corners and layers
                        for loc in locations:
                            for n in neighbors:
                                mask = copy_invasion_info[f'{item_cn}.{loc}'][n]
                                if loc != 'center':
                                    if np.any(mask):
                                        item_cluster = copy_cluster_info[f'{item_cn}.{loc}'][n]
                                        wp_item_index_list.append(np.unique(item_cluster[mask])[0])
                                elif mask:
                                    wp_item_index_list.append(copy_cluster_info[f'{item_cn}.{loc}'][n])
                        #Finally, the element
                        for loc in ['corner', 'layer']:
                            item_cluster = copy_cluster_info[f'{item}.{loc}'][pos]
                            mask = copy_cluster_info[f'{item}.{loc}'][pos] < 1
                            if np.any(mask):
                                wp_item_index_list.append(np.unique(item_cluster[mask])[0])
                        wp_index = wp_item_index_list[0]
                        wp_item_index_list = np.unique(wp_item_index_list)
                        if len(wp_item_index_list) > 1:
                            bool_merge_wp = True
                    #Setting wp index on the wp locations #SI MTM CENTER Y CORNER SON SPERADOS, ESTO NO VA
                    for loc in locations:
                        mask = copy_invasion_info[item + '.' + loc][pos]
                        copy_cluster_info[item + '.' + loc][pos,mask] = wp_index
                        #print(copy_cluster_info[item + '.' + loc][pos])

                    if bool_merge_wp:
                        #merge cluster function use the first index
                        copy_cluster_info = _if.merge_cluster(copy_cluster_info, wp_item_index_list, layer = True)
                        copy_cluster_info, bool_divided = _if.check_divide_cluster(network = pn,
                                                                               cluster_dict = copy_cluster_info,
                                                                               index = wp_index,
                                                                               layer = True,
                                                                               index_mode = 'min',
                                                                               wp = True)
                        if bool_divided:
                            print(copy_cluster_info['index_list'])
                            raise Exception('Parece que aun hay mas de un cluster')
                        else:
                            5
                    #8 Update list of p_c^e
                    for displacement2 in displacement_type:
                        if displacement2 != 'max_LC':
                                pc_dict[f'{item}.{displacement2}'][pos] = -np.inf
                    #8.4 Check if oil clusters are not divided:
                    #Scenarios: snapoff (both) or MTM_disp(both), only if any layer is formed
                    if ~np.any(mask_LF):
                        #Check the oil cluster for the old index
                        copy_cluster_info, bool_divided = _if.check_divide_cluster(network = pn,
                                                                               cluster_dict = copy_cluster_info,
                                                                               index = old_index,
                                                                               layer = True,
                                                                               index_mode = 'max',
                                                                               wp = False)
                    #Identify new nontrapped and trapped clusters
                    nontrapped_now, trapped_now = _if.obtain_nontrapped_clusters(copy_invasion_info,
                                                copy_cluster_info,
                                                connect_wp_pores = pn['pore.inlet'],
                                                connect_nwp_pores = pn['pore.outlet'],
                                                layer = True,
                                                obtain_trapped = True)
                    new_trapped = np.setdiff1d(trapped_now, trapped)
                    retired_trapped = np.setdiff1d(trapped, trapped_now)
                    new_nontrapped = np.setdiff1d(nontrapped_now, nontrapped)
                    retired_nontrapped = np.setdiff1d(nontrapped, nontrapped_now)
                    if (len(new_trapped) != 0):
                        print('New trapped clusters')
                        trapped = np.append(trapped, new_trapped)
                        pc_trapped = np.append(pc_trapped, np.ones_like(new_trapped ) *p_inv)
                        #Update pc_LC because i dont have an update function
                        for trapped_index in new_trapped:
                            for item2 in elements:
                                mask = np.any(copy_cluster_info[f'{item}.layer'] == trapped_index, axis = 1)
                                mask = np.tile(mask,(3,1)).T
                                pc_dict[f'{item}.layer_collapse'][mask] = -np.inf
                    if (len(retired_trapped) != 0):
                        print('Some trapped clusters are nontrapped now')
                        mask = np.isin(trapped, retired_trapped )
                        trapped = trapped[~mask]
                        pc_trapped = pc_trapped[~mask]
                    if (len(new_nontrapped) != 0) or (len(retired_nontrapped) != 0):
                        print('List of nontrapped clusters modified')
                        nontrapped = nontrapped_now
                #UPDATE TRAPPED CLUSTERS IF REQUIRED
                info_imbibition[status_str]['nontrapped_clusters'] = nontrapped
                info_imbibition[status_str]['trapped_clusters'] = trapped
                info_imbibition[status_str]['pc_trapped'] = pc_trapped
                print(trapped)
                print(nontrapped)
                print(copy_cluster_info['index_list'])
                print(copy_cluster_info['pore.layer'][98,:])
                print(pc_dict['pore.layer_collapse'][98,:])
                #8.5 Recalculate pressures for new possible invaded elements:
                if displacement != 'max_LC':
                    update_pressure()
                #Independent of the disp, we update its max_LC value.
                pc_dict[f'{item}.max_LC'][pos] = np.max(pc_dict[f'{item}.layer_collapse'][pos,:])

    p_next = -np.inf
    for item in elements:
        for displacement in displacement_type:
            p_disp = max(pc_dict[item + '.' + displacement])
            p_next = max(p_next, p_disp)
            print('max pressure on ' + item + ' ' + displacement + ': ' +str(p_disp))

    if np.isinf(p_next):
        print('Last invasion. Now, all clusters are trapped')
        s_inv = np.inf
    else:
        print('p_next: ' + str(p_next))
    info_imbibition[status_str]['invasion_info'] = copy.deepcopy(copy_invasion_info)
    info_imbibition[status_str]['cluster_info'] = copy.deepcopy(copy_cluster_info)
    with open(f'imbibition_process_{theta_r_sexag}_{theta_a_sexag}_{p_kPa}kPa.pkl', 'wb') as fp:
        pickle.dump(info_imbibition, fp)

