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

#SOLO PARA RED CREADA, ACTUALIZANDO NUMERO DE GARGANTAS
Np = pn.Np
Nt = pn.Nt

#Properties extracted from Valvatne - Blunt (2004) Table 1
tension = 30e-3 #N/m
water_visc = 1.05e-3 #Pa/s
oil_visc = 1.39e-3 #Pa/s
theta_r = np.pi / 180 * 80  #Respecto al agua, en sexagecimal
theta_a = np.pi / 180 * 95 #Try 120 to have layers

if theta_a < theta_r or ( (np.pi - theta_a) < theta_r):
    raise Exception('contact angle data is not compatible with the deffinition of wetting and non-wetting phases')

#Creating absolute permeability
perm_data = 1.746*0.986923e-12 #Average K from Berea Network

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


with open('check_cluster.pkl', 'rb') as fp:
    results_pdr = pickle.load(fp)

#last_invasion_stage dict
last_s_str = list(results_pdr)[-1]
drainage_info = results_pdr[last_s_str]
trapped_list = drainage_info['trapped_clusters']
pc_trapped_list = drainage_info['pc_trapped']

pmax_drn = 4000

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

#Creating inf arrays. Negative because these values are not gonna be used
p_PB = -np.inf * np.ones(Np)
p_SP = -np.inf * np.ones(Np)
p_PLD = -np.inf * np.ones(Nt)
p_ST = -np.inf * np.ones(Nt)

#Looking for neighbor throats for each pore
im = pn.create_incidence_matrix(fmt='csr') #Chequear gargantas unidas a poros.
idx=im.indices #indices de las gargantas
indptr=im.indptr #indica ubicacion inicial y final de los indices de gargantas en idx
#Para tener el indice de las gargantas del poro p,  usar idx[ indptr[p]: indptr[p+1]]

#Looking for pores for each throat
conns = pn['throat.conns']

#Pore pressures
#Checking the number of connected throats with the non-wetting phase at the center.

#Las dos siguientes tienen forma de IDX
mask_t_NW = (~copy_invasion_info['throat.center'])[idx]
mask_t_W_nontrap = np.isin( (copy_cluster_info['throat.center'])[idx] , nontrapped[nontrapped<1])

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

#Calculating for each pore
for p in range(Np):
    if (copy_cluster_info['pore.center'][p] in nontrapped[nontrapped >=1]) and (not pn['pore.boundary'][p]):
        #snap off (if at least one corner has non trapped wetting phase)
        if np.any(np.isin(copy_cluster_info['pore.corner'][p,:], nontrapped[nontrapped < 1])):
            p_SP[p] = _if.pressure_snapoff1(beta = pn['pore.half_corner_angle'][p],
                            interface = corner_info['pore.mask_AM'][p],
                            sigma = tension,
                            theta_r = theta_r,
                            theta_a = theta_a,
                            d = pn['pore.diameter'][p],
                        pc_max = pmax_drn)
        #pore body filling (if at least one throat has non trapped wetting phase at the center)
        if np.any( mask_t_W_nontrap[indptr[p]: indptr[p+1]] ):
            p_PB[p]  = _if.pressure_PB1(n_inv_t = np.sum(mask_t_NW[indptr[p]: indptr[p+1]]) ,
                        sigma = tension,
                        theta_a = theta_a,
                        d = pn['pore.diameter'][p],
                        perm = perm_data)


#Throat pressures
#Look if connected pores has a non trapped wetting phase on center
mask_wp_conn = np.isin( copy_cluster_info['pore.center'][conns],  nontrapped[nontrapped < 1])
number_wp_conn = np.any(mask_wp_conn, axis = 1)

for t in range(Nt):
    if (copy_cluster_info['throat.center'][t] in  nontrapped[nontrapped >=1]) and (not pn['throat.boundary'][t]):
         #snap off (if at least one corner has non trapped wetting phase)
        if np.any(np.isin(copy_cluster_info['throat.corner'][t,:], nontrapped[nontrapped < 1])):
            p_ST[t] = _if.pressure_snapoff1(beta = pn['throat.half_corner_angle'][t],
                            interface = corner_info['throat.mask_AM'][t],
                            sigma = tension,
                            theta_r =theta_r,
                            theta_a = theta_a,
                            d = pn['throat.diameter'][t],
                            pc_max = pmax_drn)

        #piston like displacement.
        if number_wp_conn[t]:
            p_PLD[t], _ = _if.pressure_PLD(beta = pn['throat.half_corner_angle'][t],
                        interface = corner_info['throat.mask_AM'] [t],
                        b = corner_info['throat.bi'][t],
                        sigma = tension,
                        theta_a = theta_a,
                        A = pn['throat.cross_sectional_area'][t],
                        G = pn['throat.shape_factor'][t],
                        d = pn['throat.diameter'][t],
                        p_max = pmax_drn,
                        theta_r = theta_r)

#6 Estableciendo secuencia de invasion = 1

s_inv = 0

#7 Calcular el siguiente elemento a invadir.
#7.1 calcular la mayor presión y la ubicación. Se escoge la primera ubicación. Preferencia 1 de poro sobre garganta y 2 de snapoff sobre MTM.
pc_dict = {}
pc_dict['pore.snapoff'] = p_SP
pc_dict['pore.MTM_displacement'] = p_PB
pc_dict['throat.snapoff'] = p_ST
pc_dict['throat.MTM_displacement'] = p_PLD
pc_array = np.concatenate( (p_PB, p_SP, p_PLD, p_ST) )
p_next = np.max( pc_array )

#Creando el dictionario para quiebra de layers
for item in elements:
    pc_dict[f'{item}.layer_collapse'] = np.ones_like( pn[f'{item}.half_corner_angle'] ) * -np.inf
    pc_dict[f'{item}.max_LC'] = pc_dict[f'{item}.layer_collapse'][:, 0]

#pos = np.where(pc_array == p_next)[0]

#lineas simples para saber si hay esquinas con agua en todos los poros. Es cierto.
for item in elements:
    print('At least one corner has water on all ' + item + 's?')
    print(np.all (np.any(drainage_info['cluster_info'][item + '.corner'] == 0) ) )
    print('--------')

displacement_type = ['snapoff' , 'MTM_displacement', 'max_LC']

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
                               }

#NOTA. SI QUISIERA QUE UNA ESQUINA SE MANTENGA CON OLEO (180-theta_a) < theta_r.
#Trabajando con theta_r = 0 eso no pasa

while s_inv < 100:
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
    for item in elements:
        for displacement in displacement_type:
            #FORZANDO PNEXT PARA AVERIGUAR FUNCIONAMIENTO DE CODIGO
            p_list = pc_dict[item + '.' + displacement]
            pos = np.where(p_list == p_next)[0]
            if len(pos) > 0:
                pos = pos[0]
                #Creando var bool trapped
                bool_new_trapped = False
                print('---------')
                print('status ' + str(s_inv))
                print('pressure: ' +  str(p_inv))
                print('index: ' + str(pos))
                print(item)
                print(displacement)
                print('---------')
                info_imbibition[status_str]['invaded element'] = item
                info_imbibition[status_str]['type of invasion'] = displacement
                #8 Update list of invasions
                #8.1 Update list of phase location and wp cluster info
                if displacement != 'max_LC':
                    #Invading center with the wp
                    copy_invasion_info[item + '.center'][pos] = True
                    mask_wp = copy_invasion_info[item + '.corner'][pos,:]
                    #Get old cluster index for the center
                    old_index = copy_cluster_info[item + '.center'][pos]
                    #Set the new wp index
                    if displacement == 'snapoff':
                        co_index = copy_cluster_info[item + '.corner'][pos,:]
                        wp_index = np.unique(co_index[mask_wp])[0]
                        print(wp_index)
                        raise Exception('Hubo snapoff')
                    elif item == 'throat':
                        neighbors = conns[pos,:]
                        print(neighbors)
                        raise Exception('Hubo PLD')
                    else:
                        neighbors = idx[ indptr[pos]: indptr[pos+1]]
                        print(neighbors)
                        wp_ce_index_list = copy_cluster_info['throat.center'][neighbors]
                        wp_ce_index_list = np.unique(wp_ce_index_list[wp_ce_index_list < 1])
                        wp_index = wp_ce_index_list[0]
                        print(wp_ce_index_list)
                        print(wp_index)
                        #raise Exception('Hubo PB')
                    copy_cluster_info[item + '.center'][pos] = wp_index
                    #Analysing layers
                    mask_LF = corner_info[f'{item}.mask_layer'][pos,:]
                    print('mask LF: ' + str(mask_LF))
                    #We have layer formation. Now we analyse layer collapse
                    pc_LC = corner_info[f'{item}.pressure_LC'][pos,:]
                    print('pressure LC: ' + str(pc_LC))
                    raise Exception('Hasta aqui')
                    for i in range(3):
                        mask_verify_LF = True
                        if mask_LF[i] and (i < 2) and (displacement == 'MTM_displacement'):
                            #Checking if its geometrically possible, after an MTM disp, a layer
                            mask_verify_LF = _if.verify_LF(beta = pn[f'{item}.half_corner_angle'][pos,i],
                                                           beta_max = pn[f'{item}.half_corner_angle'][pos,2],
                                                           bi_beta_max = corner_info[f'{item}.bi'][pos,2],
                                                           R_inscribed = pn[f'{item}.diameter'][pos] / 2,
                                                           sigma = tension,
                                                           theta_a = theta_a,
                                                           pressure = p_inv)
                        if mask_LF[i] and (pc_LC[i] < p_inv) and mask_verify_LF:
                            #Confirmed LF. Updating some pc_dicts
                            pc_dict[f'{item}.layer_collapse'][pos, i] = pc_LC[i]
                        else:
                            #No LF or it collapse instantly. LF ignored
                            mask_LF[i] = False
                            copy_invasion_info[item + '.layer'][pos, i] = True
                            copy_cluster_info[item + '.layer'][pos, i] = 0

                    #8.2 Set both displacementes pc as -inf, to not invade again
                    for displacement2 in displacement_type:
                        if displacement2 != 'max_LC':
                            pc_dict[item + '.' + displacement2][pos] = -np.inf
                else:
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
                #Independent of the disp, we update its max_LC value
                pc_dict[f'{item}.max_LC'][pos] = np.max(pc_dict[f'{item}.layer_collapse'][pos,:])

                #8.4 Check if oil clusters are not divided:
                #Scenarios: snapoff (both) or MTM_disp(both), only if any layer is formed
                if displacement != 'LC':
                    if ~np.any(mask_LF):
                        #Check the oil cluster for the old index
                        check_cluster = {}
                        for item2 in elements:
                            for loc in locations:
                                keyname = item2 + '.' + loc
                                check_cluster[keyname] = copy_cluster_info[keyname] == old_index
                        new_cluster = _if.cluster_advanced(pn, check_cluster, corner = True, layer = True)
                        #Lo siguiente funciona asumiendo que hay agua en todas las esquinas
                        print('Mostrando los indices de clusters creados luego de un posible division de clusters')
                        print(new_cluster['index_list'])
                        if len(new_cluster['index_list']) > 2:
                            #Se dividió el cluster
                            child_clusters =  np.delete(new_cluster['index_list'], 0)
                            for i in range(len(child_clusters) ):
                                if i > 0:
                                    new_index = np.max(copy_cluster_info['index_list']) + 1
                                    for item2 in elements:
                                        for loc in locations:
                                            keyname = item2 + '.' + loc
                                            mask = new_cluster[keyname] == child_clusters[i]
                                            if np.any(mask):
                                                copy_cluster_info[keyname][mask] = new_index
                            #Updating index list per element
                            for item2  in elements:
                                indexinfo = np.concatenate( (copy_cluster_info[item2 +'.corner'].flatten(), copy_cluster_info[item2 +'.layer'].flatten(),copy_cluster_info[item2 +'.center']))
                                copy_cluster_info[item2 + '.index'] = np.unique(indexinfo)

                            #Creating general list
                            indexinfo = np.concatenate( (copy_cluster_info['pore.index'], copy_cluster_info['throat.index']) )
                            copy_cluster_info['index_list'] = np.sort(np.unique(indexinfo))

                            #Determinar si se crearon trapped clusters
                            WP_nontrapped, NWP_nontrapped = obtain_nontrapped_clusters(copy_invasion_info,
                                                        copy_cluster_info,
                                                        pn['pore.inlet'],
                                                        pn['pore.outlet'],
                                                        layer = True)
                            #Actualizando lista de trapped cluster y sus presiones
                            nontrapped = np.union1d(WP_nontrapped, NWP_nontrapped)
                            mask_nontrap = np.isin(copy_cluster_info['index_list'], nontrapped )
                            trapped = (copy_cluster_info['index_list'])[~mask_nontrap]
                            Ntrap = len(trapped)
                            Ntrap_prev = len(info_imbibition[previous_status]['trapped_clusters'])
                            if Ntrap > Ntrap_prev:
                                #Hay qye mejorar el metodo
                                print('We have new trapped clusters')
                                bool_new_trapped = True
                                if s_inv == 64:
                                    for item2 in elements:
                                        for loc in locations:
                                            keyname = item2 + '.' + loc
                                            print(keyname)
                                            mask = copy_cluster_info[keyname] == 23
                                            print(np.where(mask))
                if bool_new_trapped:
                    new_trapped = np.setdiff1d(trapped, info_imbibition[previous_status]['trapped_clusters'])
                    print(new_trapped)
                    trapped = np.append(np.copy(info_imbibition[previous_status]['trapped_clusters']), new_trapped)
                    pressure_trapped = np.copy(info_imbibition[previous_status]['pc_trapped'])
                    pressure_trapped = np.append(pressure_trapped, np.ones_like(new_trapped ) *p_inv)
                    for index_cluster_trap in new_trapped:
                        for item2 in elements:
                            mask_cluster_trap = copy_cluster_info[item2 + '.center'] == index_cluster_trap
                            if np.any(mask_cluster_trap):
                                for disp2 in displacement_type:
                                    pc_dict[item2 + '.' + disp2][mask_cluster_trap] = -np.inf
                else:
                    print(previous_status)
                    nontrapped = np.copy(info_imbibition[previous_status]['nontrapped_clusters'])
                    trapped = np.copy(info_imbibition[previous_status]['trapped_clusters'])
                    pressure_trapped = np.copy(info_imbibition[previous_status]['pc_trapped'])
                info_imbibition[status_str]['nontrapped_clusters'] = nontrapped
                info_imbibition[status_str]['trapped_clusters'] = trapped
                print('trapped')
                print(trapped)
                info_imbibition[status_str]['pc_trapped'] = pressure_trapped
                #8.5 Know pressures for new possible invaded elements:
                if displacement != 'LC':
                    if item == 'throat':
                        #Update mask of center phase pore
                        mask_t_NW = (~copy_invasion_info['throat.center'])[idx]
                        cn = pn['throat.conns'][ pos, : ]
                        for p in cn:
                            if not copy_invasion_info['pore.center'][p]:
                                if (copy_cluster_info['pore.center'][p] in NWP_nontrapped) and (not pn['pore.boundary'][p]):
                                    new_pc = _if.pressure_PB1(n_inv_t = np.sum(mask_t_NW[indptr[p]: indptr[p+1]]) ,
                                                sigma = tension,
                                                theta_a = theta_a,
                                                d = pn['pore.diameter'][p],
                                                perm = perm_data)
                                    pc_dict['pore.MTM_displacement'][p]  = new_pc
                    else:
                        cn = idx[indptr[pos]: indptr[pos+1]]
                        for t in cn:
                            if not copy_invasion_info['throat.center'][t]:
                                if (copy_cluster_info['throat.center'][t] in NWP_nontrapped) and (not pn['throat.boundary'][t]):
                                    new_pc, _ = _if.pressure_PLD(beta = pn['throat.half_corner_angle'][t],
                                            interface = corner_info['throat.mask_AM'][t],
                                            b = corner_info['throat.bi'][t],
                                            sigma = tension,
                                            theta_a = theta_a,
                                            A = pn['throat.cross_sectional_area'][t],
                                            G = pn['throat.shape_factor'][t],
                                            d = pn['throat.diameter'][t],
                                            p_max = pmax_drn,
                                            theta_r = theta_r)
                                    pc_dict['throat.MTM_displacement'][t]  = new_pc
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
    with open('imbibition_process_layer_altering_BC.pkl', 'wb') as fp:
        pickle.dump(info_imbibition, fp)

"""
                            #REVISAR AQUI. PARA CHEQUEAR CLUSTERS EN UN MOMENTO ESPECIFICO
                            check_info = {'invasion_info': copy_invasion_info,
                                        'cluster_info': copy_cluster_info}
                            with open('checking_new_cluster.pkl', 'wb') as fp2:
                                pickle.dump(check_info, fp2)
                            raise Exception('Hola')

"""
