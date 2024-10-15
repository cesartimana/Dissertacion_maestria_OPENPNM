# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar
FUNCIONA. PERO FALTA VER EL TEMA DE CLUSTERS
"""

import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
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

resolution = 5.345e-6

#Flowrate function
def Rate_calc(network, ph, inlet, outlet, conductance):
    St_p = op.algorithms.StokesFlow(network=network, phase=ph)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=inlet, values=1)
    St_p.set_value_BC(pores=outlet, values=0)
    St_p.run()
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val

#Reading netwrork data
ws = op.Workspace()
testName_h = 'Berea_modD.pnm'
proj_h = ws.load_project(filename=testName_h)
pn = proj_h.network
Np = pn.Np
Nt = pn.Nt

#Defining boundary conditions
axis = 'x'
inlet_pores = pn['pore.' + axis + 'min']
outlet_pores = pn['pore.' + axis + 'max']

#Properties extractend from Valvatne - Blunt (2004) Table 1
tension = 30e-3 #N/m
water_visc = 1.05e-3 #Pa/s
oil_visc = 1.39e-3 #Pa/s
theta = np.pi / 180 * 0  #water phase. 0 is used by Valvatne Blunt 2004

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
water['pore.contact_angle'] = theta
water['throat.contact_angle'] = theta


#Simulating primary Drainage
pdr = _alg.Primary_Drainage(network=pn, phase=water)
pdr.set_inlet_BC(pores=inlet_pores)
pdr.set_outlet_BC(pores=outlet_pores)
pdr.run(throat_diameter = 'throat.prism_inscribed_diameter')

#Post processing
pmax_drn = 9000
p_vals =  np.array([0, pmax_drn])

#Obtaining phase distribution and clusters for each stage of invasion according to p_vals
results_pdr = pdr.postprocessing2(mode = 'pressure', inv_vals = p_vals)

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
#Extracting data
phase_ce = results_pdr['status_2']['invasion_info']['pore.center'] #True for wp present
phase_co = results_pdr['status_2']['invasion_info']['pore.corner']
R = tension / pmax_drn #interface radius at maximum pc_drainage

#Calculating saturation

#bi for pores
interface_pore = (np.tile(phase_ce, (3,1)).T != phase_co)
theta_corner_pore = np.tile(water['pore.contact_angle'],(3,1)).T #Funciona con un solo valor de theta
beta_pore = pn['pore.half_corner_angle']
bi_pore = _if.interfacial_corner_distance(R, theta_corner_pore, beta_pore, int_cond = interface_pore)

#bi for throats
interface_throat = (np.tile(results_pdr['status_2']['invasion_info']['throat.center'], (3,1)).T != results_pdr['status_2']['invasion_info']['throat.corner'])
theta_corner_throat = np.tile(water['throat.contact_angle'],(3,1)).T #Funciona con un solo valor de theta
beta_throat = pn['throat.half_corner_angle']
bi_throat = _if.interfacial_corner_distance(R, theta_corner_throat, beta_throat, int_cond = interface_throat)
#--------------end---------------

elements = ['pore', 'throat']
locations = ['center', 'corner']

"""
3 Determine non-trapped clusters
Se ubican los clusters en la red que estan relacionados a cada fase y se revisa cuales de ellos estan vinculado a los poros BC
If it is the wetting phase, non-trapped are connected with the inlet pores
If it is the non-wetting phase, non-trapped are connected with the outlet pores
"""
#--------------start---------------
def obtain_nontrapped_clusters(phase_info,
                               cluster_info,
                               inlet_pores,
                               outlet_pores):
    r"""
    Obtain list of non-trapped clusters
    We consider IMBIBITION and triangular cross sections, so the wetting phase clusters are connected to the inlet.
    and the non-wetting phase to the outlet.

    Parameters:
    ----------------
    -phase_info: Dictionary with boolean arrays. Key detail lines below
    -cluster_info: Dictionary with int arrays. Key detail lines below
    -inlet_pores: boolean array
    -oulet_pores: boolean array

    key:
    'element.location'.
    'element' can be 'pore' or 'throat'
    'location' can be 'center' or 'corner'
    Ex: 'pore.center'
    """
    #Non-trapped list
    WP_list = []
    NWP_list = []

    for wp in [True, False]:
        if wp:
            BC = inlet_pores
        else:
            BC = outlet_pores
        index = np.unique(cluster_info['pore.center'][ (phase_info['pore.center'] == wp) & BC ])
        if len(index) > 0:
            if wp:
                WP_list.append(index)
            else:
                NWP_list.append(index)
        index = np.unique(cluster_info['pore.corner'][ (phase_info['pore.corner'] == wp) & np.tile(BC, (3,1)).T ])
        if len(index) > 0:
            if wp:
                WP_list.append(index)
            else:
                NWP_list.append(index)
    return np.unique(WP_list), np.unique(NWP_list)

#DADO QUE LUEGO INVADIMOS LOS POROS DE ENTRADA, SE DECIDE CALCULAR LOS CLUSTERS LUEGO DE ELLO

#--------------end---------------

#3.5 Copiar informacion de entrada e invadir los poros de entrada con wp
#Invadir los poros y gargantas de salida con nwp si es que el conduite lo tiene. Asi se evita trapped clusters donde no hay
#Hecho. Para chequear los cambios imprimir al inicio y al final:


copy_invasion_info = copy.deepcopy(results_pdr['status_2']['invasion_info'])
copy_cluster_info = copy.deepcopy(results_pdr['status_2']['cluster_info'])

#Defining inlet and outlet throats
outlet_t = pn.find_neighbor_throats(pores=outlet_pores)
pn['throat.outlet'] = False
pn['throat.outlet'][outlet_t] = True
inlet_t = pn.find_neighbor_throats(pores=inlet_pores)
pn['throat.inlet'] = False
pn['throat.inlet'][inlet_t] = True

#invadiendo elements de entrada
for p in range(Np):
    if inlet_pores[p]:
        #Metiendo agua a los poros de entrada
        copy_invasion_info['pore.center'][p] = True
        #Colocando el mismo indice de clusters de las esquinas en el centro
        #index = ( copy_cluster_info['pore.corner'][p] ) [ copy_invasion_info['pore.corner'][p] ]
        #index = np.unique(index)
        index = 0
        copy_cluster_info['pore.center'][p] = index
for t in range(Nt):
    if pn['throat.inlet'][t]:
        #Metiendo agua a las gargantas de entrada
        copy_invasion_info['throat.center'][t] = True
        #Colocando el mismo indice de clusters de las esquinas en el centro
        #index = ( copy_cluster_info['throat.corner'][t] ) [ copy_invasion_info['throat.corner'][t] ]
        #index = np.unique(index)
        index = 0
        copy_cluster_info['throat.center'][t] = index

#invadiendo elementos de salida de acuerdo al internal pore
for t in range(Nt):
    if pn['throat.outlet'][t]:
        conns = pn['throat.conns'][t]
        for p in conns:
            if pn['pore.boundary'][p]:
                p_b = p
            else:
                p_int = p
        status = copy_invasion_info['pore.center'][p_int]
        index = copy_cluster_info['pore.center'][p_int]
        #Modifying boundary elements, center phase (I ignore corners, assuming thats OK. Not sure)
        copy_invasion_info['pore.center'][p_b] = status
        copy_invasion_info['throat.center'][t] = status
        copy_cluster_info['pore.center'][p_b] = index
        copy_cluster_info['throat.center'][t] = index

#Updating cluster index
#For pores
new_index_p =  np.unique(np.vstack(( copy_cluster_info['pore.corner'].T, copy_cluster_info['pore.center'] )))

#For throats
new_index_t =  np.unique(np.vstack(( copy_cluster_info['throat.corner'].T, copy_cluster_info['throat.center'] )))
copy_cluster_info['pore.index'] = new_index_p
copy_cluster_info['index_list'] = np.unique((new_index_p, new_index_p))

#CALCULANDO CLUSTERS AHORA SI
WP_nontrapped, NWP_nontrapped = obtain_nontrapped_clusters(copy_invasion_info,
                               copy_cluster_info,
                               inlet_pores,
                               outlet_pores)

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

#Creating advancing contact contact_angle
theta_a = np.pi / 180 * 60
#Creating absolute permeability
perm_data = 1.746*0.986923e-12

#Pore pressures
#Checking the number of connected throats with the non-wetting phase at the center.
mask_t_NW = (~copy_invasion_info['throat.center'])[idx] #Tiene forma de IDX
mask_t_W_nontrap = np.isin( (copy_cluster_info['throat.center'])[idx] , WP_nontrapped)

#Calculating for each pore
for p in range(Np):
    if (copy_cluster_info['pore.center'][p] in NWP_nontrapped) and (not pn['pore.boundary'][p]):
        #snap off
        p_SP[p] = _if.pressure_snapoff1(beta = pn['pore.half_corner_angle'][p],
                        interface = interface_pore[p],
                        sigma = tension,
                        theta_r = theta,
                        theta_a = theta_a,
                        d = pn['pore.prism_inscribed_diameter'][p],
                        pc_max = pmax_drn)
        #pore body filling
        if np.any( mask_t_W_nontrap[indptr[p]: indptr[p+1]] ):
            p_PB[p]  = _if.pressure_PB1(n_inv_t = np.sum(mask_t_NW[indptr[p]: indptr[p+1]]) ,
                        sigma = tension,
                        theta_a = theta_a,
                        d = pn['pore.prism_inscribed_diameter'][p],
                        perm = perm_data)

"""
#Calculating snap off for all pores. Funciona poros pero no esta completo
mask_p = np.isin(copy_cluster_info['pore.center'], NWP_nontrapped)
mask_t = np.isin(copy_cluster_info['throat.center'], NWP_nontrapped)

values_SP = _if.pressure_snapoff(beta = pn['pore.half_corner_angle'],
                     interface = interface_pore,
                     sigma = tension,
                     theta_r =theta,
                     theta_a = theta_a,
                     d = pn['pore.prism_inscribed_diameter'],
                     pc_max = pmax_drn)
"""

#Throat pressures
#Look if connected pores has a non trapped wetting phase on center
mask_wp_conn = np.isin( copy_cluster_info['pore.center'][conns], WP_nontrapped)
number_wp_conn = np.any(mask_wp_conn, axis = 1)

for t in range(Nt):
    if (copy_cluster_info['throat.center'][t] in NWP_nontrapped) and (not pn['throat.boundary'][t]):
        #snap off
        p_ST[t] = _if.pressure_snapoff1(beta = pn['throat.half_corner_angle'][t],
                        interface = interface_throat[t],
                        sigma = tension,
                        theta_r =theta,
                        theta_a = theta_a,
                        d = pn['throat.prism_inscribed_diameter'][t],
                        pc_max = pmax_drn)

        #piston like displacement.
        if number_wp_conn[t]:
            p_PLD[t], _ = _if.pressure_PLD(beta = pn['throat.half_corner_angle'][t],
                        interface = interface_throat[t],
                        b = bi_throat[t],
                        sigma = tension,
                        theta_a = theta_a,
                        A = pn['throat.cross_sectional_area'][t],
                        G = pn['throat.shape_factor'][t],
                        d = pn['throat.prism_inscribed_diameter'][t],
                        p_max = pmax_drn,
                        theta_r = theta)

"""
values_ST = _if.pressure_snapoff(beta = pn['throat.half_corner_angle'],
                     interface = interface_throat,
                     sigma = tension,
                     theta_r =theta,
                     theta_a = theta_a,
                     d = pn['throat.prism_inscribed_diameter'],
                     pc_max = pmax_drn)


#Pore pressures
for p in range(Np):
    if copy_cluster_info['pore.center'][p] not in NWP_nontrapped:
        pass
"""
#--------------end---------------

#6 Estableciendo secuencia de invasion = 1

s_inv = 1

#7 Calcular el siguiente elemento a invadir.
#7.1 calcular la mayor presión y la ubicación. Se escoge la primera ubicación. Preferencia 1 de poro sobre garganta y 2 de snapoff sobre MTM.
pc_dict = {}
pc_dict['pore.snapoff'] = p_SP
pc_dict['pore.MTM_displacement'] = p_PB
pc_dict['throat.snapoff'] = p_ST
pc_dict['throat.MTM_displacement'] = p_PLD
pc_array = np.concatenate( (p_PB, p_SP, p_PLD, p_ST) )
p_next = np.max( pc_array )
#pos = np.where(pc_array == p_next)[0]

#lineas simples para saber si hay esquinas con agua en todos los poros. Es cierto.
for item in elements:
    print('At least one corner has water on all ' + item + 's?')
    print(np.all (np.any(results_pdr['status_2']['cluster_info'][item + '.corner'] == 0) ) )
    print('--------')

displacement_type = ['snapoff' , 'MTM_displacement']

print(np.max(p_SP) )
print(np.max(p_PB) )
print(np.max(p_ST) )

print(copy_cluster_info.keys())
#insert cluster list

#variables to save:
info_imbibition = {}
p_inv = pmax_drn


#Saving status 0
nontrapped = np.union1d(WP_nontrapped, NWP_nontrapped)
trapped = np.setdiff1d(copy_cluster_info['index_list'], nontrapped)
info_imbibition['status 0'] = {'invasion sequence' : 0,
                               'invasion pressure' : pmax_drn,
                               'nontrapped_clusters': np.copy(nontrapped),
                               'trapped_clusters': np.copy(trapped),
                               'pc_trapped': np.ones_like(trapped) * pmax_drn,
                               'invasion_info': copy.deepcopy(copy_invasion_info),
                               'cluster_info': copy.deepcopy(copy_cluster_info)}
while s_inv < 20000:
    status_str = 'status ' + str(s_inv)
    previous_status = 'status ' + str(s_inv - 1)
    p_inv = min(p_inv, p_next)
    info_imbibition[status_str] = {'invasion sequence' : s_inv,
                                   'invasion pressure' : p_inv}
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
                print('pressure: ' +  str(round(p_next)))
                print('index: ' + str(pos))
                print(item)
                print(displacement)
                print('---------')
                info_imbibition[status_str]['invaded element'] = item
                info_imbibition[status_str]['type of invasion'] = displacement
                #8 Update list of invasions (POR AHORA SIN CAMADA)
                #8.1 Update list of phase location
                copy_invasion_info[item + '.center'][pos] = True
                copy_invasion_info[item + '.corner'][pos] = True
                #8.2 Set both element pc as -inf, to not invade again
                for displacement2 in displacement_type:
                    pc_dict[item + '.' + displacement2][pos] = -np.inf
                #8.3 Update cluster index element
                old_index = copy_cluster_info[item + '.center'][pos]
                copy_cluster_info[item + '.center'][pos] = 0
                copy_cluster_info[item + '.corner'][pos] = 0
                #8.4 Check if oil clusters are not divided:
                if item != 'throat' or displacement != 'MTM_displacement':
                    #throat.PLD Unico caso donde no hay particion de clusters
                    #Check the oil cluster for the old index
                    check_cluster = {}
                    for item2 in elements:
                        for loc in locations:
                            keyname = item2 + '.' + loc
                            check_cluster[keyname] = copy_cluster_info[keyname] == old_index
                    new_cluster = _if.cluster_advanced(pn, check_cluster, corner = True)
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
                            indexinfo = np.concatenate( (copy_cluster_info[item2 +'.corner'].flatten(), copy_cluster_info[item2 +'.center']))
                            copy_cluster_info[item2 + '.index'] = np.unique(indexinfo)

                        #Creating general list
                        indexinfo = np.concatenate( (copy_cluster_info['pore.index'], copy_cluster_info['throat.index']) )
                        copy_cluster_info['index_list'] = np.sort(np.unique(indexinfo))

                        #Determinar si se crearon trapped clusters
                        WP_nontrapped, NWP_nontrapped = obtain_nontrapped_clusters(copy_invasion_info,
                                                    copy_cluster_info,
                                                    inlet_pores,
                                                    outlet_pores)
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
                if bool_new_trapped:
                    new_trapped = np.setdiff1d(Ntrap, info_imbibition[previous_status]['trapped_clusters'])
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
                    nontrapped = np.copy(info_imbibition[previous_status]['nontrapped_clusters'])
                    trapped = np.copy(info_imbibition[previous_status]['trapped_clusters'])
                    pressure_trapped = np.copy(info_imbibition[previous_status]['pc_trapped'])
                info_imbibition[status_str]['nontrapped_clusters'] = nontrapped
                info_imbibition[status_str]['trapped_clusters'] = trapped
                info_imbibition[status_str]['pc_trapped'] = pressure_trapped
                #8.5 Know pressures for new possible invaded elements:
                if item == 'throat':
                    #Update mask of  center phase pore
                    mask_t_NW = (~copy_invasion_info['throat.center'])[idx]
                    cn = pn['throat.conns'][ pos, : ]
                    for p in cn:
                        if not copy_invasion_info['pore.center'][p]:
                            if (copy_cluster_info['pore.center'][p] in NWP_nontrapped) and (not pn['pore.boundary'][p]):
                                new_pc = _if.pressure_PB1(n_inv_t = np.sum(mask_t_NW[indptr[p]: indptr[p+1]]) ,
                                            sigma = tension,
                                            theta_a = theta_a,
                                            d = pn['pore.prism_inscribed_diameter'][p],
                                            perm = perm_data)
                                pc_dict['pore.MTM_displacement'][p]  = new_pc
                else:
                    cn = idx[indptr[pos]: indptr[pos+1]]
                    for t in cn:
                        if not copy_invasion_info['throat.center'][t]:
                            if (copy_cluster_info['throat.center'][t] in NWP_nontrapped) and (not pn['throat.boundary'][t]):
                                new_pc, _ = _if.pressure_PLD(beta = pn['throat.half_corner_angle'][t],
                                        interface = interface_throat[t],
                                        b = bi_throat[t],
                                        sigma = tension,
                                        theta_a = theta_a,
                                        A = pn['throat.cross_sectional_area'][t],
                                        G = pn['throat.shape_factor'][t],
                                        d = pn['throat.prism_inscribed_diameter'][t],
                                        p_max = pmax_drn,
                                        theta_r = theta)
                                pc_dict['throat.MTM_displacement'][t]  = new_pc

    p_next = -np.inf
    print('---------')
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
    s_inv +=1
    with open('imbibition_process.pkl', 'wb') as fp:
        pickle.dump(info_imbibition, fp)

