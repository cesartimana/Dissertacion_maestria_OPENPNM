# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar

"""

import openpnm as op
import numpy as np
import matplotlib.pyplot as plt

#importing functions for other file
import sys
sys.path.insert(1, '/home/cesar/OpenPNM_files/_funcs')
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
testName_h = 'Berea.pnm'
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
drn = _alg.Primary_Drainage(network=pn, phase=water)
drn.set_inlet_BC(pores=inlet_pores)
drn.set_outlet_BC(pores=outlet_pores)
drn.run(throat_diameter = 'throat.prism_inscribed_diameter')
pmax_drn = 9000
p_vals, invasion_info, cluster_info = drn.postprocessing(p_max = pmax_drn)
#Use the information with name['element.location']. element = pore,throat. location = center, corner

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
#cluster_ce = cluster_info['pore.center']
#cluster_co = cluster_info['pore.corner']
phase_ce = invasion_info['pore.center'] #True for wp present
phase_co = invasion_info['pore.corner']

#Calculating saturation
interface = (np.tile(phase_ce, (3,1)).T != phase_co)
theta_corner = np.tile(water['pore.contact_angle'],(3,1)).T #Funciona con un solo valor de theta
beta = pn['pore.half_corner_angle']
R = tension / pmax_drn
bi_corner = _if.interfacial_corner_distance(R, theta_corner, beta, int_cond = interface)

#bi for throats
interface_throat = (np.tile(invasion_info['throat.center'], (3,1)).T != invasion_info['throat.corner'])
theta_corner_throat = np.tile(water['throat.contact_angle'],(3,1)).T #Funciona con un solo valor de theta
beta_throat = pn['throat.half_corner_angle']
bi_throat = _if.interfacial_corner_distance(R, theta_corner_throat, beta_throat, int_cond = interface_throat)
#--------------end---------------

elements = ['pore', 'throat']
locations = ['center', 'corner']

"""
3 Determine non-trapped clusters
Se ubican los clusters en la red que estan relacionados a cada fase y se revisa cuales de ellos esta vinculado a los poros BC
If it is the wetting phase, non-trapped are connected with the inlet pores
If it is the non-wetting phase, non-trapped are connected with the outlet pores
"""
#--------------start---------------
WP_nontrapped = []
NWP_nontrapped = []

for wp in [True, False]:
    if wp:
        BC = inlet_pores
    else:
        BC = outlet_pores
    index = np.unique(cluster_info['pore.center'][ (invasion_info['pore.center'] == wp) & BC ])
    if len(index) > 0:
        if wp:
            WP_nontrapped.append(index[0])
        else:
            NWP_nontrapped.append(index[0])
    index = np.unique(cluster_info['pore.corner'][ (invasion_info['pore.corner'] == wp) & np.tile(BC, (3,1)).T ])
    if len(index) > 0:
        if wp:
            WP_nontrapped.append(index[0])
        else:
            NWP_nontrapped.append(index[0])

WP_nontrapped = np.unique(WP_nontrapped)
NWP_nontrapped = np.unique(NWP_nontrapped)
#--------------end---------------

#3.5 Copiar informacion de entrada e invadir los poros de entrada
#Hecho. Para chequear los cambios imprimir al inicio y al final:  print(copy_invasion_info['pore.center'][3960:3970])
copy_invasion_info = invasion_info.copy()
copy_cluster_info = cluster_info.copy()

for p in range(Np):
    if inlet_pores[p]:
        #Metiendo agua a los poros de entrada
        copy_invasion_info['pore.center'][p] = True
        #Colocando el mismo indice de clusters de las esquinas en el centro
        index = ( copy_cluster_info['pore.corner'][p] ) [ copy_invasion_info['pore.corner'][p] ]
        index = np.unique(index)[0]
        copy_cluster_info['pore.center'][p] = index

#4 Invadir los poros de entrada, seteando

"""
5 Calculate pressure for all possible displacements
Trabajar por grupos: poros y gargantas
Poros:
Todos los poros con fase no mojante no atrapada en el centro pueden ser invadidos por snap-off
Si un elemento está conectado a minimo 1 garganta con fase mojante no atrapada en el medio , se invade por PB
Para PB, se considera el numero de gargantas con fase no mojante, independiente de si están invadidas o no
Garganta
Todas las gargantas con fase no mojante no atrapada en el centro pueden ser invadidos por snap-off
Si un elemento está conectado a minimo 1 poro con fase mojante no atrapada en el medio, se invade por PLD

Fases atrapadas no serán desplazadas

NO HAY PRESIONES CON NAN
PRESIONES PLD ESTAN SALIENDO MAS ALTOS QUE LA PRESION MAXIMA DE DRENAJE
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
indptr=im.indptr #indica ubicacion inicial y final de los indices de gargantas
#Looking for pores for each throat
conns = pn['throat.conns']

#Creating advancing contact contact_angle
theta_a = np.pi / 180 * 60

#Pore pressures
#Checking the number of connected throats with the non-wetting phase at the center.
mask_t_NW = (~copy_invasion_info['throat.center'])[idx]
mask_t_W_nontrap = np.isin( (copy_cluster_info['throat.center'])[idx] , WP_nontrapped)

#Calculating for each pore
for p in range(Np):
    if copy_cluster_info['pore.center'][p] not in NWP_nontrapped:
        continue
    #snap off
    p_SP[p] = _if.pressure_snapoff1(beta = pn['pore.half_corner_angle'][p],
                     interface = interface[p],
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
                    perm = 1.746*0.986923e-12)

"""
#Calculating snap off for all pores. Funciona poros pero no esta completo
mask_p = np.isin(copy_cluster_info['pore.center'], NWP_nontrapped)
mask_t = np.isin(copy_cluster_info['throat.center'], NWP_nontrapped)

values_SP = _if.pressure_snapoff(beta = pn['pore.half_corner_angle'],
                     interface = interface,
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
    if copy_cluster_info['throat.center'][t] not in NWP_nontrapped:
        continue
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
#7.1 calcular la mayor presión y la ubicación. Se escoge la primera ubicación
pc_dict = {}
pc_dict['pore.snapoff'] = p_SP
pc_dict['pore.MTM_displacement'] = p_PB
pc_dict['throat.snapoff'] = p_ST
pc_dict['throat.MTM_displacement'] = p_PLD
pc_array = np.concatenate( (p_PB, p_SP, p_PLD, p_ST) )
p_next = np.max( pc_array )
#pos = np.where(pc_array == p_next)[0]

displacement_type = ['snapoff' , 'MTM_displacement']
for item in elements:
    for displacement in displacement_type:
        p_list = pc_dict[item + '.' + displacement]
        pos = np.where(p_list == p_next)[0]
        if len(pos) == 0:
            continue
        else:
            pos = pos[0]
            #8 Update list of invasions (POR AHORA SIN CAMADA)
            #8.1 Update list of phase location
            copy_invasion_info[item + '.center'][pos] = True
            #8.2 Update list of pressure of the element
            for i in displacement_type:
                pc_dict[item + '.' + displacement] = -np.inf
            #





"""

#Calculating final stats
for item  in elements:
#True if wp is present
    phase_ce = invasion_info[f'{item}.center']
    phase_co = invasion_info[f'{item}.corner']
    interface = (np.tile(phase_ce, (3,1)).T != phase_co)
    theta_corner = np.tile(water[f'{item}.contact_angle'],(3,1)).T #Funciona con un solo valor de theta
    beta = pn[f'{item}.half_corner_angle']
    R = tension / pmax_drn
    invasion_info[f'{item}.bi']= _if.interfacial_corner_distance(R, theta_corner, beta, int_cond = interface)

"""

"""
seq = np.arange(1, np.max(pd['throat.invasion_sequence']) , 10)
p_vals = []
for i in seq:
    inv_pattern = pd['throat.invasion_sequence'] <= i
    p_vals.append( max(pd['throat.invasion_pressure'][inv_pattern])  )
p_vals = np.array(p_vals)
#Obtaining phase distribution and clusters for each stage of invasion according to p_vals
p_vals, invasion_info, cluster_info = pd.postprocessing(p_vals = p_vals)

#Calculation conduit length
L = _cf.conduit_length_tubes(pn,
                             pore_length = "pore.equivalent_diameter",
                             throat_spacing = "throat.total_length",
                             L_min = resolution,
                             check_boundary = True)


#Assuming all volume is on the pores
V_sph = sum(pn['pore.volume'][pn['pore.internal']])
#Adding immobile water.f_disc is the discontinuous water
f_disc =  0.247058823529
V_t = V_sph / ( 1 - f_disc )
V_disc = V_t * f_disc


elements = ['pore', 'throat']
sat = []

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
    sph_flow = Rate_calc(pn, phase, inlet_pores , outlet_pores, label_conductance)

    #MULTIPHASE
    for i in range(len(p_vals)):
        #Extracting data
        cluster_ce = cluster_info['pore.center'][:,i]
        cluster_co = cluster_info['pore.corner'][:,:,i]
        phase_ce = invasion_info['pore.center'][:,i] == wp #True if wp is present
        phase_co = invasion_info['pore.corner'][:,:,i] == wp
        q_mph = []

        #Calculating saturation
        interface = (np.tile(phase_ce, (3,1)).T != phase_co)
        theta_corner = np.tile(water['pore.contact_angle'],(3,1)).T #Funciona con un solo valor de theta
        beta = pn['pore.half_corner_angle']
        R = water['pore.surface_tension'][0]/ p_vals[i]
        bi_corner = _if.interfacial_corner_distance(R, theta_corner, beta, int_cond = interface)
        viscosity = phase['pore.viscosity'][0]
        _, _, pore_ratio = _cf.conductance(pn, phase_ce, phase_co, theta_corner, bi_corner, viscosity, item = 'pore')
        V_mph = np.sum((pn['pore.volume'] * pore_ratio)[pn['pore.internal']])
        if wp:
            sat.append( (V_mph + V_disc) / V_t)


        #Determining continuous clusters for the phase
        continuity_list = _if.identify_continuous_cluster(cluster_ce , cluster_co, inlet_pores, outlet_pores)
        cluster_list_ce = np.unique(cluster_ce[phase_ce])
        cluster_list_co = np.unique(cluster_co[phase_co])
        cluster_list = np.union1d(cluster_list_ce, cluster_list_co)
        phase_clusters = np.intersect1d(cluster_list, continuity_list)

        #For each cluster of the same phase
        for n in phase_clusters:
            #Calculating conductance on each element
            for item in elements:
                BC = pn[f'{item}.boundary']
                status_center = (cluster_info[f'{item}.center'][:,i] == n) | BC
                status_corner = (cluster_info[f'{item}.corner'][:,:,i] == n) | np.tile(BC, (3,1)).T
                interface = (np.tile(status_center, (3,1)).T != status_corner)
                theta_corner = np.tile(water[f'{item}.contact_angle'],(3,1)).T #Funciona con un solo valor de theta
                beta = pn[f'{item}.half_corner_angle']
                R = water[f'{item}.surface_tension'][0]/ p_vals[i]
                bi_corner = _if.interfacial_corner_distance(R, theta_corner, beta, int_cond = interface)
                viscosity = phase[f'{item}.viscosity'][0]
                if item == 'pore':
                    gp_ce, gp_co, _ = _cf.conductance(pn, status_center, status_corner, theta_corner, bi_corner, viscosity, item = item, correction = True)
                else:
                    gt_ce, gt_co,_ = _cf.conductance(pn, status_center, status_corner, theta_corner, bi_corner, viscosity, item = item, correction = True)

            #Calculating conduit conductance
            g_L_mph = _cf.conduit_conductance_2phases(network = pn,
                                        pore_g_ce = gp_ce,
                                        throat_g_ce = gt_ce,
                                        conduit_length = L,
                                        pore_g_co = gp_co,
                                        throat_g_co = gt_co,
                                        corner = True,
                                        check_boundary = True)
            label_conductance = 'throat.conduit_conductance'
            phase[label_conductance] = g_L_mph

            #Calculating multiphase rate flow. Append in q_mph for each cluster
            # Conductance cannot be zero. If that, matrix A become singular (empty rows) and can not be used
            q_mph.append(Rate_calc(pn, phase, inlet_pores , outlet_pores, label_conductance))

        #Calculating k_r only if there are continuous clusters. Otherwise, k_r = 0
        if len(phase_clusters) > 0:
            mph_flow = np.sum(q_mph)
            rel_perm.append(np.squeeze(mph_flow/sph_flow))
        else:
            rel_perm.append(0)

    #Saving data
    output = np.vstack((p_vals,  sat, rel_perm)).T
    np.save('K_' + phase.name, output) #If wp, then 0. Otherwise 1
    np.savetxt('K_'+ phase.name + '.txt', output, fmt=' %.5e '+' %.5e '+' %.5e ', header=' p// sat_w // kr')
"""
