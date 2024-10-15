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

n_side = 20 #pores per side
pn = op.network.Demo(shape=[n_side , n_side , 1], spacing=1e-4)
#pn = op.network.Demo(shape=[15, 15, 1], spacing=1e-4)
Np = pn.Np
Nt = pn.Nt

#Set boundary pores
inlet_pores = pn['pore.front']
outlet_pores = pn['pore.back']
pn['pore.boundary'] = inlet_pores | outlet_pores
pn['pore.internal'] = ~pn['pore.boundary']
boundary_t = pn.find_neighbor_throats(pores=pn['pore.boundary'])
pn['throat.boundary'] = False
pn['throat.boundary'][boundary_t] = True
pn['throat.internal'] = ~pn['throat.boundary']

#Assuming diameter as cross-sectional diameter of a prism, assign beta and calculate G
elements = ['pore', 'throat']
for item in elements:
    G = np.random.rand(len( pn[f'{item}.diameter'])) * 3 ** 0.5/36
    pn[f'{item}.shape_factor'] = G
    pn[f'{item}.half_corner_angle'] = op.teste.geometry.half_angle_isosceles(G)

"""
#Checking if throats have smaller diameter than connected pores
D1, Dt, D2 = pn.get_conduit_data('diameter').T
#print( np.any( Dt > D1) or np.any( Dt > D2) ) #must be false

#setting a path to be invaded first
path = np.array([27,28,29,30,31,32,33,34,35, #main branch
                 40,41,42,43,44, #second branch
                 124, #connection second branch
                 122, #connectrion third branch
                 131,141,133,143,37,38,55,56#third branch
                 ])

#modifying throat diameter (The cross-sectional area change if d  change. Howver. Pc does not depends on that
pn['throat.diameter'] = pn['throat.diameter'] * 0.4 #reducing all diameters
pn['throat.diameter'][path] = np.minimum(D1[path], D2[path]) * 0.999 #set path with bigger diameter, but less than connected pores
pn['throat.diameter'][122] = np.minimum(D1[122], D2[122]) * 0.5 #Connection of third branch with a lower diameter than other throats from path
pn['throat.diameter'][124] = pn['throat.diameter'][27] * 0.99
"""

#After setting diameters, modify cross sectional areas
for item in elements:
    R = pn[f'{item}.diameter'] / 2
    beta = pn[f'{item}.half_corner_angle']
    pn[f'{item}.cross_sectional_area'] = R ** 2 * np.sum(1 / np.tan(beta), axis = 1)
    pn[f'{item}.perimeter'] = R ** 2 * np.sum(1 / np.tan(beta), axis = 1)

#removing throats between boundary pores
t_removed = np.concatenate((np.arange(380, 760, 20) ,  np.arange(399, 770, 20) ))
op.topotools.trim(pn, throats = t_removed)

#SOLO PARA RED CREADA, ACTUALIZANDO NUMERO DE GARGANTAS
Np = pn.Np
Nt = pn.Nt

#Properties extracted from Valvatne - Blunt (2004) Table 1
tension = 30e-3 #N/m
water_visc = 1.05e-3 #Pa/s
oil_visc = 1.39e-3 #Pa/s
theta_r = np.pi / 180 * 0  #Respecto al agua, en sexagecimal
#Creating advancing contact contact_angle
theta_a = np.pi / 180 * 120 #Con 120 todos tienen una camada. Con 100 no

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
pdr.set_inlet_BC(pores=inlet_pores)
pdr.set_outlet_BC(pores=outlet_pores)
pdr.run(throat_diameter = 'throat.diameter')

print(max(pdr['throat.invasion_pressure']))

#Post processing

elements = ['pore', 'throat']
locations = ['center', 'corner']

pmax_drn = 11000 #Invadidos todos, pcmax = 10431
p_vals =  np.array([0, pmax_drn])

#Obtaining phase distribution and clusters for each stage of invasion according to p_vals
results_pdr = pdr.postprocessing2(mode = 'pressure', inv_vals = p_vals)


#True for wp present

R = tension / pmax_drn #interface radius at maximum pc_drainage

info_corner = {}
for item in elements:
    beta = pn[f'{item}.half_corner_angle']
    phase_ce = results_pdr['status_2']['invasion_info'][f'{item}.center']
    phase_co = results_pdr['status_2']['invasion_info'][f'{item}.corner']
    interface = (np.tile(phase_ce, (3,1)).T != phase_co)
    bi = _if.interfacial_corner_distance(R = R,
                                         theta = theta_r,
                                         beta = beta,
                                         int_cond = interface)
    info_corner[f'{item}.mask_corner'] =_if.wp_in_corners(beta = beta,
                                                     theta = theta_r)

    info_corner[f'{item}.mask_layer'] =_if.nwp_in_layers(beta = beta,
                                                         theta_r = theta_r,
                                                         theta_a = theta_a)
    info_corner[f'{item}.pressure_LC'] = _if.pressure_LC(beta = beta,
                bi = bi,
                sigma = tension,
                theta_a = theta_a,
                mask = info_corner[f'{item}.mask_layer'])
    print(f'Have all {item}s at least one layer?')
    print(np.all(np.any(info_corner[f'{item}.mask_layer'], axis = 1)))
    print(f'Number of {item}s with at least one layer?')
    print(np.sum(np.any(info_corner[f'{item}.mask_layer'], axis = 1)))
    print(phase_co)
    print(np.all(phase_co))
    print( np.min((info_corner[f'{item}.pressure_LC'])[ info_corner[f'{item}.mask_layer'] ])  )
    print( np.max(info_corner[f'{item}.pressure_LC'])  )


