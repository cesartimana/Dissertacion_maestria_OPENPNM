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

Nota, la red usada tiene 3 poros con 2 gargantas isoladas
garganta 258: poros 86 , 202
garganta 259: poros 86 , 207


"""

#Preambulo

import openpnm as op
import porespy as ps
import numpy as np
import imageio as io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import sys
import math as m
import _algorithm_class as _alg
import _cond_props as _cp

import heapq as hq

np.random.seed(13)


timeZero = time.time()

ws = op.Workspace()

testName_h = 'Network_w_geo.pnm'

proj_h = ws.load_project(filename=testName_h)

pn = proj_h.network

Np = pn.Np
Nt = pn.Nt
inlet_pores = pn['pore.ymax']
outlet_pores = pn['pore.ymin']


if ('equivalent_diameter' in pn['throat'] ) and ('inscribed_diameter' in pn['throat'] ):
        #Correct inscribed diameter values
        d_corr = pn['throat.inscribed_diameter'] > pn['throat.equivalent_diameter']
        pn['throat.inscribed_diameter'][d_corr] = pn['throat.equivalent_diameter'][d_corr]

a = np.sum(pn['throat.equivalent_diameter'] < pn['throat.inscribed_diameter'])


#Add phase properties

air = op.phase.Air(network=pn,name='air')
air.add_model_collection(op.models.collections.phase.air)
air.add_model_collection(op.models.collections.physics.basic)
air.regenerate_models()
air['throat.viscosity'] = air['pore.viscosity'][0]
water = op.phase.Water(network=pn,name='water')
water['pore.surface_tension'] = 0.072
water['throat.surface_tension'] = 0.072
water.add_model_collection(op.models.collections.phase.water)
water.add_model_collection(op.models.collections.physics.basic)
water['throat.viscosity'] = water['pore.viscosity'][0]
water.regenerate_models()

#Creating contact angles with two columns (if regenerate_models is called after, theta pores is modified)
for item in ['pore', 'throat']:
    n = len( water[f'{item}.all'] )
    theta = np.ones((n,2))
    theta[:,0] = 0
    theta[:,1] = m.pi/3
    water[f'{item}.contact_angle'] = theta


#Using the algorithm
pd = _alg.Primary_Drainage(network=pn, phase=water)


pd.set_inlet_BC(pores=inlet_pores)
pd.set_outlet_BC(pores=outlet_pores)
pd.run(throat_diameter = 'throat.inscribed_diameter', advanced = False)
p_vals, invasion_info, cluster_info = pd.postprocessing(p_max = 1)

#Calculation conduit length
L = _cp.conduit_lenght_tubes(pn,
                             pore_length = "pore.length",
                             throat_spacing = "throat.total_length")

def Rate_calc(network, ph, inlet, outlet, conductance):
    St_p = op.algorithms.StokesFlow(network=network, phase=ph)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=inlet, values=1)
    St_p.set_value_BC(pores=outlet, values=0)
    St_p.run()
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val




#Assumming only one data of pressure. If more than one work woth index
#Previous calculation for conductivity
continuity_list = _cp.identify_continuous_cluster(cluster_info['pore.center'] , cluster_info['pore.corner'] , inlet_pores, outlet_pores)

n_cluster = 0 # 0 for wetting. Otherwise non wetting. Only 0,1 are continuous (see continuity_list...)
if n_cluster == 0:
    phase = water
else:
    phase = air




elements = ['pore', 'throat']
for item in elements:
    status_center = (cluster_info[f'{item}.center'] == n_cluster)
    status_corner = (cluster_info[f'{item}.corner'] == n_cluster)
    interface = (np.tile(invasion_info[f'{item}.center'], (3,1)).T != invasion_info[f'{item}.corner'])
    theta_corner = np.tile(water[f'{item}.contact_angle'][:,0],(3,1)).T
    beta = pn[f'{item}.half_corner_angle']
    R = water[f'{item}.surface_tension'][0]/ p_vals
    bi_corner = _alg._pc_funcs.interfacial_corner_distance(R, theta_corner, beta, int_cond = interface)
    viscosity = phase[f'{item}.viscosity']

    #Only for storage
    if item == 'pore':
        pore_cond_center, pore_cond_corner,ratio_p = _cp.conductance(pn, status_center, status_corner, theta_corner, bi_corner, viscosity, item = item)
        #print(ratio_p)
    else:
        throat_cond_center, throat_cond_corner,_ = _cp.conductance(pn, status_center, status_corner, theta_corner, bi_corner, viscosity, item = item)


mph_conductance = _cp.conduit_conductance_2phases(network = pn,
                                          pore_conductance_center = pore_cond_center,
                                          throat_conductance_center = throat_cond_center,
                                          conduit_length = L,
                                          pore_conductance_corner = pore_cond_corner,
                                          throat_conductance_corner = throat_cond_corner,
                                          pore_conductance_layer = None,
                                          throat_conductance_layer = None,
                                          corner = True,
                                          layer = False)




#Calculating multiphase rate flow
# Any conductance can be zero. If that, matrix A become singular (empty rows) and can not be used
#It is important to set the conduit values in Phase object because it is used by StokesFlow
label_conduc = 'throat.conduit_conductance'
mph_conductance[mph_conductance == 0] = 1e-30
phase[label_conduc] = mph_conductance
mph_flow = Rate_calc(pn, phase, inlet_pores , outlet_pores, label_conduc)
print(mph_flow)

#Calculating single phase flow

elements = ['pore', 'throat']
for item in elements:
    status_center = np.ones_like(pn[f'{item}.shape_factor'], dtype = bool)
    status_corner = np.zeros_like(pn[f'{item}.half_corner_angle'], dtype = bool)
    theta_corner = np.tile(water[f'{item}.contact_angle'][:,0],(3,1)).T
    R = water[f'{item}.surface_tension'][0]/ p_vals
    beta = pn[f'{item}.half_corner_angle']
    bi_corner = np.zeros_like(pn[f'{item}.half_corner_angle'])
    viscosity = phase[f'{item}.viscosity']
    #Only for storage
    if item == 'pore':
        pore_cond_center, pore_cond_corner,_ = _cp.conductance(pn, status_center, status_corner, theta_corner, bi_corner, viscosity, item = item)
    else:
        throat_cond_center, throat_cond_corner,_ = _cp.conductance(pn, status_center, status_corner, theta_corner, bi_corner, viscosity, item = item)

sph_conductance = _cp.conduit_conductance_2phases(network = pn,
                                          pore_conductance_center = pore_cond_center,
                                          throat_conductance_center = throat_cond_center,
                                          conduit_length = L,
                                          pore_conductance_corner = pore_cond_corner,
                                          throat_conductance_corner = throat_cond_corner,
                                          pore_conductance_layer = None,
                                          throat_conductance_layer = None,
                                          corner = True,
                                          layer = False)

label_conduc = 'throat.conductance'
sph_conductance[sph_conductance == 0] = 1e-30
phase[label_conduc] = sph_conductance
sph_flow = Rate_calc(pn, phase, inlet_pores , outlet_pores, label_conduc)
print(sph_flow)
print(mph_flow/ sph_flow)
