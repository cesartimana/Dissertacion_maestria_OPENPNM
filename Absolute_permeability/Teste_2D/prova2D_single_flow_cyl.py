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
import _algorithm_class as _alg
import _conductance_funcs as _cf
np.random.seed(13)

n_side = 10 #pores per side
pn = op.network.Demo(shape=[n_side , n_side , 1], spacing=1e-4)
Np = pn.Np
Nt = pn.Np

#Set boundary pores
inlet_pores = pn.pores('front')
outlet_pores = pn.pores('back')

#Assuming diameter as cross-sectional diameter of a prism, assign beta and calculate G
elements = ['pore', 'throat']
for item in elements:
    G = np.random.rand(len( pn[f'{item}.diameter'])) * 3 ** 0.5/36
    pn[f'{item}.shape_factor'] = G
    pn[f'{item}.half_corner_angle'] = op.teste.geometry.half_angle_isosceles(G)

#Checking if throats have smaller diameter than connected pores
D1, Dt, D2 = pn.get_conduit_data('diameter').T
print( np.any( Dt > D1) or np.any( Dt > D2) ) #must be false

#setting a path to be invaded first
#path = np.arange((n_side - 1 ) * (n_side - 6) , (n_side - 1 ) * (n_side - 5)) #front-back
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

#checking that the bigger throats are from path
a = np.argsort(pn['throat.diameter'])
print(a[-len(path):])
print(path)

#Removing throats between boundary pores
t_removed = np.concatenate((np.arange(90, 180, 10) ,  np.arange(99, 189, 10) ))
op.topotools.trim(pn, throats = t_removed)

#Updating path indexes because of removing throats (we assume the bigger diameters are found on path)
a = np.argsort(pn['throat.diameter'])
path = a[-len(path):]

#Properties extracted from Valvatne - Blunt (2004) Table 1
tension = 30e-3 #N/m
water_visc = 1.05e-3 #Pa/s
oil_visc = 1.39e-3 #Pa/s
theta = np.pi / 12 #Respecto al agua, en sexagecimal

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

#Setting boundaries and internal
pn['pore.boundary'] = False
pn['pore.boundary'][inlet_pores] = True
pn['pore.boundary'][outlet_pores] = True
pn['pore.internal'] = ~pn['pore.boundary']

#Assuming all volume is on the pores
V_sph = sum(pn['pore.volume'][pn['pore.internal']])

print(pn)

f = op.models.physics.hydraulic_conductance.generic_hydraulic
water.add_model(propname='throat.hydraulic_conductance', model=f)
print(water['throat.hydraulic_size_factors'][0:10])
print(water['throat.hydraulic_conductance'][0:10])
oil.add_model(propname='throat.hydraulic_conductance', model=f)
sf = op.algorithms.StokesFlow(network=pn, phase=water)
sf.set_value_BC(pores = inlet_pores, values = 1)
sf.set_value_BC(pores = outlet_pores, values = 0)
sf.run()
Q = sf.rate(pores=inlet_pores, mode='group')[0]
print(Q)
print(water['throat.hydraulic_size_factors'][0:10] / water_visc)
print(1 / np.sum(1 / ( water['throat.hydraulic_size_factors'][0:10] / water_visc ) , axis = 1 ))

#Dimensions for Absolute permeability (for 2D D/A = 1 / 'espesor de red eje z')
A = np.max(pn['pore.diameter'])
D = 1

K = Q * D * water['pore.viscosity'][0] / A #DeltaP is 1, accoridng to de func Rate_calc
print(f'The value of K is: {K:.2e} m2')
print(f'The value of K is: {K*1.01325e15:.2f} mD')
print(0.03 / K ** 0.5  / 10**6)
