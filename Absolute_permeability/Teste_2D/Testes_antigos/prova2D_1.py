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

n_side = 15 #pores per side
pn = op.network.Demo(shape=[n_side , n_side , 1], spacing=1e-4)
#pn = op.network.Demo(shape=[15, 15, 1], spacing=1e-4)
Np = pn.Np
Nt = pn.Np

#Assuming diameter as cross-sectional diameter of a prism, create G, beta (equilateral)

elements = ['pore', 'throat']
for item in elements:
    pn[f'{item}.shape_factor'] = 3**0.5/36
    pn[f'{item}.half_corner_angle'] = np.ones((len(pn[f'{item}.diameter']),3)) * np.pi / 6

#Checking if throats have smaller diameter than connected pores
D1, Dt, D2 = pn.get_conduit_data('diameter').T
print( np.any( Dt > D1) or np.any( Dt > D2) ) #must be false

#setting a path to be invaded first
path = np.arange((n_side - 1 ) * (n_side - 6) , (n_side - 1 ) * (n_side - 5)) #front-back

#modifying throat diameter
pn['throat.diameter'] = pn['throat.diameter'] * 0.4 #reducing all diameters
pn['throat.diameter'][path] = np.minimum(D1[path], D2[path]) * 0.999 #set path with bigger diameter, but less than connected pores

#checking that the bigger throats are from path
a = np.argsort(pn['throat.diameter'])
print(a[-(n_side - 1):])
print(path)

#Set boundary pores
inlet_pores = pn.pores('front')
outlet_pores = pn.pores('back')

#Properties extracted from Valvatne - Blunt (2004) Table 1
tension = 30e-3 #N/m
water_visc = 1.05e-3 #Pa/s
oil_visc = 1.39e-3 #Pa/s
theta = 0 #Respecto al agua, en sexagecimal

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
pd = _alg.Primary_Drainage(network=pn, phase=water)
pd.set_inlet_BC(pores=inlet_pores)
pd.set_outlet_BC(pores=outlet_pores)
pd.run(throat_diameter = 'throat.diameter')
print(pd)

inv_pattern = pd['throat.invasion_sequence'] < 15
ax = op.visualization.plot_coordinates(network=pn, pores=pn.pores('front'), c='r', s=50)
ax = op.visualization.plot_coordinates(network=pn, pores=pn.pores('front', mode='not'), c='grey', ax=ax)
op.visualization.plot_connections(network=pn, throats=inv_pattern, ax=ax, c = 'blue', linewidth = 5)
op.visualization.plot_connections(network=pn, throats=path, ax=ax, c = 'yellow', linewidth = 2)

plt.show()
