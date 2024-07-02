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
#pn = op.network.Demo(shape=[15, 15, 1], spacing=1e-4)
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

#Modifying diameter in throats between inlet pores, because I dont know how to remove them
#pn['throat.diameter'][np.arange(90, 180, 10)] = 1e-10
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


#Simulating primary Drainage
pd = _alg.Primary_Drainage(network=pn, phase=water)
pd.set_inlet_BC(pores=inlet_pores)
pd.set_outlet_BC(pores=outlet_pores)
pd.run(throat_diameter = 'throat.diameter')
print(pd)

b = np.argsort(pd['throat.entry_pressure'])
print(b[0:len(path)])
print(path)
print(np.sort(path))


inv_pattern = pd['throat.invasion_sequence'] <5
print(max(pd['throat.invasion_pressure'][inv_pattern]))
ax = op.visualization.plot_coordinates(network=pn, pores=pn.pores('front'), c='r', s=50)
ax = op.visualization.plot_coordinates(network=pn, pores=pn.pores('front', mode='not'), c='grey', ax=ax)
op.visualization.plot_connections(network=pn, throats=inv_pattern, ax=ax, c = 'blue', linewidth = 5)
op.visualization.plot_connections(network=pn, throats=path, ax=ax, c = 'yellow', linewidth = 2)
plt.show()


bool_path = np.zeros_like(inv_pattern, dtype = bool)
bool_path[path] = True

import pandas as pad

for item in elements:
    df = pad.DataFrame(
        {
            "d": pn[f'{item}.diameter'],
            "beta_0": pn[f'{item}.half_corner_angle'][:,0],
            "beta_1": pn[f'{item}.half_corner_angle'][:,1],
            "beta_2": pn[f'{item}.half_corner_angle'][:,2],
            "G": pn[f'{item}.shape_factor'],
        }
    )
    print(item)
    if item == 'throat':
        df.insert(5, "pce", pd[f'{item}.entry_pressure'])
        df.insert(6, "path", bool_path)
        df.insert(7,"P1", pn[f'{item}.conns'][:,0])
        df.insert(8,"P2", pn[f'{item}.conns'][:,1])
        #df.insert(9,"Area", pn[f'{item}.cross_sectional_area'])
    df.to_csv(f'{item}.csv', index=True)

"""
r = pn['throat.diameter'] / 2
G = pn["throat.shape_factor"]
beta = pn["throat.half_corner_angle"]
S1 = np.sum((np.cos(theta) * np.cos(theta + beta) / np.sin(beta) + theta + beta - np.pi/2), axis = 1)
S2 = np.sum((np.cos(theta + beta) / np.sin(beta)), axis = 1)
S3 = 2*np.sum((np.pi / 2 - theta - beta), axis = 1)
D = S1 - 2 * S2 * np.cos(theta) + S3
root = (1 + 4 * G * D / np.cos(theta)**2)
value = tension * (1 + root**0.5) / r
i=1
print(pd['throat.entry_pressure'][i])
print(pn['throat.half_corner_angle'][i,:])
print(S1[i])
print(S2[i])
print(S3[i])
print(D[i])
print(root[i])
print(value[i])
"""
