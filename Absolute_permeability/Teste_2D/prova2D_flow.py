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

print(pn)

#Assuming all volume is on the pores
V_sph = sum(pn['pore.volume'][pn['pore.internal']])




#Calculation conduit length
L = _cf.conduit_length_tubes(pn,
                             pore_length = "pore.diameter",
                             throat_spacing = "throat.spacing", #usually is total_length
                             min_L = 10e-7)

#Dimensions for Absolute permeability (for 2D D/A = 1 / 'espesor de red eje z')
A = np.max(pn['pore.diameter'])
D = 1

#Rate calc function
#Flowrate function
def Rate_calc(network, ph, inlet, outlet, conductance):
    St_p = op.algorithms.StokesFlow(network=network, phase=ph)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=inlet, values=1) #P_inlet = 1
    St_p.set_value_BC(pores=outlet, values=0) #P_outlet = 1
    St_p.run()
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val


elements = ['throat']

#Calculating flow rate for each phase: wp and nwp
for wp in [True,False]:

    rel_perm = []
    sat = []

    if wp:
        phase = water
    else:
        phase = oil

    #Calculating conductance on each element
    for item in elements:
        status_center = np.ones_like(pn[f'{item}.shape_factor'], dtype = bool)
        status_corner = np.ones_like(pn[f'{item}.half_corner_angle'], dtype = bool)
        theta_corner = np.zeros_like(pn[f'{item}.half_corner_angle'])
        bi_corner = np.zeros_like(pn[f'{item}.half_corner_angle'])
        viscosity = phase[f'{item}.viscosity'][0]
        if item == 'pore':
            pore_g_center, pore_g_corner, _, _ = _cf.conductance(pn, status_center, status_corner, theta_corner, bi_corner, viscosity, item = item)
            print('----pore------')
            print(pore_g_center)
            print(np.average(pore_g_center))
        else:
            throat_g_center, throat_g_corner,_,_= _cf.conductance(pn, status_center, status_corner, theta_corner, bi_corner, viscosity, item = item)
            print('----throat----')
            print(throat_g_center[0: 10])
            print(pn['throat.cross_sectional_area'][0:10])
            #print('----G=Gmax----')
            #throat_g_center = throat_g_center / pn['throat.shape_factor'] * (3**0.5 / 36)
            #print(throat_g_center[0: 10])
            print('----end--throat----')

    pore_g_center = np.ones_like(pn['pore.shape_factor']) * 1e-15

    #Calculating conduit conductance
    sph_g_L = _cf.conduit_conductance_2phases(network = pn,
                                            pore_g_ce = pore_g_center,
                                            throat_g_ce = throat_g_center,
                                            conduit_length = L,
                                            pore_g_co = None,
                                            throat_g_co = None,
                                            pore_g_la = None,
                                            throat_g_la = None,
                                            corner = False,
                                            layer = False)
    label_conductance = 'throat.conductance'
    phase[label_conductance] = sph_g_L
    #print(sph_g_L[0:10])
    #print(1/4/np.pi)
    #print(3**0.5 / 36)
    #print(0.5*1/4/np.pi)
    #print(0.6*3**0.5 / 36)
    print('-- relacion entre conductancias prisma / cilindro --')
    print( (0.6*3**0.5 / 36) / (0.5*1/4/np.pi) )

    #Calculating single phase flow
    Q = Rate_calc(pn, phase, inlet_pores , outlet_pores, label_conductance)[0]
    print('%s properties' %phase.name)
    print(f'The value of flow rate at total saturation is: {Q:.4e} m3/s ')

    #Calculating absolute permeability
    # K = Q * D * mu / (A * Delta_P) # mu and Delta_P were assumed to be 1.
    K = Q * D * phase[f'{item}.viscosity'][0] / A #DeltaP is 1, accoridng to de func Rate_calc
    print(f'The value of K is: {K:.2e} m2')
    print(f'The value of K is: {K*1.01325e15:.2f} mD')
    print(0.03 / K ** 0.5  / 10**6)

"""
cilindro

[2.45565297e-14 2.50702073e-14 3.97516817e-13 6.24879105e-13
 8.97639140e-14 7.51009477e-14 1.61622829e-13 1.84040364e-13
 1.78890659e-13 4.27132761e-15]

prisma
[1.51260736e-14 1.54550566e-14 3.12483653e-13 5.31900287e-13
 5.96649710e-14 4.95671404e-14 1.13592519e-13 1.31199872e-13
 1.27425670e-13 2.50099255e-15]

"""
