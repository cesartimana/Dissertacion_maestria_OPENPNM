# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar

"""

import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import math as m
import _conductance_funcs as _cf
import _invasion_funcs as _if
np.random.seed(13)

resolution = 5.345e-6

#Flowrate function
def Rate_calc(network, ph, inlet, outlet, conductance):
    St_p = op.algorithms.StokesFlow(network=network, phase=ph)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=inlet, values=1) #P_inlet = 1
    St_p.set_value_BC(pores=outlet, values=0) #P_outlet = 1
    St_p.run()
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val


#Calculating cross-sectional_area for cilinders
def geo_props_pores_cylinder(network, root = None, corr = False, increase_factor = 1.01, throat_area = 'throat.cross_sectional_area'):

    r"""
    Add the property 'pore.cross_sectional_area' and 'pore.shape_factor' to the network.
    Assume a cylinder.

    Assumes that te input pore.surface_area ignores throat areas.
    pore.cylinder_surface_area is the area used in calculus minus throat areas
    root choose the root criteria:
    -If  None, it choose the  more homogeneous aspect ratio (max(a,L) / min (a,L))
    -If 0, d > L
    -If 1, d < L
    -Other values are rejected
    corr: If True, the pores with lower cross sectional area than any connected throat area are modified to be bigger.
    pore.surface_area is corrected.
    For that, only the pore volume is considered and the surface area is modified
    increase_factor: Only if corr== True. for those pore whose A_p <=  max(A_t) we do A_p =  max(A_t) * increase_factor
    throat_area: Only if corr== True. Key name to call the values of the cross sectional area
    """
    V_p = network['pore.volume']
    S_p = network['pore.surface_area']
    A_t = network['throat.cross_sectional_area']
    t_conns = network['throat.conns']
    Np = network.Np
    Nt = network.Nt

    #Adding areas: solid-fluid + throat areas per pore
    As_p = np.copy(S_p)
    im = network.create_incidence_matrix()
    np.add.at(As_p, im.row, A_t[im.col])

    #Calculating geometric properties
    G_max = 1 / (4 * np.pi) #G for a cylinder
    As_min = 3 * ( 2 * np.pi ) ** (1/3) * V_p ** (2/3)
    As_used = np.fmax(As_p, As_min*(1+1e-8)) #To not have problems calculating parameters

    p = - As_used / (2 * np.pi)
    q = V_p / np.pi

    if root is None:
        r0 = 2/3**0.5*np.power(-p, 0.5)*np.cos(1/3*np.arccos(3*q*3**0.5/2/p/np.power(-p, 0.5)))
        L0 = V_p / ( np.pi * r0 ** 2 )
        r1 = 2/3**0.5*np.power(-p, 0.5)*np.cos(1/3*np.arccos(3*q*3**0.5/2/p/np.power(-p, 0.5))-2*np.pi*1/3)
        L1 = V_p / ( np.pi * r1 ** 2 )
        mask = (np.maximum(2*r0,L0) / np.minimum(2*r0,L0)) > (np.maximum(2*r1,L1) / np.minimum(2*r1,L1))
        r_p = r0 * ~mask + r1 * mask
    elif root in [0,1]:
        r_p = 2/3**0.5*np.power(-p, 0.5)*np.cos(1/3*np.arccos(3*q*3**0.5/2/p/np.power(-p, 0.5))-2*np.pi*1/3*root)
    else:
        raise Exception('root must be None, 0 or 1')
    d_p = 2 * r_p
    A_p = np.pi * r_p ** 2

    #Modifying pore_diameter if desired
    if corr:
        if increase_factor <= 1:
            increase_factor = 1.01
            print("increase_factor must be bigger than 1. Used 1.01 instead")
        #Identifying maximum throat area for each throat
        A_t_max = np.zeros((Np ))
        np.maximum.at(A_t_max, im.row, A_t[im.col])
        mask = A_p <= A_t_max
        print("The cross-sectional area of %0.3f%% of the pores were modified to be bigger than the connected throats" % (np.sum(mask) / network.Np))
        print("For these pores, we modify the prism properties")
        A_p[mask] = A_t_max[mask] * increase_factor
        d_p[mask] = ( 4 * A_p[mask] / np.pi) ** 0.5
    #print('hol')
    #print((np.sum(mask) / network.Np))
    #print(network.Np)
    #print(np.sum(mask))
    #print(np.sum(mask & network['pore.boundary']))
    #print(np.sum(mask & network['pore.internal']))
    L_p = V_p/A_p
    if corr:
        As_used[mask] = ( L_p * np.pi * d_p + 2 * A_p)[mask]
    #Substracting throat areas
    np.add.at(As_used, im.row, (-A_t)[im.col])
    network['pore.cylinder_length'] = L_p
    network['pore.cross_sectional_area'] = A_p
    network['pore.cylinder_surface_area'] = As_used
    network['pore.shape_factor'] = np.ones(Np) * G_max
    network['pore.cylinder_inscribed_diameter'] = d_p
    return

#Reading netwrork data
ws = op.Workspace()
testName_h = 'Berea_G_0.pnm' # 'Berea_G_0.pnm' or 'Berea.pnm'
proj_h = ws.load_project(filename=testName_h)
pn = proj_h.network
Np = pn.Np
Nt = pn.Nt

#Defining boundary conditions
axis = 'x'
inlet_pores = pn['pore.'+axis+'min']
index_inlet = pn.pores(axis + 'min')
outlet_pores = pn['pore.' + axis + 'max']
index_outlet = pn.pores(axis + 'max')


#Properties extractend from Valvatne - Blunt (2004) Table 1
tension = 30e-3 #N/m
water_visc = 1.05e-3 #Pa/s
oil_visc = 1.39e-3 #Pa/s
theta = 0 #water phase

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


#Assuming all volume is on the pores
pn['pore.internal'] = ~pn['pore.boundary']
V_sph = sum(pn['pore.volume'][pn['pore.internal']])

#Calculation conduit length
"""
L = _cf.conduit_length_tubes(pn,
                             pore_length = "pore.equivalent_diameter", # "pore.equivalent_diameter" or  "pore.inscribed_diameter"
                             throat_spacing = "throat.total_length",
                             min_L = resolution)
"""
L = _cf.conduit_length_spheres_cylinders(pn,
                                        pore_diameter = 'pore.inscribed_diameter',
                                        throat_diameter = 'throat.equivalent_diameter',
                                        throat_spacing = 'throat.total_length',
                                        L_min = resolution)

#Dimensions for Absolute permeability
A = (400 * resolution) ** 2
D = 400 * resolution

elements = ['pore', 'throat']

#Cambiando el area de los poros, formula de cilindro
geo_props_pores_cylinder(pn, root = None, corr = True, increase_factor = 1.01, throat_area = 'throat.cross_sectional_area')

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
        A_cs = pn[f'{item}.cross_sectional_area']
        viscosity = phase[f'{item}.viscosity'][0]
        if item == 'pore':
            pore_g_center = 0.5 * A_cs ** 2 / (4 * np.pi) / viscosity
            #Considerando pore_g_center muy grande para ignorar
            #pore_g_center = np.ones_like(pore_g_center)
        else:
            throat_g_center = 0.5 * A_cs ** 2 / (4 * np.pi) / viscosity
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
    #ignoring g_BC
    #sph_g_L[pn['throat.boundary']] = np.max(sph_g_L) * 1000

    label_conductance = 'throat.conductance'
    phase[label_conductance] = sph_g_L

    #Calculating single phase flow
    Q = Rate_calc(pn, phase, inlet_pores , outlet_pores, label_conductance)[0]
    print('%s properties' %phase.name)
    print(f'The value of flow rate at total saturation is: {Q:.4e} m3/s ')

    #Calculating absolute permeability
    # K = Q * D * mu / (A * Delta_P) # mu and Delta_P were assumed to be 1.
    K = Q * D * phase[f'{item}.viscosity'][0] / A #DeltaP is 1, accoridng to de func Rate_calc
    print(f'The value of K is: {K:.2e} m2')
    print(f'The value of K is: {K/0.98e-12*1000:.2f} mD')
    print(0.03 / K ** 0.5  / 10**6)

"""
Vol_void = np.sum(pn['pore.volume'][ ~pn['pore.boundary']])
Vol_bulk = (400 * resolution) ** 3
Poro = Vol_void / Vol_bulk
print(f'The value of Porosity is: {Poro:.4f}')
print(np.average(pn['throat.equivalent_diameter']))
print(np.average(pn['pore.equivalent_diameter']))
print(pn['pore.equivalent_diameter'])
print(np.average(pn['throat.cross_sectional_area']))
"""
