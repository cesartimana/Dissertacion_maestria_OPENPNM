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

#Reading netwrork data
ws = op.Workspace()
testName_h = 'Berea_G_0.pnm'
proj_h = ws.load_project(filename=testName_h)
pn = proj_h.network
Np = pn.Np
Nt = pn.Nt
print(pn)

#Defining internal pores, throats
pn['pore.internal'] = True
pn['pore.internal'][pn['pore.boundary']] = False
boundary_p = pn.pores('boundary')
boundary_t = pn.find_neighbor_throats(pores=boundary_p)
pn['throat.internal'] = True
pn['throat.internal'][boundary_t] = False
pn['throat.boundary'] = True
pn['throat.boundary'][pn['throat.internal']] = False

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
V_sph = sum(pn['pore.volume'][pn['pore.internal']])

#Calculation conduit length
L = _cf.conduit_length_tubes(pn,
                             pore_length = "pore.equivalent_diameter",
                             throat_spacing = "throat.total_length",
                             min_L = resolution)
#L[:, 1] = resolution

"""
mask_L = L[:,1] == resolution
print(np.sum(mask_L))
print(np.sum(mask_L[pn['throat.internal']]))
print(np.sum(mask_L[pn['throat.boundary']]))
print(pn)

"""


#Dimensions for Absolute permeability
A = (400 * resolution) ** 2
D = 400 * resolution


#Test for Area
inlets = pn.coords[index_inlet]
outlets = pn.coords[index_outlet]

Nin = np.ptp(inlets, axis=0) > 0

Nout = np.ptp(outlets, axis=0) > 0

from scipy.spatial import ConvexHull


hull_in = ConvexHull(points=inlets[:, Nin])
hull_out = ConvexHull(points=outlets[:, Nout])

#Test for lenght

from scipy.spatial import cKDTree
tree = cKDTree(data=inlets)
Ls = np.unique(np.float64(tree.query(x=outlets)[0]))


elements = ['pore', 'throat']



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
            pore_g_center, _, _, _ = _cf.conductance(pn, status_center, status_corner, theta_corner, bi_corner, viscosity, item = item)
            pore_g_center = pore_g_center / pn[f'{item}.shape_factor'] * 3**0.5/36
            #pore_g_center = np.ones_like(pore_g_center)
        else:
            throat_g_center, _,_,_= _cf.conductance(pn, status_center, status_corner, theta_corner, bi_corner, viscosity, item = item)
            throat_g_center = throat_g_center / pn[f'{item}.shape_factor'] * 3**0.5/36

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
print(water['throat.conductance'][:20])
print('hola')
print(3**0.5 / 36 * 0.6 / (0.5 / 4 / np.pi))


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
