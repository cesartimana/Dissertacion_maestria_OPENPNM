# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar

"""

import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import math as m
import _algorithm_class as _alg
import _algorithm_class_2 as _alg2
import _conductance_funcs as _cf
import _invasion_funcs as _if
np.random.seed(13)

resolution = 5.345e-6

#Cumulativo
def cum_hist(x):
    xsort = np.sort(x)
    N = len(x)
    order = np.arange(N) + 1
    return xsort, order/(N+1)

#Reading netwrork data
ws = op.Workspace()
testName_h = 'Berea.pnm'
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


pn['throat.diameter'] = pn['throat.inscribed_diameter']

#Defining boundary conditions
inlet_pores = pn['pore.ymin']
outlet_pores = pn['pore.ymax']

#Properties extractend from Valvatne - Blunt (2004) Table 1
tension = 30e-3 #N/m
water_visc = 1.05e-3 #Pa/s
oil_visc = 1.39e-3 #Pa/s
theta = np.pi / 3 * 0.5 #Water phase

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

p_max = 8000#5261.760325085836
#Obtaining phase distribution and clusters for each stage of invasion according to p_max
p_vals, invasion_info, cluster_info = pd.postprocessing(p_max = p_max)

i = np.arange(8000,8003)
print('Throat invasion information')
print('-----------------------------')
print('center')
print(invasion_info['throat.center'][i])
print('corner')
print(invasion_info['throat.corner'][i])

print(' ')
print('Throat cluster information')
print('-----------------------------')
print('center')
print(cluster_info['throat.center'][i])
print('corner')
print(cluster_info['throat.corner'][i])
print(type(invasion_info))
#Remember: invasion_info is True if the wetting phase is in that location. The first cluster is from de wetting phase and is 0
print(pd)

#Working with new clusters and the boundary conditions
#If all inlet pores are with only water, the throats connected to 2 inlet pores are trapped
#We work with that because a throat is a medium to invade a pore, which have all volume.


#Calculating bi after invade inlet pores (because bi for inlet pores is 0, all water.
#int_cond is interface condition. Only exist if center and corner are different phases
int_cond_p = np.logical_xor(invasion_info['pore.corner'], np.tile(invasion_info['pore.center'],(3,1)).T)
int_cond_t = np.logical_xor(invasion_info['throat.corner'], np.tile(invasion_info['throat.center'],(3,1)).T)
pore_bi = _if.interfacial_corner_distance(R =  tension / p_max ,
                                          theta = theta,
                                          beta = pn['pore.half_corner_angle'],
                                          int_cond = int_cond_p)
throat_bi = _if.interfacial_corner_distance(R =  tension / p_max ,
                                          theta = theta,
                                          beta = pn['throat.half_corner_angle'],
                                          int_cond = int_cond_t)


pimb = _alg2.Primary_Imbibition(network=pn, wp=water, nwp = oil)
pimb.set_inlet_BC(pores=inlet_pores)
pimb.set_outlet_BC(pores=outlet_pores)
c, _, _ = pimb._run_setup(invasion_info)
theta_a_t = np.pi / 3

invaded_t_list = np.where(~invasion_info['throat.center'])[0]
invaded_p_list = np.where(~invasion_info['pore.center'])[0]

#Calculating pressure PLD
#La funcion debe de ser capaz de alterar la tolerancia de presion

PLD_t = np.ones(Nt) * -np.inf
PLD_p = np.ones(Np) * -np.inf

for t_index in invaded_t_list:

    beta_t = pn['throat.half_corner_angle'][t_index]
    interface_t = int_cond_t[t_index]
    b_t = throat_bi[t_index]
    A_t = pn['throat.cross_sectional_area'][t_index]
    G_t = pn['throat.shape_factor'][t_index]
    d_t = pn['throat.diameter'][t_index]

    #max_iter esta porlas

    p_t, max_iter = _if.pressure_PLD(beta = beta_t,
                        interface = interface_t,
                        b = b_t,
                        sigma = tension,
                        theta_a = theta_a_t,
                        A = A_t,
                        G = G_t,
                        d = d_t,
                        p_max = p_max,
                        theta_r = theta)
    PLD_t[t_index] = p_t


for p_index in invaded_p_list:

    beta = pn['pore.half_corner_angle'][p_index]
    interface = int_cond_p[p_index]
    b = pore_bi[p_index]
    A = pn['pore.cross_sectional_area'][p_index]
    G = pn['pore.shape_factor'][p_index]
    d = pn['pore.inscribed_diameter'][p_index]

    #max_iter esta porlas

    p_p, max_iter = _if.pressure_PLD(beta = beta,
                        interface = interface,
                        b = b,
                        sigma = tension,
                        theta_a = theta_a_t,
                        A = A,
                        G = G,
                        d = d,
                        p_max = p_max,
                        theta_r = theta)
    PLD_p[p_index] = p_p


#Calculating pressure PB
PB_p = np.ones(Np) * -np.inf

prueba = _if.pressure_PB(n_inv_t = np.array([3,2]),
                sigma = tension,
                theta_a = theta_a_t,
                d = np.array([1.71163428052212e-05, 0.71163428052212e-05]),
                )
print(prueba)

#Calculating pressure pressure_snapoff



prueba_2 = _if.pressure_snapoff(beta = pn['throat.half_corner_angle'][invaded_t_list],
                     interface = int_cond_t[invaded_t_list],
                     sigma = tension,
                     theta_r = theta,
                     theta_a = theta_a_t,
                     d = pn['throat.diameter'][invaded_t_list],
                     pc_max = p_max,
                     max_it = 10,
                     tol =  0.1)
print(len(prueba_2))
