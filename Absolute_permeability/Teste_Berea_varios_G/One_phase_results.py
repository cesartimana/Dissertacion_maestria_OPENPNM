# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar
Check if the functions of one phase and two phase give the same results
#IT WORKS!
"""

import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import math as m

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
testName_h = 'Berea_G_0.pnm'
proj_h = ws.load_project(filename=testName_h)
pn = proj_h.network
Np = pn.Np
Nt = pn.Nt
conns = pn['throat.conns']

#Defining boundary conditions
axis = 'x'
inlet_pores = pn['pore.' + axis + 'min']
outlet_pores = pn['pore.' + axis + 'max']

#Properties extractend from Valvatne - Blunt (2004) Table 1
visc = 1e-3 #Pa/s

#D usado en conductance, pero requiero antes para setear L_min
pn['pore.diameter'] = pn['pore.equivalent_diameter'] #Change between extended and equivalent
pn['throat.diameter'] = pn['throat.equivalent_diameter']
D1, Dt, D2 = pn.get_conduit_data('diameter').T
conns = pn['throat.conns']

#Add phase properties
phase = op.phase.Phase(network=pn)
phase['pore.viscosity'] = visc
phase['throat.viscosity'] = visc

#conduit_length
pn['throat.spacing'] = pn['throat.total_length']

"""
#Modifying diameters to not have problems (Optional)
#---------start--------------
#diameter of boundary pores and throats
mask_tBC = pn['throat.boundary']
mask_pBC = np.isin(conns, pn.pores('boundary'))
for i in range(Nt):
    if mask_tBC[i]:
        D = D1[i]*~mask_pBC[i, 0] + D2[i]*~mask_pBC[i, 1]
        pn['pore.diameter'][conns[i, 0]] = D
        pn['pore.diameter'][conns[i, 1]] = D
        pn['throat.diameter'][i] = 0.95*D  #choose arbitrary
#Solving problems with conduits with Dt > = Dp
D1, Dt, D2 = pn.get_conduit_data('diameter').T
mask = (D1 <= Dt) | (D2 <= Dt)
print(f'Number of internal throats with Dt > = Dp: {np.sum(mask):.2f}' )
Dt[mask] = np.minimum(D1, D2)[mask] * 0.95 #choose arbitrary
pn['throat.diameter'] = Dt
#Solving problems with , supposely, a pore inscribed into another,
#equivalent = 3
L_ctc =pn['throat.spacing']
D1, Dt, D2 = pn.get_conduit_data('diameter').T
mask_L = np.abs(D1/2 - D2/2) > L_ctc
print(f'Number of conduits with a pore suposely inside another:   {np.sum(mask_L):.2f} ')
pn['throat.spacing'][mask_L] = (np.abs(D1/2 - D2/2)*1.001)[mask_L]
#---------end--------------
"""


#Calculation conduit length
pn['throat.conduit_lengths'] = _cf.conduit_length_tubes(pn,
                             pore_length = "pore.equivalent_diameter",
                             throat_spacing = "throat.spacing",
                             L_min = resolution,
                             check_boundary = True)

#Testing Formulas for mono and two-phase
label_conductance = 'throat.conductance'

#Using formula for monophase
#---------start----------
g = _cf.conductance_triangle_OnePhase(
    phase,
    correction = True)

phase[label_conductance] = g

#Calculating single phase flow
Q = Rate_calc(pn, phase, inlet_pores , outlet_pores, label_conductance)
print(Q)
#---------end----------

#Using formula for two phases
#---------start----------
elements = ['pore', 'throat']

#Calculating conductance on each element
for item in elements:
    status_center = np.ones_like(pn[f'{item}.shape_factor'], dtype = bool)
    status_corner = np.ones_like(pn[f'{item}.half_corner_angle'], dtype = bool)
    theta_corner = np.ones_like(pn[f'{item}.half_corner_angle'], dtype = bool) * 0.1
    bi_corner = np.ones_like(pn[f'{item}.half_corner_angle'], dtype = bool) * 1e-6
    viscosity = phase[f'{item}.viscosity'][0]
    g, _, _ = _cf.conductance(pn, status_center, status_corner, theta_corner, bi_corner, viscosity, item = item, correction = True)
    if item == 'pore':
        gp = g
    else:
        gt = g

g = _cf.conduit_conductance_2phases(network = pn,
                                        pore_g_ce = gp,
                                        throat_g_ce = gt,
                                        conduit_length = pn['throat.conduit_lengths'])

phase[label_conductance] = g

#Calculating single phase flow
Q = Rate_calc(pn, phase, inlet_pores , outlet_pores, label_conductance)
print(Q)
