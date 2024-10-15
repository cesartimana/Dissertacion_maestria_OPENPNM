import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import time
from parametersCubic import *
from scipy import optimize

#importing functions for other file
import sys
sys.path.insert(1, '/home/cesar/OpenPNM_files/mestrado/_funcs')
import _algorithm_class as _alg
import _conductance_funcs as _cf
import _invasion_funcs as _if

np.random.seed(seed) #Esto hara que siempre cree una misma red. Puede retirarse al final del medio

ws = op.Workspace()

pn = op.network.Cubic(shape=shape, spacing=sp, connectivity=6)

#Adding inlet and outlet elements

chosen_axis = 'x'

if chosen_axis == 'x':
    axis_index = 0
elif chosen_axis == 'y':
    axis_index = 1
elif chosen_axis == 'z':
    axis_index = 2
else:
    raise Exception('Choose one of the three axes')

p_int_inlet = pn.pores('pore.' + chosen_axis + 'min')
inlet_coords = pn['pore.coords'][ p_int_inlet ]
inlet_coords[:, axis_index] -= sp

p_int_outlet = pn.pores('pore.' + chosen_axis + 'max')
outlet_coords = pn['pore.coords'][ p_int_outlet ]
outlet_coords[:, axis_index] += sp

op.topotools.extend(network=pn, coords=inlet_coords, labels='inlet')
temp_inlet = pn.pores('inlet')
op.topotools.extend(network=pn, conns= np.column_stack((p_int_inlet, temp_inlet)), labels='inlet')

op.topotools.extend(network=pn, coords=outlet_coords, labels='outlet')
temp_outlet = pn.pores('outlet')
op.topotools.extend(network=pn, conns= np.column_stack((p_int_outlet, temp_outlet)), labels='outlet')

#Defining boundary and internal elements
pn['pore.boundary'] = pn['pore.inlet'] | pn['pore.outlet']
pn['pore.internal'] = ~pn['pore.boundary']
pn['throat.boundary'] = pn['throat.inlet'] | pn['throat.outlet']
pn['throat.internal'] = ~pn['throat.boundary']

#Defining geometry properties

#Counting elements
Nt = pn.Nt
Np = pn.Np
conns = pn['throat.conns']

G_max = 3**0.5/36
G_min = 0.001 * G_max

F_neigh_p = op.models.misc.from_neighbor_pores
pn.add_model(propname='throat.max_area', model=F_neigh_p, mode = 'min', prop = 'pore.cross_sectional_area', regen_mode = 'deferred')
pn['pore.cross_sectional_area'] = np.zeros(Np)

def set_cubic_props(a = 1):
    #Pore props
    Dp_eq =  (np.random.rand(Np) * (range_Dp[1] - range_Dp[0]) + range_Dp[0]) * sp
    Lp =  (np.random.rand(Np) * (range_Lp[1] - range_Lp[0]) + range_Lp[0]) * sp
    Vp = np.pi * np.power(Dp_eq, 3) / 6
    Ap = Vp / Lp
    Gp =  (1-1e-6) * np.ones(Np) * G_max
    betap = op.teste.geometry.half_angle_isosceles(Gp)
    #Saving data
    pn['pore.equivalent_diameter'] = Dp_eq
    pn['pore.volume'] = Vp
    pn['pore.cross_sectional_area'] = Ap
    pn['pore.length'] = Lp
    pn['pore.shape_factor'] = Gp
    pn['pore.half_corner_angle'] = betap

    #Throat props
    pn.regenerate_models(propnames = 'throat.max_area')
    At = pn['throat.max_area'] * 0.25
    Dt_eq = np.power(4 * At / np.pi, 0.5)
    Lt = sp - np.sum(Dp_eq[conns],axis =  1) / 2
    Gt =  np.random.rand(Nt) * (G_max - G_min) + G_min
    betat = op.teste.geometry.half_angle_isosceles(Gt)
    #Saving data
    pn['throat.equivalent_diameter'] = Dt_eq
    pn['throat.cross_sectional_area'] = At
    pn['throat.length'] = Lt
    pn['throat.shape_factor'] = Gt
    pn['throat.half_corner_angle'] = betat
    pn['throat.conduit_lengths'] = np.column_stack(( Dp_eq[conns[:,0]]/2, Lt, Dp_eq[conns[:,1]]/2))
    return

set_cubic_props()

#ESTE PASO NO DEBRIA AFECTAR EN UN METODO DE FIJAR PROPIEDADES PARA POROSIDAD Y PERMEABILIDAD
#POR TANTO, IRIA DESPUES
#For the boundary conduits, copy some props of the internal throats
#That is in order to share the phase distribution
for t in range(Nt):
    if pn['throat.boundary'][t]:
        conns = pn['throat.conns'][t]
        mask = pn['pore.internal'][conns]
        if np.any(mask):
            pi = conns[mask]
            pb = conns[~mask]
            G = pn['pore.shape_factor'][pi]
            beta = pn['pore.half_corner_angle'][pi,:]
            #Boundary pore
            pn['pore.shape_factor'][pb] = G
            pn['pore.half_corner_angle'][pb,:] = beta
            #Boundary throat
            pn['throat.shape_factor'][t] = G
            pn['throat.half_corner_angle'][t,:] = beta

#Add inscribed diameter and perimeter
for item in elements:
    beta = pn[f'{item}.half_corner_angle']
    A = pn[f'{item}.cross_sectional_area']
    R = np.power(A / np.sum(1 / np.tan(beta), axis = 1), 0.5)
    pn[f'{item}.diameter'] = 2 * R
    pn[f'{item}.perimeter'] = 2 * R * np.sum(1 / np.tan(beta), axis = 1)

#Saving
ws.save_project(pn.project, filename=sampleName)

#Calculating porosity
vol_box = np.prod(shape*sp)
vol_pores = np.sum(pn['pore.volume'][ pn['pore.internal']])

porosity_final = (vol_pores)/vol_box
print('porosity', porosity_final)

#Calculating permeability

def Rate_calc(network, ph, inlet, outlet, conductance):
    St_p = op.algorithms.StokesFlow(network=network, phase=ph)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=inlet, values=1)
    St_p.set_value_BC(pores=outlet, values=0)
    St_p.run() #solver ='PardisoSpsolve'
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val

#Calculating hydraulic conductance
visc = 1e-3
phase = op.phase.Phase(network=pn)
phase['pore.viscosity'] = visc
g = _cf.conductance_triangle_OnePhase(phase, correction = True, check_boundary = True)
phase['throat.hydraulic_conductance'] = g

inlet_pores = pn['pore.inlet']
outlet_pores = pn['pore.outlet']

from time import time
ws.settings.default_solver = 'ScipySpsolve'
start_time = time()
Q = Rate_calc(pn, phase, inlet_pores, outlet_pores, 'throat.hydraulic_conductance')
end_time = time()

L_net = shape[0]*sp
A_net = shape[1]*shape[2]*sp**2
K = Q[0] * L_net * visc / A_net #Q[0] * L * mu / (A * Delta_P), Delta_P = 1
print(f'The value of K is: {K} m2')
print(f'The value of K is: {K/0.9869233e-12*1000:.2f} mD')

print(pn)
