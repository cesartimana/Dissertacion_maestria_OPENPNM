import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
from time import time
from par_Spec_Custom import *
from scipy import optimize

#importing functions for other file
import sys
sys.path.insert(1, '/home/cesar/OpenPNM_files/mestrado/_funcs')
import _algorithm_class as _alg
import _conductance_funcs as _cf
import _invasion_funcs as _if

np.random.seed(seed) #Esto hara que siempre cree una misma red. Puede retirarse al final del medio

def Rate_calc(network, ph, inlet, outlet, conductance):
    St_p = op.algorithms.StokesFlow(network=network, phase=ph)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=inlet, values=1)
    St_p.set_value_BC(pores=outlet, values=0)
    St_p.run() #solver ='PardisoSpsolve'
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val

ws = op.Workspace()
ws.settings.default_solver = 'ScipySpsolve'
pn = op.network.Cubic(shape=shape, spacing=sp, connectivity=26)

#Eliminando el 30%
F_p_seed = op.models.geometry.pore_seed.random
pn.add_model(propname='pore.seed', model=F_p_seed, num_range=[0.01, 0.99])
inds = np.where(pn['pore.seed'] > 0.7)
op.topotools.trim(pn, pores=inds)

#Reducing z until the desired value
Ts = op.topotools.reduce_coordination(network=pn, z=z_mean)
op.topotools.trim(network=pn, throats=Ts)

#Eliminating non connected elements
h = op.utils.check_network_health(pn)
op.topotools.trim(network=pn, pores=h['disconnected_pores'])
#END------------------

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

#Iterations

#TUNNING POROSITY

def calc_porosity(min_seed = None, max_seed = None, Dp_eq = None):
    if Dp_eq is None:
        Dp_eq =  (np.random.rand(Np) * (max_seed - min_seed) + min_seed) * sp
    Vp = np.pi * np.power(Dp_eq, 3) / 6
    vol_pores = np.sum(Vp[ pn['pore.internal']])
    porosity_final = (vol_pores)/vol_box
    return porosity_final, Dp_eq

#Calculating starting points
vol_box = np.prod(shape*sp)
Np_int = np.sum(pn['pore.internal'])
max_porosity = Np_int  * np.pi * (sp * range_Dp[1]) ** 3 / 6 / vol_box
print(f'max porosity : {max_porosity}')
mean_por = Np_int  * np.pi * (sp * (range_Dp[0] + range_Dp[1])/2) ** 3 / 6 / vol_box
print(f'mean porosity : {mean_por}')
min_por = Np_int  * np.pi * (sp * range_Dp[0]) ** 3 / 6 / vol_box
print(f'min porosity : {min_por}')

if porosity > max_porosity:
    raise Exception('Porosity value required can not be reached. Modify Dp max')



porosity_final, D_pores = calc_porosity(min_seed = range_Dp[0], max_seed = range_Dp[1])
print(np.min(D_pores))
print(np.max(D_pores))
#porosity_final, D_pores = calc_porosity(min_seed = 0.2509765625, max_seed = 0.7509765625)
print(f'actual porosity : {porosity_final}')
#raise Exception('Poros')
#Tunning:
iter_por = 20
tol = 1e-3 #Es mejor para llegar a porosidad
error = np.abs(porosity_final - porosity) / porosity
min_value= 0
if porosity_final < porosity:
    mode = 'raise_D'
    max_value = 1 - range_Dp[1]
else:
    mode = 'reduce_D'
    max_value = range_Dp[0]

i = 0
while error > tol:
    range_tune = (min_value + max_value)/2
    if mode == 'reduce_D':
        range_tune = -1 * range_tune
    new_Dp = D_pores + range_tune * sp
    porosity_final, _ = calc_porosity(Dp_eq = new_Dp)
    error = np.abs(porosity_final - porosity) / porosity
    #print('----')
    #print(f'Range: [{range_Dp[0] + range_tune} , {range_Dp[1] + range_tune}]')
    #print(f'Final porosity: {porosity_final}')
    if porosity_final > porosity:
        max_value = range_tune
    else:
        min_value = range_tune
    i += 1
    if i > iter_por:
        raise Exception('Maximum number of iterations reached')
Dp_eq = new_Dp
print(f'Range D: [{sp * (range_Dp[0] + range_tune)} , {sp * (range_Dp[1] + range_tune)}]')
print(f'Range seed: [{(range_Dp[0] + range_tune)} , {(range_Dp[1] + range_tune)}]')
print(f'Final porosity: {porosity_final}')

#--------------------

#Calculating other pore props
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

#Calculating constant throat props
pn.regenerate_models(propnames = 'throat.max_area')
Lt = sp - np.sum(Dp_eq[conns],axis =  1) / 2
Gt =  np.random.rand(Nt) * (G_max - G_min) + G_min
betat = op.teste.geometry.half_angle_isosceles(Gt)
#Saving data
pn['throat.length'] = Lt
pn['throat.shape_factor'] = Gt
pn['throat.half_corner_angle'] = betat
pn['throat.conduit_lengths'] = np.column_stack(( Dp_eq[conns[:,0]]/2, Lt, Dp_eq[conns[:,1]]/2))
#--------------------

#TUNNING PERMEABILITY

#SEt calc permeability functions

#Calculating hydraulic conductance
visc = 1e-3
phase = op.phase.Phase(network=pn)
phase['pore.viscosity'] = visc
inlet_pores = pn['pore.inlet']
outlet_pores = pn['pore.outlet']
L_net = shape[0]*sp
A_net = shape[1]*shape[2]*sp**2

def calc_permeability():
    g = _cf.conductance_triangle_OnePhase(phase, correction = True, check_boundary = True)
    phase['throat.hydraulic_conductance'] = g
    Q = Rate_calc(pn, phase, inlet_pores, outlet_pores, 'throat.hydraulic_conductance')
    K = Q[0] * L_net * visc / A_net
    #print(f'The value of K is: {K/0.9869233e-12*1000:.2f} mD')
    pn.project.__delitem__(-1)
    return K

#Calculating permeability for the max value
min_value= 0
max_value = 0.5

def calc_throat_area(factor):
    At = pn['throat.max_area'] * factor
    Dt_eq = np.power(4 * At / np.pi, 0.5)
    pn['throat.equivalent_diameter'] = Dt_eq
    pn['throat.cross_sectional_area'] = At
    return

calc_throat_area(factor = max_value)
perm_max = calc_permeability()

if permeability > perm_max:
    raise Exception('Permeability value required can not be reached. Modify permeability value')

#Tunning
iter_perm = 20
tol = 1e-4
error = np.abs(perm_max - permeability) / permeability
i = 0

while error > tol:
    range_tune = (min_value + max_value)/2
    calc_throat_area(factor = range_tune)
    perm_final = calc_permeability()
    error = np.abs(perm_final - permeability) / permeability
    if perm_final > permeability:
        max_value = range_tune
    else:
        min_value = range_tune
    i += 1
    if i > iter_por:
        raise Exception('Maximum number of iterations reached')
print(f'Factor for throat area: {range_tune}')
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
pn.project.__delitem__(-1)
ws.save_project(pn.project, filename=sampleName)
