import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import time
from parametersCustom import *
from scipy import optimize

#importing functions for other file
import sys
sys.path.insert(1, '/home/cesar/OpenPNM_files/mestrado/_funcs')
import _algorithm_class as _alg
import _conductance_funcs as _cf
import _invasion_funcs as _if

np.random.seed(seed) #Esto hara que siempre cree una misma red. Puede retirarse al final del medio

ws = op.Workspace()

pn = op.network.Cubic(shape=shape, spacing=sp, connectivity=26)

#Preoceso de eliminar elementos

#Eliminando el 30%
F_p_seed = op.models.geometry.pore_seed.random
pn.add_model(propname='pore.seed', model=F_p_seed, num_range=[0.01, 0.99])
inds = np.where(pn['pore.seed'] > 0.7)
op.topotools.trim(pn, pores=inds)

#Reducing z until the desired value
op.topotools.reduce_coordination(network=pn, z=z_mean)

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

#Counting elements
Nt = pn.Nt
Np = pn.Np

#Defining boundary and internal elements
pn['pore.boundary'] = pn['pore.inlet'] | pn['pore.outlet']
pn['pore.internal'] = ~pn['pore.boundary']
pn['throat.boundary'] = pn['throat.inlet'] | pn['throat.outlet']
pn['throat.internal'] = ~pn['throat.boundary']

#Defining geometry properties

G_max = 3**0.5/36

F_p_size = op.models.geometry.pore_size.largest_sphere
F_p_volume= op.models.geometry.pore_volume.sphere
F_prod = op.models.misc.product
F_frac = op.models.misc.fraction
F_neigh_p = op.models.misc.from_neighbor_pores
F_scaled = op.models.misc.scaled
F_gen_fun = op.models.misc.generic_function
F_ctc = op.models.geometry.conduit_lengths.cubes_and_cuboids
F_random = op.models.misc.random
F_beta = op.teste.geometry.half_angle_isosceles

pn.add_model(propname='pore.seed', model=F_p_seed, num_range=range_seed, regen_mode = 'constant')
pn.add_model(propname='pore.max_size', model=F_p_size, iters = 10)
pn.add_model(propname='pore.equivalent_diameter', model=F_prod, props = ['pore.max_size', 'pore.seed'])
pn.add_model(propname='pore.volume', model=F_p_volume, pore_diameter = 'pore.equivalent_diameter')
pn.add_model(propname='pore.cross_sectional_area', model=F_frac,
             numerator = 'pore.volume', denominator = 'pore.equivalent_diameter')
#pn.add_model(propname='pore.shape_factor', model=F_random, element = 'pore',
             #num_range=[0.001 * G_max, G_max])
pn['pore.shape_factor'] = 0.99 * np.ones(Np) * G_max
pn.add_model(propname = 'pore.half_corner_angle', model = F_gen_fun,
             prop = 'pore.shape_factor', func = calc_beta)

pn.add_model(propname='throat.max_area', model=F_neigh_p, mode = 'min', prop = 'pore.cross_sectional_area')
pn.add_model(propname='throat.cross_sectional_area', model=F_scaled, prop = 'throat.max_area', factor = 0.23)
pn.add_model(propname='throat.equivalent_diameter', model=F_gen_fun,
             prop = 'throat.cross_sectional_area', func = eq_diam_2D)
pn.add_model(propname = 'throat.conduit_lengths', model = F_ctc,
             pore_diameter = 'pore.equivalent_diameter')
pn.add_model(propname='throat.shape_factor', model=F_random, element = 'throat',
             num_range=[0.001 * G_max, G_max])
pn.add_model(propname = 'throat.half_corner_angle', model = F_gen_fun,
             prop = 'throat.shape_factor', func = calc_beta)

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

#i = 11
#item = 'pore'
#D = pn[f'{item}.equivalent_diameter'][i]
#print(D)
#V = np.pi * D ** 3 / 6
#print(V)
#print(pn[f'{item}.volume'][i])
#print(V/D)
#print(pn[f'{item}.cross_sectional_area'][i])
#raise Exception('')

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

L_net = op.topotools.get_domain_length(pn, inlets=pn['pore.inlet'], outlets=pn['pore.outlet'])
A_net = op.topotools.get_domain_area(pn, inlets=pn['pore.inlet'], outlets=pn['pore.outlet'])
K = Q[0] * L_net * visc / A_net #Q[0] * L * mu / (A * Delta_P), Delta_P = 1
print(f'The value of K is: {K/0.9869233e-12*1000:.2f} mD')


