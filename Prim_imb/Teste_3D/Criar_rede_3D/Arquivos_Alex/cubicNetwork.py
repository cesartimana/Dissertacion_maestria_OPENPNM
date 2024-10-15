import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
import scipy.odr as odr
import scipy.stats as stats
import time
import os
from datetime import datetime
from parametersCubic import *
from scipy import optimize
#np.random.seed(seed)

ws = op.Workspace()
#ws.settings["loglevel"] = 50

plt.rcParams.update({'font.size': 20})

A_in = shape[1]*shape[2]*sp**2
vol_t = np.prod(shape*sp)
Np_th = 0.7
Np = np.prod(shape)*Np_th
N_tub = shape[1]*shape[2]*Np_th
D_t = np.float_power(permeability*A_in*128/N_tub/z_mean/100*np.pi, 0.25)
print('guess for throat diameter =', D_t)
a_por = 1/6.0
b_por = 0
#c_por = - D_t**2*z_mean/4
c_por = D_t**2*z_mean/4
d_por = -vol_t*porosity/np.pi/Np
#d_por = D_t**2*z_mean*sp/4-vol_t*porosity/np.pi/Np
roots = np.roots([a_por, b_por, c_por, d_por])
print(roots)
b = np.where(roots>0)
D_p = float(np.max(roots[b]))
print('guess for pore diameter =', D_p)
D_p_dev = np.min([(sp-D_p)/D_p, 0.1])
print('D p loc', D_p_dev)

pn = op.network.Cubic(shape=shape, spacing=sp, connectivity=6)

#   Adding a inlet and outlet super-pore
y_mean = pn.coords[:,1][pn.pores('left')].mean()
z_mean = pn.coords[:,2][pn.pores('left')].mean()
x0 = pn.coords[:,0][pn.pores('left')].mean()-sp
xL = pn.coords[:,0][pn.pores('right')].mean()+sp

op.topotools.extend(network=pn, coords=[[x0, y_mean, z_mean]], labels='inlet')
temp_inlet = np.squeeze(pn.pores('inlet'))
op.topotools.extend(network=pn, conns=[[temp_inlet, i] for i in pn.pores('left')])

op.topotools.extend(network=pn, coords=[[xL, y_mean, z_mean]], labels='outlet')
temp_outlet = np.squeeze(pn.pores('outlet'))
op.topotools.extend(network=pn, conns=[[temp_outlet, i] for i in pn.pores('right')])

#Cubic
inlet=pn.pores('inlet')
outlet=pn.pores('outlet')
internal=pn.pores('internal')
inlet_t = pn.find_neighbor_throats(pores=inlet)
outlet_t = pn.find_neighbor_throats(pores=outlet)

geo = op.geometry.GenericGeometry(network=pn, pores=pn.Ps, throats=pn.Ts)

'''geo.add_model(propname='pore.max_size',
              model=op.models.geometry.pore_size.largest_sphere,
              iters=10)

geo.add_model(propname='pore.diameter',
              model=op.models.misc.product,
              prop1='pore.max_size',
              prop2='pore.seed')
'''
#pn['pore.max_size'] = spacing*np.ones(pn.Np)
#pn['pore.diameter'] = pn['pore.max_size']

t_seeds=op.models.geometry.pore_seed.random
geo.add_model(propname='pore.seed', model=t_seeds, num_range=[0.01, 0.99])

t_seeds=op.models.geometry.throat_seed.random
geo.add_model(propname='throat.seed', model=t_seeds, num_range=[0.01, 0.99])

normal_p = op.models.geometry.pore_size.normal
geo.add_model(propname='pore.diameter', model=normal_p,
        loc=D_p, scale=D_p*D_p_dev)

normal_t = op.models.geometry.throat_size.normal
geo.add_model(propname='throat.diameter', model=normal_t,
        loc=D_t, scale=D_t*0.2, seeds='throat.seed')

geo.add_model(propname='pore.volume',
              model=op.models.geometry.pore_volume.sphere,
              pore_diameter='pore.diameter')
'''
geo.add_model(propname='throat.max_size',
               model=op.models.misc.from_neighbor_pores,
               mode='min',
               prop='pore.diameter')
geo.add_model(propname='throat.diameter',
              model=op.models.misc.scaled,
              factor=0.5,
              prop='throat.max_size')
'''

geo.add_model(propname='pore.area',
              model=op.models.geometry.pore_cross_sectional_area.sphere,
              pore_diameter='pore.diameter')


geo.add_model(propname='throat.endpoints',
              model=op.models.geometry.throat_endpoints.spherical_pores,
              pore_diameter='pore.diameter',
              throat_diameter='throat.diameter')

geo.add_model(propname='throat.length',
              model=op.models.geometry.throat_length.piecewise,
              throat_endpoints='throat.endpoints')

geo.add_model(propname='throat.surface_area',
              model=op.models.geometry.throat_surface_area.cylinder,
              throat_diameter='throat.diameter',
              throat_length='throat.length')

geo.add_model(propname='throat.volume',
              model=op.models.geometry.throat_volume.cylinder,
              throat_diameter='throat.diameter',
              throat_length='throat.length')

geo.add_model(propname='throat.area',
              model=op.models.geometry.throat_cross_sectional_area.cylinder,
              throat_diameter='throat.diameter')

geo.add_model(propname='throat.conduit_lengths',
              model=op.models.geometry.throat_length.conduit_lengths,
              throat_endpoints='throat.endpoints',
              throat_length='throat.length')
geo.show_hist()
vol_box = np.prod(shape*sp)
vol_pores = np.sum(pn['pore.volume'])
vol_throats = np.sum(pn['throat.volume'])

#print('max por size (min, max and mean) = ', geo['pore.max_size'].min(), geo['pore.max_size'].max(), geo['pore.max_size'].mean())
plt.show()
porosity_final = (vol_pores+vol_throats)/vol_box
print('porosity', porosity_final)
print('porosity pores', vol_pores/vol_box)
print('porosity throats', vol_throats/vol_box)

water = op.phases.Water(network=pn)
phys = op.physics.GenericPhysics(network=pn, phase=water, geometry=geo)
h_model = op.models.physics.hydraulic_conductance.hagen_poiseuille
phys.add_model(propname='throat.hydraulic_conductance', model=h_model)

h_mean = phys['throat.hydraulic_conductance'].mean()
phys['throat.hydraulic_conductance'][inlet_t] = 1e5*h_mean
phys['throat.hydraulic_conductance'][outlet_t] = 1e5*h_mean

sf = op.algorithms.StokesFlow(network=pn, phase=water)
sf.set_value_BC(pores=inlet, values=1)
sf.set_value_BC(pores=outlet, values=0.0)
sf.run()

Lcol = xL-x0-sp*2
perm = np.squeeze(sf.calc_effective_permeability(inlets=inlet, outlets=outlet,
        domain_length= Lcol, domain_area=A_in))
print('Permeability m2 =', perm)

print('\nErrors')
print('porosity %', (porosity_final-porosity)/porosity*100)
print('permeability %', (perm-permeability)/permeability*100)

def min_permeability_err(x):
    print('\nD_t guess = ',x)
    geo.models['throat.diameter']['loc'] = np.abs(x)
    geo.models['throat.diameter']['scale'] = np.abs(x)*0.20
    geo.regenerate_models(exclude=['pore.seed', 'throat.seed'])
    phys.regenerate_models(propnames='throat.hydraulic_conductance')
    h_mean = phys['throat.hydraulic_conductance'].mean()
    phys['throat.hydraulic_conductance'][inlet_t] = 1e5*h_mean
    phys['throat.hydraulic_conductance'][outlet_t] = 1e5*h_mean
    sf.reset()
    sf.set_value_BC(pores=inlet, values=10)
    sf.set_value_BC(pores=outlet, values=0.0)
    sf.run()
    perm = np.squeeze(sf.calc_effective_permeability(inlets=inlet, outlets=outlet,
        domain_length= Lcol, domain_area=A_in))
    error = (perm-permeability)/permeability*100
    print('error permeability %', error)
    print('permeability m2 = ', perm)
    return error

def min_porosity_err(y):
    print('D_p guess = ',y)
    geo.models['pore.diameter']['loc'] = np.abs(y)
    D_p_dev = np.max([0.01, np.min([(sp-y)/y, 0.1])])
    print('D p dev', D_p_dev)
    geo.models['pore.diameter']['scale'] = np.abs(y)*D_p_dev
    geo.regenerate_models(exclude=['pore.seed', 'throat.seed'])    
    vol_pores = np.sum(pn['pore.volume'])
    vol_throats = np.sum(pn['throat.volume'])
    porosity_final = (vol_pores+vol_throats)/vol_box
    error = (porosity_final-porosity)/porosity*100
    print('error porosity %', error)
    print('porosity = ', porosity_final)
    return error


for i in range(3):
    print ('\n\n fine tunning - permeability\n')
    timeZero = time.time()
    sol2 = optimize.root_scalar(min_permeability_err, x0=D_t, rtol=1e-4, bracket=[D_t*0.01, D_t*100.0], maxiter=25)
    print('elapsed time for tunning permeability (s) = ', time.time()-timeZero)

    print ('\n\n fine tunning - porosity\n')
    timeZero = time.time()
    sol = optimize.root_scalar(min_porosity_err, x0=D_p, rtol=1e-4, bracket=[D_p*0.00000001, sp*0.99], maxiter=20)
    print('elapsed time for tunning porosity (s) = ', time.time()-timeZero)

    D_p = sol.root
    D_t = sol2.root

print('\nRecalculating\n')
min_permeability_err(D_t)
min_porosity_err(D_p)

sf.set_value_BC(pores=inlet, values=10)
sf.set_value_BC(pores=outlet, values=0.0)
sf.run()
perm = np.squeeze(sf.calc_effective_permeability(inlets=inlet, outlets=outlet,
domain_length= Lcol, domain_area=A_in))

print('\n FINAL \n')
print('perm', perm)
vol_pores = np.sum(pn['pore.volume'])
vol_throats = np.sum(pn['throat.volume'])
porosity_final = (vol_pores+vol_throats)/vol_box
print('porosity', porosity_final)
print('porosity pores', vol_pores/vol_box)
print('porosity throats', vol_throats/vol_box)


pn.project.save_project(filename=sampleName)
#op.io.VTK.save(network=pn, filename= sampleName)
