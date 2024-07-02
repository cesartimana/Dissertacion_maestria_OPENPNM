#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:39:00 2023

@author: cesar
"""

import numpy as np
import openpnm as op
op.visualization.set_mpl_style()
np.random.seed(5)
pn = op.network.Cubic(shape=[5, 2, 1], spacing=5e-5)
pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
pn.regenerate_models()

"""
print(pn['throat.conns'])
print(pn)
print(pn['throat.spacing']) #Menor de los valores de pore.diameter de los vecinos
print(pn['throat.max_size'])
print(pn['throat.diameter'])
print(pn['throat.length']) #Esto se calcula teniendo en cuenta que Lporo² = Dporo²-Dgarganta² (pitagoras)
print(pn['pore.diameter'])
"""
print(pn['throat.cross_sectional_area'])
print(pn['throat.length'])
print(pn['throat.hydraulic_size_factors'])


water = op.phase.Phase(network=pn)

water.add_model(propname='pore.viscosity',
                model=op.models.phase.viscosity.water_correlation)
print(water)


R = pn['throat.diameter']/2
L = pn['throat.length']
mu = water['throat.viscosity']  # See ProTip below
water['throat.hydraulic_conductance'] = np.pi*R**4/(8*mu*L)
print(water['throat.hydraulic_conductance'])

sf = op.algorithms.StokesFlow(network=pn, phase=water)
print(sf)

sf.set_value_BC(pores=pn.pores('left'), values=100_000)#pressure
sf.set_rate_BC(pores=pn.pores('right'), rates=1e-10)#flow_rate
#sf.set_value_BC(pores=pn.pores('right'), values=0)
print(sf)

soln = sf.run()

print(sf)

print(sf['pore.pressure'])
print(sf['pore.all'])


'''
F_h = water['throat.hydraulic_size_factors']
water['throat.hydraulic_conductance'] = (mu * (1/F_h).sum(axis=1))**(-1)

water['throat.hydraulic_conductance']

sf = op.algorithms.StokesFlow(network=pn, phase=water)
sf.set_value_BC(pores=pn.pores('left'), values=100_000)
sf.set_rate_BC(pores=pn.pores('right'), rates=1e-10)
soln = sf.run()
sf['pore.pressure'][pn.pores('right')]
'''
