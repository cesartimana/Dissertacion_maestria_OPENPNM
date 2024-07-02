#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:20:29 2023

@author: cesar
"""

import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
#Pore network creation
np.random.seed(5)
pn = op.network.Cubic(shape=[20, 20, 1], spacing=1e-4)
pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
pn.regenerate_models()
#Phase creation
air = op.phase.Air(network=pn)

air['pore.contact_angle'] = 180 #in sexagesimal
air['pore.surface_tension'] = 0.072

air.add_model_collection(op.models.collections.phase.air)
air.add_model_collection(op.models.collections.physics.basic)
air.regenerate_models()

drn = op.algorithms.Drainage(network=pn, phase=air)
drn.set_inlet_BC(pores=pn.pores('left'))
drn.set_outlet_BC(pores=pn.pores('right'))
dP=np.linspace(5000, 25000, num=11)
drn.run()



ip = drn

water = op.phase.Water(network=pn,name='water')
water.add_model_collection(op.models.collections.phase.water)
water.add_model_collection(op.models.collections.physics.basic)
water.regenerate_models()

def sat_occ_update(network, nwp, wp, ip, i):
    r"""
    Calculates the saturation of each phase using the invasion
    sequence from the invasion percolation algorithm.
    Parameters
    ----------
    network: network
    nwp : phase
    non-wetting phase
    wp : phase
    wetting phase
    ip : IP
    invasion percolation (ran before calling this function)
    i: int
    The invasion_sequence limit for masking pores/throats that
    have already been invaded within this limit range. The
    saturation (sat) is found by adding the volume of pores and thoats
    that meet this sequence limit divided by the bulk volume.
    """
    pore_mask = (ip['pore.invasion_sequence'] < i) & (ip['pore.invasion_sequence'] >= 0)
    throat_mask = (ip['throat.invasion_sequence'] < i) & (ip['throat.invasion_sequence'] >= 0)
    sat_p = np.sum(network['pore.volume'][pore_mask])
    sat_t = np.sum(network['throat.volume'][throat_mask])
    sat1 = sat_p + sat_t
    bulk = network['pore.volume'].sum() + network['throat.volume'].sum()
    sat = sat1/bulk
    nwp['pore.occupancy'] = pore_mask
    nwp['throat.occupancy'] = throat_mask
    wp['throat.occupancy'] = 1-throat_mask
    wp['pore.occupancy'] = 1-pore_mask
    return sat

model_mp_cond = op.models.physics.multiphase.conduit_conductance
air.add_model(model=model_mp_cond, propname='throat.conduit_hydraulic_conductance',
throat_conductance='throat.hydraulic_conductance', mode='medium', regen_mode='deferred')
water.add_model(model=model_mp_cond, propname='throat.conduit_hydraulic_conductance',
throat_conductance='throat.hydraulic_conductance', mode='medium', regen_mode='deferred')

def Rate_calc(network, phase, inlet, outlet, conductance):
    phase.regenerate_models()
    St_p = op.algorithms.StokesFlow(network=network, phase=phase)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=inlet, values=1)
    St_p.set_value_BC(pores=outlet, values=0)
    St_p.run()
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val

Snwp_num=10
flow_in = pn.pores('left')
flow_out = pn.pores('right')
max_seq = np.max([np.max(ip['pore.invasion_sequence']),
np.max(ip['throat.invasion_sequence'])])
start = max_seq//Snwp_num
stop = max_seq
step = max_seq//Snwp_num
Snwparr = []
relperm_nwp = []
relperm_wp = []
for i in range(start, stop, step):
    air.regenerate_models();
    water.regenerate_models();
    sat = sat_occ_update(network=pn, nwp=air, wp=water, ip=ip, i=i)
    Snwparr.append(sat)
    Rate_abs_nwp = Rate_calc(pn, air, flow_in, flow_out, conductance = 'throat.hydraulic_conductance')
    Rate_abs_wp = Rate_calc(pn, water, flow_in, flow_out, conductance = 'throat.hydraulic_conductance')
    Rate_enwp = Rate_calc(pn, air, flow_in, flow_out, conductance = 'throat.conduit_hydraulic_conductance')
    Rate_ewp = Rate_calc(pn, water, flow_in, flow_out, conductance = 'throat.conduit_hydraulic_conductance')
    relperm_nwp.append(Rate_enwp/Rate_abs_nwp)
    relperm_wp.append(Rate_ewp/Rate_abs_wp)

plt.figure(figsize=[8,8])
plt.plot(Snwparr, relperm_nwp, '*-', label='Kr_nwp')
plt.plot(Snwparr, relperm_wp, 'o-', label='Kr_wp')
plt.xlabel('Snwp')
plt.ylabel('Kr')
plt.title('Relative Permeability in x direction')
plt.legend()
