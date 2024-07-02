# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar

"""

import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import math as m

#importing functions for other file
import sys
sys.path.insert(1, '/home/cesar/OpenPNM_files/_funcs')
import _conductance_funcs as _cf
import _invasion_funcs as _if
#end

from tqdm import tqdm
import pandas as pd


#Flowrate function
def Rate_calc(network, ph, inlet, outlet, conductance):
    St_p = op.algorithms.StokesFlow(network=network, phase=ph)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=inlet, values=1) #P_inlet = 1
    St_p.set_value_BC(pores=outlet, values=0) #P_outlet = 1
    St_p.run()
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val

Networks = np.arange(10)
Geometric = np.arange(10)

Net_array = []
Geo_array = []
Kx_cyl = []
Ky_cyl = []
Kz_cyl = []
Kx_prism = []
Ky_prism = []
Kz_prism = []
G_20 = []
G_50 = []
G_80 = []

f= op.models.physics.hydraulic_conductance.generic_hydraulic
visc = 1e-3 #Pa/s
#Creating workspace
ws = op.Workspace()

#Reading netwrork data
for net_label in Networks:
    for geo_label in Geometric:
        testName = 'Net_'+ str(int(net_label)) + '_geo_'+ str(int(geo_label)) +'.pnm'
        proj = ws.load_project(filename=testName)
        pn = proj.network
        Np = pn.Np
        Nt = pn.Nt

        #Some previous calculations
        A_t = pn['throat.cross_sectional_area']
        G_t = pn['throat.shape_factor']
        L_t = pn['throat.length']
        beta_dif = np.pi/2 - 2 * pn['throat.half_corner_angle'][:,1]
        pol = np.array([-0.17997611,  0.57966346, -0.46275726,  1.10633925])
        factor = np.polyval(pol, beta_dif)
        Fh_t = 0.6 * factor * A_t ** 2 * G_t / (L_t)

        #Gsort
        G_sort = np.sort(G_t)
        G_20.append( G_sort [ int( round(0.2 * Nt)) ] )
        G_50.append( G_sort [ int( round(0.5 * Nt)) ] )
        G_80.append( G_sort [ int( round(0.8 * Nt)) ] )

        #Creating phase model
        phase = op.phase.Phase(network=pn)
        phase['pore.viscosity'] = visc
        phase.add_model(propname = 'throat.hydraulic_conductance',
                        model = f)

        #Defining boundary conditions
        for axis in ['x', 'y', 'z']:
            pn.regenerate_models(propnames = 'throat.hydraulic_size_factors')
            phase.regenerate_models()
            inlet_pores = pn['pore.'+axis+'min']
            outlet_pores = pn['pore.' + axis + 'max']

            #Network dimensions
            D = op.topotools.get_domain_length(pn, inlets=inlet_pores, outlets=outlet_pores)
            A = op.topotools.get_domain_area(pn, inlets=inlet_pores, outlets=outlet_pores)

            #Defining internal pores, throats
            pn['pore.boundary'] = inlet_pores | outlet_pores #created line
            pn['pore.internal'] = True
            pn['pore.internal'][pn['pore.boundary']] = False
            boundary_p = pn.pores('boundary')
            boundary_t = pn.find_neighbor_throats(pores=boundary_p)
            pn['throat.internal'] = True
            pn['throat.internal'][boundary_t] = False
            pn['throat.boundary'] = True
            pn['throat.boundary'][pn['throat.internal']] = False

            #Absolute permeability, cylindrical throat
            Q_cyl = Rate_calc(pn, phase, inlet_pores, outlet_pores, 'throat.hydraulic_conductance')
            K_cyl = Q_cyl[0] * D * visc / A /0.98e-12*1000 #Q[0] * L * mu / (A * Delta_P), Delta_P = 1
            #print(f'The value of K is: {K_cyl/0.98e-12*1000:.4f} mD')


            #Assuming triangular prism for throats
            pn['throat.hydraulic_size_factors'][:, 1] = Fh_t
            phase.regenerate_models()
            #Absolute permeability, cylindrical throat
            Q_pr = Rate_calc(pn, phase, inlet_pores, outlet_pores, 'throat.hydraulic_conductance')
            K_pr = Q_pr[0] * D * visc / A /0.98e-12*1000#Q[0] * L * mu / (A * Delta_P), Delta_P = 1
            if axis == 'x':
                Kx_cyl.append(K_cyl)
                Kx_prism.append(K_pr)
            elif axis =='y':
                Ky_cyl.append(K_cyl)
                Ky_prism.append(K_pr)
            else:
                Kz_cyl.append(K_cyl)
                Kz_prism.append(K_pr)
            #print(f'The value of K is: {K_pr/0.98e-12*1000:.4f} mD')
        Net_array.append(net_label)
        Geo_array.append(geo_label)
data = {'Network': Net_array,
        'Geometric props': Geo_array,
        'Kx cylinder': Kx_cyl,
        'Ky cylinder': Ky_cyl,
        'Kz cylinder': Kz_cyl,
        'Kx prism': Kx_prism,
        'Ky prism': Ky_prism,
        'Kz prism': Kz_prism,
        'G20' : G_20,
        'G50' : G_50,
        'G80' : G_80
        }
df = pd.DataFrame(data = data)
df.to_csv("Summary.csv")
