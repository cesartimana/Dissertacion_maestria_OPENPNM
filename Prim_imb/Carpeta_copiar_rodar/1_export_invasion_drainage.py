# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar

En este archivo  se calculara todo lo referente a la invasion por Drenaje primario
Se investiga como calcula la presion capilar de entrada  de las gargantas y el proceso de invasion
El algoritmo toma commo referencia el codigo de Invasion_Percolation
El fluido a usar sera el no mojante. El angulo de contacto estará aqui.
El angulo de contacto puede sr un objeto de tamaño Nx1 o Nx2. Siendo N el numero de poros o gargantas
Si es Nx2, la primera fila tiene al menor de los valores y es el receiding contact angle

La presión capilar sera guardada en el objeto de simulacion de drenaje primario porque puede cambiar con el caso.




"""

#Preambulo

import openpnm as op
import porespy as ps
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import copy
from Properties import *
#importing functions for other file
import sys
sys.path.insert(1, '/home/cesar/OpenPNM_files/mestrado/_funcs')
import _drainage_class as _drain
import _conductance_funcs as _cf
import _invasion_funcs as _if
#end
np.random.seed(13)

#Reading pnm arquive/ network
ws = op.Workspace()
proj = ws.load_project(filename=testName)

pn = proj.network

#SOLO PARA RED CREADA, ACTUALIZANDO NUMERO DE GARGANTAS
Np = pn.Np
Nt = pn.Nt

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
water['pore.contact_angle'] = theta_r
water['throat.contact_angle'] = theta_r



#Simulating primary Drainage
pdr = _drain.Primary_Drainage(network=pn, phase=water)
pdr.set_inlet_BC(pores=pn['pore.inlet'])
pdr.set_outlet_BC(pores=pn['pore.outlet'])

#pdr._run_setup(ignore_boundary = True)
#print(np.min(pdr['throat.entry_pressure']))
#print(np.max(pdr['throat.entry_pressure']))
#print((np.sort(pdr['throat.entry_pressure']))[int(Nt *0.2)])
#print((np.sort(pdr['throat.entry_pressure']))[int(Nt *0.7)])
#print((np.sort(pdr['throat.entry_pressure']))[int(Nt *0.8)])
#print((np.sort(pdr['throat.entry_pressure']))[int(Nt *0.9)])
#raise Exception('')

results_pdr = pdr.run(ignore_boundary = True, pc_max = pmax_drn)

#last_s_str = list(results_pdr)[-2]
#print(last_s_str)
#drainage_info = results_pdr[last_s_str]
#print(drainage_info['invasion pressure'])
#print(results_pdr['status 100']['invasion pressure'])
#print(results_pdr['status 120']['invasion pressure'])
#print(results_pdr['status 700']['invasion pressure'])
#print(results_pdr['status 800']['invasion pressure'])

#raise Exception('')

with open(f'drainage_process_{theta_r_sexag}_{p_kPa}kPa.pkl', 'wb') as fp:
        pickle.dump(results_pdr, fp)
