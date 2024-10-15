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
ws.settings.default_solver = 'ScipySpsolve'
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

#Calculation conduit length
L = pn['throat.conduit_lengths']

#Volume from pores
#Assuming all volume is on the pores
V_t = sum(pn['pore.volume'][pn['pore.internal']])

with open(f'drainage_process_{theta_r_sexag}_{p_kPa}kPa.pkl', 'rb') as fpdr:
    results_pdr = pickle.load(fpdr)
last_str_s = list(results_pdr)[-1]
#invasion_dict = results_pdr[last_str_s]['invasion_info']['throat.center'])
t_center = results_pdr[last_str_s]['invasion_info']['throat.center']
#elements = ['pore', 'throat']
#locations = ['center', 'corner']

Nt_int = np.sum(pn['throat.internal'])
Nt_drn = np.sum(t_center[pn['throat.internal']])
print(f'total internal throats: {Nt_int}')
print(f' internal throats with water after drainage: {Nt_drn} , {round(Nt_drn / Nt_int * 100,2)} %')

with open(f'imbibition_process_{theta_r_sexag}_{theta_a_sexag}_{p_kPa}kPa.pkl', 'rb') as fimb:
    results_imb = pickle.load(fimb)
last_str_imb = list(results_imb)[-1]
tinv_imb = results_pdr[last_str_imb]['invasion_info']['throat.center']
Nt_imb = np.sum(tinv_imb[pn['throat.internal']])
print(f' internal throats with water after imbibition: {Nt_imb}, {round(Nt_imb / Nt_int * 100,2)} %')
