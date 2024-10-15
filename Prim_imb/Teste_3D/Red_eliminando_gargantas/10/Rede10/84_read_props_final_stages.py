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

#sp de simple spaning cluster
results_PDR = np.load(f'results_K_Pc_PDR_{theta_r_sexag}_{p_kPa}kPa.npy')
s_drn = 0
while results_PDR[s_drn, 2] == 0:
    s_drn+=1
s_drn+=1 #Para evaluar el estado en el que el simple spaning cluster se crea

results_IMB = np.load(f'results_K_Pc_imb_{theta_r_sexag}_{theta_a_sexag}_{p_kPa}kPa.npy')
s_imb = 0
while results_IMB[s_imb, 2] != 0:
    s_imb+=1

with open(f'drainage_process_{theta_r_sexag}_{p_kPa}kPa.pkl', 'rb') as fpdr:
    results_pdr = pickle.load(fpdr)

last_str_s = list(results_pdr)[-1]
sp_str_drn = 'status ' + str(s_drn)
t_center = results_pdr[last_str_s]['invasion_info']['throat.center']
t_ce_sp = results_pdr[sp_str_drn]['invasion_info']['throat.center']

Nt_int = np.sum(pn['throat.internal'])
Nt_drn = np.sum(t_center[pn['throat.internal']])
Nt_drn_sp = np.sum(t_ce_sp[pn['throat.internal']])
print(f'total internal throats: {Nt_int}')
print(f' internal throats with water after drainage ({last_str_s}): {Nt_drn} , {round(Nt_drn / Nt_int * 100,2)} %')
print(f' internal throats with water when the spanning cluster (status {s_drn}) is created: {Nt_drn_sp} , {round(Nt_drn_sp / Nt_int * 100,2)} %')

with open(f'imbibition_process_{theta_r_sexag}_{theta_a_sexag}_{p_kPa}kPa.pkl', 'rb') as fimb:
    results_imb = pickle.load(fimb)

last_str_imb = list(results_imb)[-1]
sp_str_imb = 'status ' + str(s_imb)
tinv_imb = results_imb[last_str_imb]['invasion_info']['throat.center']
tinv_sp = results_imb[sp_str_imb]['invasion_info']['throat.center']

Nt_imb = np.sum(tinv_imb[pn['throat.internal']])
Nt_imb_sp = np.sum(tinv_sp[pn['throat.internal']])
print(f' internal throats with water after imbibition ({last_str_imb}): {Nt_imb}, {round(Nt_imb / Nt_int * 100,2)} %')
print(f' internal throats with water when the spanning cluster (status {s_imb}) is created: {Nt_imb_sp} , {round(Nt_imb_sp / Nt_int * 100,2)} %')

