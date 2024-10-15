 

#Preambulo

import openpnm as op
import porespy as ps
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import copy
from tqdm import tqdm
from Properties import *
#importing functions for other file
import sys
sys.path.insert(1, '/home/cesar/OpenPNM_files/mestrado/_funcs')
import _algorithm_class as _alg
import _conductance_funcs as _cf
import _invasion_funcs as _if
#end

#Reading pnm arquive/ network
ws = op.Workspace()
ws.settings.default_solver = 'ScipySpsolve'
proj = ws.load_project(filename=testName)

pn = proj.network
Np = pn.Np
Nt = pn.Nt

#Read imbibition results
#Obtener datos de la ultima invasion registrada
with open(f'imbibition_process_{theta_r_sexag}_{theta_a_sexag}_{p_kPa}kPa.pkl', 'rb') as fp:
    info_imbibition = pickle.load(fp)

#Picking last info and transform in int
last_s_inv = int(list(info_imbibition)[-1][7:])
print(last_s_inv)
status_str = 'status ' + str(last_s_inv)

#How to extract info
#print(info_imbibition[status_str].keys())
#print(info_imbibition[status_str]['invasion_info'].keys())

#Sabiendo cuantos layers se han roto y de cuantos elementos
copy_invasion_info = copy.deepcopy(info_imbibition[status_str]['invasion_info'])
copy_cluster_info = copy.deepcopy(info_imbibition[status_str]['cluster_info'])
p_inv = info_imbibition[status_str]['invasion pressure']
print('-----------')
print(Np)
print(np.sum(copy_invasion_info['pore.broken_layer']))
print(np.sum(np.any(copy_invasion_info['pore.broken_layer'], axis = 1)))
print(np.sum(copy_invasion_info['pore.layer']))
print('-----------')
print(Nt)
print(np.sum(copy_invasion_info['throat.broken_layer']))
print(np.sum(np.any(copy_invasion_info['throat.broken_layer'], axis = 1)))
print(np.sum(copy_invasion_info['throat.layer']))
