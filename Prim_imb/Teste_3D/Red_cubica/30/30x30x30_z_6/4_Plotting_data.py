# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar
SOLO PLOTA UN DATO Y EL EXPERIMENTAL. SI QUIERES MULTIPLES DATOS
REVISAR EN OTRAS CARPETAS VECINAS

"""

#Preambulo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from Properties import *
plt.rcParams.update({'font.size': 22})


#results = np.load('results_K_Pc_PDR.npy')
angle_r = str(theta_r_sexag)
results_PDR = np.load(f'results_K_Pc_PDR_{theta_r_sexag}_{p_kPa}kPa.npy')
st_0 = np.array([1, 1, 0, 0], dtype = float) #  sat // krel_w// krel_o // pc
results_PDR = np.insert(results_PDR, 0, st_0, axis = 0)
sat_PDR = results_PDR[:, 0]
k_w_PDR = results_PDR[:, 1]
k_o_PDR = results_PDR[:, 2]
pc_PDR = results_PDR[:, 3]

angle_a = str(theta_a_sexag)
results_IMB = np.load(f'results_K_Pc_imb_{theta_r_sexag}_{theta_a_sexag}_{p_kPa}kPa.npy')
st_0 = results_PDR[-1,:] #  sat // krel_w// krel_o // pc
st_0[-1] = pmax_drn
results_IMB = np.insert(results_IMB, 0, st_0, axis = 0)
sat_IMB = results_IMB[:, 0]
k_w_IMB = results_IMB[:, 1]
k_o_IMB = results_IMB[:, 2]
pc_IMB = results_IMB[:, 3]

#Mostrar algunos resultaods:
print('ultimos resultados de drenage')
print('sat // krel_w// krel_o // pc')
print(results_PDR[-1,:])
print(f'numero de estados quasi-estaticos: {len(pc_PDR)}')

print('ultimos resultados de imbibicion')
print('sat // krel_w// krel_o // pc')
print(results_IMB[-1,:])
print(f'numero de estados quasi-estaticos: {len(pc_IMB)}')

plt.figure(1, figsize = (8,6))
plt.plot(sat_PDR, k_w_PDR, 'b1', label = 'water, pd',markersize = 7)
plt.plot(sat_PDR, k_o_PDR, 'r1', label = 'oil, pd',markersize = 7)
plt.plot(sat_IMB, k_w_IMB, 'bo', label = 'water, imb',markersize = 7)
plt.plot(sat_IMB, k_o_IMB, 'ro', label = 'oil, imb',markersize = 7)
plt.ylim([0,1])
plt.xlim([0,1])
plt.xlabel('Water saturation')
plt.ylabel(r'$k_{rel}$')
plt.legend(fontsize = "18", loc = 'upper center')#, bbox_to_anchor=(0.5, 0.99) )
plt.tight_layout()

#-- Pressure vs saturation
pc_kPa_max = np.max(pc_PDR)/1000
pc_kPa_min = np.min(pc_IMB)/1000
ylim = np.array([round(pc_kPa_min) - 1, round(pc_kPa_max) + 1])
plt.figure(2, figsize = (8,6))
plt.plot(sat_PDR, pc_PDR/1000, 'b1', label = 'pc, dr',markersize = 7)
plt.plot(sat_IMB, pc_IMB/1000, 'bo', label = 'pc, imb', markersize = 7)
plt.xlabel('Water saturation')
plt.ylabel(r'$p_c$ (kPa)')
plt.ylim(ylim)
plt.xlim([0, 1])
plt.legend(fontsize = "18")
plt.tight_layout()
plt.show()
