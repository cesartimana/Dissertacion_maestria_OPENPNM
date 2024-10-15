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

angle_r = str(theta_r_sexag)
results_PDR = np.load(f'results_K_Pc_PDR_{theta_r_sexag}_{p_kPa}kPa.npy')
s_PDR = len(results_PDR[:, 0])
angle_a = str(theta_a_sexag)
results_IMB = np.load(f'results_K_Pc_imb_{theta_r_sexag}_{theta_a_sexag}_{p_kPa}kPa.npy')
s_IMB = len(results_IMB[:, 0])
#Mostrar algunos resultaods:
print('ultimos resultados de drenage')
print('sat // krel_w// krel_o // pc')
print(results_PDR[-1,:])
print(f'numero de estados quasi-estaticos: {s_PDR}')

print('ultimos resultados de imbibicion')
print('sat // krel_w// krel_o // pc')
print(results_IMB[-1,:])
print(f'numero de estados quasi-estaticos: {s_IMB}')

#Drainage

##Retire some points
#points_PDR = np.linspace(1, s_PDR-1, 50, dtype = int)
#points_PDR = np.concatenate(([0], points_PDR))
#results_PDR = results_PDR[points_PDR, :]

#Retire some ponts, method 2
sat_PDR = np.linspace(results_PDR[0, 0], results_PDR[-1, 0], 100, dtype = float)
points_PDR = []
for i in range(len(sat_PDR)):
    diff = np.abs(results_PDR[:, 0] - sat_PDR[i])
    s_inv = np.where(diff == np.min(diff))[0]
    points_PDR.append(s_inv[0])
points_PDR = np.unique(points_PDR)
results_PDR = results_PDR[points_PDR, :]

sat_PDR = results_PDR[:, 0]
k_w_PDR = results_PDR[:, 1]
k_o_PDR = results_PDR[:, 2]
pc_PDR = results_PDR[:, 3]

#Imbibition

##Retire some points
#points_IMB = np.linspace(1, s_IMB-1, 50, dtype = int)
#points_IMB = np.concatenate(([0], points_IMB))
#results_IMB = results_IMB[points_IMB, :]

#Retire some ponts, method 2
sat_IMB = np.linspace(results_IMB[0, 0], results_IMB[-1, 0], 100, dtype = float)
points_IMB = []
for i in range(len(sat_IMB)):
    diff = np.abs(results_IMB[:, 0] - sat_IMB[i])
    s_inv = np.where(diff == np.min(diff))[0]
    points_IMB.append(s_inv[0])
points_IMB = np.unique(points_IMB)
results_IMB = results_IMB[points_IMB, :]

sat_IMB = results_IMB[:, 0]
k_w_IMB = results_IMB[:, 1]
k_o_IMB = results_IMB[:, 2]
pc_IMB = results_IMB[:, 3]


#Plotting
plt.figure(1, figsize = (8,6))
plt.plot(sat_PDR, k_w_PDR, 'bv--', label = 'water, pd',markersize = 7, fillstyle = 'none')
plt.plot(sat_PDR, k_o_PDR, 'bo--', label = 'oil, pd',markersize = 7, fillstyle = 'none')
plt.plot(sat_IMB, k_w_IMB, 'rv--', label = 'water, imb',markersize = 7, fillstyle = 'none')
plt.plot(sat_IMB, k_o_IMB, 'ro--', label = 'oil, imb',markersize = 7, fillstyle = 'none')
plt.ylim([-0.02,1])
plt.xlim([0,1])
plt.xlabel('Water saturation')
plt.ylabel(r'$k_{rel}$')
plt.legend(fontsize = "18", loc = 'upper center')#, bbox_to_anchor=(0.5, 0.99) )
plt.tight_layout()

plt.figure(3, figsize = (8,6))
plt.plot(sat_PDR, k_w_PDR, 'bv--', label = 'water, pd',markersize = 7, fillstyle = 'none')
plt.plot(sat_PDR, k_o_PDR, 'bo--', label = 'oil, pd',markersize = 7, fillstyle = 'none')
plt.plot(sat_IMB, k_w_IMB, 'rv--', label = 'water, imb',markersize = 7, fillstyle = 'none')
plt.plot(sat_IMB, k_o_IMB, 'ro--', label = 'oil, imb',markersize = 7, fillstyle = 'none')
plt.ylim([1e-10,1])
plt.xlim([0,1])
plt.yscale('log')
plt.xlabel('Water saturation')
plt.ylabel(r'$k_{rel}$')
#plt.legend(fontsize = "18", loc = 'upper center')#, bbox_to_anchor=(0.5, 0.99) )
plt.tight_layout()

#-- Pressure vs saturation
pc_kPa_max = 2#np.max(pc_IMB)/1000
pc_kPa_min = 0#np.min(pc_IMB)/1000
ylim = np.array([round(pc_kPa_min) - 1, round(pc_kPa_max) + 1])

plt.figure(2, figsize = (8,6))
plt.plot(sat_PDR, pc_PDR/1000, 'bv--', label = 'pc, dr',markersize = 7, fillstyle = 'none')
plt.plot(sat_IMB, pc_IMB/1000, 'rv--', label = 'pc, imb', markersize = 7, fillstyle = 'none')
plt.xlabel('Water saturation')
plt.ylabel(r'$p_c$ (kPa)')
plt.ylim(ylim)
plt.xlim([0, 1])
plt.legend(fontsize = "18")
plt.tight_layout()
plt.show()
