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
angle = str(theta_r)
results = np.load('results_K_Pc_PDR_' + angle + '.npy')
st_0 = np.array([1, 1, 0, 0], dtype = float) #  sat // krel_w// krel_o // pc
results = np.insert(results, 0, st_0, axis = 0)
sat = results[:, 0]
k_w = results[:, 1]
k_o = results[:, 2]
pc = results[:, 3]

plt.figure(1, figsize = (8,6))
plt.plot(sat, k_w, 'b', label = 'water')
plt.plot(sat, k_o, 'r', label = 'oil')
plt.ylim([0,1])
plt.xlim([0,1])
plt.xlabel('Water saturation')
plt.ylabel(r'$k_{rel}$')
plt.legend(fontsize = "18", loc = 'center left')#, bbox_to_anchor=(0.5, 0.99) )
plt.tight_layout()

#-- Pressure vs saturation
pc_kPa_max = np.max(pc)/1000
ylim = np.array([0, round(pc_kPa_max) + 1])
plt.figure(2, figsize = (8,6))
plt.plot(sat, pc/1000, 'b', label = r'pc')
plt.xlabel('Water saturation')
plt.ylabel(r'$p_c$ (kPa)')
plt.ylim(ylim)
plt.xlim([0, 1])
#plt.legend(fontsize = "18")
plt.tight_layout()
plt.show()
