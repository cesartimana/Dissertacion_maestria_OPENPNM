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
plt.rcParams.update({'font.size': 22})

k_water = np.load('K_water.npy')
k_oil = np.load('K_oil.npy')

plt.figure(1, figsize = (8,8))
plt.plot(k_water[:,1], k_water[:,2], 'b--', label = r'water, sim.')
plt.plot(k_oil[:,1], k_oil[:,2], 'r--', label = r'oil, sim.')
plt.ylim([0,1])
plt.xlim([0,1])
plt.xlabel('Water saturation')
plt.ylabel(r'$k_{rel}$')
plt.legend(fontsize = "18", loc = "center left")
plt.tight_layout()
#plt.legend(fontsize = "14", loc = 'upper center', bbox_to_anchor=(0.7, 0.99) )

#-- Pressure vs saturation
ylim = np.array([0, 25])
plt.figure(2, figsize = (8,8))
plt.plot(k_water[:,1], k_water[:,0] / 1000, 'b-', label = r'sim.')
plt.xlabel('Water saturation')
plt.ylabel(r' $p_c$ (kPa)')
plt.ylim(ylim)
plt.xlim([0, 1])
plt.tight_layout()
plt.show()
