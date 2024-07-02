# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar


"""

#Preambulo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

columns_to_keep = ['sat', 'k']
df_water = pd.read_csv("Data_agua_Berea.dat", sep='  ', engine='python', decimal=',')
df_water = df_water.to_numpy(dtype= float)
df_oil = pd.read_csv("Data_oil_Berea.dat", sep='  ', engine='python', decimal=',')
df_oil = df_oil.to_numpy(dtype= float)
k_water0 = np.load('K_water_theta0.npy')
k_oil0 = np.load('K_oil_theta0.npy')
k_water20 = np.load('K_water_theta20.npy')
k_oil20 = np.load('K_oil_theta20.npy')


f_disc =  0.247058823529

plt.figure(1, figsize = (10,10))
plt.plot(df_water[:,0], df_water[:,1], 'b^', label = 'water, exp.')
plt.plot(df_oil[:,0], df_oil[:,1], 'r^', label = 'oil, exp.')
plt.plot(k_water0[1:,1], k_water0[1:,2], 'b--', label = r'water, sim. $\theta=0^\circ$')
plt.plot(k_oil0[1:,1], k_oil0[1:,2], 'r--', label = r'oil, sim. $\theta=0^\circ$')
plt.plot(k_water20[1:,1], k_water20[1:,2], 'b:', label = r'water, sim. $\theta=20^\circ$')
plt.plot(k_oil20[1:,1], k_oil20[1:,2], 'r:', label = r'oil, sim. $\theta=20^\circ$')
plt.plot(np.ones_like(k_oil0[1:,2]) * f_disc, k_oil0[1:,2] , 'k', label = ' sat = 0.247', linewidth = 0.5)

plt.ylim([0,1])
plt.xlim([0,1])
plt.xlabel('Water saturation')
plt.ylabel('Relative permeability')
plt.legend()

#-- Pressure vs saturation
ylim = np.array([0, 17500])
plt.figure(2, figsize = (10,10))
plt.plot(k_water0[1:,1], k_water0[1:,0], 'b-', label = r'sim.  $\theta=0^\circ$')
plt.plot(k_water20[1:,1], k_water20[1:,0], 'b:', label = r'sim.  $\theta=20^\circ$')
plt.plot(np.ones_like(ylim) * f_disc, ylim , 'k', label = 'sat = 0.247', linewidth = 0.5)
plt.xlabel('Water saturation')
plt.ylabel('Capillary pressure (Pa)')
plt.ylim(ylim)
plt.xlim([0, 1])
plt.legend()

plt.show()
