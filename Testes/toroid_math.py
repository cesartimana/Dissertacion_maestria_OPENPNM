#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:13:32 2023

@author: cesar
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt

#part 1 plot beta max and pc max for theta if drainage is analyzed

theta = np.linspace(90, 180 , 91)
theta = np.deg2rad(theta) 
rt_R = 10000





beta_max = theta -  m.pi + np.arcsin( np.sin(theta)/ (1 + rt_R) )
Pc = -1 * np.cos(theta - beta_max) / (1 + 1/rt_R * (1 - np.cos(beta_max))) #es PC * rt/2 sigma


theta = np.rad2deg(theta) 
beta_max = np.rad2deg(beta_max) 

fig, axs = plt.subplots(2)
fig.suptitle('Beta_max e P_c vs theta')
axs[0].plot(theta, beta_max)
axs[1].plot(theta, Pc)



#part 2 plot Pc for all beta
theta = np.linspace(90, 180 , 4)
theta = np.deg2rad(theta) 
rt_R = 10

beta = np.linspace(-90, 90 , 181)
beta_deg = beta
beta = np.deg2rad(beta) 




fig, axs = plt.subplots(len(theta))
fig.suptitle('P_c vs beta for each theta')


for i in range(len(theta)):
    Pc =  np.cos(theta[i] - beta) / (1 + 1/rt_R * (1 - np.cos(beta))) #es PC * rt/2 sigma, sem o menos
    axs[i].plot(beta_deg, Pc)

