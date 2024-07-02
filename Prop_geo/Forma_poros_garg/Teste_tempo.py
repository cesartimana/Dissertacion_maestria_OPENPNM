# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar

En este archivo se va a redactar la funcion para asignar G, beta a las gargantas

Se considera como preambulo que se cuenta con un objeto tipo network que contiene la 
propiedad throat.voxels

Por ahora, dado que aun no se crea una red directo con esa propiedad
la etapa de asignar throat voxels estará en Preambulo

"""

#Preambulo

import openpnm as op
import porespy as ps
import numpy as np
import imageio as io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import sys
import math as m

np.random.seed(13)

n = 5
a = np.random.rand(n)
b = np.random.rand(n) < 0.5
timeZero = time.time()
c = a*b + 2*a * ~b
print('Elapsed time using arrays = ', (time.time()-timeZero)/60, ' min')


timeZero = time.time()
c = []
for i  in range(len(a)):
    if b[i] == True:
        c.append(a)
    else:
        c.append(2*a)
print('Elapsed time using for, if = ', (time.time()-timeZero)/60, ' min')
    
"""
d = np.random.rand(5,5)
print(d)
d = np.sort(d,1)
print(d)
"""


def half_ang_isos(G, prob = 0.5):
    r"""
    Calculates the half angles considering isosceles triangle

    prob: chance to use an isosceles triangle with different angle >= pi/3 (beta>=pi/6). Can be an array.
    Must be between 0 and 1
    """
    t_bool = np.squeeze(np.random.rand(len(G),1)) < prob
    c = np.arccos(-12*3**0.5*G)/3+ t_bool * 4*m.pi/3
    beta2 = np.arctan(2/3**0.5*np.cos( c ))
    beta = np.sort( np.concatenate(([beta2], [beta2], [m.pi/2-beta2*2])).T , 1)

    return beta

G_max = 0.04811252243
GGG = a * G_max
beta = half_ang_isos(GGG)
print(beta)

