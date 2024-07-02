# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar

En este archivo se va a redactar la funcion para asignar G, beta a los poros

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

ws = op.Workspace()

timeZero = time.time()
resolution = 5.345e-6
testName_s = 'Berea_std.pnm'
testName_h = 'Berea_high.pnm'


proj_s = ws.load_project(filename=testName_s)
proj_h = ws.load_project(filename=testName_h)
print('Elapsed time reading ',testName_h ,'= ', (time.time()-timeZero)/60, ' min')

pn_s = proj_s.network
pn_h = proj_h.network

#------------------------------------------------------------------------

#Function


def geo_props_eq_pores(network):
    
    r"""
    Add the properties 'pore.length', 'pore.cross_sectional_area', 'pore.shape_factor' and  'pore.half_cont_angle' to the network.
    Assume a triangular tube (equilateral triangle).
    
    """
    
    V_p = network['pore.volume']
    As_p = network['pore.surface_area']
    n = len(V_p)
    G_max = 0.04811252243 #G for an equilateral triangle with 10 decimals

    As_min = 3**1.5*2**(1/3)*np.power(V_p, 2/3)

    As_used = np.fmax(As_p, As_min*1.0001) #To not have problems calculating parameters

    p = - 2 * As_used / 3**0.5
    q = 8 * V_p

    a_p = 2/3**0.5*np.power(-p, 0.5)*np.cos(1/3*np.arccos(3*q*3**0.5/2/p/np.power(-p, 0.5))-2*m.pi*1/3)
    A_p = np.power(a_p,2)*3**0.5/4
    h_p = V_p/A_p

    network['pore.length'] = h_p
    network['pore.cross_sectional_area'] = A_p
    network['pore.shape_factor'] = np.ones(n) * G_max
    network['pore.half_cont_angle'] = np.ones((n,3))*m.pi/6

    return

print(pn_h)

geo_props_eq_pores(pn_h)
print(pn_h)

