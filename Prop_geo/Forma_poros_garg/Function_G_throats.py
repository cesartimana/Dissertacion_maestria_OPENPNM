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

v_t = pn_s['throat.cross_sectional_area']/resolution**2 #Unico uso del standard
pn_h['throat.voxels'] = v_t
print(pn_h)

#------------------------------------------------------------------------

#Function

def Gcorr_isos(G_porespy, voxels, prob = 0.5, min_vox = 30):
    r"""
    Calculates the corrected shape factor G from G calculated by porespy, considering isosceles triangle

    prob: chance to use an isosceles triangle with different angle >= pi/3 (beta>=pi/6). Can be an array.
    Must be between 0 and 1
    min_vox: Minimum number of voxels to correct. If less, G = Gmax
    """
    G_max = 0.04811252243 #G for an equilateral triangle with 10 decimals
    t_bool = np.squeeze(np.random.rand(len(G_porespy),1)) < prob
    v_bool = voxels > min_vox
    c = ((0.24012/np.log10(np.log10(voxels)) + 0.79409) * t_bool  + (0.21148/np.log(np.log(voxels)) + 0.82177) * ~t_bool)*v_bool
    G = np.min([0.5 * voxels**(c-1) * G_porespy**c, np.ones_like(G_porespy)*G_max], axis = 0)* v_bool + G_max * ~v_bool

    return G

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

def Gcorr_beta_throats(network, prob  = 0.5):
    
    r"""
    Add the properties 'throat.shape_factor' and  'throat.half_cont_angle' to the network,
    previous correction of the perimeter data
    
    prob: chance to use an isosceles triangle with different angle >= pi/3 (beta>=pi/6). Can be an array.
    Must be between 0 and 1
    """
    
    A_t = network['throat.cross_sectional_area']
    P_t = network['throat.perimeter']
    v_t = network['throat.voxels']
    G_t = A_t/np.power(P_t,2) #Shape factor to correct


    n = len(A_t)
    t_bool = np.squeeze(np.random.rand(n,1) < prob)
    G = Gcorr_isos(G_t, v_t, prob = t_bool)
    beta = half_ang_isos(G, prob = t_bool)

    network['throat.shape_factor'] = G
    network['throat.half_cont_angle'] = beta

    return

Gcorr_beta_throats(pn_h)
print(pn_h)

    
