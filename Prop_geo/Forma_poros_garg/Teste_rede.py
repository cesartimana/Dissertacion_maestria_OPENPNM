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
testName_h = 'Berea_high.pnm'


proj_h = ws.load_project(filename=testName_h)
print('Elapsed time reading ',testName_h ,'= ', (time.time()-timeZero)/60, ' min')

pn_h = proj_h.network

print(pn_h)

def geo_props_eq_pores(network):

    r"""
    Add the properties 'pore.length', 'pore.cross_sectional_area', 'pore.shape_factor' and  'pore.half_cont_angle' to the network.
    Assume a triangular tube (equilateral triangle).
    pore.surface_area ignores throat areas.
    pore.prism_surface_area is the area used in calculus minus throat areas


    """

    V_p = network['pore.volume']
    S_p = network['pore.surface_area']
    A_t = network['throat.cross_sectional_area']
    t_conns = network['throat.conns']

    #Adding areas: solid-fluid + throat areas per pore
    Np = len(V_p)
    Nt = len(A_t)

    pore_throat =  np.tile(np.arange(Np),(Nt,1)) #matrix Nt x Np. Each row have all pore numbers
    neigh1 = np.tile(np.array(t_conns)[:,0], (Np,1)).T #matrix Nt x Np.
    neigh2  = np.tile(np.array(t_conns)[:,1], (Np,1)).T #matrix Nt x Np.
    area_t_per_p =  np.sum(np.tile(A_t, (Np,1)).T * ((pore_throat == neigh1) |  (pore_throat == neigh2)), axis=0)
    As_p = S_p + area_t_per_p

    #Doing geometry
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
    network['pore.prism_surface_area'] = As_used - area_t_per_p
    network['pore.shape_factor'] = np.ones(Np) * G_max
    network['pore.half_cont_angle'] = np.ones((Np,3))*m.pi/6

    return

#geo_props_eq_pores(pn_h)


op.teste.geometry.geo_props_eq_pores(pn_h)
op.teste.geometry.Gcorr_beta_throats(pn_h, prob  = 0.5)

print(pn_h)
