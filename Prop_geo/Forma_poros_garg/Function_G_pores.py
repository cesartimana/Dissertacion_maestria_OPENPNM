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
testName_h = 'Berea_high.pnm'


proj_h = ws.load_project(filename=testName_h)
print('Elapsed time reading ',testName_h ,'= ', (time.time()-timeZero)/60, ' min')

pn_h = proj_h.network

#------------------------------------------------------------------------

#Function


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

    As_min = 3**1.5*2**(1/3)*np.power(V_p, 2/3)

    As_used = np.fmax(As_p, As_min*1.00001) #To not have problems calculating parameters

    p = - 2 * As_used / 3**0.5
    q = 8 * V_p

    a0_p = 2/3**0.5*np.power(-p, 0.5)*np.cos(1/3*np.arccos(3*q*3**0.5/2/p/np.power(-p, 0.5)))
    a1_p = 2/3**0.5*np.power(-p, 0.5)*np.cos(1/3*np.arccos(3*q*3**0.5/2/p/np.power(-p, 0.5))-2*m.pi*1/3)
    h0_p = V_p/(np.power(a0_p,2)*3**0.5/4)
    h1_p = V_p/(np.power(a1_p,2)*3**0.5/4)

    ratio_0 = np.maximum(a0_p, h0_p) / np.minimum(a0_p, h0_p)
    ratio_1 = np.maximum(a1_p, h1_p) / np.minimum(a1_p, h1_p)

    #Correction study
    area_bom = As_p >= As_min
    V_A = V_p**2/As_used**3
    analysis_VA = V_A > 0.00021652
    print(min(V_A))
    print(np.average(analysis_VA))
    print(np.average(area_bom))

    #True if ratio 1 better than ratio_0 (near to zero)
    crit_1 = (ratio_1 < ratio_0)

    L_a0 =  h0_p - a0_p
    L_a1 =  h1_p - a1_p

    crit_2 = (np.abs(L_a1)) < (np.abs(L_a0))
    crit_3 = np.abs(1-a1_p / h1_p) <np.abs(1-a0_p / h0_p)
    print(np.all(crit_1 == crit_3))
    sss = 1#np.arange(0,3)
    print('Informação do poro %0.0f da amostra da Berea' % sss)
    print(' ')
    print('V = %0.6e '% V_p[sss])
    print('A_s = %0.6e '% As_p[sss])
    print(' ')
    print('Primeira raiz')
    print('a = %0.6e '% a0_p[sss])
    print('L = %0.6e ' % h0_p[sss])
    print('max(a,L) / min(a,L) = %0.3f'% ratio_0[sss])
    print('abs(L-a) = %0.6e'% abs(h0_p[sss] - a0_p[sss]))
    print('abs(L/a-1) = %0.6e' % abs(h0_p[sss]/a0_p[sss]-1))
    print(' ')
    print('Segunda raiz')
    print('a = %0.6e'% a1_p[sss])
    print('L = %0.6e'% h1_p[sss])
    print('max(a,L) / min(a,L) = %0.3f'% ratio_1[sss])
    print('abs(L-a) = %0.6e'% abs(h1_p[sss] - a1_p[sss]))
    print('abs(L/a-1) = %0.6e' % abs(h1_p[sss]/a1_p[sss]-1))
    print(' ')

    sss = 0#np.arange(0,3)
    print('Informação do poro %0.0f da amostra da Berea' % sss)
    print(' ')
    print('V = %0.6e '% V_p[sss])
    print('A_s = %0.6e '% As_p[sss])
    print(' ')
    print('Primeira raiz')
    print('a = %0.6e '% a0_p[sss])
    print('L = %0.6e ' % h0_p[sss])
    print('max(a,L) / min(a,L) = %0.3f'% ratio_0[sss])
    print('abs(L-a) = %0.6e'% abs(h0_p[sss] - a0_p[sss]))
    print('abs(L/a-1) = %0.6e' % abs(h0_p[sss]/a0_p[sss]-1))
    print(' ')
    print('Segunda raiz')
    print('a = %0.6e'% a1_p[sss])
    print('L = %0.6e'% h1_p[sss])
    print('max(a,L) / min(a,L) = %0.3f'% ratio_1[sss])
    print('abs(L-a) = %0.6e'% abs(h1_p[sss] - a1_p[sss]))
    print('abs(L/a-1) = %0.6e' % abs(h1_p[sss]/a1_p[sss]-1))

    #A_p = np.power(a_p,2)*3**0.5/4
    #h_p = V_p/A_p

    #network['pore.length'] = h_p
    #network['pore.cross_sectional_area'] = A_p
    #network['pore.shape_factor'] = np.ones(n) * G_max
    #network['pore.half_cont_angle'] = np.ones((n,3))*m.pi/6

    return

print(pn_h)

geo_props_eq_pores(pn_h)
#print(pn_h)

