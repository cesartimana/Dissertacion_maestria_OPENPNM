# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:03:56 2023

@author: cesar
"""

import openpnm as op
import porespy as ps
import numpy as np
import imageio as io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import sys
import math as m

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

#--------------------------------
#Doing G and angles with throats

A_t = pn_h['throat.cross_sectional_area']
P_t = pn_h['throat.perimeter']
Gp_t = A_t/np.power(P_t,2)





def half_ang_isos(G, prob = 0.5):
    #G can be a number or a 1D numpy array with values just for triangles
    #prob is the chance to get the results of the isosceles triangle with the different angle >= pi/3
    #numpy library must be imported as np
    
    #Tratamiento cuando G solo tiene un dato
    if isinstance(G,float):
        G = np.append(G,G)
    
    G = np.asarray(G)
    n = len(G)
    beta = np.zeros((n,3))
    azar = np.random.rand(n,1) - prob
    
    for i in range(n):
        if azar[i] <= 0:
            #Hacer triangulo con angulo dif mayor a pi/3
            beta2 = m.atan(2/3**0.5*m.cos(m.acos(-12*3**0.5*G[i])/3+4*m.pi/3))
            beta1 = beta2
            beta3 = m.pi/2-beta2*2
        else:
            beta2 = m.atan(2/3**0.5*m.cos(m.acos(-12*3**0.5*G[i])/3))
            beta3 = beta2
            beta1 = m.pi/2-beta2*2
        beta[i] = np.array([beta1, beta2, beta3]) 
    
    
    return beta

def corr_half_ang_isos(G, v_t, prob = 0.5):
    #G can be a number or a 1D numpy array
    #v_t is an array with the number of voxels in order to make the correction of G
    #prob is the chance to get the results of the isosceles triangle with the different angle >= pi/3
    #numpy library must be imported as np
    
    G_max = m.tan(m.pi/6)**2/4/m.tan(m.pi/3)-10e-10 #aprox 0.0481, se le resta para evitar problemas de calculo cerca del limite
    
    p = 0
    
    #Tratamiento cuando G solo tiene un dato
    if isinstance(G,float):
        p = 1
        G = np.append(G,G)
        v_t = np.append(v_t, v_t)
    
    
    G = np.asarray(G)
    n = len(G)
    beta = np.zeros((n,3))
    azar = np.random.rand(n,1) - prob
    
    for i in range(n):
        if G[i] >= G_max or v_t[i] < 30:
            G[i] = G_max
            beta2 = m.pi/6 #Recordar que es la mitad de 60
            beta[i] = np.array([beta2, beta2, beta2]) 
        elif azar[i] <= 0:
            c = 0.24012/m.log(m.log(v_t[i],10),10) + 0.79409
            G[i] =  min((0.5 * v_t[i]**(c-1) * G[i]**c, G_max)) #ingresar formula para angulo diferente mayor a 60
            beta2 = m.atan(2/3**0.5*m.cos(m.acos(-12*3**0.5*G[i])/3+4*m.pi/3))
            beta[i] = np.array([beta2, beta2, m.pi/2-beta2*2])
        else:
            c = 0.21148/m.log(m.log(v_t[i],10),10) + 0.82177
            G[i] = min((0.5 * v_t[i]**(c-1) * G[i]**c, G_max)) #ingresar formula para angulo diferente menor a 60. o sea solo acutangulos
            beta2 = m.atan(2/3**0.5*m.cos(m.acos(-12*3**0.5*G[i])/3))
            beta[i] = np.array([m.pi/2-beta2*2, beta2, beta2])
        
    #Tratamiento de G si tiene un solo dato
    if p == 1:
        G = G[0]
        beta = beta[0]
    
    
    return G, beta
    
G_t, beta_t =  corr_half_ang_isos(Gp_t, v_t)
pn_h['throat.corrected_G'] = G_t
pn_h['throat.half_contact_angle'] = beta_t


#-----------------------------------------
#Calculation of dimensions for pores

V_p = pn_h['pore.volume']*resolution**3 #problema de resolucion en high
As_p = pn_h['pore.surface_area']

As_min = 3**1.5*2**(1/3)*np.power(V_p, 2/3)
a = np.zeros_like(V_p)
h = np.zeros_like(V_p)


"""
for i in range(len(V_p)):
    if As_p[i] 
    
"""

As_used = np.fmax(As_p, As_min)

p = - 2 * As_used / 3**0.5
q = 8 * V_p

a_p = 2/3**0.5*np.power(-p, 0.5)*np.cos(1/3*np.arccos(3*q*3**0.5/2/p/np.power(-p, 0.5))-2*m.pi*1/3)
A_p = np.power(a_p,2)*3**0.5/4
h_p = V_p/A_p

def half_ang_equi(number):
    """
    Se encarga de asignar G, beta en figuras con triangulos equilateros
    number = tamaño del array necesario para crear con G, beta
    """
    G_max = m.tan(m.pi/6)**2/4/m.tan(m.pi/3)-10e-10 #equilatero
    
    G = np.ones((number,1))*G_max
    
    beta = np.ones((number,3))*m.pi/6
    
    return G, beta
    

G_p, beta_p = half_ang_equi(len(V_p))

pn_h['pore.shape_factor'] = G_p
pn_h['pore.half_contact_angle'] = beta_p

print(pn_h)
print(m.tan(m.pi/6)**2/4/m.tan(m.pi/3))


#Los problemas con tubos triangulares aumentan si tomamos los poros de contorno (que son como paredes)
#Pero no hay problema
#Region volume es el producto del numero de voxels por el volumen/area del voxel. Depende de la resolucion
#volume en high calcula el volumen usando marching cubes, pero el codigo actual no toma en cuenta la resolución.
#Equivalent diameter trabaja con los voxels siempre, o sea con region_volume
