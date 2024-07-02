#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:05:33 2023

@author: cesar
"""

import numpy as np
import openpnm as op
import math as m
import random as rnd

#Fixing random numbers
np.random.seed(5)
rnd.seed(5)

#Creating the network
pn = op.network.Cubic(shape=[5, 5, 5], spacing=5e-5)

"""
print(pn)
print(len(pn['pore.coords'])) #Number of pores
print(len(pn['throat.conns'])) #Number of throats
"""

#Creating geometric functions. For now without calling for instance the network object

def shape_factor(n,Gmin=1e-5,Gmax=np.sqrt(3)/36):
    """
    Creates an array of shape factor values for each element
    n= number of elements (pore or throats)
    create Gmin to evade G=0 
    Gmax=sqrt(3)/36 for equilateral triangle
    """
    G=(Gmax-Gmin)*np.random.rand(n)+Gmin
    return G 

def half_angle(n,G):
    """
    Creates an array of three random half angles for each element
    n=number of elements (pore or throats)
    G=shape factor
    beta1<=beta2<=beta3
    Valvatne and Blunt 2004 / Patzek 2001
    """
    beta=np.random.rand(n,3)
    beta_2_min=np.arctan(2/m.sqrt(3)*np.cos(np.arccos(-12*m.sqrt(3)*G)/3+4*m.pi/3))
    beta_2_max=np.arctan(2/m.sqrt(3)*np.cos(np.arccos(-12*m.sqrt(3)*G)/3))
    beta[:,1]=(beta_2_max-beta_2_min)*beta[:,1]+beta_2_min #beta_2
    beta[:,0]=-0.5*beta[:,1]+0.5*np.arcsin((np.tan(beta[:,1])+4*G)/(np.tan(beta[:,1])-4*G)*np.sin(beta[:,1]))
    beta[:,2]=m.pi/2-beta[:,1]-beta[:,0]
    return beta

#Creating geometric properties

G_pore=shape_factor(len(pn['pore.coords']))
G_throat=shape_factor(len(pn['throat.conns']))
ha_pore=half_angle(len(pn['pore.coords']), G_pore)
ha_throat=half_angle(len(pn['throat.conns']), G_throat)

#h=half_angle(2, prova_G)

