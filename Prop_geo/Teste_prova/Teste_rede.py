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
print(ws)
print('Elapsed time reading ',testName_h ,'= ', (time.time()-timeZero)/60, ' min')

pn_h = proj_h.network

#Eliminating isolated pores
h = op.utils.check_network_health(pn_h)
op.topotools.trim(network=pn_h, pores=h['disconnected_pores'])

print(pn_h)
print(pn_h['throat.conns'])

#calculating geo props
op.teste.geometry.geo_props_eq_pores(pn_h)
op.teste.geometry.Gcorr_beta_throats(pn_h, prob  = 0.5)

ws.save_project(proj_h, filename='Network_w_geo')



