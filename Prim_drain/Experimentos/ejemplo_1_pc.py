# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar

En este archivo  se calculara todo lo referente a la invasion por Drenaje primario
Se investiga como calcula la presion capilar de entrada  de las gargantas y el proceso de invasion
El algoritmo toma commo referencia el codigo de Invasion_Percolation
El fluido a usar sera el no mojante. El angulo de contacto estará aqui.
El angulo de contacto puede sr un objeto de tamaño Nx1 o Nx2. Siendo N el numero de poros o gargantas
Si es Nx2, la primera fila tiene al menor de los valores y es el receiding contact angle

La presión capilar sera guardada en el objeto de simulacion de drenaje primario porque puede cambiar con el caso.

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

testName_h = 'Network_w_geo.pnm'


proj_h = ws.load_project(filename=testName_h)

print('Elapsed time reading ',testName_h ,'= ', (time.time()-timeZero)/60, ' min')
pn = proj_h.network
print(pn)


#a = np.arange(-180,-100)
#print(pn['throat.perimeter'][a])
#print(pn['throat.shape_factor'][a])
print(pn['throat.perimeter'].min())
print(pn['throat.inscribed_diameter'].min())
print(pn['throat.inscribed_diameter'].max())
print(pn['throat.shape_factor'].max())
print(pn['throat.shape_factor'].min())
#Add phase properties

air = op.phase.Air(network=pn,name='air')
air['pore.surface_tension'] = 0.072
air['pore.contact_angle'] = 180.0
air.add_model_collection(op.models.collections.phase.air)
air.add_model_collection(op.models.collections.physics.basic)
air.regenerate_models()
water = op.phase.Water(network=pn,name='water')
water.add_model_collection(op.models.collections.phase.water)
water.add_model_collection(op.models.collections.physics.basic)
water.regenerate_models()
