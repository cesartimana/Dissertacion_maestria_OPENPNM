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

import numpy as np

np.random.seed(13)

#Reading pnm arquive/ network
ws = op.Workspace()
proj = ws.load_project(filename='C10Z6meanTun.pnm')

pn = proj.network

#SOLO PARA RED CREADA, ACTUALIZANDO NUMERO DE GARGANTAS
Np = pn.Np
Nt = pn.Nt

print(pn['pore.cross_sectional_area'])
print(pn['throat.cross_sectional_area'])

#i = 365
#t_check = pn.find_neighbor_throats(pores = [i])
#print(t_check)
#print(pn['pore.cross_sectional_area'][i])
#print(pn['throat.cross_sectional_area'][t_check])
#print('----')
#print(pn['pore.diameter'][i])
#print(pn['throat.diameter'][t_check])
#print('----')
#print(pn['pore.half_corner_angle'][i])
#print(pn['throat.half_corner_angle'][t_check])
