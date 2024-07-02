# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar

"""

import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import _conductance_funcs as _cf
current_directory = os.getcwd()

np.random.seed(13)


#Reading pnm arquive/ network
ws = op.Workspace()

timeZero = time.time()
resolution = 5.345e-6
testName = 'Berea.pnm'


proj = ws.load_project(filename=testName)

print('Elapsed time reading ',testName ,'= ', (time.time()-timeZero)/60, ' min')

pn = proj.network

#Sobre conduit length

Li = _cf.conduit_length_tubes(pn,
                             pore_length = "pore.inscribed_diameter",
                             throat_spacing = "throat.total_length",
                             min_L = resolution)
Le = _cf.conduit_length_tubes(pn,
                             pore_length = "pore.equivalent_diameter",
                             throat_spacing = "throat.total_length",
                             min_L = resolution)
maski = Li[:,1] == resolution
maske = Le[:,1] == resolution
print('-----')
print(np.sum(maski))
print(np.sum(maski & pn['throat.boundary']))
print(np.sum(maski & pn['throat.internal']))
print(np.sum(maske))
print(np.sum(maske & pn['throat.boundary']))
print(np.sum(maske & pn['throat.internal']))

Li = _cf.conduit_length_tubes(pn,
                             pore_length = "pore.inscribed_diameter",
                             throat_spacing = "throat.total_length",
                             min_L = 0)
Le = _cf.conduit_length_tubes(pn,
                             pore_length = "pore.equivalent_diameter",
                             throat_spacing = "throat.total_length",
                             min_L = 0)

maski = Li[:,1] == 0
maske = Le[:,1] == 0
print('-----')
print(np.sum(maski))
print(np.sum(maski & pn['throat.boundary']))
print(np.sum(maski & pn['throat.internal']))
print(np.sum(maske))
print(np.sum(maske & pn['throat.boundary']))
print(np.sum(maske & pn['throat.internal']))

print('-----')
print(np.sum(pn['throat.total_length'] < resolution ))
print(np.sum(pn['throat.direct_length'] < resolution ))


# Para exportar info a Paraview
#path_to_file = current_directory
#op.io._vtk.project_to_vtk(pn.project, filename=path_to_file+'/Paraview_net')
