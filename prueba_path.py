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

#importing functions for other file
import sys

sys.path.insert(1, '/home/cesar/OpenPNM_files/_funcs')
print(sys.path)
import _conductance_funcs as _cf
import _invasion_funcs as _if
#end


