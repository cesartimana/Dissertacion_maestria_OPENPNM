# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar
ES UNA RED CON PUROS PRISMAS
lUEGO DE IMPORTAR LA RED, MODIFICAMOS LOS DIAMETROS DE GARGANTAS PEQUEÑAS Y LUEGO RECALCULAMOS K_ABS
"""

import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import math as m

#importing functions for other file
import sys
sys.path.insert(1, '/home/cesar/OpenPNM_files/_funcs')
import _algorithm_class as _alg
import _conductance_funcs as _cf
import _invasion_funcs as _if
#end

np.random.seed(13)

resolution = 5.345e-6

#Flowrate function
def Rate_calc(network, ph, inlet, outlet, conductance):
    St_p = op.algorithms.StokesFlow(network=network, phase=ph)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=inlet, values=1) #P_inlet = 1
    St_p.set_value_BC(pores=outlet, values=0) #P_outlet = 1
    St_p.run()
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val


#Reading network data
ws = op.Workspace()
testName_h = 'Berea_G_0.pnm' # 'Berea_G_0.pnm' or 'Berea.pnm'
proj_h = ws.load_project(filename=testName_h)
pn = proj_h.network
Np = pn.Np
Nt = pn.Nt

#EXTRAYENDO DATA DE DIAMETROS EQUIVALENTES
D1, Dt, D2 = pn.get_conduit_data('equivalent_diameter').T
mask = pn['throat.voxels'] < 30
F_v1 = 0.7 #Fijo

#NEW: MODIFYING DIAMETER
# D_t = D_t + (F min(D_j, D_k) - D_t) * f, F = 0.7
#---------start--------------
F_v2 = 0.38126
print('-----')
_Dt = Dt + ( F_v1 * np.minimum(D1, D2) - Dt ) * F_v2
pn['throat.equivalent_diameter'][mask] = _Dt[mask]
A = np.pi * _Dt ** 2 / 4
pn['throat.cross_sectional_area'][mask] = A[mask]
pn['throat.perimeter'][mask] = ( A[mask] / pn['throat.shape_factor'][mask] ) ** 0.5
d = 2*(A * 4 * pn['throat.shape_factor']) ** 0.5
pn['throat.prism_inscribed_diameter'][mask] = d[mask]

#---------end--------------
ws.save_project(proj_h, filename='Berea_modD')
