import openpnm as op
import numpy as np
from Properties import *

#importing functions for other file
import sys
sys.path.insert(1, '/home/cbeteta/_funcs')
import _algorithm_class as _alg
import _conductance_funcs as _cf
import _invasion_funcs as _if
#Reading pnm arquive/ network
ws = op.Workspace()
proj = ws.load_project(filename=testName)
#proj = ws.load_project(filename='Berea.pnm')
pn = proj.network

Np_side = 10
sp = 5e-4

#Calculating porosity
vol_box = (Np_side * sp)**3
vol_pores = np.sum(pn['pore.volume'][ pn['pore.internal']])

porosity_final = (vol_pores)/vol_box
print('porosity', porosity_final)

#Calculating permeability

def Rate_calc(network, ph, inlet, outlet, conductance):
    St_p = op.algorithms.StokesFlow(network=network, phase=ph)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=inlet, values=1)
    St_p.set_value_BC(pores=outlet, values=0)
    St_p.run() #solver ='PardisoSpsolve'
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val

#Calculating hydraulic conductance
visc = 1e-3
phase = op.phase.Phase(network=pn)
phase['pore.viscosity'] = visc
g = _cf.conductance_triangle_OnePhase(phase, correction = True, check_boundary = True)
phase['throat.hydraulic_conductance'] = g

inlet_pores = pn['pore.inlet']
outlet_pores = pn['pore.outlet']

from time import time
ws.settings.default_solver = 'ScipySpsolve'
start_time = time()
Q = Rate_calc(pn, phase, inlet_pores, outlet_pores, 'throat.hydraulic_conductance')
end_time = time()

L_net = Np_side * sp
A_net = (Np_side * sp)**2
K = Q[0] * L_net * visc / A_net #Q[0] * L * mu / (A * Delta_P), Delta_P = 1
print(f'The value of K is: {K/0.9869233e-12*1000:.2f} mD')
print(f'The value of K is: {K} m2')

