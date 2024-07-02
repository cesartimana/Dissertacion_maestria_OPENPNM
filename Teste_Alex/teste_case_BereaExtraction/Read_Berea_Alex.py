import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

filePP = 'Berea3.pnm'

prjPP = op.io.PNM.load_project(filePP)
pnPP = prjPP.network
print(pnPP)
pnPP['pore.diameter'] = pnPP['pore.inscribed_diameter']
geoPP = op.geometry.Imported(network=pnPP)

pn = pnPP

#Defining internal pores, throats
pn['pore.internal'] = True
pn['pore.internal'][pn['pore.boundary']] = False
boundary_p = pn.pores('boundary')
boundary_t = pn.find_neighbor_throats(pores=boundary_p)
pn['throat.internal'] = True
pn['throat.internal'][boundary_t] = False
pn['throat.boundary'] = ~pn['throat.internal']

Np_int = np.sum(pn['pore.internal'])
Nt_int = np.sum(pn['throat.internal'])
print(f'Internal elements: {Np_int} pores and  {Nt_int} throats')

cn = pn['throat.conns']

#Reading diameter problems
print('---Using throat equivalent diameter---')
Dteq = pn['throat.equivalent_diameter']

print('---For pore equivalent---')
D1eq = pn['pore.equivalent_diameter'][cn[:, 0]]
D2eq = pn['pore.equivalent_diameter'][cn[:, 1]]
mask = ((D1eq <= Dteq) | (D2eq <= Dteq)) & pn['throat.internal']
perc = np.sum(mask) / Nt_int * 100
print(f'Percentage of internal throats bigger than pores: {perc:.3f} %')

print('---For pore extended---')
D1ex = pn['pore.extended_diameter'][cn[:, 0]]
D2ex = pn['pore.extended_diameter'][cn[:, 1]]
mask = ((D1ex <= Dteq) | (D2ex <= Dteq)) & pn['throat.internal']

perc = np.sum(mask) / Nt_int * 100
print(f'Percentage of internal throats bigger than pores: {perc:.3f} %')


"""







"""
