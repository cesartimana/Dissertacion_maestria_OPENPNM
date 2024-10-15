# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:41:13 2023

@author: cesar
PARA CALCULAR CLUSTERS CONSIDERANDO CONEXCIONES DE POROS Y GARGANTAS

"""

import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sprs
import scipy.sparse.csgraph as csgraph
#importing functions for other file
import sys
sys.path.insert(1, '/home/cesar/OpenPNM_files/mestrado/_funcs')
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
    St_p.set_value_BC(pores=inlet, values=1)
    St_p.set_value_BC(pores=outlet, values=0)
    St_p.run()
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val

#Reading netwrork data
ws = op.Workspace()
testName_h = 'Berea_modD.pnm'
proj_h = ws.load_project(filename=testName_h)
pn = proj_h.network
Np = pn.Np
Nt = pn.Nt

#Defining boundary conditions
axis = 'x'
inlet_pores = pn['pore.' + axis + 'min']
outlet_pores = pn['pore.' + axis + 'max']

#Properties extractend from Valvatne - Blunt (2004) Table 1
tension = 30e-3 #N/m
water_visc = 1.05e-3 #Pa/s
oil_visc = 1.39e-3 #Pa/s
theta = np.pi / 180 * 0  #water phase. 0 is used by Valvatne Blunt 2004

#Add phase properties
#OIL
oil = op.phase.Phase(network=pn, name='oil')
oil['pore.surface_tension'] = tension
oil['throat.surface_tension'] = tension
oil['pore.viscosity'] = oil_visc
oil['throat.viscosity'] = oil_visc

#WATER
water = op.phase.Phase(network=pn,name='water')
water['pore.surface_tension'] = tension
water['throat.surface_tension'] = tension
water['pore.viscosity'] = water_visc
water['throat.viscosity'] = water_visc
water['pore.contact_angle'] = theta
water['throat.contact_angle'] = theta


#Simulating primary Drainage
drn = _alg.Primary_Drainage(network=pn, phase=water)
drn.set_inlet_BC(pores=inlet_pores)
drn.set_outlet_BC(pores=outlet_pores)
drn.run(throat_diameter = 'throat.prism_inscribed_diameter')
pmax_drn = 9000
p_vals, invasion_info, cluster_info = drn.postprocessing(p_max = pmax_drn)
#Use the information with name['element.location']. element = pore,throat. location = center, corner


"""
Base del codigo.

1)scipy.sparse.csr_matrix
Ejemplos:
--------------------
where data, row_ind and col_ind satisfy the relationship a[row_ind[k], col_ind[k]] = data[k].
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
RESULTADO
array([[1, 0, 2],
       [0, 0, 3],
       [4, 5, 6]])
------------------
is the standard CSR representation where the column indices for row i are stored in
indices[indptr[i]:indptr[i+1]] and their corresponding values are stored in data[indptr[i]:indptr[i+1]].
indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
RESPUESTA
array([[1, 0, 2],
       [0, 0, 3],
       [4, 5, 6]])
-----------------

2)scipy.sparse.csgraph.connected_components
Te da los clusters
---------------
n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
RESPUESTA
n_components
2
labels
array([0, 0, 0, 1, 1], dtype=int32)
----------------
"""

def cluster_advanced(network, phase_locations):
    r"""
    Determina los tipos de cluster revisando la coneccion de poros y gargamtas y la ubicación de las fases.
    Hago una superlista de poros y gargantas
    Parameters:
    ------------------
    pn: Network
    phase_locations: main dictionary with the secondary dictionarys pore, throat; and the keywords center, corner, layer (not yet)

    Returns:
    -------------------
    cluster_info: dictionary with index information. Have the dictionaries 'pore', 'throat',
    with the keywords 'center', 'corner', 'index'
    """
    conns = network['throat.conns']
    Np = network.Np
    Nt = network.Nt
    #Matriz of (Np + Nt) x (Np + Nt). gargantas de 0 a Nt-1. poros de Nt a Np-1
    connections = np.zeros((Np+Nt, Np+Nt), dtype = bool)
    #Completando las conecciones
    for t in range(Nt):
        #Revisando la presencia de la fase en gargantas
        mask_t_center = phase_locations['throat.center'][t]
        mask_t_corner = np.any(phase_locations['throat.corner'][t,:])
        if mask_t_center or mask_t_corner:
            #Evaluando presencia de la fase en poros vecino:
            for cn in [0,1]:
                p = conns[t,cn]
                mask_p_center = phase_locations['pore.center'][p]
                mask_p_corner = np.any(phase_locations['pore.corner'][p,:])
                #Estableciendo conexion (Tabla 4.2 disertacion)
                if (mask_t_center and mask_p_center) or (mask_t_corner and mask_t_corner):
                    connections[t, Nt + p] = True
    #Aplicando el algoritmo scipy de cluster
    connections = sprs.csr_matrix(connections)
    clusters = csgraph.connected_components(csgraph=connections, directed=False, return_labels=True)[1] + 1
    throat_clusters = clusters[0:Nt]
    pore_clusters = clusters[Nt:(Nt+Np)]

    #Adding information

    cluster_info = {}
    cluster_info['pore.center'] = pore_clusters * phase_locations['pore.center']
    cluster_info['pore.corner'] = np.tile(pore_clusters,(3,1)).T * phase_locations['pore.corner']
    cluster_info['throat.center'] = throat_clusters * phase_locations['throat.center']
    cluster_info['throat.corner'] = np.tile(throat_clusters,(3,1)).T * phase_locations['throat.corner']

    #Creating index list per element
    for item  in ['pore', 'throat']:
        indexinfo = np.concatenate( (cluster_info[item +'.corner'].flatten(), cluster_info[item +'.center']))
        cluster_info[item + '.index'] = np.unique(indexinfo)

    #Creating general list
    indexinfo = np.concatenate( (cluster_info['pore.index'], cluster_info['throat.index']) )
    cluster_info['index_list'] = np.sort(np.unique(indexinfo))

    #Use continuous numbers
    newlist = np.arange(len(  cluster_info['index_list'] ))

    for item in ['pore', 'throat']:
        for i in range( len(newlist) ):
            if np.isin(cluster_info['index_list'][i], cluster_info[item + '.index']):
                mask = cluster_info[item + '.index'] == cluster_info['index_list'][i]
                cluster_info[item + '.index'][ mask ] = newlist[i]
                for loc in ['corner', 'center']:
                    mask = cluster_info[item + '.' + loc] == cluster_info['index_list'][i]
                    cluster_info[item + '.' + loc][ mask ] = newlist[i]
    return cluster_info



#Cluster wetting phase
cc = cluster_advanced(pn, invasion_info)
print(cc.keys())
print(cc['index_list'])
print(cc['pore.center'] == 1)
print(cluster_info['pore.center'] == 0)
print(np.all( (cc['pore.center'] == 1) == (cluster_info['pore.center'] == 0) ) ) #Check
print(np.all( (cc['throat.center'] == 1) == (cluster_info['throat.center'] == 0) ) ) #Check
print(np.all( (cc['pore.corner'] == 1) == (cluster_info['pore.corner'] == 0) ) ) #Check
print(np.all( (cc['throat.corner'] == 1) == (cluster_info['throat.corner'] == 0) ) ) #Check

nw_ph = {}
for item in ['pore', 'throat']:
    for loc in ['center', 'corner']:
        nw_ph[item + '.' + loc] = ~invasion_info[item + '.' + loc]

#Cluster non wetting phase
cc_nw = cluster_advanced(pn, nw_ph)
print(cc_nw['pore.center'][0:50])
print(cluster_info['pore.center'][0:50])
print(np.unique(cc_nw['pore.center']))
print(np.unique(cluster_info['pore.center']))
print(np.all( (cc_nw['pore.center'] == 31) == (cluster_info['pore.center'] == 10) ) ) #Check
print(np.all( (cc_nw['throat.center'] == 31) == (cluster_info['throat.center'] == 10) ) ) #Check
print(np.all( (cc_nw['pore.corner'] == 31) == (cluster_info['pore.corner'] == 10) ) ) #Check
print(np.all( (cc_nw['throat.corner'] == 31) == (cluster_info['throat.corner'] == 10) ) ) #Check
