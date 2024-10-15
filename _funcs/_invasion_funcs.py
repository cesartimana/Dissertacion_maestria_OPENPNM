import logging
import numpy as np
from openpnm.models import _doctxt
from scipy.optimize import fsolve
import openpnm as op
from collections import namedtuple
import scipy.sparse as sprs
import scipy.sparse.csgraph as csgraph
logger = logging.getLogger(__name__)


__all__ = [
    "MSP_prim_drainage",
]


#En las funciones creadas antes, se colocaba en una linea anterior
#lo siguiente: @_doctxt
#Esto lo unico que hace es substituir %(nombre)s por el texto del archivo _doctxt



def MSP_prim_drainage(phase,
	     throat_diameter = 'throat.diameter',
	     throat_contact_angle = 'throat.contact_angle'):
    r"""
    Computes the capillary entry pressure of all throats.

    Parameters
    ----------
    phase: The invaded/wetting one. The network associted with must have these properties:
        -throat.surface_tension
        -throat.contact_angle (the key name can be changed)
        -throat.diameter (the key name can be changed)
        -throat.shape_factor


    Returns
    -------
    Capillary entry pressure

    Notes
    -----
    -Reference of the method: Oren 1998 / Valvatne and Blunt 2004
    -A constraint was considered for the correct calculation. In case G <= 10-3  AND theta <= pi/180, then 4GD/cos2theta = -1
    If not, math errors can appear.
    The maximum percentage error when assuming this is 0.082%

    """
    network = phase.network
    sigma = phase["throat.surface_tension"]
    theta = phase[throat_contact_angle]
    theta_r =    np.tile(theta,(3,1)).T #matrix Nt x 3. All columns are the same
    r = network[throat_diameter] / 2
    G = network["throat.shape_factor"]
    beta = network["throat.half_corner_angle"]
    condition = wp_in_corners(beta, theta)  #just these angles are valid for the addition
    S1 = np.sum((np.cos(theta_r) * np.cos(theta_r + beta) / np.sin(beta) + theta_r + beta - np.pi/2) * condition, axis = 1)
    S2 = np.sum((np.cos(theta_r + beta) / np.sin(beta)) * condition, axis = 1)
    S3 = 2*np.sum((np.pi / 2 - theta_r - beta) * condition, axis = 1)
    D = S1 - 2 * S2 * np.cos(theta) + S3
    value = sigma * np.cos(theta) * (1 + np.sqrt(1 + 4 * G * D / np.cos(theta)**2)) / r
    return value

def update_cluster_index(cluster_dict, layer = False, list_string = 'index_list'):
    r"""
    Function used to update the indexes of the cluster dictionary
    Assumes that center and corner positions exist
    """
    elements = ['pore', 'throat']
    locations = ['center', 'corner']
    if layer:
        locations.append('layer')
    index_list = []
    for item in elements:
        for loc in locations:
            index_list = np.append(index_list,np.unique(cluster_dict[f'{item}.{loc}']))
    cluster_dict[list_string] = np.unique(index_list)
    return(cluster_dict)

def merge_cluster(cluster_dict, cluster_indexes, layer = False, list_string = 'index_list'):
    r"""
    Function used to update the indexes of the cluster dictionary, merging two clusters
    Assumes that center and corner positions exist
    Choose the lower index
    The first number in the cluster index is chosen to be the new index

    Parameters:
    -------------------
    cluster_dict : Dictionary
    cluster_indexes : The indexes of the merged clusters
    """
    new_index = cluster_indexes[0]
    elements = ['pore', 'throat']
    locations = ['center', 'corner']
    if layer:
        locations.append('layer')
    for item in elements:
        for loc in locations:
            mask = np.isin(cluster_dict[f'{item}.{loc}'], cluster_indexes)
            if np.any(mask):
                cluster_dict[f'{item}.{loc}'][mask] = new_index
    cluster_dict = update_cluster_index(cluster_dict, layer, list_string)
    return cluster_dict

def check_divide_cluster(network, cluster_dict, index, layer = False, list_string = 'index_list', index_mode = 'max', wp = False):
    r"""
    Function used to check if a cluster was divided
    Assumes that center and corner positions exist
    Choose the lower index

    Parameters:
    -------------------
    cluster_dict : Dictionary
    cluster_indexes : The indexes of the merged clusters
    index_mode: just two modex: 'max' or 'min'. Set the new index

    Return
    cluster_dict: The dictionary, updated if the cluster was divided
    """
    index == int(index)
    modes = ['max', 'min']
    if index_mode not in modes:
        raise Exception('Only two index modes: max, min.')
    bool_divided = False
    elements = ['pore', 'throat']
    locations = ['center', 'corner']
    if layer:
        locations.append('layer')
    check_cluster = {}
    for item in elements:
        for loc in locations:
            check_cluster[f'{item}.{loc}'] = (cluster_dict[f'{item}.{loc}'] == index)
    if layer and wp:
        new_cluster_dict = cluster_advanced_WP_CLC(network, check_cluster)
    else:
        new_cluster_dict = cluster_advanced(network, check_cluster, corner = True, layer = layer)
    if len(new_cluster_dict['index_list']) > 2:
        #Divided cluster
        bool_divided = True
        #print(new_cluster_dict['index_list'])
        if layer and wp:
            child_clusters = new_cluster_dict['index_list'][new_cluster_dict['index_list'] < 1]
        else:
            child_clusters =  np.delete(new_cluster_dict['index_list'], 0)
        #print(child_clusters)
        for i in range(len(child_clusters) ):
            if i > 0:
                if index_mode == 'max':
                    new_index = np.max(cluster_dict['index_list']) + 1
                else:
                    new_index = np.min(cluster_dict['index_list']) - 1
                for item in elements:
                    for loc in locations:
                        keyname = item + '.' + loc
                        mask = new_cluster_dict[keyname] == child_clusters[i]
                        if np.any(mask):
                            cluster_dict[keyname][mask] = new_index

            cluster_dict = update_cluster_index(cluster_dict, layer, list_string)
    return cluster_dict, bool_divided

def obtain_nontrapped_clusters(wp_dict,
                               cluster_dict,
                               connect_wp_pores,
                               connect_nwp_pores,
                               layer = False,
                               obtain_trapped = False):
    r"""
    Obtain list of non-trapped clusters
    We consider IMBIBITION and triangular cross sections, so the wetting phase clusters are connected to the inlet.
    and the non-wetting phase to the outlet.

    Parameters:
    ----------------
    -phase_info: Dictionary with boolean arrays. Key detail lines below
    -cluster_info: Dictionary with int arrays. Key detail lines below
    -connect_wp_pores: Pores that allows the inlet or outlet of the wetting phase
    -connect_nwp_pores: Pores that allows the inlet or outlet of the non-wetting phase
    -inlet_pores: boolean array
    -oulet_pores: boolean array

    key:
    'element.location'.
    'element' can be 'pore' or 'throat'
    'location' can be 'center' or 'corner'
    Ex: 'pore.center'
    """
    #Non-trapped list
    nontrapped_list = np.array([])

    if layer:
        corner_locations = ['corner', 'layer']
    else:
        corner_locations = ['corner']

    for wp in [True, False]:
        if wp:
            BC = connect_wp_pores
        else:
            BC = connect_nwp_pores
        index = np.unique(cluster_dict['pore.center'][ (wp_dict['pore.center'] == wp) & BC ])
        if len(index) > 0:
            nontrapped_list = np.append(nontrapped_list, index)
        for loc in corner_locations:
            index = np.unique(cluster_dict[f'pore.{loc}'][ (wp_dict[f'pore.{loc}'] == wp) & np.tile(BC, (3,1)).T ])
            if len(index) > 0:
               nontrapped_list = np.append(nontrapped_list, index)
    nontrapped_list = np.unique(nontrapped_list)
    if obtain_trapped:
        trapped_list = np.setdiff1d(cluster_dict['index_list'], nontrapped_list )
        return nontrapped_list, trapped_list
    else:
        return nontrapped_list

def wp_in_corners(beta, theta):
    r"""
    Function that allows us to know if the corners of a triangle
    can contain the wetting phase (True) if the center phase is non-wetting
    beta: Array Nx3
    theta: Array Nx1
    """
    theta =    np.tile(theta,(3,1)).T
    condition = (beta < (np.pi / 2 - theta))

    return condition

def interfacial_corner_distance(R, theta, beta, int_cond = True, outer_AM = False):
    r"""
    Calculate the distance between each corner of the triangular cross section and the location of the solid-phase 1 - phase 2 interface.

    Parameters:
    -R: Radius of curvature of the interface. Equivalent to tension/ capillary_pressure
    -theta: float or np.array (same size as beta) with the contact angles at each corner.
    -beta: Half corner angle.
    -int_cond: interface condition for each corner. True or 3-columns boolean array.
     If False, there is no interface between center and corner, and bi = 0 for those corners

    """
    if outer_AM:
        bi = -np.abs(R) * np.cos(theta - beta) / np.sin(beta)
    else:
        bi = R * np.cos(theta + beta) / np.sin(beta)
    if isinstance(bi, np.ndarray):
        bi[bi < 0] = 0
        if type(int_cond) == bool:
            if int_cond:
                pass
            else:
                bi = np.zeros_like(bi, dtype = False)
        else:
            bi[~int_cond] = 0
    return bi


def identify_continuous_cluster(cluster_pore_center , cluster_pore_corner, inlet_pores, outlet_pores, cluster_pore_layer = None, layer = False):
    r"""
    This function returns a list of cluster indices that have a continuity.
    That is, the phase can enter and exit the porous medium.
    Parameters:
    cluster_pore_center / corner: lists of cluster indexes per pore
    inlet/outlet_pores: boundary conditions


    Return a list of clusters
    """
    continuous_list = []
    if layer:
        cluster_list = np.unique(np.vstack(([cluster_pore_center], cluster_pore_corner.T, cluster_pore_layer.T)))
        for i  in cluster_list:
            cond_inlet = (((cluster_pore_center == i) | np.any(cluster_pore_corner == i, axis = 1) | np.any(cluster_pore_layer == i, axis = 1))[inlet_pores]).any()
            cond_outlet = (((cluster_pore_center == i) | np.any(cluster_pore_corner == i, axis = 1) | np.any(cluster_pore_layer == i, axis = 1))[outlet_pores]).any()
            if cond_inlet and cond_outlet:
                continuous_list.append(i)
    else:
        cluster_list = np.unique(np.vstack((cluster_pore_corner.T, [cluster_pore_center])))
        for i  in cluster_list:
            cond_inlet = (((cluster_pore_center == i) | np.any(cluster_pore_corner == i, axis = 1))[inlet_pores]).any()
            cond_outlet = (((cluster_pore_center == i) | np.any(cluster_pore_corner == i, axis = 1))[outlet_pores]).any()
            if cond_inlet and cond_outlet:
                continuous_list.append(i)
    return np.array(continuous_list)

def corner_area(beta, theta, bi):
    r"""
    Calculation of corner area
    """
    S1 = (bi * np.sin(beta) / (np.cos(theta + beta)))
    S2 = (np.cos(theta) * np.cos(theta + beta) / np.sin(beta) + theta + beta - np.pi / 2  )
    return S1 ** 2 * S2

def cluster_advanced(network, phase_locations, corner = False, layer = False):
    r"""
    Set index clusters to all phase locations according to its presence and location,
    building a general connection list for pores and throat. NOT ADEQUATE IF WE CONSIDER THE WETTING PHASE AND LAYERS
    Connections accepted:
    corner-corner
    layer-layer
    layer-center
    center-center
    Determina los tipos de cluster revisando la coneccion de poros y gargamtas y la ubicación de las fases.
    Hago una superlista de poros y gargantas
    Parameters:
    ------------------
    pn: Network
    phase_locations: main dictionary with the secondary dictionarys pore, throat; and the keywords center, corner, layer
    corner: bool. If True, check locations 'center' and 'corner'
    layer: bool. Used only if corner == True. If True, check also the location 'False'

    Returns:
    -------------------
    cluster_info: dictionary with index information. Have the dictionaries 'pore', 'throat',
    with the keywords 'center', 'corner', 'layer', 'index'
    """
    conns = network['throat.conns']
    Np = network.Np
    Nt = network.Nt
    #Matriz of (Np + Nt) x (Np + Nt). gargantas de 0 a Nt-1. poros de Nt a Np-1
    connections = np.zeros((Np+Nt, Np+Nt), dtype = bool)
    if corner:
        if layer:
            #Completando las conecciones
            for t in range(Nt):
                #Revisando la presencia de la fase en gargantas
                mask_t_center = phase_locations['throat.center'][t]
                mask_t_corner = np.any(phase_locations['throat.corner'][t,:])
                mask_t_layer = np.any(phase_locations['throat.layer'][t,:])
                if mask_t_center or mask_t_corner or mask_t_layer:
                    #Evaluando presencia de la fase en poros vecino:
                    for cn in [0,1]:
                        p = conns[t,cn]
                        mask_p_center = phase_locations['pore.center'][p]
                        mask_p_corner = np.any(phase_locations['pore.corner'][p,:])
                        mask_p_layer = np.any(phase_locations['pore.layer'][p,:])
                        #Estableciendo conexion (Tabla 3.4 disertacion)
                        if ((mask_t_layer or mask_t_center) and (mask_p_layer or mask_p_center)) or (mask_t_corner and mask_p_corner):
                            connections[t, Nt + p] = True
        else:
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
                        #Estableciendo conexion (Tabla 3.4 disertacion)
                        if (mask_t_center and mask_p_center) or (mask_t_corner and mask_p_corner):
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
    if layer:
        cluster_info['pore.layer'] = np.tile(pore_clusters,(3,1)).T * phase_locations['pore.layer']
        cluster_info['throat.layer'] = np.tile(throat_clusters,(3,1)).T * phase_locations['throat.layer']

    #Creating index list per element
    for item  in ['pore', 'throat']:
        if layer:
            indexinfo = np.concatenate( (cluster_info[item +'.corner'].flatten(), cluster_info[item +'.layer'].flatten(), cluster_info[item +'.center']))
        else:
            indexinfo = np.concatenate( (cluster_info[item +'.corner'].flatten(), cluster_info[item +'.center']))
        cluster_info[item + '.index'] = np.unique(indexinfo)

    #Creating general list
    indexinfo = np.concatenate( (cluster_info['pore.index'], cluster_info['throat.index']) )
    cluster_info['index_list'] = np.sort(np.unique(indexinfo))

    #Using consecutive numbers as index
    newlist = np.arange(len(  cluster_info['index_list'] ))

    if layer:
        locations = ['corner', 'center', 'layer']
    else:
        locations = ['corner', 'center']

    for item in ['pore', 'throat']:
        for i in range( len(newlist) ):
            if np.isin(cluster_info['index_list'][i], cluster_info[item + '.index']):
                mask = cluster_info[item + '.index'] == cluster_info['index_list'][i]
                cluster_info[item + '.index'][ mask ] = newlist[i]
                for loc in locations:
                    mask = cluster_info[item + '.' + loc] == cluster_info['index_list'][i]
                    cluster_info[item + '.' + loc][ mask ] = newlist[i]
    cluster_info['index_list'] = newlist
    return cluster_info

def cluster_advanced_WP_CLC(network, phase_locations):
    r"""
    Set index clusters to all phase locations for the wetting phase (wp) according to its presence and location,
    building a general connection list for pores and throat.
    First create clusters considering the corner locations (if all elements has at least one corner with the wp, we have one large cluster with index 0)
    Then , we set cluster index considering only the center and layer locations. For each cluster found, if at least one elements is saturated with the wp, this cluster has index 0.

    If an element has the wetting phase, it has to be, at least, on the corner

    Connections accepted:
    corner-corner
    layer-layer
    layer-center
    center-center
    Determina los tipos de cluster revisando la coneccion de poros y gargamtas y la ubicación de las fases.
    Hago una superlista de poros y gargantas
    Parameters:
    ------------------
    pn: Network
    phase_locations: main dictionary with the secondary dictionarys pore, throat; and the keywords center, corner, layer. True for all phase locations with the wp
    index_start: First index used to set the clusters that are not part of the large cluster

    Returns:
    -------------------
    cluster_info: dictionary with index information. Have the dictionaries 'pore', 'throat',
    with the keywords 'center', 'corner', 'layer', 'index'
    """
    elements = ['pore', 'throat']
    locations = ['center', 'layer', 'corner']
    conns = network['throat.conns']
    Np = network.Np
    Nt = network.Nt
    mask_wp_co_array = np.zeros(Np+Nt, dtype = bool)
    mask_wp_la_array = np.zeros(Np+Nt, dtype = bool)
    mask_wp_ce_array = np.zeros(Np+Nt, dtype = bool)
    for t in range(Nt):
        mask_wp_co_array[t] = np.any(phase_locations['throat.corner'][t,:])
        mask_wp_la_array[t] = np.any(phase_locations['throat.layer'][t,:])
        mask_wp_ce_array[t] = phase_locations['throat.center'][t]
    for p in range(Np):
        mask_wp_co_array[Nt + p] = np.any(phase_locations['pore.corner'][p,:])
        mask_wp_la_array[Nt + p] = np.any(phase_locations['pore.layer'][p,:])
        mask_wp_ce_array[Nt + p] = phase_locations['pore.center'][p]
    mask_connected_wp = mask_wp_co_array & mask_wp_la_array #conexion ce-la-co
    #Matriz of (Np + Nt) x (Np + Nt). gargantas de 0 a Nt-1. poros de Nt a Np-1
    connections = np.zeros((Np+Nt, Np+Nt), dtype = bool)
    #Averiguando conecciones de esquinas
    for t in range(Nt):
        #Revisando la presencia de la fase en gargantas
        if mask_wp_co_array[t]:
            #Evaluando presencia de la fase en poros vecino:
            for cn in [0,1]:
                p = conns[t,cn]
                #Estableciendo conexion (Tabla 3.4 disertacion)
                if mask_wp_co_array[Nt + p]:
                    connections[t, Nt + p] = True
    connections = sprs.csr_matrix(connections)
    clusters = csgraph.connected_components(csgraph=connections, directed=False, return_labels=True)[1] * -1 #To have negative values
    clusters[~mask_wp_co_array] = 1
    corner_index = -1
    for index in np.unique(clusters[clusters < 1]):
        mask = clusters == index
        clusters[mask] = corner_index
        corner_index -= 1
    center_index = np.min(clusters) - 1
    #La fase de las esquinas forma un unico clusters
    #Identificando clusters para fases centro y layer
    connections = np.zeros((Np+Nt, Np+Nt), dtype = bool)
    for t in range(Nt):
        #Revisando la presencia de la fase en gargantas
        if mask_wp_ce_array[t]:
            #Evaluando presencia de la fase en poros vecino:
            for cn in [0,1]:
                p = conns[t,cn]
                #Estableciendo conexion (Tabla 3.4 disertacion)
                if mask_wp_ce_array[Nt + p]:
                    connections[t, Nt + p] = True
    connections = sprs.csr_matrix(connections)
    clusters_ce = csgraph.connected_components(csgraph=connections, directed=False, return_labels=True)[1] * -1 - 1 + center_index
    #Identifying elements filled completely with wp
    clusters_ce[~mask_wp_ce_array] = 1
    bool_all_connected = True
    for index in np.unique(clusters_ce[clusters_ce < 1]):
        mask = clusters_ce == index
        if np.any(mask_connected_wp[mask]):
            clusters_ce[mask] = clusters[mask]
        else:
            clusters_ce[mask] = center_index
            center_index -= 1
    if not bool_all_connected:
        raise Exception('Center phase separada de otros elementos')
    throat_clusters = clusters_ce[0:Nt]
    pore_clusters = clusters_ce[Nt:(Nt+Np)]
    #Adding information
    cluster_info = {}
    for item in elements:
        if item =='pore':
            N = Np
            cluster_index_co = clusters[Nt:(Nt+Np)]
            cluster_index_ce = clusters_ce[Nt:(Nt+Np)]
        else:
            N = Nt
            cluster_index_co = clusters[0:Nt]
            cluster_index_ce = clusters_ce[0:Nt]
        cluster_info[f'{item}.center'] = np.ones(N, dtype = int)
        cluster_info[f'{item}.center'][phase_locations[f'{item}.center']] = cluster_index_ce[phase_locations[f'{item}.center']]
        cluster_info[f'{item}.layer'] = np.ones((N,3), dtype = int)
        cluster_info[f'{item}.layer'][phase_locations[f'{item}.layer']] = np.tile(cluster_index_ce,(3,1)).T [phase_locations[f'{item}.layer']]
        cluster_info[f'{item}.corner'] = np.ones((N,3), dtype = int)
        cluster_info[f'{item}.corner'][phase_locations[f'{item}.corner']] = np.tile(cluster_index_co,(3,1)).T [phase_locations[f'{item}.corner']]
    cluster_info['index_list'] = np.unique(np.concatenate((clusters,clusters_ce)))

    return cluster_info



    #Aplicando el algoritmo scipy de cluster
    """
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
    if layer:
        cluster_info['pore.layer'] = np.tile(pore_clusters,(3,1)).T * phase_locations['pore.layer']
        cluster_info['throat.layer'] = np.tile(throat_clusters,(3,1)).T * phase_locations['throat.layer']


    #Creating index list per element
    for item  in ['pore', 'throat']:
        if layer:
            indexinfo = np.concatenate( (cluster_info[item +'.corner'].flatten(), cluster_info[item +'.layer'].flatten(), cluster_info[item +'.center']))
        else:
            indexinfo = np.concatenate( (cluster_info[item +'.corner'].flatten(), cluster_info[item +'.center']))
        cluster_info[item + '.index'] = np.unique(indexinfo)

    #Creating general list
    indexinfo = np.concatenate( (cluster_info['pore.index'], cluster_info['throat.index']) )
    cluster_info['index_list'] = np.sort(np.unique(indexinfo))

    #Using consecutive numbers as index
    newlist = np.arange(len(  cluster_info['index_list'] ))

    if layer:
        locations = ['corner', 'center', 'layer']
    else:
        locations = ['corner', 'center']

    for item in ['pore', 'throat']:
        for i in range( len(newlist) ):
            if np.isin(cluster_info['index_list'][i], cluster_info[item + '.index']):
                mask = cluster_info[item + '.index'] == cluster_info['index_list'][i]
                cluster_info[item + '.index'][ mask ] = newlist[i]
                for loc in locations:
                    mask = cluster_info[item + '.' + loc] == cluster_info['index_list'][i]
                    cluster_info[item + '.' + loc][ mask ] = newlist[i]
    cluster_info['index_list'] = newlist
    """


def make_contiguous_clusters(network, invaded_throats, inlet_pores = None):
    r"""
    Relates the central phase of each element to a cluster index that belongs to a contiguous
    range starting at 1. Non-invaded elements have a value of "0" (possible if theta_r <= pi/3).
    _perctools.find_cluster() assumes that an invaded throat has as "invaded" its two connected pores,
    which is possible on a primary drainage invasion

    Parameters:
    ------------------
    network: Network object
    invaded_throats: Boolean array, Nt-sized, with the throats with non wetting phases on the center
    inlet_pores: Boolean array, Np-sized. If given, inlet pores that are not connected to an invaded throat has a unique cluster index
    """

    clusters = op.topotools._perctools.find_clusters(network, invaded_throats)
    c_pores = np.copy(clusters.pore_labels)
    c_throats = np.copy(clusters.throat_labels)
    cluster_list = np.union1d(clusters.pore_labels, clusters.throat_labels)
    if np.any(cluster_list == -1):
        index = 0
    else:
        index = 1
    for i in cluster_list:
        c_pores[clusters.pore_labels == i] = index
        c_throats[clusters.throat_labels == i] = index
        index += 1

    if inlet_pores is not None:
        if inlet_pores.dtype != bool:
            raise Exception('Inlet pores must be a boolean array of Np length')
        if np.size(inlet_pores) == network.Np:
            trapped_pores = np.where(inlet_pores * (c_pores == 0))[0]#[i for i in range(network.Np) if (inlet_pores * (c_pores == 0))]
            if len(trapped_pores) > 0:
                for i in trapped_pores:
                    c_pores[i] = index
                    index += 1
        else:
            raise Exception('Inlet pores must be a boolean array of Np length')
    clusters = namedtuple('clusters', ['pore_clusters', 'throat_clusters'])
    return clusters(c_pores, c_throats)

def pc_theta_a(pc_max,
               theta_a,
               theta_r,
               beta_i):
    r"""
    Used to calculate the capillary pressure at which the contact angle reaches its maximum value

    Parameters:
    ---------------------
    pc_max: Maximum capillary pressure, or maximum delta_p between the inlet and outlet pores
    theta_a: advancing contact angle. Maximum value for the contact angle. Can be a number or 1D
    theta_r: receiding contact angle. Minimum value for the contact angle. Can be a number or 1D
    beta_i: half corner angle(s). Can be a number or a 1D, 2D array
    """

    #if beta_i is a 2D and theta_a, theta_r are 1D, expand theta:
    if isinstance(beta_i, np.ndarray):
        if beta_i.ndim == 2:
            n = len(beta_i[0])
            if isinstance(theta_a, np.ndarray):
                theta_a = np.tile(theta_a, (n,1)).T
            if isinstance(theta_r, np.ndarray):
                theta_r = np.tile(theta_r, (n,1)).T
    pc = pc_max * np.cos(theta_a + beta_i) / np.cos(theta_r + beta_i)
    return pc

def pressure_PLD(beta,
                 interface,
                 b,
                 sigma,
                 theta_a,
                 A,
                 G,
                 d,
                 p_max = 1e10,
                 theta_r = 0,
                 tol = 0.1,
                 max_it = 10):
    r"""
    Calculate entry capillary pressure of one element assuming piston like displacement and a triangular cross section.
    First we assume that b_i is fixed. If any angle theta > theta_a, we start to calculate the correct b_i
    The uknown variables are 7: r_imb, A_eff, L_nws, L_nww and up to three theta_hi and three bi (if the element has 3 AMs)
    NOTES: For now we are going to assume that at least one interface exist

    Parameters:
    --------------------
    beta: half corner angles. Array with 3 elements
    interface: boolean array. Indicates if an interface exist in each corner. True if exist. Same size as beta_i
    b: distance between interface and corner, assuming the contact angle is hinging. Same size as beta_i
    sigma: surface tension. Float
    theta_a: advancing contact angle. Float.
    A: cross sectional area
    G: Shape factor
    d = inscribed diameter
    """
    r = d / 2
    n = np.sum(interface) #How many interfaces exist
    index_corner = np.where(interface)[0]
    p = p_max
    p_old = p_max

    j = 0

    while j < max_it:
        S1 = 0
        S2 = 0
        S3 = 0
        r_imb = sigma / p
        for i in range(n):
            beta_i = beta[index_corner[i]]
            theta_h_i = np.arccos(p * np.cos(theta_r + beta_i) / p_max) - beta_i
            if theta_h_i > theta_a:
                theta_h_i = theta_a
                b_i = r_imb * np.cos(theta_a + beta_i) / np.sin(beta_i)
            else:
                b_i = b[index_corner[i]]
            S1 = S1 + np.cos(theta_h_i) * np.cos(theta_h_i + beta_i) / np.sin(beta_i) + theta_h_i + beta_i - np.pi / 2
            S2 = S2 + b_i
            S3 = S3 + np.arcsin( b_i * np.sin(beta_i) / r_imb)
        #Eq for A_eff
        A_eff = A - r_imb ** 2 * S1
        #Eq for L_nws
        L_nws = r / (2*G) - 2 * S2
        #Eq for L_nww
        L_nww = 2 * r_imb * S3

        p = sigma * (L_nww + L_nws * np.cos(theta_a)) / A_eff
        err = abs(p - p_old)
        p_old = p
        j += 1
        if err < tol:
            break
        elif j == max_it:
            raise Exception('Maximum number of iterations (%i) reached. Pressure can not be calculated' % max_it)

    return p, j


def pressure_PB(n_inv_t,
                sigma,
                theta_a,
                d,
                perm = 3.70e-12,
                par_value = None
                ):
    r"""
    Calculate entry capillary pressure of elements assuming pore body filling, used for all pores.
    For the coefficientes c_i, we are going to use the criteria of Blunt (1998), cited in Blunt's Book
    c_i = 1 / average(r_t), except that c_o0 = 0

    Parameters:
    --------------------
    n_inv_t: number of connected invaded throats, with the non wetting phase at the center
    sigma: surface tension. Float
    theta_a: advancing contact angle. Float.
    d: pore inbscribed diameter
    perm: network permeability, in m2. Default: 3.70e-12 (Value of a Berea sandstone)
    par_values (optional): The number used for the arbitrary parameters, if permeability is not available/desired.

    Reference:
    --------------------
    Blunt (2004) : Predictive pore-scale modeling of two-phase flow in mixed wet media
    Blunt (1998) : Physically-based network modeling of multiphase flow in intermediate-wet porous media

    Note:
    -------------------
    Blunt (1998) explain that using c_i = 0.015 um-1 for a Berea sandstone is adequate
    Then, Blunt (2004) propose the method to calculate the coefficients c_i using the absolute permeability.
    The value perm = 3.70e-12 used in the formula proposed by Blunt (2004) returns a value of c_i similar to the
    one obtained by Blunt (1998).

    """
    if isinstance(n_inv_t, (int, float)):
        n_inv_t = np.array([n_inv_t])
    Np = len(n_inv_t)
    r = d / 2
    max_t = np.max(n_inv_t)
    x_i = np.random.rand(Np, max_t)
    c_i = np.ones_like(x_i)
    c_i[:, 0] = 0
    mask = np.zeros_like(x_i)
    for i in range(Np):
        mask[i, : n_inv_t[i]] = 1
    if par_value is None:
        c_i = c_i * 0.03 / perm ** 0.5
    else:
        c_i = c_i * par_value
    p = 2 * sigma * np.cos(theta_a) / r - sigma * np.sum(x_i * c_i * mask, axis = 1)
    return p

def pressure_PB1(n_inv_t,
                sigma,
                theta_a,
                d,
                perm = 3.70e-12,
                par_value = None
                ):
    r"""
    Calculate entry capillary pressure of ONLY ONE ELEMENT assuming pore body filling, used for all pores.
    For the coefficientes c_i, we are going to use the criteria of Blunt (1998), cited in Blunt's Book
    c_i = 1 / average(r_t), except that c_o0 = 0

    Parameters:
    --------------------
    n_inv_t: number of connected invaded throats, with the non wetting phase at the center
    sigma: surface tension. Float
    theta_a: advancing contact angle. Float.
    d: pore inbscribed diameter
    perm: network permeability, in m2. Default: 3.70e-12 (Value of a Berea sandstone)
    par_values (optional): The number used for the arbitrary parameters, if permeability is not available/desired.

    Reference:
    --------------------
    Blunt (2004) : Predictive pore-scale modeling of two-phase flow in mixed wet media
    Blunt (1998) : Physically-based network modeling of multiphase flow in intermediate-wet porous media

    Note:
    -------------------
    Blunt (1998) explain that using c_i = 0.015 um-1 for a Berea sandstone is adequate
    Then, Blunt (2004) propose the method to calculate the coefficients c_i using the absolute permeability.
    The value perm = 3.70e-12 used in the formula proposed by Blunt (2004) returns a value of c_i similar to the
    one obtained by Blunt (1998).

    """
    r = d / 2
    x_i = np.random.rand(n_inv_t)
    if par_value is None:
        if n_inv_t <= 1:
            c_i = 0
        else:
            c_i = np.ones_like(x_i) * 0.03 / perm ** 0.5
            c_i[0] = 0
    else:
        c_i = par_value
    p = 2 * sigma * np.cos(theta_a) / r - sigma * np.sum(x_i * c_i)
    return p


def pressure_snapoff(beta,
                     interface,
                     sigma,
                     theta_r,
                     theta_a,
                     d,
                     pc_max,
                     max_it = 10,
                     tol =  0.1):
    r"""
    Calculate entry capillary pressure of a list of elements (pores or throats) assuming snap off and a triangular cross section.
    Required to have at least two AM.
    Function used with corner-center AM
    Assumes the sharpest corner is in the first column


    Parameters:
    --------------------
    beta: half corner angles. Array with 3 elements
    interface: boolean array. Indicates if an interface exist in each corner. True if exist. Same size as beta_i
    b: distance between interface and corner, assuming the contact angle is hinging. Same size as beta_i
    sigma: surface tension. Float
    theta_a: advancing contact angle. Float.
    A: cross sectional area
    G: Shape factor
    d = throat diameter

    Reference:
    --------------------
    Blunt (2004) : Predictive pore-scale modeling of two-phase flow in mixed wet media
    """
    if beta.ndim == 1:
        beta = np.array([beta])
    if interface.ndim == 1:
        interface = np.array([interface])

    r = d / 2
    n = np.sum(interface, axis = 1) #How many interfaces exist

    mask_1 = (n >= 2) #NO USADO PERO PUEDE SERVIR PARA EL CONDICIONAL
    index_corner = np.where(interface)[0]
    beta_1 = beta[:,0]
    beta_2 = beta[:,1]
    beta_3 = beta[:,2]
    #If n = 2, beta_3 does not have an interface. Therefore, beta_2 is used
    beta_3[n == 2] = beta_2[n == 2]
    mask_forced = (theta_a >= ( np.pi / 2 - beta_1) )

    #Pc to reach theta_a in the sharpest corner
    p1 = pc_max * np.cos(np.minimum(theta_a + beta_1, np.pi)) / np.cos(theta_r + beta_1)

    #PC spontaneous, only one interface (beta_1) is moving
    # Iteration required to calculate p2 and theta_h3
    p2 = np.ones_like(r) * pc_max
    p2_old = np.copy(p2)
    i = 0

    while i < max_it:
        theta_h3 = np.minimum(np.arccos(p2 / pc_max * np.cos(theta_r + beta_3)) - beta_3, theta_a)
        theta_h2 = np.minimum(np.arccos(p2 / pc_max * np.cos(theta_r + beta_2)) - beta_2, theta_a)
        p2 = sigma / r * (np.cos(theta_a) / np.tan(beta_1) - np.sin(theta_a) + np.cos(theta_h3) / np.tan(beta_3) - np.sin(theta_h3)) / (1 / np.tan(beta_1) + 1 / np.tan(beta_2))
        err = np.abs(p2 - p2_old)
        p2_old = p2
        i += 1
        if  np.all(err < tol):
            break
        elif i == max_it:
            print(p2)
            print(err)
            raise Exception('Maximum number of iterations (%i) reached. Pressure can not be calculated' % max_it)

    #Pc spontaneous, two or more interfaces ar moving
    p3 = sigma / r * (np.cos(theta_a) - 2 * np.sin(theta_a) / (1 / np.tan(beta_1) + np.tan(beta_2)))

    p = p1 * mask_forced + np.maximum(p2, p3) * ~mask_forced
    return p

def pressure_snapoff1(beta,
                     interface,
                     sigma,
                     theta_r,
                     theta_a,
                     d,
                     pc_max,
                     max_it = 20,
                     tol =  0.1):
    r"""
    Calculate entry capillary pressure of ONE element (pore or throat) assuming snap off and a triangular cross section.
    Required to have at least two AM.
    Function used with corner-center AM

    Parameters:
    --------------------
    beta: half corner angles. Array with 3 elements
    interface: boolean array. Indicates if an interface exist in each corner. True if exist. Same size as beta_i
    FALTA HACER CONDICIONAL PARA CUANDO SOLOO TUVIERA 1 INTERFACE
    b: distance between interface and corner, assuming the contact angle is hinging. Same size as beta_i
    sigma: surface tension. Float
    theta_a: advancing contact angle. Float.
    A: cross sectional area
    G: Shape factor
    d = throat diameter

    Reference:
    --------------------
    Blunt (2004) : Predictive pore-scale modeling of two-phase flow in mixed wet media
    """
    n = np.sum(interface) #How many interfaces exist
    if n < 2:
        #Solo hay una interfase,
        return -np.inf

    r = d / 2
    beta_1 = beta[0]
    beta_2 = beta[1]
    #If n = 2, beta_3 does not have an interface. Therefore, beta_2 is used
    if n == 2:
        beta_3 = beta_2
    else:
        beta_3 = beta[2]

    if theta_a >= ( np.pi / 2 - beta_1):
        #Pc to reach theta_a in the sharpest corner (Useful if its forced)
        p = pc_max * np.cos(np.min( (theta_a + beta_1, np.pi) )) / np.cos(theta_r + beta_1)
    else:
        #PC spontaneous
        p2 = p2_old = pc_max
        i = 0
        while i < max_it:
            theta_h3 = min(np.arccos(p2 / pc_max * np.cos(theta_r + beta_3)) - beta_3, theta_a)
            theta_h2 = min(np.arccos(p2 / pc_max * np.cos(theta_r + beta_2)) - beta_2, theta_a)
            p2 = sigma / r * (np.cos(theta_a) / np.tan(beta_1) - np.sin(theta_a) + np.cos(theta_h3) / np.tan(beta_3) - np.sin(theta_h3)) / (1 / np.tan(beta_1) + 1 / np.tan(beta_3))
            err = abs(p2 - p2_old)
            p2_old = p2
            i += 1
            if  np.all(err < tol):
                break
            elif i == max_it:
                print(err)
                print(tol)
                raise Exception('Maximum number of iterations (%i) reached. Pressure can not be calculated' % max_it)

        #two or more interfaces ar moving
        p3 = sigma / r * (np.cos(theta_a) - 2 * np.sin(theta_a) / (1 / np.tan(beta_1) + 1 / np.tan(beta_2)))

        #Choosing  the best option
        p = max(p2, p3)
    return p

def nwp_in_layers(beta, theta_r, theta_a):
    r"""
    Function that allows us to know if the corners of a triangle
    can contain a sandwiched layer with the non wetting phase (True)
    The center and corner phases are occupied by the wp
    beta: Array Nx3
    theta_r, theta_a: Array Nx3 or number
    """
    condition = (beta < (np.pi / 2 - theta_r)) & (beta < (theta_a - np.pi / 2))
    return condition

def pressure_LC(beta,
                bi,
                sigma,
                theta_a,
                mask = None):
    r"""
    Method to calculate the pressure for a layer to collapse
    If bi == 0, its impossible to have a layer

    Parameters:
    --------------
    beta: half corner angles
    bi: Inner interface distance
    sigma: interfacial tension
    theta_a: advancing contact angle. Number of an array like beta
    mask: corners where the layers exist

    Return:
    ------------------
    p: presure for a layer to collapse
    """
    if mask is None:
        mask = bi != 0
    else:
        mask = mask & (bi != 0)
    if isinstance(theta_a, (int, float)):
        theta_a = np.ones_like(beta) * theta_a
    p = np.ones_like(beta) * -np.inf #If no layer, positive inf to be reported as breaked on imbibition´
    p[mask] = sigma / bi[mask] * np.cos( np.arccos( (2 * np.sin(beta) + np.cos(theta_a) )[mask] ) + beta[mask] ) / np.sin(beta[mask])
    return p


def verify_LF(beta,
              beta_max,
              bi_beta_max,
              R_inscribed,
              sigma,
              theta_a,
              pressure):
    r"""
    Verifying, geometrically, if the layer formation can be formed after a MTM displacement.
    Calculated for only one corner. Assuming triangular cross section
    True if is possible

    Parameters:
    --------------
    beta: the half corner angle of the corner that we are verified
    beta_max: maximum half corner angle
    bi_beta_max: interface distance related to the corner with the maximum half corner angle
    R_inscribed: cross sectional inscribed radius
    sigma: interfacial tension
    theta_a: advancing contact angle.
    pressure: actual capillary pressure

    Return:
    ------------------
    boolean result. True if a layer can be created
    """
    R = R_inscribed
    a = R *(1/np.tan(beta) + 1/np.tan(beta_max))
    bo = sigma / pressure * np.cos(theta_a - beta) / np.sin(beta)
    return (bo + bi_beta_max) < a
