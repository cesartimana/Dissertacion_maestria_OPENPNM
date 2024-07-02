import logging
import numpy as _np
from openpnm.models import _doctxt
import math as _m
import openpnm as op
from collections import namedtuple

logger = logging.getLogger(__name__)


__all__ = [
    "MSP_prim_drainage",
]


#En las funciones creadas antes, se colocaba en una linea anterior
#lo siguiente: @_doctxt
#Esto lo unico que hace es substituir %(nombre)s por el texto del archivo _doctxt



def MSP_prim_drainage(phase,
	     throat_diameter = 'throat.diameter'):
    r"""
    Computes the capillary entry pressure of all throats.

    Parameters
    ----------
    phase: The invaded/wetting one. The network associted with must have these properties:
        -throat.surface_tension
        -throat.contact_angle
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
    if phase["throat.contact_angle"].ndim == 1:
          theta = phase["throat.contact_angle"]
    else:
          theta = phase["throat.contact_angle"][:,0]
    theta_r =    _np.tile(theta,(3,1)).T #matrix Nt x 3. All columns are the same
    r = network[throat_diameter] / 2
    G = network["throat.shape_factor"]
    beta = network["throat.half_corner_angle"]
    condition = wp_in_corners(beta, theta)  #just these angles are valid for the addition
    S1 = _np.sum((_np.cos(theta_r) * _np.cos(theta_r + beta) / _np.sin(beta) + theta_r + beta - _m.pi/2) * condition, axis = 1)
    S2 = _np.sum((_np.cos(theta_r + beta) / _np.sin(beta)) * condition, axis = 1)
    S3 = 2*_np.sum((_m.pi / 2 - theta_r - beta) * condition, axis = 1)
    D = S1 - 2 * S2 * _np.cos(theta) + S3
    value = sigma * (1 + _np.sqrt(1 + 4 * G * D / _np.cos(theta)**2)) / r
    return value

def wp_in_corners(beta, theta):
    r"""
    Function that allows us to know if the corners of a triangle
    can contain the wetting phase (True)
    beta: Array Nx3
    theta: Array Nx1
    """
    theta =    _np.tile(theta,(3,1)).T
    condition = (beta < (_m.pi / 2 - theta))

    return condition

def interfacial_corner_distance(R, theta, beta, int_cond = True):
    r"""
    Calculate the distance between each corner of the triangular cross section and the location of the solid-phase 1 - phase 2 interface.

    Parameters:
    -R: Radius of curvature of the interface. Equivalent to tension/ capillary_pressure
    -theta: Array with the contact angles at each corner.
    -beta: Half corner angle.
    -int_cond: interface condition for each corner. True or 3-columns boolean array.
     If False, there is no interface between center and corner, and bi = 0 for those corners

    """
    if _m.isinf(R):
        #Impossible to calculate for geometry
        bi = _np.zeros_like(beta)
    else:
        bi = R * _np.cos(theta + beta) / _np.sin(beta)
        bi[bi < 0] = 0
        if int_cond.all():
            pass
        else:
            bi[~int_cond] = 0
    return bi


def identify_continuous_cluster(cluster_pore_center , cluster_pore_corner, inlet_pores, outlet_pores):
    r"""
    This function returns a list of cluster indices that have a continuity.
    That is, the phase can enter and exit the porous medium.
    Parameters:
    cluster_pore_center / corner: lists of cluster indexes per pore
    inlet/outlet_pores: boundary conditions


    Return a list of clusters
    """

    cluster_list = _np.unique(_np.hstack((cluster_pore_center, cluster_pore_center)))
    continuous_list = []
    for i  in cluster_list:
        cond_inlet = (((cluster_pore_center == i) | _np.any(cluster_pore_corner == i, axis = 1))[inlet_pores]).any()
        cond_outlet = (((cluster_pore_center == i) | _np.any(cluster_pore_corner == i, axis = 1))[outlet_pores]).any()
        if cond_inlet and cond_outlet:
            continuous_list.append(i)
    return continuous_list

def corner_area(beta, theta, bi):
    r"""
    Calculation of corner area
    """
    S1 = (bi * _np.sin(beta) / (_np.cos(theta + beta)))
    S2 = (_np.cos(theta) * _np.cos(theta + beta) / _np.sin(beta) + theta + beta - _m.pi / 2  )
    return S1 ** 2 * S2

def make_contiguous_clusters(network, invaded_throats, inlet_pores = None):
    r"""
    Relates the central phase of each element to a cluster index that belongs to a contiguous
    range starting at 1. Non-invaded elements have a value of "0".
    _perctools.find_cluster() assumes that an invaded throat has as "invaded" its two connected pores,
    which is possible on a primary drainage invasion

    """

    clusters = op.topotools._perctools.find_clusters(network, invaded_throats)
    c_pores = _np.copy(clusters.pore_labels)
    c_throats = _np.copy(clusters.throat_labels)
    cluster_list = _np.union1d(clusters.pore_labels, clusters.throat_labels)
    if _np.any(cluster_list == -1):
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
            trapped_pores = _np.where(inlet_pores * (c_pores == 0))[0]#[i for i in range(network.Np) if (inlet_pores * (c_pores == 0))]
            if len(trapped_pores) > 0:
                for i in trapped_pores:
                    c_pores[i] = index
                    index += 1
        else:
            raise Exception('Inlet pores must be a boolean array of Np length')
    clusters = namedtuple('clusters', ['pore_clusters', 'throat_clusters'])
    return clusters(c_pores, c_throats)
