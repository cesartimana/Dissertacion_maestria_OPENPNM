import logging
import numpy as _np
from openpnm.models import _doctxt
import collections.abc
import math as _m

logger = logging.getLogger(__name__)


__all__ = [
    "conductance",
]


#En las funciones creadas antes, se colocaba en una linea anterior
#lo siguiente: @_doctxt
#Esto lo unico que hace es substituir %(nombre)s por el texto del archivo _doctxt

def corner_area(beta, theta, bi):
    r"""
    Calculation of corner area
    """
    S1 = (bi * _np.sin(beta) / (_np.cos(theta + beta)))
    S2 = (_np.cos(theta) * _np.cos(theta + beta) / _np.sin(beta) + theta + beta - _m.pi / 2  )
    return S1 ** 2 * S2

def conductance(
    network,
    status_center,
    status_corner,
    theta_corner,
    bi_corner,
    viscosity,
    item = None, ):
  r"""
  The conductance per unit length g/L for a conduit is calculated for the phase volume
  corresponding to a cluster.
  The procedure is the next:

  The conductance is calculated for the "center" and "corner" locations.
    -If there is the same phase between a corner and the center, the conductance for
    that corner is not calculated and center conductance consider that corner
    -For multiple corners with the same phase, g/L (total) = sum(g)/L
  Then the conductance is calculated per conduite pore - throat - pore, with each pore
  contributing half of its conductance value to the conduite.
  It is necessary that the 3 elements contain the evaluated cluster to calculate the conductance.
  Otherwise, the value is zero.
  At the moment it does not consider the appearance of layers between center and corner locations.
  Reference: Valvatne & Blunt (2004)

  Parameters:
  network: Network object
  status... : Boolean arrays that indicate whether said location contains a phase that belongs
  to the cluster to be analyzed (True). There are 4:
  status_center: Boolean. If True, the phase is present at the center
  status_corner: Boolean with 3 columns, one per corner. If True, the phase is present in that corner
  theta_corner: contact angle in each corner. Same lenght as status_corner
  bi_corner: distance between the corner and the interfase. If there is not interfase, 0. Same lenght as status_corner
  viscosity: Float or int value

  Note
  Corner area calculates independento for the pore area. So a checking must be done. WE assume that corner doesnt exist, and all is center

  """
  #Checking data
  if (item == 'pore') or (item == 'throat'):
    pass
  else:
    raise Exception('item must be a str: pore or throat')

  if item == 'pore':
    N = network.Np
  if item == 'throat':
    N = network.Nt
  if (len(status_center) == N) and (len(status_corner) == N) and (status_center.dtype) == 'bool' and (status_corner.dtype) == 'bool':
    pass
  else:
    raise Exception('status_center and status_corner must be boolean arrays with a 1D lenght equal to the number of ' +  item + 's')

  #Calculating center and corner area.

  beta = network[f'{item}.half_corner_angle']

  area = network[f'{item}.cross_sectional_area']
  G = network[f'{item}.shape_factor']
  A_corner = corner_area(beta, theta_corner, bi_corner)
  #checking corner area
  mask = _np.sum(A_corner, axis = 1) > area
  mask = _np.tile(mask,(3,1)).T
  A_corner[mask] = 0

  #Calculating Area form center
  A_center = area - _np.sum(A_corner, axis = 1)

  #Center Conductance

  cond_center = 3/5 * A_center ** 2 * G / viscosity * status_center

  #Corner conductance
  bi = _np.copy(bi_corner)
  #Corner perimeter. bi cannot be zero due to division problems. Using err = min bi * 1e-10
  err = 1e-25
  bi[bi_corner == 0 ] = err
  P_corner = 2 * bi  * (1 - _np.sin(beta) / _np.cos(beta + theta_corner) * (theta_corner + beta - _m.pi / 2))
  A_corner[A_corner == 0] = err
  G_corner = A_corner  / P_corner ** 2
  G_mod = _np.sin(beta) * _np.cos(beta) / (2 + 2 * _np.sin(beta))**2
  C = 0.364 + 0.28 * G_mod / G_corner
  cond_corner = _np.zeros_like(bi)
  cond_corner[status_corner] = (C * A_corner ** 2 * G_corner)[status_corner]
  cond_corner[bi_corner == 0] = 0
  cond_corner = _np.sum(cond_corner, axis = 1) / viscosity
  ratio_phase = (A_center*status_center + _np.sum(A_corner* status_corner, axis = 1)) / area



  return cond_center, cond_corner, ratio_phase

def conduit_lenght_tubes(
    network,
    pore_length = "pore.length",
    throat_spacing = "throat.spacing"):

    r"""
    Calculates conduit lengths in the network assuming pores and throats as
    tubes.

    A conduit is defined as ( 1/2 pore - full throat - 1/2 pore ).

    Parameters
    ----------
    network: Network object
    pore_length
    throat_spacing: distance considered  between the two pore centers

    Returns
    -------
    lengths : ndarray
        Array (Nt by 3) containing conduit values for each element
        of the pore-throat-pore conduits. The array is formatted as
        ``[pore1, throat, pore2]``.

    """

    P12 = network["throat.conns"]
    L1 = network[pore_length][P12[:, 0]] / 2
    L2 = network[pore_length][P12[:, 1]] / 2
    L_ctc = network[throat_spacing]
    # Handle throats w/ overlapping pores
    #Distributing the lengths in proportion to the pore length
    _L1 = L1/(L1 + L2) * L_ctc
    mask = L_ctc - (L1 + L2) < 0
    L1[mask] = _L1[mask]
    L2[mask] = (L_ctc - L1)[mask]
    Lt = _np.maximum(L_ctc - (L1 + L2), 1e-15)
    return _np.vstack((L1, Lt, L2)).T


def conduit_conductance_2phases(
    network,
    pore_conductance_center,
    throat_conductance_center,
    conduit_length,
    pore_conductance_corner = None,
    throat_conductance_corner = None,
    pore_conductance_layer = None,
    throat_conductance_layer = None,
    corner = False,
    layer = False,
    ):
    r"""
    This function allows you to calculate the conductivity of the porous medium for each conduit.
    A conduite is made up of 1/2 pore - 1 throat - 1/2 pore.
    It is necessary to indicate whether phase is allowed in the corners and layers.

    Parameters:
    -------------

    network: Network object
    conduit_length:Nt x 3 array. The first and last column represent the length of half the pore length.
    The middle column represent the length of all the throat.

    Reference: Valvatne and Blunt(2004)
    """

    Np = network.Np
    Nt = network.Nt
    P1 = network['throat.conns'][:,0]
    P2 = network['throat.conns'][:,1]

    #Check conductance in center
    if (len(pore_conductance_center) == Np) and  (len(throat_conductance_center) == Nt):
        pass
    else:
        raise Exception('pore_conductance_center and throat_conductance_center must be defined as arrays witha lenght of Np and Nt respectively')

    #Check conduit length:
    if (len(conduit_length) == Nt) and  (len(conduit_length[0]) == 3):
        pass
    else:
        raise Exception('shape of conduit length must be (Nt,3)')


    if corner:
        #Check conductance in corner
        if (len(pore_conductance_corner) == Np) and  (len(throat_conductance_corner) == Nt):
            pass
        else:
            raise Exception('pore_conductance_corner and throat_conductance_corner must be defined as arrays witha lenght of Np and Nt respectively')
        if layer:
            #Check conductance in layer
            if (len(pore_conductance_layer) == Np) and  (len(throat_conductance_layer) == Nt):
                pass
            else:
                raise Exception('pore_conductance_layer and throat_conductance_layer must be defined as arrays witha lenght of Np and Nt respectively')
    #Layers can only exist if corner is True
    if layer and not corner:
        raise Exception('Layers can only exist if corner is True')

    if not layer:
        pore_conductance_layer = _np.zeros_like(pore_conductance_center)
        throat_conductance_layer = _np.zeros_like(throat_conductance_center)
    if not corner:
        pore_conductance_corner = _np.zeros_like(pore_conductance_center)
        throat_conductance_corner = _np.zeros_like(throat_conductance_center)
    pore_conductance = pore_conductance_center + pore_conductance_corner  + pore_conductance_layer
    throat_conductance = throat_conductance_center + throat_conductance_corner  + throat_conductance_layer
    P1_cond = pore_conductance[P1]
    P2_cond = pore_conductance[P2]

    P12_cond = _np.vstack((P1_cond , throat_conductance, P2_cond)).T
    continuity = _np.all(P12_cond > 0, axis = 1)
    continuity_3 = _np.tile(continuity, (3,1)).T

    #Beacuse we need to divide by zero
    P12_cond[P12_cond == 0 ] = 1e-15
    g_L = _np.sum(conduit_length / P12_cond, axis = 1 )**(-1)
    g_L[~continuity] = 0
    return g_L


















    return 0

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

