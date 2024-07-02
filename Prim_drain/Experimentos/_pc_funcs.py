import logging
import numpy as _np
from openpnm.models import _doctxt
import math as _m

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
    Computes the capillary entry pressure of all throats

    Parameters
    ----------
    phase: The wetting one
    with the network properties/keys
    throat.surface_tension
    throat.contact_angle
    throat.diameter (the key name can be changed)
    throat.shape_factor


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
    S3 = _np.sum((_m.pi / 2 - theta_r - beta) * condition, axis = 1)
    D = S1 - 2 * S2 * _np.cos(theta) + S3
    C1 = 4 * G * D / _np.cos(theta)**2
    C1_bool = (G <= 10-3) & (theta<= _m.pi/180) #constrain
    C1 = C1*~C1_bool + (-1) * C1_bool
    value = sigma * _np.cos(theta) * (1 + _np.sqrt(1 + C1)) / r
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
    R: Radius of curvature of the interface. Equivalent to tension/ capillary_pressure
    theta: Array with the contact angles at each corner.
    beta: Half corner angle.
    int_cond: interface condition for each corner. True or 3-columns boolean array.
    If False, there is no interface between center and corner, and bi = 0 for those corners

    Note: Check if for R large or negative, the function works
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
