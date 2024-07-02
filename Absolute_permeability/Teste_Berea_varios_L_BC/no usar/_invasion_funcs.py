import logging
import numpy as np
from openpnm.models import _doctxt
from scipy.optimize import fsolve
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

def wp_in_corners(beta, theta):
    r"""
    Function that allows us to know if the corners of a triangle
    can contain the wetting phase (True)
    beta: Array Nx3
    theta: Array Nx1
    """
    theta =    np.tile(theta,(3,1)).T
    condition = (beta < (np.pi / 2 - theta))

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
    if np.isinf(R):
        #Impossible to calculate for geometry
        bi = np.zeros_like(beta)
    else:
        bi = R * np.cos(theta + beta) / np.sin(beta)
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

    cluster_list = np.unique(np.hstack((cluster_pore_center, cluster_pore_center)))
    continuous_list = []
    for i  in cluster_list:
        cond_inlet = (((cluster_pore_center == i) | np.any(cluster_pore_corner == i, axis = 1))[inlet_pores]).any()
        cond_outlet = (((cluster_pore_center == i) | np.any(cluster_pore_corner == i, axis = 1))[outlet_pores]).any()
        if cond_inlet and cond_outlet:
            continuous_list.append(i)
    return continuous_list

def corner_area(beta, theta, bi):
    r"""
    Calculation of corner area
    """
    S1 = (bi * np.sin(beta) / (np.cos(theta + beta)))
    S2 = (np.cos(theta) * np.cos(theta + beta) / np.sin(beta) + theta + beta - np.pi / 2  )
    return S1 ** 2 * S2

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
    Calculate entry capillary pressure of an element assuming pore body filling, used for all pores.
    For the coefficientes c_i, we are going to use the criteria of Blunt (1998), cited in Blunt's Book
    c_i = 1 / average(r_t), except that c_o0 = 0

    Parameters:
    --------------------
    n_inv_t: number of connected invaded throats
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

    mask_1 = (n >= 2)
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
        raise Exception('Maximum number of iterations (%i) reached. Pressure can not be calculated' % max_it)

    #Pc spontaneous, two or more interfaces ar moving
    p3 = sigma / r * (np.cos(theta_a) - 2 * np.sin(theta_a) / (1 / np.tan(beta_1) + np.tan(beta_2)))

    p = p1 * mask_forced + np.maximum(p2, p3) * ~mask_forced
    return p
