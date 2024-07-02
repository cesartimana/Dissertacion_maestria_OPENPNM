import logging
import numpy as np
from openpnm.models import _doctxt
import collections.abc
import math as _m
import _invasion_funcs as _if
import openpnm.models as opmods

logger = logging.getLogger(__name__)


__all__ = [
    "conductance",
]


#En las funciones creadas antes, se colocaba en una linea anterior
#lo siguiente: @_doctxt
#Esto lo unico que hace es substituir %(nombre)s por el texto del archivo _doctxt

def conductance(
    network,
    status_center,
    status_corner,
    theta_corner,
    bi_corner,
    viscosity,
    item = None,
    cross_sectional_area = 'cross_sectional_area',
    null_value = 1e-30,
    correction = False):
    r"""
    Conductance for one type of elements assuming triangles, JUST CENTER AND CORNER LOCATIONS
    Assumptions:
    -status center and status_corner are physically possible
    -bi_corner is geometricaly viable
    -All the locations one an element, that contains the same phase belongs to the same cluster


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
    Otherwise, the value is 1e-20 (not zero due to its use in flow calculation)
    At the moment it does not consider the appearance of layers between center and corner locations.
    Reference: Valvatne & Blunt (2004)

    Parameters:
    -----------
    -network: Network object
    -status... : Boolean arrays that indicate whether said location contains a phase that belongs
     to the cluster to be analyzed (True):
        -status_center: Boolean. If True, the phase is present at the center
        -status_corner: Boolean with 3 columns, one per corner. If True, the phase is present in that corner
    -theta_corner: contact angle in each corner. Same lenght as status_corner
    -bi_corner: distance between the corner and the interfase. If there is not interfase, 0. Same lenght as status_corner
    -viscosity: Float or int value
    -item: To choose between "pore" or "throat"
    -one_phase_corr: If True, modify the center conductance with a factor obtained with fitting with data from White(2002) for isosceles triangles

    Note
    Corner area calculates independent for the pore area. So a checking must be done. WE assume that corner doesnt exist, and all is center

    """
    #Checking data
    if item not in ['pore', 'throat']:
        raise Exception('item must be a str: pore or throat')

    if item == 'pore':
        N = network.Np
    if item == 'throat':
        N = network.Nt
    if (len(status_center) != N) or (len(status_corner) != N) or (status_center.dtype) != 'bool' or (status_corner.dtype) != 'bool':
        raise Exception('status_center and status_corner must be boolean arrays with a 1D lenght equal to the number of ' +  item + 's')

    #Picking data
    beta = network[f'{item}.half_corner_angle']
    area = network[f'{item}.' + cross_sectional_area]
    G = network[f'{item}.shape_factor']

    #Corner area independent from phase location
    A_corner = _if.corner_area(beta, theta_corner, bi_corner)

    #If corner and center share the same phase, A_corner = 0. A_center include those corners
    A_center = status_center * (area - np.sum(A_corner * ~status_corner, axis = 1) )
    A_corner = (A_corner * status_corner) * ~ np.tile(status_center,(3,1)).T

    if np.any(A_center < 0) or np.any(A_corner < 0):
        raise Exception ('Some areas are negative. Posible geometrial incongruence. Check bi, or theta')

    #If sum (A_corner) > A_center, there is only one phase #CHECK THAT ON IMBIBITION
    #mask = np.sum(A_corner * status_corner, axis = 1) > area
    #mask = np.tile(mask,(3,1)).T
    #A_corner[mask] = 0

    #Calculating Area form center
    #A_center = area - np.sum(A_corner, axis = 1)

    #Center Conductance
    cond_center = np.ones_like(status_center) * null_value
    cond_center[status_center] = 3/5 * A_center[status_center] ** 2 * G[status_center] / viscosity

    #Corner conductance
    mask = A_corner == 0
    A_corner[mask] = null_value
    bi_corner[mask] = null_value
    P_corner = 2 * bi_corner  * (1 - np.sin(beta) / np.cos(beta + theta_corner) * (theta_corner + beta - _m.pi / 2))
    G_corner = A_corner / P_corner ** 2
    G_mod = np.sin(beta) * np.cos(beta) / (2 + 2 * np.sin(beta))**2
    C = 0.364 + 0.28 * G_mod / G_corner
    cond_corner = np.ones_like(A_corner) * null_value
    cond_corner[status_corner] = (C * A_corner ** 2 * G_corner)[status_corner] / viscosity
    cond_corner[mask] = null_value
    cond_corner = np.sum(cond_corner, axis = 1)
    ratio_phase = (A_center*status_center + np.sum(A_corner* status_corner, axis = 1)) / area

    #Code for layer conductance (in development)
    cond_layer = np.zeros_like(A_corner)
    if correction:
        beta_dif = np.pi/2 - 2 * beta[:,1]
        pol = np.array([-0.17997611,  0.57966346, -0.46275726,  1.10633925])
        factor = np.polyval(pol, beta_dif)
        cond_center = cond_center * factor
    return cond_center, cond_corner, ratio_phase

def conductance_triangle_OnePhase(
    phase,
    viscosity="viscosity",
    shape_factor="shape_factor",
    cross_sectional_area="cross_sectional_area",
    conduit_lengths="throat.conduit_lengths",
    half_corner_angle = 'half_corner_angle',
    correction = False,
    check_boundary = False,
    boundary = 'boundary',
    ):
    r"""
    Use valvatne and blunt formula for shape factor and correct it using
    a correlation according to White (2002) to isosceles triangle
    If correction = False, it is equal to hydraulic_conductance_valvatne_blunt
    """
    network = phase.network
    conns = network["throat.conns"]
    L = network[conduit_lengths]
    for item in ['pore', 'throat']:
        mu = phase[f'{item}.' + viscosity]
        A = network[f'{item}.' + cross_sectional_area]
        G = network[f'{item}.' + shape_factor]
        if not correction:
            F = 1
        else:
            beta = np.sort(network[f'{item}.' + half_corner_angle], axis = 1)
            beta_dif = np.pi/2 - 2 * beta[:,1]
            pol = np.array([-0.17997611,  0.57966346, -0.46275726,  1.10633925])
            F = np.polyval(pol, beta_dif)
        value = 0.6 * F * A ** 2 * G / mu
        if item == 'pore':
            gp = value
        else:
            gt = value
    g = conduit_conductance_2phases(network, gp, gt, conduit_length = L, check_boundary = check_boundary, boundary = boundary)
    return g

def HydraulicSizeFactors_SpheresCylinders_improved(
    network,
    diameter= "diameter",
    conduit_length = "throat.conduit_lengths",
    check_boundary = False,
    boundary = 'boundary',
    ):
    r"""
    OpenPNM model considering boundary elements
    If check_boundary = True, internal pores of boundary conduits are considered as cylinders
    """
    D1, Dt, D2 = network.get_conduit_data(diameter).T
    L1, Lt, L2 = network[conduit_length].T
    # Fi is the integral of (1/A^2) dx, x = [0, Li]
    a = 4 / (D1**3 * np.pi**2)
    b = 2 * D1 * L1 / (D1**2 - 4 * L1**2) + np.arctanh(2 * L1 / D1)
    F1 = a * b
    a = 4 / (D2**3 * np.pi**2)
    b = 2 * D2 * L2 / (D2**2 - 4 * L2**2) + np.arctanh(2 * L2 / D2)
    F2 = a * b
    Ft = Lt / (np.pi / 4 * Dt**2)**2

    if check_boundary:
        conns = network['throat.conns']
        mask_tBC = network['throat.' + boundary]
        mask_pBC = np.isin(conns, network.pores(boundary))
        #For internal pores of boundary conduits
        F1[ mask_tBC &  ~mask_pBC[:, 0] ] = ( L1 / (np.pi / 4 * D1**2)**2)[ mask_tBC &  ~mask_pBC[:, 0] ]
        F2[ mask_tBC &  ~mask_pBC[:, 1] ] = ( L2 / (np.pi / 4 * D2**2)**2)[ mask_tBC &  ~mask_pBC[:, 1] ]

    # I is the integral of (y^2 + z^2) dA, divided by A^2
    I1 = I2 = It = 1 / (2 * np.pi)

    # S is 1 / (16 * pi^2 * I * F)
    S1 = 1 / (16 * np.pi**2 * I1 * F1)
    St = 1 / (16 * np.pi**2 * It * Ft)
    S2 = 1 / (16 * np.pi**2 * I2 * F2)

    if check_boundary:
        St[mask_tBC] = np.inf
        if np.any(mask_pBC[:, 0]):
            S1[mask_pBC[:, 0]] = np.inf
        if np.any(mask_pBC[:, 1]):
            S2[mask_pBC[:, 1]] = np.inf
    return np.vstack([S1, St, S2]).T


def conduit_length_tubes(
    network,
    pore_length = "pore.length",
    throat_spacing = "throat.spacing",
    L_min = 1e-15,
    check_boundary = False,
    boundary = 'boundary'):

    r"""
    Calculates conduit lengths in the network assuming pores and throats as
    tubes.

    A conduit is defined as ( 1/2 pore - full throat - 1/2 pore ).

    Parameters
    ----------
    network: Network object
    pore_length
    throat_spacing: distance considered  between the two pore centers
    min_L: Minimum value for Lt, to avoid zeros/negative values.
    boundary: If True, for the internal pore of the boundary conduits L = max(pore_length, conduit_length)
    boundary throats can be set on network using the following lines
    #boundary_p = pn.pores('boundary')
    #boundary_t = pn.find_neighbor_throats(pores=boundary_p)
    #pn['throat.boundary'] = False
    #pn['throat.boundary'][boundary_t] = True


    Returns
    -------
    lengths : ndarray
        The array is formatted as
        ``[pore1, throat, pore2]``.

    Note:
    -------
    Functions to calculate conduit_lenght are implemented in OpenPNM but they use
    pore diameter to calculate L1, L2

    """
    conns = network["throat.conns"]
    Lp = network[pore_length]
    L_ctc = network[throat_spacing]
    Nt = network.Nt

    #Construct L_min array
    if isinstance(L_min, np.ndarray):
        if len(L_min) != Nt:
            raise Exception('L_min must be a positive number or a Nt-size numpy array')
    else:
        try:
            if L_min > 0:
                L_min = np.ones(Nt) * L_min
            else:
                raise Exception('L_min must be a positive number or a Nt-size numpy array')
        except:
            raise Exception('L_min must be a positive number or a Nt-size numpy array')

    L1 = Lp[conns[:, 0]] / 2
    L2 = Lp[conns[:, 1]] / 2

    if check_boundary:
        mask_pBC = np.isin(conns, network.pores(boundary))
        mask_tBC = np.any(mask_pBC, axis = 1)
        L1[mask_pBC[:,0]] = L_min[mask_pBC[:,0]]
        L2[mask_pBC[:,0]] = L_min[mask_pBC[:,0]]

    Lt = (L_ctc - (L1 + L2))
    if check_boundary:
        Lt[mask_tBC] = L_min[mask_tBC]

    mask = Lt < L_min
    if np.any(mask):
        _L1 = L1/(L1 + L2) * (L_ctc - L_min)
        _L2 = L2/(L1 + L2) * (L_ctc - L_min)
        L1[mask] = _L1[mask]
        L2[mask] = _L2[mask]
        Lt[mask] = L_min[mask]

    return np.vstack((L1, Lt, L2)).T


def conduit_length_spheres_cylinders(
    network,
    pore_diameter = 'pore.diameter',
    throat_diameter = 'throat.diameter',
    throat_spacing = 'throat.spacing',
    L_min = 1e-20):

    r"""
    A modified function of openpnm.models.geometry.conduit_lengths.spheres_and_cylinders.
    Includes a minium value of length (L_min) and a treatment for pores with Dp <= Dt.
    For a pore with Dp <= Dt, Lp = L_min

    Parameters:
    -----------------
    network:
    pore_diameter:
    throat_diameter:
    throat_spacing: str related to the conduit length
    L_min: float or an Nt-size array. Minimum lenght of elements

    Returns
    -------
    lengths : ndarray
        The array is formatted as
        ``[pore1, throat, pore2]``.
    """

    #Obtaining data
    L_ctc = network[throat_spacing]
    D1, Dt, D2 = network.get_conduit_data(pore_diameter.split(".", 1)[-1]).T
    Dt = network[throat_diameter]
    Nt = network.Nt

    #Construct L_min array
    if isinstance(L_min, np.ndarray):
        if len(L_min) != Nt:
            raise Exception('L_min must be a positive number or a Nt-size numpy array')
    else:
        try:
            if L_min > 0:
                L_min = np.ones(Nt) * L_min
            else:
                raise Exception('L_min must be a positive number or a Nt-size numpy array')
        except:
            raise Exception('L_min must be a positive number or a Nt-size numpy array')

    #Identifying pores with no diameter problem
    mask1 = D1 > Dt
    mask2 = D2 > Dt

    #Calculating values.
    L1 = np.copy(L_min)
    L2 = np.copy(L_min)
    L1[mask1] = np.sqrt( (D1**2 - Dt**2)[mask1] ) / 2
    L2[mask2] = np.sqrt( (D2**2 - Dt**2)[mask2] ) / 2
    Lt = L_ctc - (L1 + L2)

    #Managing overlapping pores
    mask = Lt < L_min
    if np.any(mask):
        L1[mask & ~mask2] = L_ctc[mask & ~mask2] - 2*L_min[mask & ~mask2]
        L2[mask & ~mask1] = L_ctc[mask & ~mask1] - 2*L_min[mask & ~mask1]
        f1 = D1 / (D1 + D2)
        f2 = D2 / (D1 + D2)
        L1[mask & mask1 & mask2] = (L_ctc[mask & mask1 & mask2] - L_min[mask & mask1 & mask2]) * f1[mask & mask1 & mask2]
        L2[mask & mask1 & mask2] = (L_ctc[mask & mask1 & mask2] - L_min[mask & mask1 & mask2]) * f2[mask & mask1 & mask2]
        Lt = L_ctc - (L1 + L2)
    return np.vstack((L1, Lt, L2)).T

def conduit_length_spheres_cylinders_improved(
    network,
    pore_diameter = 'pore.diameter',
    throat_diameter = 'throat.diameter',
    throat_spacing = 'throat.spacing',
    L_min = 1e-20,
    check_boundary = False,
    boundary = 'boundary'):

    r"""
    An improved version of the function of openpnm.models.geometry.conduit_lengths.spheres_and_cylinders.
    Includes a minium value of length (L_min) and a treatment for pores with Dp <= Dt.
    Also treatment for boundary conduits. If check_boundary = True, for each boundary conduit:
        ACTUAL, the internal pore has an L = CONDUIT LENGTH
        OLD, NOT USED the internal pore has an L = 0.95 * R_{internal}
    For a pore with Dp <= Dt, Lp = L_min

    Parameters:
    -----------------
    network:
    pore_diameter:
    throat_diameter:
    throat_spacing: str related to the conduit length
    L_min: float or an Nt-size array. Minimum lenght of elements

    Returns
    -------
    lengths : ndarray
        The array is formatted as
        ``[pore1, throat, pore2]``.
    """

    #Obtaining data
    L_ctc = network[throat_spacing]
    R1, Rt, R2 = network.get_conduit_data(pore_diameter.split(".", 1)[-1]).T / 2
    Dt = network[throat_diameter]
    Nt = network.Nt

    #Construct L_min array
    if isinstance(L_min, np.ndarray):
        if len(L_min) != Nt:
            raise Exception('L_min must be a positive number or a Nt-size numpy array')
    else:
        try:
            if L_min > 0:
                L_min = np.ones(Nt) * L_min
            else:
                raise Exception('L_min must be a positive number or a Nt-size numpy array')
        except:
            raise Exception('L_min must be a positive number or a Nt-size numpy array')

    #Identifying pores with no diameter problem
    mask1 = R1 > Rt
    mask2 = R2 > Rt
    if check_boundary:
            mask_tBC = network['throat.' + boundary]
            conns = network['throat.conns']
            mask_pBC = np.isin(conns, network.pores(boundary))

    if np.any(~mask1) or np.any(~mask2):
        print(f'From {Nt:.0f} conduits, {np.sum(~mask1 | ~mask2):.0f} conduits has a throat with bigger diameter than, at least, one connected pore')
        if check_boundary:
            print(f'From {np.sum(~mask_tBC):.0f} internal conduits, {np.sum((~mask1 | ~mask2) & ~mask_tBC ):.0f} internal conduits has a throat with bigger diameter than, at least, one connected pore')

    #Calculating values, setting L1,L2 = Lmin for pores with a bigger throat
    L1 = np.copy(L_min)
    L2 = np.copy(L_min)
    L1[mask1] = np.sqrt( (R1**2 - Rt**2)[mask1] )
    L2[mask2] = np.sqrt( (R2**2 - Rt**2)[mask2] )
    Lt = L_ctc - (L1 + L2)

    #Managing overlapping pores
    mask = Lt < L_min
    if np.any(mask):
        #Conduits with overlapping pores and one of them is smaller than the connected throat
        L1[mask & ~mask2] = L_ctc[mask & ~mask2] - 2*L_min[mask & ~mask2]
        L2[mask & ~mask1] = L_ctc[mask & ~mask1] - 2*L_min[mask & ~mask1]
        #Conduits without diameter problems
        _L1 = ( R1**2 + (L_ctc - L_min)**2 - R2**2 ) / (2 * L_ctc)
        _L2 = ( R2**2 + (L_ctc - L_min)**2 - R1**2 ) / (2 * L_ctc)
        L1[mask & mask1 & mask2] = np.maximum(L_min, _L1)[mask & mask1 & mask2]
        L2[mask & mask1 & mask2] = np.maximum(L_min, _L2)[mask & mask1 & mask2]
        Lt = L_ctc - (L1 + L2)

    #Solving pores problems with L>=R (THIS IS PROBABLY BECAUSE OF LMIN)
    prob_1 =L1 >= R1
    prob_2 = L2 >= R2

    if check_boundary:
        #For internal pores of boundary conduits
        L1[ mask_tBC &  ~mask_pBC[:, 0] ] = L_ctc[ mask_tBC &  ~mask_pBC[:, 0] ]
        L2[ mask_tBC &  ~mask_pBC[:, 1] ] = L_ctc[ mask_tBC &  ~mask_pBC[:, 1] ]
        if np.any(prob_1[~mask_tBC]) or np.any(prob_2[~mask_tBC]):
            print(f'WARNING: Some internal pores ({np.sum(prob_1[~mask_tBC]) + np.sum(prob_2[~mask_tBC]):.0f}) has L > R, maybe because of L_min.')
            print('We will make L = R * 0.95. However, we recommend to reduce L_min')
            L1[prob_1 & ~mask_tBC] = 0.95 * R1[prob_1 & ~mask_tBC]
            L2[prob_2 & ~mask_tBC] = 0.95 * R2[prob_2 & ~mask_tBC]
    elif np.any(prob_1) or np.any(prob_2):
        print(f'WARNING: Some pores ({np.sum(prob_1) + np.sum(prob_2):.0f}) has L > R, maybe because of L_min.')
        print('We will make L = R * 0.95. However, we recommend to reduce L_min')
        L1[prob_1] = 0.95 * R1[prob_1]
        L2[prob_2] = 0.95 * R2[prob_2]


    return np.vstack((L1, Lt, L2)).T

def conduit_conductance_2phases(
    network,
    pore_g_ce,
    throat_g_ce,
    conduit_length,
    pore_g_co = None,
    throat_g_co = None,
    pore_g_la = None,
    throat_g_la = None,
    corner = False,
    layer = False,
    null_g = 1e-30,
    check_boundary = False,
    boundary = 'boundary',
    ):
    r"""
    This function allows you to calculate the conductivity of the porous medium for each conduit.
    A conduite is made up of 1/2 pore - 1 throat - 1/2 pore.
    It is necessary to indicate whether phase is allowed in the corners and layers.

    Parameters:
    -------------

    -network: Network object
    -conduit_length:Nt x 3 array. The first and last column represent the length of half the pore length.
     The middle column represent the length of all the throat.
    -pore_g_... and throat_g_... : Conductances from pores and throats. ce for center, co for corner, la for layer
    -Corner: Boolean value. If True, pore_g_co and throat_g_co must be given
    -Layer: Boolean value. If True, pore_g_la and throat_g_la must be given
    -null_g: Minimum value for conduit conductance, to avoid zeros.

    Reference: Valvatne and Blunt(2004)
    """

    Np = network.Np
    Nt = network.Nt
    P1 = network['throat.conns'][:,0]
    P2 = network['throat.conns'][:,1]

    #Check conductance in center
    if (len(pore_g_ce) != Np) or  (len(throat_g_ce) != Nt):
        raise Exception('pore_conductance_center and throat_conductance_center must be defined as arrays witha lenght of Np and Nt respectively')

    #Check conduit length:
    if (len(conduit_length) != Nt) or (len(conduit_length[0]) != 3):
        raise Exception('shape of conduit length must be (Nt,3)')

    if corner:
        #Check conductance in corner
        if (len(pore_g_co) != Np) or (len(throat_g_co) != Nt):
            raise Exception('pore_conductance_corner and throat_conductance_corner must be defined as arrays witha lenght of Np and Nt respectively')
        if layer:
            #Check conductance in layer
            if (len(pore_g_la) != Np) or (len(throat_g_la) != Nt):
                raise Exception('pore_conductance_layer and throat_conductance_layer must be defined as arrays witha lenght of Np and Nt respectively')
    else:
        pore_g_co = np.ones_like(pore_g_ce) * null_g
        throat_g_co = np.ones_like(throat_g_ce) * null_g

    #Layers can only exist if corner is True
    if layer and not corner:
        raise Exception('Layers can only exist if corner is True')
    elif not layer:
        pore_g_la = np.ones_like(pore_g_ce) * null_g
        throat_g_la = np.ones_like(throat_g_ce) * null_g


    pore_g = pore_g_ce + pore_g_co  + pore_g_la
    throat_g = throat_g_ce + throat_g_co  + throat_g_la
    if check_boundary:
            pore_g[network['pore.' + boundary]] = np.inf
            throat_g[network['throat.' + boundary]] = np.inf
    P1_g = pore_g[P1]
    P2_g = pore_g[P2]
    P12_g = np.vstack((P1_g , throat_g, P2_g)).T
    g_L = np.sum(conduit_length / P12_g, axis = 1 )**(-1)
    return g_L



