#Modified by Cesar
#Updated 2023/10/05

import numpy as np

__all__ = [
    "geo_props_pores_equilateral",
    "G_corrected_isosceles",
    "half_angle_isosceles",
    "Gcorrected_beta_throats",
    "prism_triangle_props"
]


def geo_props_pores_equilateral(network, root = None, corr = False, increase_factor = 1.01, throat_area = 'throat.cross_sectional_area'):

    r"""
    Add the properties 'pore.length', 'pore.cross_sectional_area', 'pore.shape_factor' and  'pore.half_cont_angle' to the network.
    Assume a triangular tube (equilateral triangle).
    It calculates new properties using the pore length L and the triangle edge a
    pore.surface_area ignores throat areas.
    pore.prism_surface_area is the area used in calculus minus throat areas
    root choose the root criteria:
    -If  None, it choose the  more homogeneous aspect ratio (max(a,L) / min (a,L))
    -If 0, a > L
    -If 1, a < L
    -Other values are rejected
    corr: If True, the pores with lower cross sectional area than any connected throat area are modified to be bigger.
    pore.surface_area is corrected.
    For that, only the pore volume is considered and the surface area is modified
    increase_factor: Only if corr== True. for those pore whose A_p <=  max(A_t) we do A_p =  max(A_t) * increase_factor
    throat_area: Only if corr== True. Key name to call the values of the cross sectional area
    """

    V_p = network['pore.volume']
    S_p = network['pore.surface_area']
    A_t = network['throat.cross_sectional_area']
    t_conns = network['throat.conns']
    Np = network.Np
    Nt = network.Nt

    #Adding areas: solid-fluid + throat areas per pore
    As_p = np.copy(S_p)
    im = network.create_incidence_matrix()
    np.add.at(As_p, im.row, A_t[im.col])

    #Calculating geometric properties
    G_max = 0.04811252243 #G for an equilateral triangle with 10 decimals
    As_min = 3**1.5*2**(1/3)*np.power(V_p, 2/3)
    As_used = np.fmax(As_p, As_min*(1+1e-8)) #To not have problems calculating parameters

    p = - 2 * As_used / 3**0.5
    q = 8 * V_p

    if root is None:
        a0 = 2/3**0.5*np.power(-p, 0.5)*np.cos(1/3*np.arccos(3*q*3**0.5/2/p/np.power(-p, 0.5)))
        L0 = V_p / np.power(a0,2)*3**0.5/4
        a1 = 2/3**0.5*np.power(-p, 0.5)*np.cos(1/3*np.arccos(3*q*3**0.5/2/p/np.power(-p, 0.5))-2*np.pi*1/3)
        L1 = V_p / np.power(a1,2)*3**0.5/4
        mask = (np.maximum(a0,L0) / np.minimum(a0,L0)) > (np.maximum(a1,L1) / np.minimum(a1,L1))
        a_p = a0 * ~mask + a1 * mask
    elif root in [0,1]:
        a_p = 2/3**0.5*np.power(-p, 0.5)*np.cos(1/3*np.arccos(3*q*3**0.5/2/p/np.power(-p, 0.5))-2*np.pi*1/3*root)
    else:
        raise Exception('root must be None, 0 or 1')
    d_p = a_p / 3**0.5
    A_p = np.power(a_p,2)*3**0.5/4

    #Modifying pore_diameter if desired
    if corr:
        if increase_factor <= 1:
            increase_factor = 1.01
            print("increase_factor must be bigger than 1. Used 1.01 instead")
        #Identifying maximum throat area for each throat
        A_t_max = np.zeros((Np ))
        np.maximum.at(A_t_max, im.row, A_t[im.col])
        mask = A_p <= A_t_max
        print("The cross-sectional area of %0.3f%% of the pores were modified to be bigger than the connected throats" % (np.sum(mask) / network.Np))
        print("For these pores, we modify the prism properties")
        A_p[mask] = A_t_max[mask] * increase_factor
        a_p[mask] = (4 * A_p[mask] / 3 ** 0.5) ** 0.5
        d_p[mask] = a_p[mask] / 3**0.5
    #print('hol')
    #print((np.sum(mask) / network.Np))
    #print(network.Np)
    #print(np.sum(mask))
    #print(np.sum(mask & network['pore.boundary']))
    #print(np.sum(mask & network['pore.internal']))
    L_p = V_p/A_p
    if corr:
        As_used[mask] = (3 * L_p * a_p + 2 * A_p)[mask]
    #Substracting throat areas
    np.add.at(As_used, im.row, (-A_t)[im.col])

    network['pore.prism_length'] = L_p
    network['pore.cross_sectional_area'] = A_p
    network['pore.prism_surface_area'] = As_used
    network['pore.shape_factor'] = np.ones(Np) * G_max
    network['pore.half_corner_angle'] = np.ones((Np,3))*np.pi/6
    network['pore.prism_inscribed_diameter'] = d_p
    return

def geo_props_pores_equilateral_legacy(network, root = None, corr = False, increase_factor = 1.01, throat_diameter = 'throat.diameter'):

    r"""
    LEGACY: MODIFY DIAMETER OF PORES
    Add the properties 'pore.length', 'pore.cross_sectional_area', 'pore.shape_factor' and  'pore.half_cont_angle' to the network.
    Assume a triangular tube (equilateral triangle).
    It calculates new properties using the pore length L and the triangle edge a
    pore.surface_area ignores throat areas.
    pore.prism_surface_area is the area used in calculus minus throat areas
    root choose the root criteria:
    -If  None, it choose the  more homogeneous aspect ratio (max(a,L) / min (a,L))
    -If 0, a > L
    -If 1, a < L
    -Other values are rejected
    corr: If True, the pores with lower diameter than any connected throat diameter are modified to be bigger.
    And pore inscribed_diameter and surface_area are corrected.
    For that, only the pore volume is considered and the surface area is modified
    increase_factor: Only if d_corr== True. for those pore whose d_p <=  max(d_t) we do d_p =  max(d_t) * increase_factor
    throat_diameter: Only if d_corr== True. Key name to call the values of the throat diameters
    """

    V_p = network['pore.volume']
    S_p = network['pore.surface_area']
    A_t = network['throat.cross_sectional_area']
    t_conns = network['throat.conns']
    Np = network.Np
    Nt = network.Nt

    #Adding areas: solid-fluid + throat areas per pore
    As_p = np.copy(S_p)
    im = network.create_incidence_matrix()
    np.add.at(As_p, im.row, A_t[im.col])

    #Calculating geometric properties
    G_max = 0.04811252243 #G for an equilateral triangle with 10 decimals
    As_min = 3**1.5*2**(1/3)*np.power(V_p, 2/3)
    As_used = np.fmax(As_p, As_min*(1+1e-8)) #To not have problems calculating parameters

    #Identifying maximum throat area
    A_t_max = np.zeros((Np ))
    np.maximum.at(A_t_max, im.row, A_t[im.col])

    p = - 2 * As_used / 3**0.5
    q = 8 * V_p

    if root is None:
        a0 = 2/3**0.5*np.power(-p, 0.5)*np.cos(1/3*np.arccos(3*q*3**0.5/2/p/np.power(-p, 0.5)))
        L0 = V_p / np.power(a0,2)*3**0.5/4
        a1 = 2/3**0.5*np.power(-p, 0.5)*np.cos(1/3*np.arccos(3*q*3**0.5/2/p/np.power(-p, 0.5))-2*np.pi*1/3)
        L1 = V_p / np.power(a1,2)*3**0.5/4
        mask = (np.maximum(a0,L0) / np.minimum(a0,L0)) > (np.maximum(a1,L1) / np.minimum(a1,L1))
        a_p = a0 * ~mask + a1 * mask
    elif root in [0,1]:
        a_p = 2/3**0.5*np.power(-p, 0.5)*np.cos(1/3*np.arccos(3*q*3**0.5/2/p/np.power(-p, 0.5))-2*np.pi*1/3*root)
    else:
        raise Exception('root must be None, 0 or 1')
    d_p = a_p / 3**0.5

    #Modifying pore_diameter if desired
    if corr:
        if increase_factor <= 1:
            increase_factor = 1.01
            print("increase_factor must be bigger than 1. Used 1.01 instead")
        d_t = network[throat_diameter]
        d_t_max = np.zeros((Np ))
        np.maximum.at(d_t_max, im.row, d_t[im.col])
        mask = (d_p <= d_t_max)
        print("The diameter of %0.3f%% of the pores were modified to be bigger than the connected throats" % (np.sum(mask) / network.Np))
        print("For that pores, we modify the prism properties considering these diameters")
        d_p[mask] = d_t_max[mask] * increase_factor
        a_p[mask] = d_p[mask] * 3 ** 0.5
    A_p = np.power(a_p,2)*3**0.5/4
    mask2 = A_p < A_t_max
    #print('hol')
    #print((np.sum(mask2) / network.Np))
    #print(network.Np)
    #print(np.sum(mask2))
    #print(np.sum(mask2 & network['pore.boundary']))
    #print(np.sum(mask2 & network['pore.internal']))
    L_p = V_p/A_p
    if d_corr:
        As_used[mask] = (3 * L_p * a_p + 2 * A_p)[mask]
    #Substracting throat areas
    np.add.at(As_used, im.row, (-A_t)[im.col])

    network['pore.prism_length'] = L_p
    network['pore.cross_sectional_area'] = A_p
    network['pore.prism_surface_area'] = As_used
    network['pore.shape_factor'] = np.ones(Np) * G_max
    network['pore.half_corner_angle'] = np.ones((Np,3))*np.pi/6
    network['pore.prism_inscribed_diameter'] = d_p
    return


def G_corrected_isosceles(G_porespy, voxels, prob = 0.5, min_vox = 30):
    r"""
    Calculates the corrected shape factor G from G calculated by porespy, considering isosceles triangle

    prob: chance to use an isosceles triangle with different angle >= pi/3 (beta>=pi/6). Can be an array.
    Must be between 0 and 1
    min_vox: Minimum number of voxels to correct. If less, G = Gmax
    voxels +0.0001 to avoid problems dividing
    """
    G_max = 0.04811252243 #G for an equilateral triangle with 10 decimals
    t_bool = np.squeeze(np.random.rand(len(G_porespy),1)) < prob
    v_bool = voxels >= min_vox
    c = ((0.23913/np.log10(np.log10(voxels+0.0001)) + 0.79507) * t_bool  + (0.21057/np.log10(np.log10(voxels+0.0001)) + 0.82264) * ~t_bool)*v_bool
    G = np.min([0.5 * voxels**(c-1) * G_porespy**c, np.ones_like(G_porespy)*G_max], axis = 0)* v_bool + G_max * ~v_bool
    return G

def half_angle_isosceles(G, prob = 0.5):
    r"""
    Calculates the half angles considering isosceles triangle

    prob: chance to use an isosceles triangle with different angle >= pi/3 (beta>=pi/6). Can be an array.
    Must be between 0 and 1
    If G is les than 1e-16 beta can be zero or negative, so a restriction was added.
    A restriction for G > 0.04811252243 (equilateral triangle) was added
    """
    G = np.where(G < 1e-16, 1e-16, G) #added
    G = np.where(G > 0.04811252243, 0.04811252243, G) #added
    t_bool = np.squeeze(np.random.rand(len(G),1)) < prob
    c = np.arccos(-12*3**0.5*G)/3+ t_bool * 4*np.pi/3
    beta2 = np.arctan(2/3**0.5*np.cos( c ))
    beta = np.sort( np.concatenate(([beta2], [beta2], [np.pi/2-beta2*2])).T , 1)
    return beta

def Gcorrected_beta_throats(network, prob  = 0.5, min_vox = 30):

    r"""
    Add the properties 'throat.shape_factor' and  'throat.half_cont_angle' to the network,
    previous correction of the perimeter data.
    Correct the properties 'throat.perimeter' and  'throat.inscribed_diameter'
    prob: chance to use an isosceles triangle with different angle >= pi/3 (beta>=pi/6). Can be an array.
    Must be between 0 and 1
    """

    A_t = network['throat.cross_sectional_area']
    P_t = network['throat.perimeter']
    v_t = network['throat.voxels']
    G_t = A_t/np.power(P_t,2) #Shape factor to correct

    n = len(A_t)
    t_bool = np.squeeze(np.random.rand(n,1) < prob)
    G = G_corrected_isosceles(G_t, v_t, prob = t_bool, min_vox = min_vox)
    beta = half_angle_isosceles(G, prob = t_bool)
    r = (A_t * 4 * G) ** 0.5 #Inscribed radii

    network['throat.shape_factor'] = G
    network['throat.half_corner_angle'] = beta
    network['throat.perimeter'] = (A_t/G)**0.5
    network['throat.prism_inscribed_diameter'] = 2 * r
    return


def prism_triangle_props(network,
                         prob = 0.5,
                         min_vox = 30,
                         p_root = None,
                         corr = False,
                         increase_factor = 1.01,
                         throat_area = 'throat.cross_sectional_area'):
    r"""
    Add pore and throats shape factor and half corner angle, among other properties, to the network object.
    Constant isosceles triangle cross section is assumed (equilateral on pores)

    Parameters:
    ----------------
    network:
    prob: For throats, to choose one triangle group of another
    min_vox: For throats. minimum number of voxels to correct throat shape factor
    roots: For pores, to choose what root for the triangle side
    corr, increase_factor, throat_area: Used onlyu if corr = True. See geo_props_pores_equilateral for details

    """
    Gcorrected_beta_throats(network,
                            prob  = prob,
                            min_vox = min_vox)
    geo_props_pores_equilateral(network,
                                root = p_root,
                                corr = corr,
                                increase_factor = increase_factor,
                                throat_area = throat_area )
    return

