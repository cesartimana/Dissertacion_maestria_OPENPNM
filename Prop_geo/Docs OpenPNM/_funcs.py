#Modified by Cesar
#Updated 2023/10/05

import numpy as np
import math as m

__all__ = [
    "geo_props_pores_equilateral",
    "G_corrected_isosceles",
    "half_angle_isosceles",
    "Gcorrected_beta_throats",
    "prism_triangle_props"
]


def geo_props_pores_equilateral(network, root = None, d_corr = False, increase_factor = 1.01, throat_diameter = 'throat.diameter'):

    r"""
    Add the properties 'pore.length', 'pore.cross_sectional_area', 'pore.shape_factor' and  'pore.half_cont_angle' to the network.
    Assume a triangular tube (equilateral triangle).
    pore.surface_area ignores throat areas.
    pore.prism_surface_area is the area used in calculus minus throat areas
    root choose the root criteria:
    -If  None, it choose the  more homogeneous aspect ratio (max(a,L) / min (a,L))
    -If 0, a > L
    -If 1, L < a
    -Other values are rejected
    d_corr: If True, the pores with lower diameter than any connected throat diameter are modified to be bigger.
    And pore inscribed_diameter and surface_area are corrected.
    For that, only the pore volume is considered and the surface area is modified
    increase_factor: Only if d_corr== True. for those pore whose d_p <=  max(d_t) we do d_p =  max(d_t) * increase_factor
    throat_diameter: Only if d_corr== True. Key name to call the values of the throat diameters
    """

    V_p = network['pore.volume']
    S_p = network['pore.surface_area']
    A_t = network['throat.cross_sectional_area']
    t_conns = network['throat.conns']

    #Adding areas: solid-fluid + throat areas per pore
    Np = len(V_p)
    Nt = len(A_t)

    pore_throat =  np.tile(np.arange(Np),(Nt,1)) #matrix Nt x Np. Each row have all pore numbers
    neigh1 = np.tile(np.array(t_conns)[:,0], (Np,1)).T #matrix Nt x Np.
    neigh2  = np.tile(np.array(t_conns)[:,1], (Np,1)).T #matrix Nt x Np.
    area_t_per_p =  np.sum(np.tile(A_t, (Np,1)).T * ((pore_throat == neigh1) |  (pore_throat == neigh2)), axis=0)
    As_p = S_p + area_t_per_p

    #Doing geometry
    G_max = 0.04811252243 #G for an equilateral triangle with 10 decimals

    As_min = 3**1.5*2**(1/3)*np.power(V_p, 2/3)

    As_used = np.fmax(As_p, As_min*(1+1e-8)) #To not have problems calculating parameters

    p = - 2 * As_used / 3**0.5
    q = 8 * V_p

    if root is None:
        a0 = 2/3**0.5*np.power(-p, 0.5)*np.cos(1/3*np.arccos(3*q*3**0.5/2/p/np.power(-p, 0.5)))
        L0 = V_p / np.power(a0,2)*3**0.5/4
        a1 = 2/3**0.5*np.power(-p, 0.5)*np.cos(1/3*np.arccos(3*q*3**0.5/2/p/np.power(-p, 0.5))-2*m.pi*1/3)
        L1 = V_p / np.power(a1,2)*3**0.5/4
        mask = (np.maximum(a0,L0) / np.minimum(a0,L0)) > (np.maximum(a1,L1) / np.minimum(a1,L1))
        a_p = a0 * ~mask + a1 * mask
    elif root in [0,1]:
        a_p = 2/3**0.5*np.power(-p, 0.5)*np.cos(1/3*np.arccos(3*q*3**0.5/2/p/np.power(-p, 0.5))-2*m.pi*1/3*root)
    else:
        raise Exception('root must be None, 0 or 1')
    d_p = a_p / 3**0.5
    if d_corr:
        if increase_factor <= 1:
            increase_factor = 1.01
            print("increase_factor must be bigger than 1. Use 1.01 instead")
        d_t = network[throat_diameter]
        im = network.create_incidence_matrix()
        values = np.zeros((network.Np, ))
        np.maximum.at(values, im.row, d_t[im.col])
        mask = (d_p <= values)
        mod = np.sum(mask) / network.Np
        print("The diameter of %0.3f%% of the pores were modified to be bigger than the connected throats" % (np.sum(mask) / network.Np))
        print("The surface_area of those pores were corrected")
        d_p[mask] = values[mask] * increase_factor
        a_p[mask] = d_p[mask] * 3 ** 0.5
    A_p = np.power(a_p,2)*3**0.5/4
    L_p = V_p/A_p
    As_used[mask] = (3 * L_p * a_p + 2 * A_p)[mask]

    network['pore.length'] = L_p
    network['pore.cross_sectional_area'] = A_p
    network['pore.prism_surface_area'] = As_used - area_t_per_p
    network['pore.shape_factor'] = np.ones(Np) * G_max
    network['pore.half_corner_angle'] = np.ones((Np,3))*m.pi/6
    network['pore.inscribed_diameter'] = d_p
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
    c = np.arccos(-12*3**0.5*G)/3+ t_bool * 4*m.pi/3
    beta2 = np.arctan(2/3**0.5*np.cos( c ))
    beta = np.sort( np.concatenate(([beta2], [beta2], [m.pi/2-beta2*2])).T , 1)
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
    network['throat.inscribed_diameter'] = 2 * r
    return


def prism_triangle_props(network,
                         prob = 0.5,
                         min_vox = 30,
                         p_root = None,
                         d_corr = False,
                         increase_factor = 1.01,
                         throat_diameter = 'throat.diameter'):
    r"""
    Add pore and throats shape factor and half corner angle, among other properties, to the network object.
    Constant isosceles triangle cross section is assumed (equilateral on pores)

    Parameters:
    ----------------
    network:
    prob: For throats, to choose one triangle group of another
    min_vox: For throats. minimum number of voxels to correct throat shape factor
    roots: For pores, to choose what root for the triangle side
    d_corr, increase_factor, throat_diameter: Used onlyu if d_corr = True. See geo_props_pores_equilateral for details

    """
    Gcorrected_beta_throats(network,
                            prob  = prob,
                            min_vox = min_vox)
    geo_props_pores_equilateral(network,
                                root = p_root,
                                d_corr = d_corr,
                                increase_factor = increase_factor,
                                throat_diameter = throat_diameter )
    return

