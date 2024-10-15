import numpy as np
import openpnm as op
from openpnm.algorithms import Algorithm
from numba import jit, njit
import heapq as hq
import math as m
import warnings
import _invasion_funcs as _if
from openpnm.utils import Docorator
import copy
docstr = Docorator()


__all__ = [
    'Primary_Drainage',
] #Only when use the code from <module> import *


@docstr.get_sections(base='Prm_Drn_Settings',
                     sections=['Parameters', 'Other Parameters']) #Probably to read the information only. Not use in math

class Imb_Settings:
    r"""

    Parameters
    ----------
    %(AlgorithmSettings.parameters)s
    pore_volume : str
        The dictionary key for the pore volume array
    throat_volume : str
        The dictionary key for the throat volume array
    entry_pressure : str
        The dictionary key for the throat capillary pressure

    """
    phase = ''



class Imbibition(Algorithm):
    r"""
    An algorithm to run the imbibition case assuming prisms with a triangular cross sections.
    This algorithm use the information from the invaded phase
    Optimized for speed with numba

    Parameters
    ----------
    network : Network
        The Network upon which the invasion will occur

    """

    def __init__(self, phase, name='prdr_?', **kwargs):
        super().__init__(name=name, **kwargs)
        self.settings._update(Imb_Settings()) #It seems to be just information
        self.settings['phase'] = phase.name #Phase information
        self['pore.bc.inlet'] = False
        self['pore.bc.outlet'] = False
        self['throat.bc.inlet'] = False
        self['throat.bc.outlet'] = False
        self.reset()

    def reset(self):
        self['pore.invasion_sequence'] = -1
        self['throat.invasion_sequence'] = -1
        self.settings['status'] = 'reseted'

    def set_inlet_BC(self, pores=None, mode='add'):
        r"""
        Specifies which pores are treated as inlets for the invading phase

        Parameters
        ----------
        pores : ndarray
            The indices of the pores from which the invading fluid invasion
            should start
        mode : str or list of str, optional
            Controls how the boundary conditions are applied. Options are:

            ============ =====================================================
            mode         meaning
            ============ =====================================================
            'add'        (default) Adds the supplied boundary conditions to
                         the given locations. Raises an exception if values
                         of any type already exist in the given locations.
            'overwrite'  Adds supplied boundary conditions to the given
                         locations, including overwriting conditions of the
                         given type or any other type that may be present in
                         the given locations.
            'remove'     Removes boundary conditions of the specified type
                         from the specified locations. If ``bctype`` is not
                         specified then *all* types are removed. If no
                         locations are given then values are remvoed from
                         *all* locations.
            ============ =====================================================

            If a list of strings is provided, then each mode in the list is
            handled in order, so that ``['remove', 'add']`` will give the same
            results add ``'overwrite'``.

        """
        self.set_BC(pores=pores, bcvalues=True, bctype='inlet', mode=mode)
        self.reset()
        self['pore.invasion_sequence'][self['pore.bc.inlet']] = 0
        throats = self.project.network.find_neighbor_throats(pores=pores)
        self['throat.bc.inlet'][throats] = True

    def set_outlet_BC(self, pores=None, mode='add'):
        r"""
        Specifies which pores are treated as outlets for the defending phase

        This must be specified if trapping is to be considered.

        Parameters
        ----------
        pores : ndarray
            The indices of the pores from which the defending fluid exits the
            domain
        mode : str or list of str, optional
            Controls how the boundary conditions are applied. Options are:

            ============ =====================================================
            mode         meaning
            ============ =====================================================
            'add'        (default) Adds the supplied boundary conditions to
                         the given locations. Raises an exception if values
                         of any type already exist in the given locations.
            'overwrite'  Adds supplied boundary conditions to the given
                         locations, including overwriting conditions of the
                         given type or any other type that may be present in
                         the given locations.
            'remove'     Removes boundary conditions of the specified type
                         from the specified locations. If ``bctype`` is not
                         specified then *all* types are removed. If no
                         locations are given then values are remvoed from
                         *all* locations.
            ============ =====================================================

            If a list of strings is provided, then each mode in the list is
            handled in order, so that ``['remove', 'add']`` will give the same
            results add ``'overwrite'``.

        """
        self.set_BC(pores=pores, bcvalues=True, bctype='outlet', mode=mode)
        throats = self.project.network.find_neighbor_throats(pores=pores)
        self['throat.bc.outlet'][throats] = True

    def run(self, status_drn_dict, throat_diameter = 'throat.diameter', ignore_boundary = False, pc_max = None, pc_min = -np.inf):
        r"""
        Performs the algorithm for the given number of steps
        Assume the key for the throat diameter is 'throat.diameter'

        Input parameters:
        -----------------------------
        status_drn_dict: Dictionary with all the information of the final stage at the drainage.
        throat_diameter: string related to the throat diameter property
        ignore_boundary: If True, a boundary conduit is invaded when the internal pore is connected to an invaded internal throat

        Note of the status_drn_dict:
        ------------------------------
        The important keys are
        -'invasion pressure': minimum cappilarry pressure to achieve this status
        'trapped_clusters': array with the index of the trapped clusters during the drainage process
        -'pc_trapped':  array with the capillary pressure values The trapped cluster index at the position[i] has a capillary pressure at the same position[i]
        -'invasion_info': dictionary with boolean information about the presence of the wetting phase in each phase locations
        -'cluster_info': dictionary with the cluster information for each phase locations

        'invasion_info' and 'cluster_info' has the following keys: 'pore.center', 'throat.center', 'pore.corner', 'throat.corner'.
        """

        invasion_info, cluster_info = self._run_setup(throat_diameter, ignore_boundary)

        results_dict = self._run_algorithm(wp_dict = invasion_info,
                                           cluster_dict = cluster_info,
                                           pc_max = np.inf)
        return results_dict

    def _run_algorithm(self, wp_dict, cluster_dict, pc_max = np.inf):
        r"""
        Detail how the algorithm works for invasion
        """
        #Obtain required data
        network = self.network
        conns = network['throat.conns']
        Nt = network.Nt
        Np = network.Np
        elements = ['pore', 'throat']
        locations = ['center', 'corner']

        # Create incidence matrix with info of neighbor throats for each pore
        im = self.network.create_incidence_matrix(fmt='csr')
        idx=im.indices
        indptr=im.indptr

        #Preparing inlet throats to be invaded
        p_list = np.ones(Nt) * np.inf
        Ts = network.find_neighbor_throats(pores=self['pore.bc.inlet'])
        p_list[Ts] = self['throat.entry_pressure'][Ts]
        p_next = np.min(p_list)

        #Check if the maximum capillary pressure is adequate
        if p_next > pc_max:
            raise Exception('The chosen capillary pressure is too low to invade a throat')

        #Create info to save status
        s_inv = 0
        p_inv = 0
        info_drainage = {}
        nontrapped = [-1] #Wetting phase
        trapped = []
        status_str = 'status ' + str(s_inv)

        #Save status 0
        info_drainage[status_str] = {'invasion sequence' : s_inv,
                               'invasion pressure' : p_inv,
                               'nontrapped_clusters': np.copy(nontrapped),
                               'trapped_clusters': np.copy(trapped),
                               'pc_trapped': [],
                               'invasion_info': copy.deepcopy(wp_dict),
                               'cluster_info': copy.deepcopy(cluster_dict)}

        while p_next <= pc_max:
            if p_next > p_inv:
                #New quasi-static state
                s_inv +=1
                status_str = 'status ' + str(s_inv)
                previous_status = 'status ' + str(s_inv - 1)
                p_inv = p_next
                info_drainage[status_str] = {'invasion sequence' : s_inv,
                                               'invasion pressure' : p_inv}
            else:
                #Maintain actual status
                previous_status = 'status ' + str(s_inv)
            #Invade throat
            pos = np.where(p_list == p_next)[0]
            #Choose only one element
            pos = pos[0]
            #Creando bool of new cluster
            bool_new_wp_cluster = False
            info_drainage[status_str]['invaded throat'] = pos
            #Invading throat with the nwp
            mask_corner_t = self['throat.mask_wp_corner'][pos,:]
            wp_index = cluster_dict['throat.center'][pos]
            wp_dict['throat.center'][pos] = False
            wp_dict['throat.corner'][pos] = mask_corner_t
            #Modify pc from the invaded throat
            p_list[pos] = np.inf
            #Choose the nwp cluster index
            Ps = conns[pos,:]
            if np.any(~wp_dict['pore.center'][Ps]):
                #At least one pore woth the nwp
                index = np.unique([x for x in cluster_dict['pore.center'][Ps] if x > 0])
                if len(index) > 1:
                    #Two different nwp clusters are merged
                    new_index = np.min(index)
                    cluster_dict = _if.merge_cluster(cluster_dict, index)
                else:
                    new_index = index[0]
            else:
                if len(cluster_dict['index_list']) == 1:
                    new_index = 1 #Start with one
                else:
                    new_index = np.max(cluster_dict['index_list']) + 1
                cluster_dict['index_list'] = np.append(cluster_dict['index_list'], new_index)
            #Setting nwp center index
            cluster_dict['throat.center'][pos] = new_index
            #Setting nwp corners indexes, if required
            if np.any(~mask_corner_t):
                cluster_dict['throat.corner'][pos,~mask_corner_t] = new_index
            #IF WE HAVE INLET OR OUTLET PORES, INVADE AT THE SAME TIME WITH THE THROAT
            mask_BC = self['pore.bc.inlet'] | self['pore.bc.outlet']
            if np.any(mask_BC[Ps]):
                pb = Ps[mask_BC[Ps]][0]
                mask_corner_p = self['pore.mask_wp_corner'][pb,:]
                wp_dict['pore.center'][pb] = False
                wp_dict['pore.corner'][pb] = mask_corner_p
                cluster_dict['pore.center'][pb] = new_index
                if np.any(~mask_corner_p):
                    cluster_dict['pore.corner'][pb,~mask_corner_p] = new_index
            #If the throat is saturated with nwp, check if the wp cluster is divided
            if np.all(~mask_corner_t):
                cluster_dict, bool_new_wp_cluster = _if.check_divide_cluster(network, cluster_dict, wp_index, index_mode = 'min')
            #Determine if trapped clusters are formed
            nontrapped_1, trapped_1 = _if.obtain_nontrapped_clusters(wp_dict,
                                                            cluster_dict,
                                                            connect_wp_pores = self['pore.bc.outlet'],
                                                            connect_nwp_pores = self['pore.bc.inlet'],
                                                            obtain_trapped = True)
            trapped = np.copy(info_drainage[previous_status]['trapped_clusters'])
            pressure_trapped = np.copy(info_drainage[previous_status]['pc_trapped'])
            if bool_new_wp_cluster:
                Ntrap = len(trapped_1)
                Ntrap_prev = len(trapped)
                if Ntrap > Ntrap_prev:
                    #Hard process to maintain the order of the indexes and pressures
                    new_trapped = np.setdiff1d(trapped_1, trapped)
                    trapped = np.append(trapped, new_trapped)
                    pressure_trapped = np.append(pressure_trapped, np.ones_like(new_trapped ) *p_inv)
            else:
                trapped = np.copy(info_drainage[previous_status]['trapped_clusters'])
            info_drainage[status_str]['nontrapped_clusters'] = nontrapped_1
            info_drainage[status_str]['trapped_clusters'] = trapped
            info_drainage[status_str]['pc_trapped'] = pressure_trapped
            #invade neighbor pores
            for p in Ps:
                #print(f'pore : {p}')
                if wp_dict['pore.center'][p] and cluster_dict['pore.center'][p] in nontrapped_1:
                    bool_new_wp_cluster = False
                    #Now we invade that pore
                    wp_index = cluster_dict['pore.center'][p]
                    mask_corner_p = self['pore.mask_wp_corner'][p,:]
                    wp_dict['pore.center'][p] = False
                    wp_dict['pore.corner'][p] = mask_corner_p
                    #Choose the nwp cluster index
                    Ts = idx[indptr[p]:indptr[p+1]]
                    if np.sum(~wp_dict['throat.center'][Ts]) > 1:
                        #Two different nwp clusters are merged
                        index = np.unique([x for x in cluster_dict['throat.center'][Ts] if x > 0])
                        new_index = np.min(index)
                        cluster_dict = _if.merge_cluster(cluster_dict, index)
                    #Setting nwp center index
                    cluster_dict['pore.center'][p] = new_index
                    #Setting nwp corners indexes, if required
                    if np.any(~mask_corner_p):
                        cluster_dict['pore.corner'][p,~mask_corner_p] = new_index
                    #If the pore is saturated with nwp, check if the wp cluster is divided
                    if len(Ts) == 1:
                        #The invaded throat is the only linked throat
                        if np.all(~wp_dict['throat.corner'][pos,:]) and np.any(wp_dict['pore.corner'][p,:]):
                            bool_new_wp_cluster = True
                            wp_index = np.min(cluster_dict['index_list']) - 1
                            cluster_dict['index_list'] = np.append(cluster_dict['index_list'], wp_index)
                            cluster_dict['pore.corner'][p, ~mask_corner_p] = wp_index
                    elif np.all(~mask_corner_p):
                        cluster_dict, bool_new_wp_cluster = _if.check_divide_cluster(network, cluster_dict, wp_index, index_mode = 'min')
                    #Determine if trapped clusters are formed
                    nontrapped_2, trapped_2 = _if.obtain_nontrapped_clusters(wp_dict,
                                                                    cluster_dict,
                                                                    connect_wp_pores = self['pore.bc.outlet'],
                                                                    connect_nwp_pores = self['pore.bc.inlet'],
                                                                    obtain_trapped = True)
                    trapped = np.copy(info_drainage[status_str]['trapped_clusters'])
                    pressure_trapped = np.copy(info_drainage[status_str]['pc_trapped'])
                    if bool_new_wp_cluster:
                        Ntrap = len(trapped_2)
                        Ntrap_prev = len(trapped)
                        if Ntrap > Ntrap_prev:
                            #Hard process to maintain the order of the indexes and pressures
                            new_trapped = np.setdiff1d(trapped_2, trapped)
                            trapped = np.append(trapped, new_trapped)
                            pressure_trapped = np.append(pressure_trapped, np.ones_like(new_trapped ) *p_inv)
                    info_drainage[status_str]['nontrapped_clusters'] = nontrapped_2
                    info_drainage[status_str]['trapped_clusters'] = trapped
                    info_drainage[status_str]['pc_trapped'] = pressure_trapped
                    #Setting new possible throat pc
                    for t in Ts:
                        if wp_dict['throat.center'][t] and (cluster_dict['throat.center'][t] in nontrapped_2):
                            p_list[t] = self['throat.entry_pressure'][t]
            for t in range(Nt):
                if cluster_dict['throat.center'][t] in trapped:
                    p_list[t] = np.inf
            p_next = np.min(p_list)
            info_drainage[status_str]['invasion_info'] = copy.deepcopy(wp_dict)
            info_drainage[status_str]['cluster_info'] = copy.deepcopy(cluster_dict)
            if s_inv > 200:
                p_next = np.inf
                print('Forced to end invasion.')
            if np.isinf(p_next):
                print('we can not invade more throats')
                break
            #else:
                #print(f'p_next: {p_next}')
            if p_next > pc_max:
                print('We reach the maximum capillary pressure')

        return info_drainage

    def _run_setup(self, throat_diameter = 'throat.diameter', ignore_boundary = False):
        phase = self.get_phase()
        network = self.network
        elements = ['pore', 'throat']
        locations = ['center', 'corner']
        self['throat.entry_pressure'] = _if.MSP_prim_drainage(phase, throat_diameter = throat_diameter)
        #Assigning values to invade the boundary conduit when a neighbor internal throat is invaded
        if ignore_boundary:
            mask_p = self['pore.bc.outlet'] | self['pore.bc.inlet']
            mask_t = self['throat.bc.outlet'] | self['throat.bc.inlet']
            for t in range(network.Nt):
                if mask_t[t]:
                    #boundary throat
                    conns = network['throat.conns'][t,:]
                    for p in conns:
                        if mask_p[p]:
                            #boundary pore
                            pb = p
                        else:
                            #internal pore
                            pi = p
                    throats = self.project.network.find_neighbor_throats(pores=pi)
                    throats = throats[ throats != t ]
                    pc_min = np.min(self['throat.entry_pressure'][throats])
                    self['throat.entry_pressure'][t] = pc_min
        #Check if, after invasion, the wp presence is possible
        for item in elements:
            theta = phase[f"{item}.contact_angle"]
            beta = network[f'{item}.half_corner_angle']
            self[f'{item}.mask_wp_corner'] = _if.wp_in_corners(beta = beta, theta = theta)

        #Creating invasion and cluster dictionaries
        inv_dict = {} #True if wp is present
        cluster_dict = {} #Start with -1 as wp index
        for item in elements:
            for loc in locations:
                row = len(self[f'{item}.bc.inlet'])
                if loc == 'center':
                    col = 1
                else:
                    col = 3
                inv_dict[f'{item}.{loc}'] = np.squeeze(np.ones((row,col), dtype = bool))
                cluster_dict[f'{item}.{loc}'] = np.squeeze(np.ones((row,col), dtype = int) * -1)
        cluster_dict['index_list'] = [-1]
        return inv_dict, cluster_dict


    def get_phase(self):
        r"""
        Get the invaded/wetting phase object
        """
        project = self.project
        for item in project.phases:
            if item.name == self.settings['phase']:
                phase = item
                break
        return phase

    def _set_wp_corner_presence(self, pore = True, throat = True):
        r"""
        To determine the presence of the wetting phase according to the determined invasion sequence.
        The values are automatically inserted in the algorithm
        If element.wp_center_presence is added in the algorithm, the information is taken into account
        If not, it assumes all elements invaded
        """
        phase = self.get_phase()
        element = []
        if pore:
            element.append('pore')
        if throat:
            element.append('throat')
        if len(element) == 0:
            wanings.warn('At least one type of element (pore or throat) to be analyzed must be indicated. Can not continue.')
        else:
            for item in element:
                if phase[f"{item}.contact_angle"].ndim == 1:
                    theta = phase[f"{item}.contact_angle"]
                else:
                    theta = phase[f"{item}.contact_angle"][:,0]
                cond = _if.wp_in_corners(beta = self.network[f'{item}.half_corner_angle'],
                                         theta = theta)
                if 'center_presence' in self[f'{item}']:
                     self[f'{item}.wp_corner_presence'] = ((np.tile(self[f'{item}.wp_center_presence'],(3,1)).T) | cond)
                else:
                    self[f'{item}.wp_corner_presence'] = cond
        return
