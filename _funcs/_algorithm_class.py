import numpy as np
import openpnm as op
from openpnm.algorithms import Algorithm
from numba import jit, njit
import heapq as hq
import math as m
import warnings
import _invasion_funcs as _if
from openpnm.utils import Docorator
docstr = Docorator()


__all__ = [
    'Primary_Drainage',
] #Only when use the code from <module> import *


@docstr.get_sections(base='Prm_Drn_Settings',
                     sections=['Parameters', 'Other Parameters']) #Probably to read the information only. Not use in math

class Prm_Drn_Settings:
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



class Primary_Drainage(Algorithm):
    r"""
    An algorithm to run the primary drainage case assuming prisms with a triangular cross sections.
    This algorithm use the information from the invaded phase
    Optimized for speed with numba

    Parameters
    ----------
    network : Network
        The Network upon which the invasion will occur

    Notes
    -----
    This algorithm uses a `binary heap <https://en.wikipedia.org/wiki/Binary_heap>`_
    to store a list of all accessible throats, sorted according to entry
    pressure.  This means that item [0] in the heap is the most easily invaded
    throat that is currently accessible by the invading fluid, so looking up
    which throat to invade next is computationally trivial. In order to keep
    the list sorted, adding new throats to the list takes more time; however,
    the heap data structure is very efficient at this.

    """

    def __init__(self, phase, name='prdr_?', **kwargs):
        super().__init__(name=name, **kwargs)
        self.settings._update(Prm_Drn_Settings()) #It seems to be just information
        self.settings['phase'] = phase.name #Phase information
        self['pore.bc.inlet'] = False
        self['pore.bc.outlet'] = False
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

    def run(self, throat_diameter = 'throat_diameter'):
        r"""
        Performs the algorithm for the given number of steps
        Assume the key for the throat diameter is 'throat.diameter'

        Properties added/modified to the PrimaryDrainage object (pores and throats)
        ----------------------
        invasion_sequence: Invasion order
        invasion_pressure: If an invaded throat has a entry pressure lower than the invaded previous one,
        the invaded pressure of this invaded throat is equal to the pressure of the previous one



        """
        # Setup arrays and info
        # TODO: This should be called conditionally so that it doesn't
        # overwrite existing data when doing a few steps at a time
        self._run_setup(throat_diameter)
        n_steps = np.inf

        # Create incidence matrix with info of neghbor throats for each pore
        #to use in _run_accelerated which is jit
        im = self.network.create_incidence_matrix(fmt='csr')

        # Perform initial analysis on input pores
        Ts = self.project.network.find_neighbor_throats(pores=self['pore.bc.inlet'])
        t_start = self['throat.order'][Ts]


        t_inv, p_inv, p_inv_t, order, t_ref = \
            _run_accelerated(
                t_start=t_start,
                t_sorted=self['throat.sorted'],
                t_order=self['throat.order'],
                t_inv=self['throat.invasion_sequence'],
                p_inv=self['pore.invasion_sequence'],
                p_inv_t=np.zeros_like(self['pore.invasion_sequence']),
                conns=self.project.network['throat.conns'],
                idx=im.indices,
                indptr=im.indptr,
                n_steps=n_steps)
        # Transfer results onto algorithm object
        self['throat.invasion_sequence'] = t_inv
        self['pore.invasion_sequence'] = p_inv
        self['throat.invasion_pressure'] = np.zeros_like(t_inv, dtype = float)
        self['throat.invasion_pressure'][order] = np.copy(self['throat.entry_pressure'][t_ref])
        self['pore.invasion_pressure'] = self['throat.invasion_pressure'][p_inv_t]
        # Set pressure of uninvaded (isolated, not connected with the boundary pores) elements as inf (Check)
        #self['throat.invasion_pressure'][self['throat.invasion_sequence'] < 0] = float('inf')
        #self['pore.invasion_pressure'][self['pore.invasion_sequence'] < 0] = float('inf')
        # Set invasion pressure of inlets to 0
        self['pore.invasion_pressure'][self['pore.invasion_sequence'] == 0] = 0.0

        #Set run as executed
        self.settings['status'] = 'executed'

    def get_phase(self):
        r"""
        Get the invaded phase object
        """
        project = self.project
        for item in project.phases:
            if item.name == self.settings['phase']:
                phase = item
                break
        return phase

    def _run_setup(self, throat_diameter = 'throat_diameter'):
        self['pore.invasion_sequence'][self['pore.bc.inlet']] = 0
        phase = self.get_phase()
        self['throat.entry_pressure'] = _if.MSP_prim_drainage(phase, throat_diameter = throat_diameter)
        # Generated indices into t_entry giving a sorted list
        self['throat.sorted'] = np.argsort(self['throat.entry_pressure'], axis=0)
        self['throat.order'] = 0
        self['throat.order'][self['throat.sorted']] = np.arange(0, self.Nt)


    def postprocessing(self, p_vals=None, points = None, p_max=None, p_min = None):
        r"""
        Get information about the location of phases and clusters.
        For this procedure inlet pores that do not have an invaded neighbor throat are considered as uninvaded
        Assumes the wetting phase is at least in one corner of each element.

        Parameters:
        ------------

        p_vals: Array of pressure values for which current network information is
        desired. If None, the network is considered as completely invaded.

        points: number of pressure values to get the information.
        Works if p_vals is None. Else, p_vals is the priority
        If it is equal to 1, the value of p_max is considered.
        If it is 2 or more, an array is constructed between p_min and p_max.
        If p_max is None. the network is considered as completely invaded.
        If p_min is None, is assumed 0

        Returns:

        p_vals: Pressure values evaluated
        invasion_info : Boolean info related to the invaded / wetting phase. True if this phase is located in
        a center or corner. If the element has only wp, center and corners are True
        cluster_info: Information of the clusters related to the phases. Starting with 0 for the
        initial wetting cluster and probably the only cluster of the wetting phase. Use invasion_info

        """
        #Checking pressure data

        if p_vals is None and points is None:
            p_vals = [np.max(self['throat.invasion_pressure'])]
            points = 1
        elif p_vals is not None:
            if isinstance(p_vals, np.ndarray):
                pass
            else:
                raise Exception('p_vals must be a numpy array. If it is just one value, set p_max instead')
        elif not isinstance(points, int):
            raise Exception('points must be an integer')

        if p_max is None:
            p_max = np.max(self['throat.invasion_pressure'])
        if p_min is None:
            p_min = 0.0

        if points is not None:
            if points == 1:
                if isinstance(p_max, (int, float)):
                    p_vals = [p_max]
                else:
                    raise Exception('p_max must be an int or float')
            elif points >= 2:
                if isinstance(p_max, (int, float)) and isinstance(p_min, (int, float)):
                    if p_min < p_max:
                        p_vals = np.linspace(p_min, p_max, points)
                    else:
                        raise Exception('p_min must be less than p_max')
                else:
                    raise Exception('p_min and p_max must be int or float')
            else:
                raise Exception('points must be positive')

        #Starting

        index = 0
        Np = self.network.Np
        Nt = self.network.Nt

        #Calculating corner presence in pores and throats assuming all invaded
        self._set_wp_corner_presence(pore = True, throat = True)
        bool_corner_p = self['pore.wp_corner_presence']
        bool_corner_t = self['throat.wp_corner_presence']

        #For each pressure value calculating phase location (invasion_info) and cluster_info
        for i in p_vals:
            invaded_pores = self['pore.invasion_pressure'] <= i
            invaded_throats = self['throat.invasion_pressure'] <= i
            #Calculating clusters on the network
            center_clusters = _if.make_contiguous_clusters(self.network, invaded_throats) #inlet_pores = self['pore.bc.inlet'])
            #Assumes the wetting phase of all elements are connected due to the wp on corners.
            invaded_pores[center_clusters.pore_clusters == 0] = False #Set inlet pores with non invaded neighbor throats as False

            #info about the invaded phase
            inv_loc_center_p =  ~invaded_pores
            inv_loc_corner_p = bool_corner_p | (np.tile(~invaded_pores,(3,1)).T)

            inv_loc_center_t = ~invaded_throats
            inv_loc_corner_t = bool_corner_t  | (np.tile(~invaded_throats,(3,1)).T)

            #info about the clusters indexes
            cluster_corner_p = (np.tile(center_clusters.pore_clusters,(3,1)).T) * ~inv_loc_corner_p
            cluster_corner_t = (np.tile(center_clusters.throat_clusters,(3,1)).T) * ~inv_loc_corner_t

            #Saving phase location and cluster info
            if index == 0:
                invaded_center_info_p = inv_loc_center_p
                invaded_corner_info_p = inv_loc_corner_p
                cluster_center_info_p = center_clusters.pore_clusters
                cluster_corner_info_p = cluster_corner_p
                invaded_center_info_t = inv_loc_center_t
                invaded_corner_info_t = inv_loc_corner_t
                cluster_center_info_t = center_clusters.throat_clusters
                cluster_corner_info_t = cluster_corner_t
            else:
                invaded_center_info_p = np.vstack((invaded_center_info_p, inv_loc_center_p ))
                invaded_corner_info_p = np.dstack((invaded_corner_info_p, inv_loc_corner_p ))
                cluster_center_info_p = np.vstack((cluster_center_info_p, center_clusters.pore_clusters))
                cluster_corner_info_p = np.dstack((cluster_corner_info_p, cluster_corner_p))
                invaded_center_info_t = np.vstack((invaded_center_info_t, inv_loc_center_t ))
                invaded_corner_info_t = np.dstack((invaded_corner_info_t, inv_loc_corner_t ))
                cluster_center_info_t = np.vstack((cluster_center_info_t, center_clusters.throat_clusters))
                cluster_corner_info_t = np.dstack((cluster_corner_info_t, cluster_corner_t))
            index +=1

        invasion_info = {}
        invasion_info['pore.center'] = invaded_center_info_p.T
        invasion_info['pore.corner'] = invaded_corner_info_p
        invasion_info['throat.center'] = invaded_center_info_t.T
        invasion_info['throat.corner'] = invaded_corner_info_t


        cluster_info = {}
        cluster_info['pore.center'] = cluster_center_info_p.T
        cluster_info['pore.corner'] = cluster_corner_info_p
        cluster_info['throat.center'] = cluster_center_info_t.T
        cluster_info['throat.corner'] = cluster_corner_info_t



        return p_vals, invasion_info, cluster_info

    def set_wp_presence(self, pore_center, pore_corner, throat_center, throat_corner):
        r"""
        Set presence on pores and throats and put on the Algorithm object
        """



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


@njit
def _run_accelerated(t_start, t_sorted, t_order, t_inv, p_inv, p_inv_t,
                     conns, idx, indptr, n_steps):  # pragma: no cover
    r"""
    Numba-jitted run method for InvasionPercolation class.

    Parameters:
    -----------------------

    t_start: t_order, but just for the throats connected to the inlet pores
    t_sorted: Array with the throat indexes. The index of throat with the lower entry pressure is first. Then, the index of the second lower and so on.
    t_order: Ordinary number related to the throat entry pressure
    t_inv: Ascending order for the throat entry pressure
    p_inv: Ascending order for the pore entry pressure
    p_inv_t: Null matrix with the same size as p_inv
    conns: Throat conns (pores connected)
    t_pc: throat entry capillary pressure
    idx: im.indices. Throat indexes that are connected to ecah pore
    indptr=im.indptr. Positions in idx that indicates for each pore, the corresponding throat indexes
    n_steps: numer of steps, usually set as inf

    Returns:
    ---------------------
    t_inv: invasion sequence for throats
    p_inv: invasion sequence for pores
    p_inv_t: Info of which throat is the responsible for the pore invasion
    order: throat indexes ordered according to the invasion sequence. The first element is the index for the first invaded throat and so on
    t_ref: array with the throat indexes used to know the invasion pressure of the throats in array "order"

    Notes
    -----
    ``idx`` and ``indptr`` are properties are the network's incidence
    matrix, and are used to quickly find neighbor throats.

    Numba doesn't like foreign data types (i.e. Network), and so
    ``find_neighbor_throats`` method cannot be called in a jitted method.

    Nested wrapper is for performance issues (reduced OpenPNM import)
    time due to local numba import

    """
    # TODO: The following line is supposed to be numba's new list, but the
    # heap does not work with this
    # queue = List(t_start)
    queue = list(t_start)
    hq.heapify(queue)
    count = 1
    order = []
    t_ref = []
    while count < (n_steps + 1):
        # Find throat at the top of the queue
        t = hq.heappop(queue)
        # Extract actual throat number/index
        t_next = t_sorted[t]
        order.append(t_next)
        #--------------
        if len(order) == 1:
            t_ref.append(t_next)
        elif t_order[t_ref[-1]] > t_order[t_next]:
            t_ref.append(t_ref[-1])
        else:
            t_ref.append(t_next)
            count +=1
        #--------------
        t_inv[t_next] = count
        # If throat is duplicated
        while len(queue) > 0 and queue[0] == t:
            _ = hq.heappop(queue)
        # Find pores connected to newly invaded throat from am in coo format
        Ps = conns[t_next]
        # Remove already invaded pores from Ps
        Ps = Ps[p_inv[Ps] < 0]
        # If either of the neighboring pores are uninvaded (-1), set it to
        # invaded and add its neighboring throats to the queue
        if len(Ps) > 0:
            p_inv[Ps] = count
            p_inv_t[Ps] = t_next
            for i in Ps:
                # Get neighboring throat numbers from im in csr format
                Ts = idx[indptr[i]:indptr[i+1]]
                # Keep only throats which are uninvaded
                Ts = Ts[t_inv[Ts] < 0]
            for i in Ts:  # Add throat to the queue
                hq.heappush(queue, t_order[i])
        if len(queue) == 0:
            break
    order = np.array(order)
    return t_inv, p_inv, p_inv_t, order, t_ref











