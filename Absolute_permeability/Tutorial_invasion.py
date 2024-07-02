import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
op.visualization.set_mpl_style()

np.random.seed(5)
pn = op.network.Demo(shape=[20, 20, 1], spacing=1e-4)
air = op.phase.Air(network=pn)
air['pore.contact_angle'] = 120
air['pore.surface_tension'] = 0.072
f = op.models.physics.capillary_pressure.washburn
air.add_model(propname='throat.entry_pressure',
              model=f,
              surface_tension='throat.surface_tension',
              contact_angle='throat.contact_angle',
              diameter='throat.diameter',)

ip = op.algorithms.InvasionPercolation(network=pn, phase=air)
pn['pore.volume'][1] = 0.0
ip.set_inlet_BC(pores=[1])
np.random.seed(5)
ip.run()

inv_pattern = ip['throat.invasion_pressure'] <= 5000
ax = op.visualization.plot_coordinates(network=pn, pores=pn.pores('left'), c='r', s=50)
ax = op.visualization.plot_coordinates(network=pn, pores=pn.pores('left', mode='not'), c='grey', ax=ax)
op.visualization.plot_connections(network=pn, throats=inv_pattern, ax=ax)
plt.show()

clusters = op.topotools._perctools.find_clusters(pn, inv_pattern)
c_pores = np.copy(clusters.pore_labels)
c_throats = np.copy(clusters.throat_labels)

print(c_pores)
print(c_throats)
