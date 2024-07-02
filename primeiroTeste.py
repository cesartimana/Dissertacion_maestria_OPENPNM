import numpy as np
import openpnm as op
op.visualization.set_mpl_style()
np.random.seed(10)
#%matplotlib inline
np.set_printoptions(precision=5)
pn = op.network.Cubic(shape=[15, 15, 15], spacing=1e-6)



pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
pn.regenerate_models()
phase = op.phase.Phase(network=pn)
phase['pore.viscosity']=1.0
phase.add_model_collection(op.models.collections.physics.basic)
phase.regenerate_models()
inlet = pn.pores('left')
outlet = pn.pores('right')

"""

flow = op.algorithms.StokesFlow(network=pn, phase=phase)
flow.set_value_BC(pores=inlet, values=1)
flow.set_value_BC(pores=outlet, values=0)
flow.run()
phase.update(flow.soln)
ax = op.visualization.plot_connections(pn)
ax = op.visualization.plot_coordinates(pn, ax=ax, color_by=phase['pore.pressure'])

"""

print(pn)

print(type(pn))
