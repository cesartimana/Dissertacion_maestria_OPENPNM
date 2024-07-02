import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

filePP = 'Berea1.pnm'

prjPP = op.io.PNM.load_project(filePP)
pnPP = prjPP.network
#pnPP['throat.equivalent_diameter'] = ( pnPP['throat.area'] * 4 / np.pi) **0.5 # Modifica un 2% el resultado
print(pnPP)
#pnPP['pore.diameter'] = pnPP['pore.extended_diameter']
geoPP = op.geometry.Imported(network=pnPP)

#geoPP['throat.equivalent_diameter'] = ( geoPP['throat.area'] * 4 / np.pi) **0.5 #No afecta al resultado

water = op.phases.Water(network=pnPP)
phys = op.physics.GenericPhysics(network=pnPP, phase=water, geometry=geoPP)
shape_model = op.models.physics.flow_shape_factors.ball_and_stick
phys.add_model(propname='throat.flow_shape_factors', model=shape_model)
h_model = op.models.physics.hydraulic_conductance.hagen_poiseuille
phys.add_model(propname='throat.hydraulic_conductance', model=h_model)

inlet = pnPP.pores('top')
outlet = pnPP.pores('bottom')
#X: left right / Y : front-back / Z: top-bottom

sf = op.algorithms.StokesFlow(network=pnPP, phase=water)
sf.set_value_BC(pores=inlet, values=1)
sf.set_value_BC(pores=outlet, values=0.0)
sf.run()
Lx = pnPP['pore.coords'][:, 0].max() - pnPP['pore.coords'][:, 0].min()
Ly = pnPP['pore.coords'][:, 1].max() - pnPP['pore.coords'][:, 1].min()
Lz = pnPP['pore.coords'][:, 2].max() - pnPP['pore.coords'][:, 2].min()

A_xx = Ly*Lz  # Since the network is cubic Lx = Ly = Lz
print('Lx', Lx, Ly, Lz)

perm = np.squeeze(sf.calc_effective_permeability(inlets=inlet, outlets=outlet,
        domain_length= Lx, domain_area=A_xx))
print('X-permeability m2 =', perm)
print('permeability D =', perm * 1013250273830.8866)

porosity = pnPP['pore.region_volume'].sum()/Lx/A_xx
print('Porosity =', porosity)

#op.io.VTK.export_data(network=pnPP, phases=water, filename= 'BereaParaview')

