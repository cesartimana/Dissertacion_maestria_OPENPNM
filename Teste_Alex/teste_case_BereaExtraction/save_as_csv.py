import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

filePP = 'Berea.pnm'

prjPP = op.io.PNM.load_project(filePP)
pnPP = prjPP.network
op.io.CSV.export_data(network=pnPP, filename='Berea')
