import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

pn = op.io.network_from_csv('Berea.csv')
