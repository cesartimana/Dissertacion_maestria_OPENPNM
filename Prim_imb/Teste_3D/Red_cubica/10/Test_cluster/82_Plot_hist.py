import openpnm as op
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import copy
from tqdm import tqdm
from Properties import *
#importing functions for other file
import sys
sys.path.insert(1, '/home/cbeteta/_funcs')
import _algorithm_class as _alg
import _conductance_funcs as _cf
import _invasion_funcs as _if
plt.rcParams.update({'font.size': 22})
#end

#Reading pnm arquive/ network
ws = op.Workspace()
proj = ws.load_project(filename=testName)
#proj = ws.load_project(filename='Berea.pnm')
pn = proj.network

#SOLO PARA RED CREADA, ACTUALIZANDO NUMERO DE GARGANTAS
Np = pn.Np
Nt = pn.Nt
print(pn)

#Plotting properties

def cum_hist(G):
    Gsort = np.sort(G)
    N = len(G)
    order = np.arange(N) + 1
    return Gsort, order/(N+1)

#Dt, y_Dt = cum_hist(pn['throat.diameter'])
Dp, y_Dp = cum_hist(pn['pore.equivalent_diameter'][pn['pore.internal']])
fig1, ax1 = plt.subplots(figsize = (7,7))
#ax1.plot(Dt * 1e5,  y_Dt,'b', linewidth = 3, label = 'throats')
ax1.plot(Dp * 1e6,  y_Dp,'r', linewidth = 3, label = 'pores')
xticks = np.arange(60, 80, 10)
#ax1.set_xticks(xticks)
#ax1.set_xlim([xticks[0], xticks[-1]])
ax1.set_xlabel(r'$x$ ($\mu m$)')
ax1.set_ylabel(r'$\mathbb{P}(D_{eq, p} < x)$')
ax1.set_ylim([0, 1])
#ax1.legend()
plt.tight_layout()

Gt, y_Gt = cum_hist(pn['throat.shape_factor'][pn['throat.internal']])
fig2, ax2 = plt.subplots(figsize = (7,7))
ax2.plot(Gt,  y_Gt,'b', linewidth = 3, label = 'throats')
ax2.plot(np.ones_like(Gt) * 3 ** 0.5/36 ,  y_Gt,'k--', linewidth = 2, label = '$x = \sqrt{3}/36$')
xticks = np.arange(0, 0.06, 0.01)
ax2.set_xticks(xticks)
ax2.set_xlim([xticks[0], xticks[-1]])
ax2.set_xlabel('$x$')# (x $10 ^{-5}$)')
ax2.set_ylabel('$\mathbb{P}(G_t < x)$')
ax2.set_ylim([0, 1])
ax2.legend()
plt.tight_layout()

beta_rep = (pn['throat.half_corner_angle'][pn['throat.internal']])[:,1]
beta_isol = np.pi / 2 - 2 * beta_rep

beta1, y_beta1 = cum_hist(beta_rep)
beta2, y_beta2 = cum_hist(beta_isol)
fig3, ax3 = plt.subplots(figsize = (7,7))
ax3.plot(beta1,  y_beta1,'b', linewidth = 3, label = r'$\beta_{rep}$')
ax3.plot(beta2,  y_beta2,'r', linewidth = 3, label = r'$\beta_{isol}$')
ax3.plot(np.ones_like(Gt) * np.pi / 2 ,  y_Gt,'k--', linewidth = 2, label = r'$x = \pi / 2$')
xticks = np.arange(0, 2.1, 0.5)
ax3.set_xticks(xticks)
ax3.set_xlim([xticks[0], 1.7])
ax3.set_xlabel('$x$ (rad)')# (x $10 ^{-5}$)')
ax3.set_ylabel(r'$\mathbb{P}(\beta < x)$')
ax3.set_ylim([0, 1.])
ax3.legend( loc = 'lower center', bbox_to_anchor=(0.7, 0.02) )
plt.tight_layout()

At, y_At = cum_hist(pn['throat.cross_sectional_area'][pn['throat.internal']])
fig4, ax4 = plt.subplots(figsize = (7,7))
ax4.plot(At * 1e12,  y_At,'r', linewidth = 3, label = 'throats')
xticks = np.array([1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6])
ax4.set_xticks(xticks)
#ax4.set_xlim([xticks[0], xticks[-1]])
ax4.set_xlim([At[0]*1e12, At[-1]*1e12])
ax4.set_xscale('log')
#ax4.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax4.set_xlabel(r'$x (\mu m^2)$')
ax4.set_ylabel(r'$\mathbb{P}(A_{t} < x)$')
ax4.set_ylim([0, 1.02])
#ax1.legend()
plt.tight_layout()

plt.show()
