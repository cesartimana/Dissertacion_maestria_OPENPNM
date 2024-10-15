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
sys.path.insert(1, '/home/cesar/OpenPNM_files/mestrado/_funcs')
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
sp = 5e-4
#SOLO PARA RED CREADA, ACTUALIZANDO NUMERO DE GARGANTAS
Np = pn.Np
Nt = pn.Nt

#Plotting properties

def cum_hist(G, add_final_point = False, ratio_fp = 100):
    Gsort = np.sort(G)
    N = len(G)
    order = np.arange(N) + 1
    if add_final_point:
        order = np.append(order, order[-1] + 1)
        Gsort = np.append(Gsort, Gsort[-1] * ratio_fp)
    return Gsort, order/(N+1)

Dt, y_Dt = cum_hist(pn['throat.equivalent_diameter'][pn['throat.internal']], add_final_point = True)
Dp, y_Dp = cum_hist(pn['pore.equivalent_diameter'][pn['pore.internal']], add_final_point = True)
fig1, ax1 = plt.subplots(figsize = (7,7))
ax1.plot(Dt * 1e6,  y_Dt,'b', linewidth = 3, label = 'throats')
ax1.plot(Dp * 1e6,  y_Dp,'r', linewidth = 3, label = 'pores')
xticks = np.arange(0, sp*1*1e6 , sp*0.2*1e6)
ax1.set_xticks(xticks)
ax1.set_xlim([xticks[0], xticks[-1]])
ax1.set_xlabel(r'$x$ ($\mu m$)')
ax1.set_ylabel(r'$\mathbb{P}(D_{eq} < x)$')
yticks = np.arange(0,1.1,0.25)
ax1.set_yticks(yticks)
ax1.set_ylim([0, 1.01])
ax1.grid(True)
ax1.legend()
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
yticks = np.arange(0,1.1,0.25)
ax2.set_yticks(yticks)
ax2.set_ylim([0, 1.01])
ax2.grid(True)
ax2.legend()
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
yticks = np.arange(0,1.1,0.25)
ax4.set_yticks(yticks)
ax4.set_ylim([0, 1.01])
ax4.grid(True)
#ax1.legend()
plt.tight_layout()


#Histograma de z
#Este histograma solo considera poros internos y gargantas internas
op.topotools.trim(network=pn, throats=pn['throat.boundary'])

# Create incidence matrix with info of neighbor throats for each pore
im = pn.create_incidence_matrix(fmt='csr')
idx=im.indices
indptr=im.indptr

mask = pn['pore.internal']
connectivity = []
for p in range(Np):
    if mask[p]:
        Ts = idx[indptr[p]:indptr[p+1]]
        connectivity.append(len(Ts))
print(np.mean(connectivity))
print(np.min(connectivity))
print(np.max(connectivity))
fig5, ax5 = plt.subplots(figsize = (9,7))
bins = np.arange(np.min(connectivity) - 0.5, np.max(connectivity) + 1.5)
ax5.hist(connectivity, bins = bins, density = True, edgecolor='black', zorder = 10)
xticks = np.arange(np.min(connectivity), np.max(connectivity) + 1)
yticks = np.arange(0,0.25,0.05)
ax5.set_xticks(xticks)
ax5.set_yticks(yticks)
ax5.set_xlabel('x')
ax5.set_ylabel('$\mathbb{P}(Z = x)$')
ax5.grid(True, axis = 'y', zorder = 0)
plt.tight_layout()

#PLOTAR USANDO PI, MODALIDAD ESPECIAL

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$%s/%s$'%(latex,den)
                #return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$-%s/%s$'%(latex,den)
                #return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$%s%s/%s$'%(num,latex,den)
                #return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

beta_rep = (pn['throat.half_corner_angle'][pn['throat.internal']])[:,1]
beta_isol = np.pi / 2 - 2 * beta_rep

beta1, y_beta1 = cum_hist(beta_rep)
beta2, y_beta2 = cum_hist(beta_isol)
fig3, ax3 = plt.subplots(figsize = (7,7))
ax3.plot(beta1,  y_beta1,'b', linewidth = 3, label = r'$\beta_{rep}$')
ax3.plot(beta2,  y_beta2,'r', linewidth = 3, label = r'$\beta_{isol}$')
ax3.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 6))
ax3.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter(denominator = 6)))
ax3.set_xlim([0, 1.7])
ax3.set_xlabel('$x$ (rad)')# (x $10 ^{-5}$)')
ax3.set_ylabel(r'$\mathbb{P}(\beta < x)$')
yticks = np.arange(0,1.1,0.25)
ax3.set_yticks(yticks)
ax3.set_ylim([0, 1.])
ax3.legend( loc = 'lower center', bbox_to_anchor=(0.7, 0.02) )
ax3.grid(True)
plt.tight_layout()

plt.show()
