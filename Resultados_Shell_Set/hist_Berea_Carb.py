import numpy as np
import scipy as sp
import openpnm as op
import time
import matplotlib.pyplot as plt  #matplot lbi version 3.5.2
import math as m
import scipy.stats as stats
import matplotlib

#Berea tiene 10001 gargantas
plt.rcParams.update({'font.size': 20})

ws = op.Workspace()

resolution_C = 2.85e-6 #For berea : 5.345e-6 / For carbonate: 2.85e-6

timeZero = time.time()
testName = 'Carbonatehigh.pnm' #'Carbonate_std_legacy.pnm' #'Carbonate_adv_legacy.pnm'
net_type = 'high' #standard or high

proj_C = ws.load_project(filename=testName)
print('Elapsed time reading ',testName ,'= ', (time.time()-timeZero)/60, ' min')

#----------------------------------------------
#For Berea
resolution_B = 5.345e-6 #For berea : 5.345e-6 / For carbonate: 2.85e-6

timeZero = time.time()
testName = 'Bereahigh.pnm' #'Berea_std_legacy.pnm' #'Berea_adv_legacy.pnm'
net_type = 'high' #standard or high

proj_B = ws.load_project(filename=testName)
print('Elapsed time reading ',testName ,'= ', (time.time()-timeZero)/60, ' min')

pn_B = proj_B.network
pn_C = proj_C.network

#print(pn_B)
#print(pn_C)

def cum_hist2(G):
    Gsort = np.sort(G)
    N = len(G)
    order = np.arange(N) + 1
    return Gsort, order/(N+1)

i = 0

def hist_plots(pn, resolution, ind):
    Nt = pn.Nt
    voxel_V = resolution**3
    pn['pore.internal'] = True
    pn['pore.internal'][pn['pore.boundary']] = False
    boundary_p = pn.pores('boundary')
    boundary_t = pn.find_neighbor_throats(pores=boundary_p)
    pn['throat.internal'] = True
    pn['throat.internal'][boundary_t] = False

    #To plot a bar chart
    #op.teste.geometry.Gcorr_beta_throats(pn, prob  = 0.5)
    op.teste.geometry.geo_props_eq_pores(pn)

    #More pore data
    A = pn['pore.cross_sectional_area'][pn['pore.internal']]
    L = pn['pore.length'][pn['pore.internal']]
    a = (4*A/3**0.5)**0.5
    a_L = a/L
    max_min_aL = np.maximum(a,L)/ np.minimum(a,L)
    nv_p = pn['pore.region_volume'][pn['pore.internal']] / voxel_V

    #Pore data sorted

    A_sort, A_hist = cum_hist2(A)
    L_sort, L_hist = cum_hist2(L)
    a_sort, a_hist = cum_hist2(a)
    a_L_sort, a_L_hist = cum_hist2(a_L)
    maxmin_sort, maxmin_hist = cum_hist2(max_min_aL)
    nv_p_sort, nv_p_hist = cum_hist2(nv_p)

    #Gráfico 2
    if Nt > 10000:
        ax2.plot(a_L_sort,  a_L_hist,'k', linewidth = 3, label = 'Berea data')
    else:
        ax2.plot(a_L_sort,  a_L_hist,'b', linewidth = 3, label = 'Carbonate data')
        #plotting sqrt(3) linewidth
        a_L_max = np.ones_like(a_L_sort)*3**0.5
        ax2.plot(a_L_max, a_L_hist , '--r', linewidth = 2, label = '$a/L = \sqrt{3}$')

    #Grafico 3
    if Nt > 10000:
        ax3.plot(nv_p_sort,  nv_p_hist,'k', linewidth = 3, label = 'Berea data')
    else:
        ax3.plot(nv_p_sort,  nv_p_hist,'b', linewidth = 3, label = 'Carbonate data')
fig2, ax2 = plt.subplots(1, figsize=(7, 6))
fig3, ax3 = plt.subplots(1, figsize=(7, 6))
hist_plots(pn_B, resolution_B, i)
hist_plots(pn_C, resolution_C, i)

xmax = 1 #exponent of 10
xmin = -2 #exponent of 10
ax2.set_xlabel('$a/L$')
ax2.set_ylabel('$Fr_a$')
ax2.set_xlim([10**xmin, 10**xmax])
ax2.set_xticks([0.01, 0.1, 1, 10])
ax2.set_ylim([0, 1])
ax2.set_xscale('log')
ax2.set_xticks(np.logspace(xmin,xmax,xmax-xmin+1))
ax2.legend(fontsize="16")
ax2.grid(True)
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
fig2.savefig(fname = 'P2_'+net_type+'_port.pdf', format = 'pdf', pad_inches=0)

xmax = 5 #exponent of 10
xmin = 1 #exponent of 10
ax3.set_xlabel('$n_v$')
ax3.set_ylabel('$Fr_a$')
ax3.set_xlim([10**xmin, 10**xmax])
ax3.set_ylim([0, 1])
ax3.set_xscale('log')
ax3.set_xticks(np.logspace(xmin,xmax,xmax-xmin+1))
ax3.legend(fontsize="16")
ax3.grid(True)
ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
fig3.savefig(fname = 'P3_'+net_type+'_port.pdf', format = 'pdf', pad_inches=0)

"""
    #Grafico 2
    plt.figure(2, figsize = [7, 6])
    if Nt > 10000:
        plt.plot(a_L_sort,  a_L_hist,'k', linewidth = 3, label = 'Berea data')
    else:
        plt.plot(a_L_sort,  a_L_hist,'b', linewidth = 3, label = 'Carbonate data')
        #plotting sqrt(3) linewidth
        a_L_max = np.ones_like(a_L_sort)*3**0.5
        plt.plot(a_L_max, a_L_hist , '--r', linewidth = 2, label = '$a/L = \sqrt{3}$' )


    #Grafico 3
    plt.figure(3, figsize = [7, 6])
    if Nt > 10000:
        plt.plot(nv_p_sort,  nv_p_hist,'k', linewidth = 3, label = 'Berea data')
    else:
        plt.plot(nv_p_sort,  nv_p_hist,'b', linewidth = 3, label = 'Carbonate data')

hist_plots(pn_B, resolution_B, i)
hist_plots(pn_C, resolution_C, i)

xmax = 1 #exponent of 10
xmin = -2 #exponent of 10
plt.figure(2)
plt.xlabel('$a/L$')
plt.ylabel('$Fr_a$')
plt.xlim([10**xmin, 10**xmax])
plt.ylim([0, 1])
plt.xscale('log')
plt.xticks(np.logspace(xmin,xmax,xmax-xmin+1))
plt.legend(fontsize="16")
plt.grid(True)
plt.savefig(fname = 'P2_'+net_type+'_port.pdf', format = 'pdf', pad_inches=0)

xmax = 5 #exponent of 10
xmin = 1 #exponent of 10
plt.figure(3)
plt.xlabel('$n_v$')
plt.ylabel('$Fr_a$')
plt.xlim([10**xmin, 10**xmax])
plt.ylim([0, 1])
plt.xscale('log')
plt.xticks(np.logspace(xmin,xmax,xmax-xmin+1))
plt.legend(fontsize="16")
plt.grid(True)
plt.savefig(fname = 'P3_'+net_type+'_port.pdf', format = 'pdf', pad_inches=0)
"""
