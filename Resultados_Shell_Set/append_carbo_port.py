import numpy as np
import scipy as sp
import openpnm as op
import time
import matplotlib.pyplot as plt  #matplot lbi version 3.5.2
import math as m
import scipy.stats as stats

plt.rcParams.update({'font.size': 20})

ws = op.Workspace()

resolution = 2.85e-6 #For berea : 5.345e-6 / For carbonate: 2.85e-6
voxel_area = resolution**2
voxel_V = resolution**3

timeZero = time.time()
testName = 'Carbonatehigh.pnm' #'Carbonate_std_legacy.pnm' #'Carbonate_adv_legacy.pnm'
net_type = 'high' #standard or high

proj = ws.load_project(filename=testName)
print('Elapsed time reading ',testName ,'= ', (time.time()-timeZero)/60, ' min')

pn = proj.network

pn['pore.internal'] = True
pn['pore.internal'][pn['pore.boundary']] = False
boundary_p = pn.pores('boundary')
boundary_t = pn.find_neighbor_throats(pores=boundary_p)
pn['throat.internal'] = True
pn['throat.internal'][boundary_t] = False

#print(pn)

def shape_factor(A,P):
    G= np.divide(A,np.power(P,2))
    return G

"""
Analysis for internal throats
We haver cross sectional area and perimeter
We are going to calculate G and classify according to its value (Blunt 2017)
Between 0 and 0.0481 - Triangle (T)
Between 0.0481 and 0.0625 - Rectangle (R)
Between 0.0625 and 0.0796 - Ellipse (E)
More than 0.0796 - Others (O)
"""

A_i=pn['throat.cross_sectional_area'][pn['throat.internal']]
P_i=pn['throat.perimeter'][pn['throat.internal']]
d_i=pn['throat.inscribed_diameter'][pn['throat.internal']]
de_i=pn['throat.equivalent_diameter'][pn['throat.internal']]
ratio_d = de_i/d_i #ratio between equivalent and inscribed diameter
#print(len([x for x in ratio_d if x <= 1]))

Ac_i = m.pi*np.power(d_i,2)/4
At_i = 3*m.sqrt(3)*np.power(d_i,2)/4

tda = abs(A_i - Ac_i)/A_i #diferença de areas antigas = a real - a circular inscrita
tdn = abs(A_i - At_i)/A_i #diferença de areas atual = area real -a triangular
t_res = tda - tdn #pore results old - new
tbo = len([x for x in t_res if x <= 0]) #number of throats better old
tbn = len(t_res) - tbo #number of throats better old
#print('gargantas boas com triangulo equilatero',round(100*tbn/len(t_res),2))


G_ti=shape_factor(A_i,P_i)


def min_max_av(array):
    minimum = min(array)
    maximum = max(array)
    av = sum(array)/len(array)
    R = [minimum, maximum, av]
    return R

#print(min_max_av(A_i))
#print(min_max_av(P_i))
#print(min_max_av(G_ti))


nG_ti=len([x for x in G_ti if x <= 1/4/m.pi]) #colocar x for cada x in G_ti que cumpla la condicion if ... no olvidar []
#print('numeros reales de Gi', nG_ti)
#print('porcentaje de numeros reales de Gi:', round(100*nG_ti/len(G_ti),2))

#-------------------------------------
#To plot a bar chart

def cum_hist(G, xmin = -5, xmax = 0, nbin = 20):
    #xmin, xmax exponents of 10
    rel, bins = np.histogram(G, bins=np.logspace(xmin,xmax,nbin),  weights = np.ones_like(G) / (len(G)+1)) #Recomendacion de Paulo F = i/(N+1)
    absol = np.zeros_like(rel)
    absol[0] = rel[0]
    for i in range(len(absol)-1):
      absol[i+1] = absol[i] + rel[i+1]
    bin_centres = (bins[:-1] + bins[1:])/2
    return bin_centres, absol


def cum_hist2(G):
    Gsort = np.sort(G)
    N = len(G)
    order = np.arange(N) + 1
    return Gsort, order/(N+1)


nbin = 20
xmax = 0 #exponent of 10
xmin = -4 #exponent of 10

#bin_old, abs_old = cum_hist(G_ti, xmin = xmin, xmax = xmax, nbin = nbin)
bin_old, abs_old = cum_hist2(G_ti)

op.teste.geometry.Gcorr_beta_throats(pn, prob  = 0.5)
op.teste.geometry.geo_props_eq_pores(pn)
G_new = pn['throat.shape_factor'][pn['throat.internal']]

#bin_new, abs_new = cum_hist(G_new, xmin = xmin, xmax = xmax, nbin = nbin)
bin_new, abs_new = cum_hist2(G_new)

plt.figure(1, figsize = [7, 6])
y = np.linspace(0,1,50)
y_1 = np.ones_like(y) #vector of 1 and shape of y
plt.plot(bin_old, abs_old,'k', linewidth = 3)
plt.plot(bin_new, abs_new,'r', linewidth = 3)
plt.plot(m.sqrt(3)/36*y_1, y, 'b--', 0.0625*y_1, y, 'r--', 1/4/m.pi*y_1, y, 'g--' )
plt.xlabel('$G$')
plt.ylabel('$Fr_a$')
plt.xlim([10**xmin, 10**xmax])
plt.ylim([0, 1])
plt.xscale('log')
plt.xticks(np.logspace(xmin,xmax,xmax-xmin+1))
plt.grid(True)
plt.legend(['Dados antigos' , 'Dados corrigidos', '$G_{max}$ para triângulo', '$G_{max}$ para rectângulo', '$G_{max}$ para elipse'], fontsize="16")
plt.savefig(fname = 'Carb_G_'+net_type+'_port.pdf', format = 'pdf', pad_inches=0)

