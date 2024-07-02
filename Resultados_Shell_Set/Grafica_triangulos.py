import numpy as np
import math as  m
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16}) #modify all fonts in plots
from matplotlib.ticker import StrMethodFormatter # to modify decimals in axis
from scipy.odr import *
import scipy.stats as stats

#R2
def R2(y_actual, y_predicted):
  SSR = np.sum((y_actual-y_predicted)**2)
  SST = np.sum((y_actual - np.mean(y_actual))**2)
  return 1-SSR/SST

vdata = np.load('info_v_G_Gp_type.npy') #v/G/Gp/type
vetor_v = np.unique(vdata[:,0])

v=200

pos = sum([x for x in vetor_v if x < v])
pos = int(pos)

vdata_fil = vdata[pos:pos+v,:]

pos1 = sum([1 for x in vdata_fil[:,3] if x == 0 ])

Gp_total = vdata_fil[:,2]
G_total = vdata_fil[:,1]
xdata_0 = Gp_total[0:pos1]
xdata_1 = Gp_total[pos1:]
b0 = (xdata_0 != 1/v)
b1 = (xdata_1 != 1/v)
xdata_0 = np.array([x for x in xdata_0*b0 if x != 0])
xdata_1 = np.array([x for x in xdata_1*b1 if x != 0])
ydata_0 = G_total[0:pos1]
ydata_1 = G_total[pos1:]
ydata_0 = np.array([x for x in ydata_0*b0 if x != 0])
ydata_1 = np.array([x for x in ydata_1*b1 if x != 0])
print(b0)

N_0 = len(xdata_0)
N_1 = len(xdata_1)

# Define a function (o fit the data with.
#No le cambio el nombre para saber donde esta
def exp_expo(p, x): #Exponencial
    return  (x*v)**p/(2*v)

p_expo = [1.]

# Create a model for fitting.
exp_expo_model = Model(exp_expo)

# Create a RealData object using our initiated data from above.
#data = RealData(xdata, ydata, sx=xdatadev,  sy=ydatadev)  # ponderando com os desvios padrão de x e y => ODR
# data = RealData(xdata, ydata, sy=ydatadev)  # ponderando com os desvios padrão só de y => mínimos quadrados ponderados (chiSquare)
data_0 = RealData(xdata_0, ydata_0)  # sem ponderar a função objetivo com os desvios padrão => mínimos quadrados simples
data_1 = RealData(xdata_1, ydata_1)

# Set up ODR with the model and data.
odr_expo_0 = ODR(data_0, exp_expo_model, beta0=p_expo)
odr_expo_1 = ODR(data_1, exp_expo_model, beta0=p_expo)

# Run the regression.
out_expo_0 = odr_expo_0.run()
out_expo_1 = odr_expo_1.run()

# Use the in-built pprint method to give us results.
out_expo_0.pprint()
out_expo_1.pprint()

# o segundo parãmetro de t.interval é
# o número de graus de liberadade = números de dados menos o número de parãmetros
t_expo_0 = stats.t.interval(0.95,N_0-1)
t_expo_1 = stats.t.interval(0.95,N_1-1)


R_2_e0 = R2(ydata_0, exp_expo(out_expo_0.beta, xdata_0))
R_2_e1 = R2(ydata_1, exp_expo(out_expo_1.beta, xdata_1))

plt.figure(1, figsize = (7,7))

plt.plot(Gp_total, G_total, 'p', label='data for $G_p = G_{p,min}$', zorder = 1)
plt.plot(xdata_0, ydata_0, 'pg', label= 'data for '+ r'$ \phi \leq \pi /3$', zorder = 3)
plt.plot(xdata_1, ydata_1, 'pr', label= 'data for '+ r'$ \phi > \pi /3$', zorder = 2)

x_d_plot = np.linspace(1/v, xdata_1.max()*1.05, 100)
"""
plt.plot(x_d_plot, exp_expo(out_expo_0.beta, x_d_plot), 'b-',label='corr: c=%6.3f' % tuple(out_expo_0.beta)+
         '\nerro95 = %6.3f' % tuple(t_expo_0[1]*out_expo_0.sd_beta)+'\n$R^2$ =%6.4f '% R_2_e0)
plt.plot(x_d_plot, exp_expo(out_expo_1.beta, x_d_plot), 'm-',label='corr: c=%6.3f' % tuple(out_expo_1.beta)+
         '\nerro95 = %6.3f' % tuple(t_expo_1[1]*out_expo_1.sd_beta)+'\n$R^2$ =%6.4f '% R_2_e1)
"""
#plt.plot(x_d_plot, (x_d_plot*v)**1.25/(2*v), 'k-',label='prova')

y_vetor = np.linspace(0,xdata_1.max()*1.05)
plt.plot(y_vetor,y_vetor,'--k',label = "$G_p$ = $G$")
plt.xlim(0,xdata_1.max()*1.05)
plt.ylim(0,ydata_1.max()*1.05)

plt.xlabel('$G_p$', fontsize = 15)

plt.ylabel('$G$', fontsize = 15)

plt.legend(fontsize = 14)

plt.show()

