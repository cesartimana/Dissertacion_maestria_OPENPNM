import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *
import scipy.stats as stats

np.random.seed(5)


beta_rep = np.linspace( np.pi / 360 , 89 * np.pi / 360,num = 100)
beta_dif = np.pi/2 - 2 * beta_rep

G = 1/4 * np.tan(beta_dif) * np.tan(beta_rep) / np.tan(beta_dif + beta_rep)


beta_dif_white = np.array([0,10,20,30,40,50,60,70,80,90]) * np.pi / 180
beta_rep_white = 0.5 * (np.pi/2 - beta_dif_white)
G_white = 1/4 * np.tan(beta_dif_white) * np.tan(beta_rep_white) / np.tan(beta_dif_white + beta_rep_white)
c_white = 32 / np.array([48, 51.6, 52.9, 53.3,
                       52.9, 52.0, 51.1, 49.5,
                       48.3, 48.0])
factor_corr = c_white / 0.6


def exp_op_1(a, x): #Option 1, c=a/v.
  return a/np.log10(np.log10(x))

def exp_op_2(p, x): #Option 2, y=ax**3 + bx**2 + cx**1 + d.
    a,b,c,d = p
    return a*x**3 + b*x**2 + c*x**1 + d

def exp_op_3(p, x): #Option 3, y = a + b * sin (c * x + d)
    a,b,c,d = p
    return a + b * np.sin(c * x + d)

N = len(G_white)

par_1 = [1] #For expression with 1 parameter
par_4 = [0.1,0.1,0.1,0.1] #For expression with 4 parameters

# Create a model for fitting.
#exp_1 = Model(exp_op_1)
exp_2 = Model(exp_op_2)
exp_3 = Model(exp_op_3)

# Create a RealData object using our initiated data from above.
data_beta = RealData(beta_dif_white, factor_corr)

# Set up ODR with the model and data.
#odr_1 = ODR(data, exp_1, beta0 = par_1)
odr_2 = ODR(data_beta, exp_2, beta0 = par_4)
odr_3 = ODR(data_beta, exp_3, beta0 = par_4)

# Run the regression.
#out_1 = odr_1.run()
out_2 = odr_2.run()
out_3 = odr_3.run()

#Calculating t
#t_1 = stats.t.interval(0.95,N-1) #For expression with 1 parameter
t_2 = stats.t.interval(0.95,N-4) #For expression with 2 parameters
t_3 = stats.t.interval(0.95,N-4) #For expression with 3 parameters

"""
plt.figure(1,figsize = (5,5))
plt.plot(G_white, c_white, 'ok')
plt.xlabel('G')
plt.ylabel('constant')


plt.figure(2,figsize = (5,5))
plt.plot(G_white, factor_corr , '*b')
i = 6 #1, 6 comparten el mismo G
plt.plot(G_white[i], factor_corr[i] , 'pk')
plt.xlabel('G')
plt.ylabel('correction factor')
"""

plt.figure(3,figsize = (7,7))
plt.plot(beta_dif_white, factor_corr , '*b', label = 'Table White')
plt.plot(beta_dif,exp_op_2(out_2.beta, beta_dif), '--k', label = 'Cubic fit')
plt.plot(beta_dif,exp_op_3(out_3.beta, beta_dif), '--r', label = 'Trigon. (sin) fit')
plt.xlabel('beta diff')
plt.ylabel('correction factor')
plt.legend()

plt.show()

print(out_2.beta)
print(out_3.beta)

pol = np.array([-0.17997611,  0.57966346, -0.46275726,  1.10633925])
factorprint = np.polyval(pol, beta_dif_white)
print(factorprint)

