import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

theta_a_sexagec = 180
theta_a = np.pi / 180 * theta_a_sexagec
theta_r = np.linspace(0, min(theta_a, np.pi / 2), 5)
theta_r[-1] = theta_r[-1] - np.pi / 180
theta_r[0] = theta_r[0] + np.pi / 180
beta = np.linspace(np.pi/2 - theta_a, np.pi / 2 , 500)

ones_array = np.ones(10)

plt.figure(1, figsize = (8,6))

for i in range(len(theta_r)):
    ratio_p = np.cos(theta_a + beta) / np.cos(theta_r[i] + beta)
    theta_r_sexagec =  round(theta_r[i] / np.pi * 180)
    plt.scatter(beta / np.pi * 180, ratio_p, s= 2, label = r'$\theta _r$ = ' + str(theta_r_sexagec) + r'$^o$')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$p_c / p_{c,max}$')
plt.title(r'$\theta_a = $' + str(theta_a_sexagec) + r'$^o$')
plt.xlim([0 , 90])
plt.ylim([-10,0])
plt.legend(loc = 'upper right')
plt.tight_layout()


print(np.sin(theta_a + beta))

plt.figure(2, figsize = (8,6))
plt.scatter(beta / np.pi * 180, np.cos(theta_a + beta), label = r'$\theta_a = $' + str(theta_a_sexagec) )
plt.scatter(beta / np.pi * 180, np.cos(theta_r[0] + beta), label = r'$\theta_r = $' + str(int(theta_r[0] / np.pi * 180)) )
plt.legend(loc = 'upper right')
plt.xlabel(r'$\beta$')
plt.ylabel(r'cos($\theta + \beta$)')
plt.xlim([0 , 90])
plt.tight_layout()
plt.show()
