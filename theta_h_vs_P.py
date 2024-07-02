import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

beta = np.array([0.001, 0.101, 0.2, 0.3, 0.399]) * np.pi
ratio_p = np.linspace(-0.6, 1, 50)
theta_r = np.pi * 0.1

#Revisar si hay fase mojante en las esquinas: beta < pi/2 - theta_r
print(beta)
print(np.pi / 2 - theta_r)
print(beta < (np.pi / 2 - theta_r))

plt.figure(1, figsize = (8,8))
for i in range(len(beta)):
    theta_h = np.arccos(ratio_p * np.cos(theta_r + beta[i]) ) - beta[i]
    plt.plot(ratio_p, theta_h, label = r'$ \beta_{%i} = %0.3f \pi$' %tuple([i + 1, beta[i]/np.pi]))
plt.plot(ratio_p, np.ones_like(ratio_p) * np.pi / 2, '--k', label = r'$\theta_h = \pi / 2$')
plt.grid(True)
plt.legend()
plt.xlim([ratio_p[0], ratio_p[-1]])
y_ticks = np.linspace(0.25, 2.5, 10)
plt.yticks(y_ticks)
plt.ylim(y_ticks[0], y_ticks[-1])
plt.show()
