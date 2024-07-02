import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

K_in  = np.load('K_inscribed.npy')
K_ex  = np.load('K_extended.npy')
K_eq = np.load('K_equivalent.npy')


fig1 , ax1 = plt.subplots(1, figsize = (9,7))
ax1.plot(K_in[:, 0], K_in[:, 1], label = 'inscribed', linewidth = 3)
ax1.plot(K_ex[:, 0], K_ex[:, 1], label = 'extended', linewidth = 3)
ax1.plot(K_eq[:, 0], K_eq[:, 1], label = 'equivalent', linewidth = 3)
ax1.set_xlabel('L/R')
ax1.set_ylabel(r'$K_{abs}$')
ax1.set_xscale('log')
ax1.set_xticks([0.01, 0.04, 0.1, 0.4,1])
ax1.set_xlim([0.01, 1])
ax1.set_ylim([0, 1200])
ax1.legend(loc = 'lower left')
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

fig2 , ax2 = plt.subplots(1, figsize = (9,7))
ax2.plot(K_in[:, 0], 1 - K_in[:, 1] / K_in[0, 1], label = 'inscribed', linewidth = 3)
ax2.plot(K_ex[:, 0], 1 - K_ex[:, 1] / K_ex[0, 1], label = 'extended', linewidth = 3)
ax2.plot(K_eq[:, 0], 1 - K_eq[:, 1] / K_eq[0, 1], label = 'equivalent', linewidth = 3)
ax2.set_xlabel('L/R')
ax2.set_ylabel(r'$1 - K_{abs}/ K_{abs, L/R = 0.01}$')
ax2.set_xscale('log')
ax2.set_xticks([0.01, 0.04, 0.1, 0.4,1])
ax2.set_xlim([0.01, 1])
ax2.set_ylim([0, 0.1])
ax2.legend(loc = 'upper left')
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


fig3 , ax3 = plt.subplots(1, figsize = (7,7))
ax3.plot(K_in[:, 0], 100 - 100*K_in[:, 1] / K_in[0, 1], label = 'inscribed', linewidth = 3)
ax3.plot(K_ex[:, 0], 100 - 100*K_ex[:, 1] / K_ex[0, 1], label = 'extended', linewidth = 3)
ax3.plot(K_eq[:, 0], 100 - 100*K_eq[:, 1] / K_eq[0, 1], label = 'equivalent', linewidth = 3)
#ax3.plot([0.8, 1], np.ones_like([0.8, 1]) * 0.05, 'k--', label = 'y = 0.05')
ax3.plot([0.8, 1], np.ones_like([0.8, 1]) * 2.7, 'b--', label = r'$e_{rel} = 2.7 \%$')
ax3.plot(np.ones_like([0, 10]) * 0.95 , [0, 10] , 'r--', label = r'$L/R = 0.95$')
ax3.set_xlabel('L/R')
#ax3.set_ylabel(r'$1 - K_{abs}/ K_{abs, L/R = 0.01}$')
ax3.set_ylabel(r'$e_{rel} (\%)$')
#ax3.set_xscale('log')
ax3.set_xticks([0.8, 0.85, 0.9, 0.95, 1])
ax3.set_xlim([0.8, 1])
ax3.set_ylim([0, 10])
ax3.legend(loc = 'upper left')
#ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.show()

