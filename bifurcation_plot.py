import numpy as np
import matplotlib.pyplot as plt

K = 10**4; mu = 0.136
sigma = 0.25; c = 2.7
b = 0.95; a = (1 - b)/(c*b); rho = 0; nu = 2*mu
gamma_crit = mu*(a + sigma/(mu + c*sigma))

gamma = np.linspace(0,0.5,100)
beta_crit = (1 + rho/nu)*(gamma - 2*a*mu - sigma*(a*c+1) + np.sqrt(gamma**2 + 2*gamma*sigma*(a*c - 1) + sigma**2*(a*c + 1)**2))/(2*a)

plt.axvline(x=gamma_crit, color='black', linestyle='--', label=r'$\gamma = \gamma^*$')
plt.plot(gamma, beta_crit, 'k-', label=r'$\beta = \beta^*(\gamma)$')
plt.xlim(0, 0.3); plt.ylim(0, 1)
plt.xlabel(r'$\gamma$',fontsize=20); plt.ylabel(r'$\beta$',rotation=0, labelpad = 30, fontsize=20)
plt.legend(fontsize=10)
plt.text(0.2, 0.45, "Persistence", fontsize=10)
plt.text(0.09, 0.45, "Extinction \nby Stressor", fontsize=10, horizontalalignment='center')
plt.text(0.022, 0.45, "Forced \nExtinction", fontsize=10, horizontalalignment='center')
plt.fill_between(gamma, beta_crit, 10, color='silver')
plt.savefig("bif_plot.pdf",bbox_inches='tight')

plt.plot([0.15, 0.15], [0.2, 0.8], 'r.',markersize=20)
plt.savefig("bif_plot_dot.pdf",bbox_inches='tight')
plt.show()