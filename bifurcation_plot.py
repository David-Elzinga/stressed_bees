import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from bee_model import odes

'''
This code generates 2 pdf images. The first is the bifurcation plot, and the 
second has time series pulled from parameter combinations that are 
highlighted on the bifurcation plot. 
'''

# Define parameters for the bifurcation plot.
K = 10**4; mu = 0.136
sigma = 0.25; c = 2.7
b = 0.95; a = (1 - b)/(c*b); rho = 0; nu = 2*mu

# Define the critical gamma value so phi > 0 is possible. 
gamma_crit = mu*(a + sigma/(mu + c*sigma))

# For various gamma values, determine the critical beta value so phi > 0 holds. 
gamma = np.linspace(0,0.5,100)
beta_crit = (1 + rho/nu)*(gamma - 2*a*mu - sigma*(a*c+1) + np.sqrt(gamma**2 + 2*gamma*sigma*(a*c - 1) + sigma**2*(a*c + 1)**2))/(2*a)

# Plot the results! 
plt.axvline(x=gamma_crit, color='black', linestyle='--', label=r'$\gamma = \gamma^*$')
plt.plot(gamma, beta_crit, 'k-', label=r'$\beta = \beta^*(\gamma)$')
plt.fill_between(gamma, beta_crit, 10, color='silver')
plt.xlim(0, 0.3); plt.ylim(0, 1)
plt.xlabel(r'$\gamma$',fontsize=20); plt.ylabel(r'$\beta$',rotation=0, labelpad = 30, fontsize=20)
plt.legend(fontsize=10)

plt.text(0.2, 0.45, "Persistence", fontsize=10)
plt.text(0.09, 0.45, "Extinction \nby Stressor", fontsize=10, horizontalalignment='center')
plt.text(0.022, 0.45, "Forced \nExtinction", fontsize=10, horizontalalignment='center')

plt.plot([0.15, 0.15], [0.2, 0.8], 'r.',markersize=20) # adds the red dots. 
plt.savefig("bif_plot_dot.pdf", bbox_inches='tight')

# Run two time series with parameters corresponding to the dots from above.
parm = {}
parm['gamma'] = 0.15; parm['b'] = 0.95; parm['K_expn'] = 4; parm['w_expn'] = -6
parm['c'] = 2.7; parm['sigma'] = 0.25; parm['y_expn'] = -6; parm['mu'] = 0.136
parm['auto'] = True; parm['phi_expn'] = 3.5

parm['p'] = 0; parm['nu'] = 2*parm['mu']; parm['rho'] = 0
parm['t0'] = 0; parm['t1'] = 180

fig, ax = plt.subplots(1,2, figsize=(15,5))
num_years = 2; tsol = np.linspace(0, 180*num_years, num_years*100)
for n, parm['beta'] in enumerate([0.2, 0.8]):
    [H, FU, FI] = solve_ivp(fun=odes, t_span=[tsol.min(), tsol.max()], t_eval=tsol, y0=[200, 50, 0], args=(parm,)).y
    S = parm['sigma']*(H - parm['c']*FU*(1 + parm['p']*FI/(FU + 10**parm['y_expn'])))

    ax[n].plot(tsol, H, label=r'$H$', linewidth=2)
    ax[n].plot(tsol, FU, label=r'$F_U$', linewidth=2)
    ax[n].plot(tsol, FI, label=r'$F_I$', linewidth=2)

    ax_twin = ax[n].twinx()
    ax_twin.plot(tsol, [H[k]/S[k] if H[k] > 5 else np.nan for k in range(len(tsol))], color='red', linestyle='--', linewidth=2)
    ax_twin.set_ylim(0, 18)
    ax_twin.spines['right'].set_color('red'); ax_twin.tick_params(axis='y', colors='red')
    ax_twin.set_ylabel(r'AAOF', rotation=90, fontsize=14, labelpad=10, color='red')

ax[0].legend()
plt.subplots_adjust(wspace=0.5)
plt.savefig('two_stressor_comparison.pdf')