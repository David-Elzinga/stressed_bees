import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from bee_model import odes
import matplotlib.gridspec as gridspec

'''
This code generates 2 pdf images. The first is the bifurcation plot, and the 
second has time series pulled from parameter combinations that are 
highlighted on the bifurcation plot. 
'''

# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)
plt.figure(figsize=(18,20))

# Define parameters for the bifurcation plot.
K = 10**4; mu = 0.1356
sigma = 0.2769; c = 2.7
b = 0.875; a = (1 - b)/(c*b); rho = 0.2; nu = 2*mu

# Define the critical gamma value so phi > 0 is possible. 
gamma_crit = mu*(a + sigma/(mu + c*sigma))

# For various gamma values, determine the critical beta value so phi > 0 holds. 
gamma = np.linspace(0,0.15,100)
beta_crit = (1 + rho/nu)*(gamma - 2*a*mu - sigma*(a*c+1) + np.sqrt(gamma**2 + 2*gamma*sigma*(a*c - 1) + sigma**2*(a*c + 1)**2))/(2*a)

# Plot the results! 
ax = plt.subplot(gs[0, :])
ax.axvline(x=gamma_crit, color='black', linestyle='--', label=r'$\gamma = \gamma^*$')
ax.plot(gamma, beta_crit, 'k-', label=r'$\beta = \beta^*(\gamma)$')
ax.fill_between(gamma, beta_crit, 10, color='silver')
ax.set_xlim(0, 0.15); ax.set_ylim(0, 0.4)
ax.set_xlabel(r'$\gamma$',fontsize=28); ax.set_ylabel(r'$\beta$',rotation=0, labelpad = 30, fontsize=28)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.legend(fontsize=18)

ax.text(0.12, 0.26, "Persistence", fontsize=18)
ax.text(0.07, 0.25, "Extinction \nby Stressor", fontsize=18, horizontalalignment='center')
ax.text(0.02, 0.25, "Forced \nExtinction", fontsize=18, horizontalalignment='center')

ax.plot([0.0625, 0.0625], [0.05, 0.15], 'r.',markersize=26) # adds the red dots. 

# Run two time series with parameters corresponding to the dots from above.
parm = {}
parm['gamma'] = 0.0625; parm['b'] = b; parm['K_expn'] = 4; parm['w_expn'] = -6
parm['c'] = c; parm['sigma'] = sigma; parm['y_expn'] = -6; parm['mu'] = mu
parm['auto'] = True; parm['phi_expn'] = 3.5

parm['p'] = 0; parm['nu'] = nu; parm['rho'] = rho
parm['t0'] = 0; parm['t1'] = 180

num_years = 10; tsol = np.linspace(0, 180*num_years, num_years*100)
for n, parm['beta'] in enumerate([0.05, 0.15]):
    [H, FU, FI] = solve_ivp(fun=odes, t_span=[tsol.min(), tsol.max()], t_eval=tsol, y0=[200, 50, 0], args=(parm,)).y
    S = parm['sigma']*(H - parm['c']*FU*(1 + parm['p']*FI/(FU + 10**parm['y_expn'])))

    ax = plt.subplot(gs[1, n])
    ax.plot(tsol, H, label=r'$H$', linewidth=2)
    ax.plot(tsol, FU, label=r'$F_U$', linewidth=2)
    ax.plot(tsol, FI, label=r'$F_I$', linewidth=2)
    ax.set_xlabel(r'$t$ (days)', fontsize=22)
    ax.set_ylabel('Bees', fontsize=22)
    ax.set_title(r'$\beta = $' + str(parm['beta']), fontsize=24)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax_twin = ax.twinx()
    ax_twin.plot(tsol, [H[k]/S[k] if H[k] + FU[k] + FI[k]> 100 else np.nan for k in range(len(tsol))], color='red', linestyle='--', linewidth=2)
    ax_twin.set_ylim(0, 25)
    ax_twin.spines['right'].set_color('red'); ax_twin.tick_params(axis='y', colors='red')
    ax_twin.tick_params(axis='y', labelsize=16)
    ax_twin.set_ylabel(r'AAOF', rotation=90, fontsize=22, labelpad=10, color='red')

    if n == 0:
        ax.legend(fontsize=18)

plt.subplots_adjust(hspace=0.2, wspace=0.4)
plt.savefig('bif.pdf', bbox_inches='tight')