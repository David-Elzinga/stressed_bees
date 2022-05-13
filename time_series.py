import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from bee_model import odes

'''
This script runs the model for a given set of parameters and plots 
the time series.
'''

# Define model parameters. 
parm = {}
parm['gamma'] = 0.15; parm['b'] = 0.95; parm['K_expn'] = 4; parm['w_expn'] = -6
parm['c'] = 2.7; parm['sigma'] = 0.25; parm['y_expn'] = -6; parm['mu'] = 0.136
parm['auto'] = True; parm['phi_expn'] = 3.5

# These parameters dictate the stressor characteristics. 
parm['p'] = 0; parm['beta'] = 0.01; parm['nu'] = 100; parm['rho'] = 0.5
parm['t0'] = 0; parm['t1'] = 180

# Solve the IVP and plot. 
num_years = 10; tsol = np.linspace(0, 180*num_years, num_years*100)
[H, FU, FI] = solve_ivp(fun=odes, t_span=[tsol.min(), tsol.max()], t_eval=tsol, y0=[3500, 500, 500], args=(parm,)).y
S = parm['sigma']*(H - parm['c']*FU*(1 + parm['p']*FI/(FU + 10**parm['y_expn'])))

# Plot the time series
fig, ax = plt.subplots(figsize=(5,15))
ax.plot(tsol, H, label=r'$H$', linewidth=2)
ax.plot(tsol, FU, label=r'$F_U$', linewidth=2)
ax.plot(tsol, FI, label=r'$F_I$', linewidth=2)
ax.set_xlabel(r'$t$ (Days)', fontsize=14); ax.set_ylabel('Bees', fontsize=14); ax.legend(fontsize=14)

ax_twin = ax.twinx()
ax_twin.plot(tsol, H/S, color='red', linestyle='-', linewidth=2)
ax_twin.spines['right'].set_color('red'); ax_twin.tick_params(axis='y', colors='red')
ax_twin.set_ylabel(r'AAOF', rotation=90, fontsize=14, labelpad=10, color='red')

fig.subplots_adjust(right=0.8)
plt.show()