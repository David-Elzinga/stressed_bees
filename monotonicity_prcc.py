import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from bee_model import odes

# Define baseline model parameters. 
parm = {}
parm['gamma'] = 0.08; parm['b'] = 0.95; parm['K_expn'] = 4; parm['w_expn'] = -6
parm['c'] = 2.7; parm['sigma'] = 0.25; parm['y_expn'] = -6; parm['mu'] = 0.136
parm['auto'] = True; parm['phi_expn'] = 3.5
parm['p'] = 0; parm['beta'] = 0.05; parm['nu'] = 2*parm['mu']; parm['rho'] = 0
parm['t0'] = 0; parm['t1'] = 180

# Define parameter ranges.
ranges = {}
ranges['gamma'] = [0.001, 0.11]; ranges['b'] = [0.8, 1]; ranges['K_expn'] = [3, 5]; ranges['w_expn'] = [-7, -5]
ranges['c'] = [2.16, 3.24]; ranges['sigma'] = [0.8*0.25, 1.2*0.25]; ranges['y_expn'] = [-7, -5]; ranges['mu'] = [0.116, 0.163]

# Varying one parameter at a time, we measure the response varibles (population at
# 1 and 5 years, value of AAOF over 1 and 5 years), while all other parameters
# are kept constant. 

# We use a parameter m to define the number of values we test within each range. 
m = 15

# Set up a plot, each subplot corresponds to a parameter. Now we iterate through them. 
fig, axes = plt.subplots(2, 4)
for n, parm_name in enumerate(ranges):

    # Define the values within the range split into m pieces. 
    vals = np.linspace(min(ranges[parm_name]), max(ranges[parm_name]), m)
    pop_one = []; pop_five = []; aaof_one = []; aaof_five = []

    # Create a dictionary to hold parameter values, with all parameters at default 
    # except the one we are currently varying.
    temp_parm = parm.copy()

    # Iterate over the values of our parameter.
    for val in vals:
        temp_parm[parm_name] = val # overwrite the default value of our parameter in the dictionary.

        # Run the model! Calculate our measurements. 
        [H, FU, FI] = solve_ivp(fun=odes, t_span=[0, 1825], t_eval=[365, 1825], y0=[3500, 500, 500], args=(temp_parm,)).y
        pop_one.append(H[0] + FU[0] + FI[0]); pop_five.append(H[1] + FU[1] + FI[1])
        S = temp_parm['sigma']*(H - temp_parm['c']*FU*(1 + temp_parm['p']*FI/(FU + 10**temp_parm['y_expn'])))
        aaof_one.append((H[0] > 10)*H[0]/S[0]); aaof_five.append((H[1] > 10)*H[1]/S[1])

    # On the correct subplot, plot the measurement over the range it was varied on.
    axes[math.floor(n/4), n % 4].plot(vals, pop_one, 'k-')
    axes[math.floor(n/4), n % 4].plot(vals, pop_five, 'k--')
    axes[math.floor(n/4), n % 4].set_ylabel('Bees', fontsize=14)

    ax_twin = axes[math.floor(n/4), n % 4].twinx()
    ax_twin.plot(vals, aaof_one, color='red', linestyle='-', linewidth=2)
    #ax_twin.plot(vals, aaof_five, color='red', linestyle='--', linewidth=2)
    ax_twin.spines['right'].set_color('red'); ax_twin.tick_params(axis='y', colors='red')
    #ax_twin.set_ylabel(r'AAOF', rotation=90, fontsize=14, labelpad=10, color='red')
    ax_twin.set_ylim(0,25)

# Add in a whole bunch of labels so we know which parameter corresponds to each subplot.
x_labels = [r'$\gamma$', r'$b$', r'$\log_{10}(K)$', r'$\log_{10}(w)$', r'$c$', r'$\sigma$', r'$\log_{10}(y)$', r'$\mu$']
for n in range(8):
    axes[math.floor(n/4), n % 4].set_xlabel(x_labels[n], fontsize=14)

# On the first plot add a legend. 
axes[0,0].legend([r'$N(365)$', r'$N(5\cdot 365)$', 'AAOF 1', 'AAOF 2'], loc='upper left')
fig.subplots_adjust(hspace=0.25, wspace=0.3, bottom=0.07, top = 0.93, left=0.07, right=0.93)
plt.show()