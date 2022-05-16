import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from bee_model import odes

'''
This code validates the monotonicity of the model with respect to four measurements
as each individual parameter varies within a defined range. This script saves two
.pdf figures that show this monotonic relationship.
'''

# Define baseline model parameters. 
parm = {}
parm['gamma'] = 0.0625; parm['b'] = 0.875; parm['K_expn'] = 4; parm['w_expn'] = -6
parm['c'] = 2.7; parm['sigma'] = 0.2769; parm['y_expn'] = -6; parm['mu'] = 0.1356
parm['auto'] = True; parm['phi_expn'] = 3.5
parm['p'] = 0; parm['beta'] = 0; parm['nu'] = 0; parm['rho'] = 0
parm['t0'] = 0; parm['t1'] = 180

# Define parameter ranges.
ranges = {}
ranges['gamma'] = [0.0169, 0.1081]; ranges['b'] = [0.8, 0.95]; ranges['K_expn'] = [3, 5]; ranges['w_expn'] = [-7, -5]
ranges['c'] = [2.16, 3.24]; ranges['sigma'] = [0.05, 0.5]; ranges['y_expn'] = [-7, -5]; ranges['mu'] = [0.1111, 0.1667]

# Varying one parameter at a time, we measure the response varibles (population at
# 1 and 5 years, value of AAOF at 1 and 5 years), while all other parameters
# are kept constant. 

# We use a parameter m to define the number of values we test within each range. 
m = 15

# Set up a plot, each subplot corresponds to a parameter. Now we iterate through them. 
fig_pop, axes_pop = plt.subplots(2, 4, figsize=(15,8))
fig_aaof, axes_aaof = plt.subplots(2, 4, figsize=(15,8))
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
        [H, FU, FI] = solve_ivp(fun=odes, t_span=[0, 1825], t_eval=[365, 1825], y0=[200, 50, 0], args=(temp_parm,)).y
        pop_one.append(H[0] + FU[0] + FI[0]); pop_five.append(H[1] + FU[1] + FI[1])
        S = temp_parm['sigma']*(H - temp_parm['c']*FU*(1 + temp_parm['p']*FI/(FU + 10**temp_parm['y_expn'])))

        # Record AAOF depending on if the population went extinct or not. 
        if H[0] + FU[0] + FI[0] < 100:
            aaof_one.append(np.nan)
        else:
            aaof_one.append(H[0]/S[0])
        
        if H[1] + FU[1] + FI[1] < 100:
            aaof_five.append(np.nan)
        else:
            aaof_five.append(H[1]/S[1])

    # On the correct subplot, plot the measurement over the range it was varied on.
    axes_pop[math.floor(n/4), n % 4].plot(vals, pop_one, 'k-')
    axes_pop[math.floor(n/4), n % 4].plot(vals, pop_five, 'k--')
    axes_pop[math.floor(n/4), n % 4].set_ylabel('Bees', fontsize=14)

    axes_aaof[math.floor(n/4), n % 4].plot(vals, aaof_one, 'k-')
    axes_aaof[math.floor(n/4), n % 4].plot(vals, aaof_five, 'k--')
    axes_aaof[math.floor(n/4), n % 4].set_ylabel('AAOF', fontsize=14)
    axes_aaof[math.floor(n/4), n % 4].set_ylim(0,50)

# Add in a whole bunch of labels so we know which parameter corresponds to each subplot.
x_labels = [r'$\gamma$', r'$b$', r'$\log_{10}(K)$', r'$\log_{10}(w)$', r'$c$', r'$\sigma$', r'$\log_{10}(y)$', r'$\mu$']
for n in range(8):
    axes_pop[math.floor(n/4), n % 4].set_xlabel(x_labels[n], fontsize=14)
    axes_aaof[math.floor(n/4), n % 4].set_xlabel(x_labels[n], fontsize=14)


# On the first plot add a legend. 
axes_pop[0,0].legend([r'$N(365)$', r'$N(5\cdot 365)$'], loc='upper left')
axes_aaof[0,0].legend([r'AAOF, $t=365$', r'AAOF, $t=5\cdot 365$'], loc='upper left')
fig_pop.subplots_adjust(wspace=0.5); fig_aaof.subplots_adjust(wspace=0.5)
fig_pop.savefig('monotonicity_pop.pdf'); fig_aaof.savefig('monotonicity_aaof.pdf')