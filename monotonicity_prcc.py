import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def odes(t, Z, parm):
    # Unpack the states, preallocate ODE evaluations. 
    H, F, I = Z
    ODEs = [0,0,0]
    
    # Fill in ODEs
    parm['a'] = (1 - parm['b'])/(parm['c']*parm['b']) # Define a in this case as b changes.  
    E = parm['gamma']*(F + parm['p']*I)/(parm['a']*H + F + parm['p']*I + 10**parm['w_expn']) * (1 - F/(10**parm['K_expn']))
    S = parm['sigma']*(H - parm['c']*F*(1 + parm['p']*I/(F + 10**parm['y_expn'])))
    ODEs[0] = E*H - S
    ODEs[1] = S - parm['beta']*F - parm['mu']*F
    ODEs[2] = parm['beta']*F - parm['nu']*I
    return ODEs

# Define baseline model parameters. 
parm = {}
parm['gamma'] = 0.05; parm['K_expn'] = 4
parm['sigma'] = 0.25; parm['c'] = 2.7
parm['mu'] = 0.136
parm['w_expn'] = -6; parm['y_expn'] = -6
parm['p'] = 0; parm['beta'] = 0; parm['nu'] = 0
parm['b'] = 0.99; parm['a'] = (1 - parm['b'])/(parm['c']*parm['b'])

# Define model ranges.
ranges = {}
ranges['gamma'] = [0.001, 0.11]; ranges['K_expn'] = [3, 5]
ranges['sigma'] = [0.8*0.25, 1.2*0.25]; ranges['c'] = [2.16, 3.24]
ranges['mu'] = [0.116, 0.163]
ranges['w_expn'] = [-7, -5]; ranges['y_expn'] = [-7, -5]
ranges['b'] = [0.8, 1]

# Varying one parameter at a time, we measure the response varibles (population at
# 1 and 5 years) while all other parameters are kept constant. Define a number of values to test, M.
# Set up a plot to hold the results. 
M = 100
fig, axes = plt.subplots(2, 4)
for n, parm_name in enumerate(ranges):
    vals = np.linspace(min(ranges[parm_name]), max(ranges[parm_name]), M)
    pop_one = [None]*M; pop_five = [None]*M
    temp_parm = parm.copy()

    for v, val in enumerate(vals):
        temp_parm[parm_name] = val
        Z = solve_ivp(fun=odes, t_span=[0, 1825], t_eval=[365, 1825], y0=[100, 30, 0], args=(temp_parm,))
        pop_one[v] = np.sum(Z.y,axis=0)[0]
        pop_five[v] = np.sum(Z.y,axis=0)[1]

    axes[math.floor(n/4), n % 4].plot(vals, pop_one, 'k-')
    axes[math.floor(n/4), n % 4].plot(vals, pop_five, 'k--')

x_labels = [r'$\gamma$', r'$\log_{10}(K)$', r'$\sigma$', r'$c$', r'$\mu$', r'$\log_{10}(w)$', r'$\log_{10}(y)$', r'$b$']
for n in range(8):
    axes[math.floor(n/4), n % 4].set_xlabel(x_labels[n], fontsize=12)
    axes[math.floor(n/4), n % 4].set_ylabel('Bees', fontsize=12)

axes[0,0].legend([r'$N(365)$', r'$N(5\cdot 365)$'], loc='upper left')
fig.subplots_adjust(hspace=0.25, wspace=0.3, bottom=0.07, top = 0.93, left=0.07, right=0.93)
plt.show()