import numpy as np
from bee_model import simulate
import matplotlib.pyplot as plt

# Define the parameters that are stressor-independent. 
parm = {}
parm['gamma'] = 0.15; parm['K_expn'] = 4
parm['c'] = 2.7; parm['sigma'] = 0.25
parm['mu'] = 0.136
parm['w_expn'] = -6; parm['y_expn'] = -6
parm['b'] = 0.95; parm['a'] = (1 - parm['b'])/(parm['c']*parm['b'])

# Simulating a 180 day season,  we introduce on day d (between 0 and 180), and terminate
# the stressor after l days. We require that d + l <= 180. Define possible combinations.
m = 50
d_vals, l_vals = np.meshgrid(np.linspace(1,180,m), np.linspace(1,180,m))

# For each combination of temporal stressor strategies, simulate the model. Record
# the population at the end of 5 years of repeating the same strategy.
five_yr_pop = []
for d, l in zip(d_vals.flatten(), l_vals.flatten()):
    if d + l < 180:
        for year in range(5):
            
            # Simulate the model until time d is reached. Consider cases on the initial condition. 
            tsol = np.linspace(180*year, 180*year + d, 1000)
            if year == 0:
                parm['beta'] = 0; parm['p'] = 0; parm['nu'] = 0; parm['rho'] = 0
                Z = simulate(tsol, [3500, 500, 500], parm, system = "nonautonomous")
            else:
                Z = simulate(tsol, Z.y[:,-1], parm, system = "nonautonomous")

            # Simulate the model from time d to time d + l. Turn on the stressor.
            parm['beta'] = 0.6; parm['p'] = 0; parm['nu'] = 1.5*parm['mu']; parm['rho'] = 0.2
            tsol = np.linspace(180*year + d, 180*year + d + l, 1000)
            Z = simulate(tsol, Z.y[:,-1], parm, system = "nonautonomous")

            # Simulate the model from time d + l to 180. Turn off the stressor.
            parm['beta'] = 0; parm['p'] = 0; parm['nu'] = 0; parm['rho'] = 0
            tsol = np.linspace(180*year + d + l, 180*year + 180, 1000)
            Z = simulate(tsol, Z.y[:,-1], parm, system = "nonautonomous")
        five_yr_pop.append(sum(Z.y[:,-1]))
    else:
        five_yr_pop.append(np.nan)

fig, ax = plt.subplots(1,1)
CS = ax.contourf(d_vals,l_vals,np.array(five_yr_pop).reshape(m,m), cmap='bone')
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('Five Year Population', fontsize=15)

ax.set_xlabel('Stressor Start Day', fontsize=15)
ax.set_ylabel('Stressor Application Duration', fontsize=15)
plt.show()

import pdb; pdb.set_trace()