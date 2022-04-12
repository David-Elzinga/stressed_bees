import numpy as np
import matplotlib.pyplot as plt
from bee_model import simulate

# Define model parameters. 
parm = {}
parm['gamma'] = 0.15; parm['K_expn'] = 4
parm['c'] = 2.7; parm['sigma'] = 0.25
parm['mu'] = 0.136
parm['w_expn'] = -6; parm['y_expn'] = -6
parm['p'] = 0; parm['beta'] = 0.4; parm['nu'] = parm['mu']; parm['rho'] = 0
parm['b'] = 0.95; parm['a'] = (1 - parm['b'])/(parm['c']*parm['b'])

# Solve the IVP.
tsol = np.linspace(0,18000,10000)
Z = simulate(tsol, [3500, 500, 500], parm, system = "autonomous")
plt.plot(tsol, Z.y[0,:], label='H')
plt.plot(tsol, Z.y[1,:], label='F')
plt.plot(tsol, Z.y[2,:], label='I')
plt.legend()
plt.show()