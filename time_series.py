import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def odes(t, Z, parms):
    # Unpack the states, preallocate ODE evaluations. 
    H, F, I = Z
    ODEs = [0,0,0]
    
    # Fill in ODEs 
    E = parm['gamma']*(F + parm['p']*I)/(parm['a']*H + F + parm['p']*I + parm['w']) * (1 - F/parm['K'])
    S = parm['sigma']*(H - parm['c']*F*(1 + parm['p']*I/(F + parm['y'])))
    ODEs[0] = E*H - S
    ODEs[1] = S - parm['beta']*F - parm['mu']*F
    ODEs[2] = parm['beta']*F - parm['nu']*I

    return ODEs

# Define model parameters. 
parm = {}
parm['gamma'] = 0.05; parm['K'] = 10**4
parm['c'] = 2.7; parm['sigma'] = 0.25 # ???
parm['mu'] = 0.136
parm['w'] = 10**(-3); parm['y'] = 10**(-3)
parm['p'] = 0; parm['beta'] = 0; parm['nu'] = 0
parm['b'] = 0.99; parm['a'] = (1 - parm['b'])/(parm['c']*parm['b'])

# Solve the IVP.
Z = solve_ivp(fun=odes, t_span=[0, 1825], t_eval=[365, 1825], y0=[100, 30, 0], args=(parm,))