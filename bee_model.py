import numpy as np

'''
This script runs the bee model ODEs. The parameters must be specified, 
including if the system is autonomous. 
'''

def odes(t, Z, parm):
    # Unpack the states, preallocate ODE evaluations. 
    H, FU, FI = Z
    ODEs = [0,0,0]

    # Calculate the number of functioning foragers. 
    FF = FU + parm['p']*FI
    
    # Define a (depends on b and c used, which might change during LHS runs). 
    parm['a'] = (1 - parm['b'])/(parm['c']*parm['b'])

    # Define carrying capacity K (depends on if the system is autonomous). 
    K = 10**parm['K_expn'] + (not parm['auto'])*10**(parm['phi_expn'])*np.cos(np.pi*(t%180-30)/150)

    # Define the transmission rate. It's always turned on if the system is autonomous, or else 
    # it is only turned on if t0 < t < t1.
    beta = (parm['auto'] or t%180 > parm['t0'] and t%180 < parm['t1'])*parm['beta']

    # Define the eclosion and social inhibition terms. 
    E = parm['gamma']*FF/(parm['a']*H + FF + 10**parm['w_expn']) * (1 - FF/K)
    S = parm['sigma']*(H - parm['c']*FU*(1 + parm['p']*FI/(FU + 10**parm['y_expn'])))

    # Calculate the ODEs.
    ODEs[0] = E*H - S
    ODEs[1] = S - (beta + parm['mu'])*FU + parm['rho']*FI
    ODEs[2] = beta*FU - (parm['nu'] + parm['rho'])*FI
    
    return ODEs