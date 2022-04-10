from scipy.integrate import solve_ivp
import numpy as np

def simulate(tsol, init_cond, parm, system):

    def autonomous_odes(t, Z, parm):
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

    def nonautonomous_odes(t, Z, parm):
        # Unpack the states, preallocate ODE evaluations. 
        H, F, I = Z
        ODEs = [0,0,0]
        
        # Fill in ODEs
        parm['a'] = (1 - parm['b'])/(parm['c']*parm['b']) # Define a in this case as b changes.  
        E = parm['gamma']*(F + parm['p']*I)/(parm['a']*H + F + parm['p']*I + 10**parm['w_expn']) * (1 - F/(10**parm['K_expn'] + 10**(3.9)*np.cos(np.pi*(t%180-30)/150)))
        #print(F/(10**parm['K_expn'] + 10**(3.9)*np.cos(np.pi*(t%180-30)/150)))
        S = parm['sigma']*(H - parm['c']*F*(1 + parm['p']*I/(F + 10**parm['y_expn'])))
        ODEs[0] = E*H - S
        ODEs[1] = S - parm['beta']*F - parm['mu']*F
        ODEs[2] = parm['beta']*F - parm['nu']*I
        return ODEs
    
    if system == 'autonomous':
        Z = solve_ivp(fun=autonomous_odes, t_span=[tsol.min(), tsol.max()], t_eval=tsol, y0=init_cond, args=(parm,))
    else:
        Z = solve_ivp(fun=nonautonomous_odes, t_span=[tsol.min(), tsol.max()], t_eval=tsol, y0=init_cond, args=(parm,))

    return Z