import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp
import multiprocessing

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

def worker(obj):

    # Define the parameters that are fixed for this realization. 
    parm = {}
    parm['gamma'] = 0.05; parm['K_expn'] = 4
    parm['sigma'] = 0.25; parm['c'] = 2.7
    parm['mu'] = 0.136
    parm['w_expn'] = -6; parm['y_expn'] = -6
    parm['b'] = 0.95; parm['a'] = (1 - parm['b'])/(parm['c']*parm['b'])

    # Unpack the stress params.
    parm['beta'], parm['nu'], parm['p'] = obj

    # Solve the ODEs with these parameters. Record evaluations at 1 year.
    Z = solve_ivp(fun=odes, t_span=[0, 365], t_eval=[365], y0=[100, 30, 0], args=(parm,))
    pop_one = np.sum(Z.y,axis=0)[0]

    return pop_one

def main(pool):
    # Construct a grid of values for the stress parameters
    n=100j; m = int(np.imag(n))
    beta_vals, nu_vals = np.mgrid[0:1:n, 0:1:n]
    beta_nu_grid = np.vstack([beta_vals.ravel(), nu_vals.ravel()]).T

    # Iterate through p values. Create a contour object each time. Store the survival results. 
    survival_results = []
    for p_val, ls in zip([0, 0.25, 0.75, 1], ['solid', 'dotted', 'dashed', 'dashdot']):
        print(p_val)
        # Define the stress grid with beta, nu, and p defined for each parameter set. Run all parameter sets
        stress_grid = np.append(beta_nu_grid, np.full((m*m,1), p_val), axis=1)
        pop_one = pool.map(worker, stress_grid)
        survival = np.array(pop_one).reshape(m,m) > 1 # a 1 indicates survival, a 0 indicates extn. 
        survival_results.append(survival)

        # Create a line on the graph that divides where pop_one > 1 bee and < 1 bee. 
        CS = plt.contour(beta_vals, nu_vals, survival, levels = [0.5], colors=('black',), linestyles=(ls,), linewidths=(2,))
        CS.collections[0].set_label(r'$p = $' + str(p_val))

    plt.contourf(beta_vals, nu_vals, survival_results[0] | np.logical_not(survival_results[-1]), cmap=mpl.colors.ListedColormap(['lightgrey','white']))
 
    # Beautify the plots. 
    plt.xlabel(r'$\beta$', fontsize=15); plt.ylabel(r'$\nu$', fontsize=15)
    plt.text(0.04, 0.3, 'Persistence', rotation=90)
    plt.text(0.4, 0.4, 'Extinction')
    plt.legend(fontsize=10, loc='upper right')
    plt.xlim(0,0.7); plt.ylim(0,0.7)
    plt.show()

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    main(pool)