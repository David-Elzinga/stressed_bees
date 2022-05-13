import numpy as np
import pandas as pd
import multiprocessing
import os
import argparse
import matplotlib.pyplot as plt
from itertools import product
from scipy.integrate import solve_ivp
from bee_model import odes
import time


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--ncores", type=int, help="number of cores", default=os.cpu_count() - 1)

def worker(obj):

    # Define the parameters that are fixed for this realization. 
    parm = {}
    parm['gamma'] = 0.15; parm['b'] = 0.95; parm['K_expn'] = 4; parm['w_expn'] = -6
    parm['c'] = 2.7; parm['sigma'] = 0.25; parm['y_expn'] = -6; parm['mu'] = 0.136
    parm['auto'] = False; parm['phi_expn'] = 3.5
    parm['beta'] = 1.5; parm['nu'] = 1; parm['rho'] = 0.5; parm['p'] = 0

    # Unpack the stress timing parameters.
    parm['t0'], parm['t1'] = obj

    if parm['t0'] > parm['t1']:
        return np.nan

    # Solve the ODEs with these parameters. Record evaluations the terminal populations (after 10 years).
    num_years = 5; tsol = np.linspace(0, 180*num_years, num_years*100)
    [H, FU, FI] = solve_ivp(fun=odes, t_span=[tsol.min(), tsol.max()], t_eval=[tsol[-1]], y0=[200, 50, 0], args=(parm,)).y

    return H + FU + FI

def main(pool): 

    # Define the possible values for t0
    m = 70
    t0_range = np.linspace(0, 180, m)
    t1_range = np.linspace(0, 180, m)

    # Make all possible parameter stress timing combinations and put them in a dtaframe. 
    # Simulate the model with these in parallel. 
    # Model returns nan if t1 > t0.
    df = pd.DataFrame(list(product(t0_range, t1_range)), columns=['t0', 't1'])

    starting = time.time()
    df['term_pop'] = pool.map(worker, df.values)
    ending = time.time()
    print(df.shape[0], ending - starting)
    
    fig, ax = plt.subplots(1,1)
    CS = ax.contourf(df['t0'].values.reshape(m,m), df['t1'].values.reshape(m,m), df['term_pop'].values.reshape(m,m), cmap='bone', levels=[1000*k for k in [0, 1, 5, 10, 15, 20, 25, 30]])
    cbar = fig.colorbar(CS)

    cbar.ax.set_ylabel('Five Year Population')
    ax.set_xlabel('Stressor Start Date')
    ax.set_ylabel('Stressor End Date')

    fig.savefig('temporal_comparison.pdf')


if __name__ == '__main__':
    args = parser.parse_args()
    pool = multiprocessing.Pool(processes=args.ncores)
    main(pool)

# # Define the parameters that are stressor-independent. 
# parm = {}
# parm['gamma'] = 0.15; parm['K_expn'] = 4
# parm['c'] = 2.7; parm['sigma'] = 0.25
# parm['mu'] = 0.136
# parm['w_expn'] = -6; parm['y_expn'] = -6
# parm['b'] = 0.95; parm['a'] = (1 - parm['b'])/(parm['c']*parm['b'])

# # Simulating a 180 day season,  we introduce on day d (between 0 and 180), and terminate
# # the stressor after l days. We require that d + l <= 180. Define possible combinations.
# m = 50
# d_vals, l_vals = np.meshgrid(np.linspace(1,180,m), np.linspace(1,180,m))

# # For each combination of temporal stressor strategies, simulate the model. Record
# # the population at the end of 5 years of repeating the same strategy.
# five_yr_pop = []
# for d, l in zip(d_vals.flatten(), l_vals.flatten()):
#     if d + l < 180:
#         for year in range(5):
            
#             # Simulate the model until time d is reached. Consider cases on the initial condition. 
#             tsol = np.linspace(180*year, 180*year + d, 1000)
#             if year == 0:
#                 parm['beta'] = 0; parm['p'] = 0; parm['nu'] = 0; parm['rho'] = 0
#                 Z = simulate(tsol, [3500, 500, 500], parm, system = "nonautonomous")
#             else:
#                 Z = simulate(tsol, Z.y[:,-1], parm, system = "nonautonomous")

#             # Simulate the model from time d to time d + l. Turn on the stressor.
#             parm['beta'] = 0.6; parm['p'] = 0; parm['nu'] = 1.5*parm['mu']; parm['rho'] = 0.2
#             tsol = np.linspace(180*year + d, 180*year + d + l, 1000)
#             Z = simulate(tsol, Z.y[:,-1], parm, system = "nonautonomous")

#             # Simulate the model from time d + l to 180. Turn off the stressor.
#             parm['beta'] = 0; parm['p'] = 0; parm['nu'] = 0; parm['rho'] = 0
#             tsol = np.linspace(180*year + d + l, 180*year + 180, 1000)
#             Z = simulate(tsol, Z.y[:,-1], parm, system = "nonautonomous")
#         five_yr_pop.append(sum(Z.y[:,-1]))
#     else:
#         five_yr_pop.append(np.nan)

# fig, ax = plt.subplots(1,1)
# CS = ax.contourf(d_vals,l_vals,np.array(five_yr_pop).reshape(m,m), cmap='bone')
# cbar = fig.colorbar(CS)
# cbar.ax.set_ylabel('Five Year Population', fontsize=15)

# ax.set_xlabel('Stressor Start Day', fontsize=15)
# ax.set_ylabel('Stressor Application Duration', fontsize=15)
# plt.show()

# import pdb; pdb.set_trace()