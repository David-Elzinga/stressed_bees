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
    parm['gamma'] = 0.0625; parm['b'] = 0.875; parm['K_expn'] = 4; parm['w_expn'] = -6
    parm['c'] = 2.7; parm['sigma'] = 0.2769; parm['y_expn'] = -6; parm['mu'] = 0.1356
    parm['auto'] = False; parm['phi_expn'] = 3.5
    parm['beta'] = 0.3; parm['nu'] = 0.2; parm['rho'] = 0.5; parm['p'] = 0.75

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
    m = 50
    t0_range = np.linspace(0, 180, m)
    t1_range = np.linspace(0, 180, m)

    # Make all possible parameter stress timing combinations and put them in a dataframe. 
    # Simulate the model with these in parallel. 
    # Model returns nan if t1 > t0.
    df = pd.DataFrame(list(product(t0_range, t1_range)), columns=['t0', 't1'])

    starting = time.time()
    df['term_pop'] = pool.map(worker, df.values)
    ending = time.time()
    print(df.shape[0], ending - starting)
    
    fig, ax = plt.subplots(1,1)
    CS = ax.contourf(df['t0'].values.reshape(m,m), df['t1'].values.reshape(m,m), df['term_pop'].values.reshape(m,m), cmap='bone', levels=[1000*k for k in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]])
    cbar = fig.colorbar(CS)

    cbar.ax.set_ylabel('Five Year Population')
    ax.set_xlabel('Stressor Start Date')
    ax.set_ylabel('Stressor End Date')

    fig.savefig('temporal_comparison.pdf')


if __name__ == '__main__':
    args = parser.parse_args()
    pool = multiprocessing.Pool(processes=args.ncores)
    main(pool)