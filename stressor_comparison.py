import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import multiprocessing
import argparse
import os
from itertools import product
from scipy.integrate import solve_ivp
from bee_model import odes

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--ncores", type=int, help="number of cores", default=os.cpu_count() - 1)

def worker(obj):

    # Define the parameters that are fixed for this realization. 
    parm = {}
    parm['gamma'] = 0.0625; parm['b'] = 0.875; parm['K_expn'] = 4; parm['w_expn'] = -6
    parm['c'] = 2.7; parm['sigma'] = 0.2769; parm['y_expn'] = -6; parm['mu'] = 0.1356
    parm['auto'] = True; parm['phi_expn'] = 3.5
    parm['t0'] = 0; parm['t1'] = 180

    # Unpack the stress parameters.
    parm['beta'], parm['nu'], parm['rho'], parm['p'] = obj

    # Solve the ODEs with these parameters. Record evaluations the terminal populations (after 10 years).
    num_years = 10; tsol = np.linspace(0, 180*num_years, num_years*100)
    [H, FU, FI] = solve_ivp(fun=odes, t_span=[tsol.min(), tsol.max()], t_eval=[tsol[-1]], y0=[200, 50, 0], args=(parm,)).y

    return H + FU + FI

def main(pool):

    # Define ranges for stress parameters beta and nu.  
    m = 70
    beta_range = np.linspace(0, 1, m)
    nu_range = np.linspace(0.01, 1, m)
    p_range = [0, 0.25, 0.75, 1]
    rho_range = [0, 0.5]

    # Create a dataframe to hold all parameter combinations. Simulate the model with these
    # in parallel. Check for survival. 
    df = pd.DataFrame(list(product(beta_range, nu_range, p_range, rho_range)), columns=['beta', 'nu', 'p', 'rho'])
    print(df.shape)
    df['term_pop'] = pool.map(worker, df.values)
    df['survival'] = 1*(df['term_pop'] > 5)

    # Iterate through rho and p values to make plot.
    fig, ax = plt.subplots(1,2, figsize=(20, 10)); survival_shading = []
    for n, (rho, p, ls) in enumerate(zip([0]*4 + [0.5]*4, p_range*2, ['solid', 'dotted', 'dashed', 'dashdot']*2)):

        # Subset data to this value of rho and p. 
        x = df[(df['rho'] == rho) & (df['p'] == p)].beta.values.reshape(m,m)
        y = df[(df['rho'] == rho) & (df['p'] == p)].nu.values.reshape(m,m)
        z = df[(df['rho'] == rho) & (df['p'] == p)].survival.values.reshape(m,m)
        survival_shading.append(z)
        # Create a contour line to show where survival happens.
        CS = ax[int(n/4)].contour(x,y,z, levels = [0.5], colors=('black',), linestyles=(ls,), linewidths=(2,))
    
    # Construct the legend
    lines = [Line2D([0], [0], color='black', linewidth=2, linestyle=ls) for ls in ['solid', 'dotted', 'dashed', 'dashdot']]
    labels = [r'$p = $' + str(p) for p in p_range]
    ax[0].legend(lines, labels, loc='upper right')

    # Shade between p = 0 and p = 1 lines. 
    ax[0].contourf(x, y, survival_shading[0] | np.logical_not(survival_shading[3]), cmap=mpl.colors.ListedColormap(['lightgrey','white']))
    ax[1].contourf(x, y, survival_shading[4] | np.logical_not(survival_shading[7]), cmap=mpl.colors.ListedColormap(['lightgrey','white']))

    # Beautify the plot
    ax[0].set_xlabel(r'$\beta$', fontsize=18); ax[0].set_ylabel(r'$\nu$', fontsize=18)
    ax[1].set_xlabel(r'$\beta$', fontsize=18); ax[1].set_ylabel(r'$\nu$', fontsize=18)
    ax[0].set_title(r'$\rho = 0$', fontsize=18); ax[1].set_title(r'$\rho = 0.5$', fontsize=18)

    ax[0].text(0.017, 0.15, 'Persistence', rotation=90, fontsize=14)
    ax[0].text(0.5, 0.5, 'Extinction', fontsize=14)

    ax[1].text(0.015, 0.1, 'Persistence', rotation=0, fontsize=14)
    ax[1].text(0.5, 0.5, 'Extinction', fontsize=14)

    fig.savefig('many_stressor_comparison.pdf')

if __name__ == '__main__':
    args = parser.parse_args()
    pool = multiprocessing.Pool(processes=args.ncores)
    main(pool)