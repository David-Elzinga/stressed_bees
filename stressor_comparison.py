import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from bee_model import simulate
import multiprocessing

def worker(obj):

    # Define the parameters that are fixed for this realization. 
    parm = {}
    parm['gamma'] = 0.15; parm['K_expn'] = 4
    parm['sigma'] = 0.25; parm['c'] = 2.7
    parm['mu'] = 0.136
    parm['w_expn'] = -6; parm['y_expn'] = -6
    parm['b'] = 0.95; parm['a'] = (1 - parm['b'])/(parm['c']*parm['b'])

    # Unpack the stress params.
    parm['beta'], parm['nu'], parm['p'] = obj

    # Solve the ODEs with these parameters. Record evaluations the terminal populations (after 10 years).
    tsol = np.array([0, 365*10])
    Z = simulate(tsol, [3000, 500, 0], parm, system = "autonomous")
    term_pop = Z.y[:,-1].sum()
    return term_pop

def main(pool):
    # Construct a grid of values for the stress parameters
    n=50j; m = int(np.imag(n))
    beta_vals, nu_vals = np.mgrid[0:3:n, 0:3:n]
    beta_nu_grid = np.vstack([beta_vals.ravel(), nu_vals.ravel()]).T

    # Iterate through p values. Create a contour object each time. Store the survival results. 
    survival_results = []
    for p_val, ls in zip([0, 0.25, 0.75, 1], ['solid', 'dotted', 'dashed', 'dashdot']):
        print(p_val)
        # Define the stress grid with beta, nu, and p defined for each parameter set. Run all parameter sets
        stress_grid = np.append(beta_nu_grid, np.full((m*m,1), p_val), axis=1)
        term_pop = pool.map(worker, stress_grid)
        print(term_pop)
        survival = np.array(term_pop).reshape(m,m) > 1 # a 1 indicates survival, a 0 indicates extn. 
        survival_results.append(survival)

        # Create a line on the graph that divides where term_pop > 1 bee and < 1 bee. 
        CS = plt.contour(beta_vals, nu_vals, survival, levels = [0.5], colors=('black',), linestyles=(ls,), linewidths=(2,))
        CS.collections[0].set_label(r'$p = $' + str(p_val))

    plt.contourf(beta_vals, nu_vals, survival_results[0] | np.logical_not(survival_results[-1]), cmap=mpl.colors.ListedColormap(['lightgrey','white']))
 
    # Beautify the plots. 
    plt.xlabel(r'$\beta$', fontsize=15); plt.ylabel(r'$\nu$', fontsize=15)
    plt.text(0.01, 1, 'Persistence', rotation=0, fontsize=14)
    plt.text(1.6, 1.9, 'Extinction', fontsize=14)
    plt.legend(fontsize=10, loc='upper right')
    plt.xlim(0,3); plt.ylim(0,3)
    plt.savefig('stressor_comparison.pdf')

    plt.plot([2], [0.65], 'r.',markersize=20)
    plt.savefig('stressor_comparison_dot.pdf')
    plt.show()

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    main(pool)