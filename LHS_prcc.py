import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import qmc
import multiprocessing
import pingouin as pg
import pandas as pd


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
    
    # Unpack the parms.
    parm = {}
    parm['gamma'], parm['K_expn'], parm['sigma'], parm['c'], parm['mu'], parm['w_expn'], parm['y_expn'], parm['b'] = obj
    parm['p'] = 0; parm['beta'] = 0; parm['nu'] = 0

    # Solve the ODEs with these parameters. Record evaluations at 1 year and 5 years. 
    Z = solve_ivp(fun=odes, t_span=[0, 1825], t_eval=[365, 1825], y0=[100, 30, 0], args=(parm,))
    pop_one = np.sum(Z.y,axis=0)[0]
    pop_five = np.sum(Z.y,axis=0)[1]

    return (pop_one, pop_five)

def main(pool):

    # Make our LHS
    n=1000
    LHS = qmc.LatinHypercube(d=8)
    parm_list = LHS.random(n)
    l_bounds = [0.001, 3, 0.8*0.25, 2.16, 0.116, -7, -7, 0.8]
    u_bounds = [0.11, 5, 1.2*0.25, 3.24, 0.163, -5, -5, 1]
    bounded_parm_list = qmc.scale(parm_list, l_bounds, u_bounds)

    # Run all parameter sets. 
    LHS_results = pool.map(worker, bounded_parm_list)

    # Stick results onto the bounded_parm_list so that the 1 year population is 
    # column 8 and 5 year population is column 9
    data = np.zeros((n,10))
    data[:,:8] = bounded_parm_list; data[:,8] = [n[0] for n in LHS_results]; data[:,9] = [n[1] for n in LHS_results]
    
    # Rank transform the data
    data_ranked = data #data.argsort(axis=0)

    # Create a pandas df.
    col_names = ['x1','x2','x3','x4','x5','x6','x7','x8','y1','y2']
    data_ranked_df = pd.DataFrame(data_ranked, columns=col_names)

    # For each measurement, calculate the prcc value, a 95% CI, and p-value for each parameter.
    prcc_point = [[],[]]; prcc_conf_intv =  [[],[]]; prcc_pval = [[],[]]
    for n, measure in enumerate([-2, -1]):
        for p in range(8):
            # Define the covariates as all other parameters that are varied in the LHS. 
            covars = col_names[:p] + col_names[p+1:-2]
            
            # Do the calculation and store the results! 
            z = pg.partial_corr(data_ranked_df, x=col_names[p], y=col_names[measure], covar=covars, method='spearman')
            prcc_point[n].append(z['r'][0]); prcc_conf_intv[n].append(abs(z['CI95%'][0] - z['r'][0]).tolist()); prcc_pval[n].append(z['p-val'][0])
    
    ## Plotting! 

    # Set width of each bar and their position. 
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
    br1 = np.arange(len(prcc_point[0]))
    br2 = [x + barWidth for x in br1]

    # Make the plot
    plt.bar(br1, prcc_point[0], color ='cyan', width = barWidth, yerr = np.array(prcc_conf_intv[0]).T, capsize=8, error_kw = {'label':'95% CI'},
        edgecolor ='black', label =r'$N(365)$')
    plt.bar(br2, prcc_point[1], color ='green', width = barWidth, yerr = np.array(prcc_conf_intv[1]).T, capsize=8,
        edgecolor ='black', label =r'$N(5\cdot 365)$')

    # Beautify
    plt.ylim(-1,1)
    plt.ylabel('Partial Rank Correlation Coefficient', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(prcc_point[0]))],
            [r'$\gamma$', r'$\log_{10}(K)$', r'$\sigma$', r'$c$', r'$\mu$', r'$\log_{10}(w)$', r'$\log_{10}(y)$', r'$b$'], fontsize=15, rotation=45)
    plt.legend(fontsize=15)
    plt.hlines(y=0,xmin=0,xmax=8, color='k')

    import pdb; pdb.set_trace()

    
    
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    main(pool)