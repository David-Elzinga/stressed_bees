import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from bee_model import odes
from scipy.stats import qmc
import pingouin as pg

'''
This code creates a Latin Hypercube (LHS) and calculate PRCC values for each parameter according 
to each measure (see manuscript). Two figures are generated and saved as pdf files. Runs in less than  
a couple of minutes for 10k samples. 
'''

def worker(obj):
    
    # Unpack the parameters from the LHS.
    parm = {}
    parm['gamma'], parm['K_expn'], parm['sigma'], parm['c'], parm['mu'], parm['w_expn'], parm['y_expn'], parm['b'] = obj
    
    # Specify the parameters that correspond to the absence of a stressor. 
    parm['auto'] = True; parm['phi_expn'] = 3.5; parm['t0'] = 0; parm['t1'] = 180
    parm['p'] = 0; parm['beta'] = 0; parm['nu'] = 0; parm['rho'] = 0

    # Solve the ODEs with these parameters. 
    [H, FU, FI] = solve_ivp(fun=odes, t_span=[0, 1825], t_eval=[365, 1825], y0=[200, 50, 0], args=(parm,)).y
    S = parm['sigma']*(H - parm['c']*FU*(1 + parm['p']*FI/(FU + 10**parm['y_expn'])))

    # Measure the response varibles (population at 1 and 5 years, value of AAOF at 1 and 5 years)
    pop_one = H[0] + FU[0] + FI[0]; pop_five = H[1] + FU[1] + FI[1]
    if pop_one < 100: # < 5 bees indicates an extinction, return nan for aaof.
        aaof_one = np.nan
    else:
        aaof_one = H[0]/S[0]
    
    if pop_five < 100: # < 5 bees indicates an extinction, return nan for aaof.
        aaof_five = np.nan
    else:
        aaof_five = H[1]/S[1]

    return (pop_one, pop_five, aaof_one, aaof_five)

def main(pool):

    # Make our LHS
    n=10000
    LHS = qmc.LatinHypercube(d=8)
    parm_list = LHS.random(n)

    # Order of parameters: gamma, K_expn, sigma, c, mu, w_expn, y_expn, b.
    l_bounds = [0.001, 3, 0.8*0.25, 2.16, 0.116, -7, -7, 0.8]
    u_bounds = [0.11, 5, 1.2*0.25, 3.24, 0.163, -5, -5, 1]
    bounded_parm_list = qmc.scale(parm_list, l_bounds, u_bounds)

    # Run all parameter sets. 
    LHS_results = pool.map(worker, bounded_parm_list)

    # Stick results onto the bounded_parm_list so that the 1 year population is 
    # column 8 and 5 year population is column 9, 1 year aaof is column 10, 5 year aaof is column 11.
    data = np.zeros((n,12))
    data[:,:8] = bounded_parm_list
    data[:,8:] = [n for n in LHS_results]

    # Create a pandas df.
    col_names = ['x1','x2','x3','x4','x5','x6','x7','x8','y1','y2','y3','y4']
    data_composite_df = pd.DataFrame(data, columns=col_names)

    # For each measurement, calculate the prcc value, a 95% CI, and p-value for each parameter.
    prcc_point = [[],[],[],[]]; prcc_conf_intv =  [[],[],[],[]]; prcc_pval = [[],[],[],[]]
    for n, measure in enumerate([-4, -3, -2, -1]):
        for p in range(8):

            # Drop rows where this measurement is nan.
            data_df = data_composite_df[data_composite_df[col_names[measure]].notna()]
            
            # Define the covariates as all other parameters that are varied in the LHS. 
            covars = col_names[:p] + col_names[p+1:-4]
            
            # Do the calculation and store the results! 
            z = pg.partial_corr(data_df, x=col_names[p], y=col_names[measure], covar=covars, method='spearman')
            prcc_point[n].append(z['r'][0]); prcc_conf_intv[n].append(abs(z['CI95%'][0] - z['r'][0]).tolist()); prcc_pval[n].append(z['p-val'][0])
    
    ## Plotting! We make two figures, one for population size and one for aaof. 
    for m1, m2, name, legend_labels in zip([0, 2], [1, 3], ['lhs_pop', 'lhs_aaof'], [[r'$N(365)$', r'$N(5\cdot 365)$'], [r'AAOF, $t=365$', r'AAOF, $t=5\cdot 365$']]):
        # Set width of each bar and their position. 
        barWidth = 0.25
        fig, ax = plt.subplots(figsize =(12, 8))
        br1 = np.arange(len(prcc_point[m1]))
        br2 = [x + barWidth for x in br1]

        # Make the bar plots plot. Legend depends on if we are making the plot for population or aaof.
        edge_colors_m1 = ['red' if prcc_pval[m1][n] < 0.05 else 'black' for n in range(8)]
        edge_colors_m2 = ['red' if prcc_pval[m2][n] < 0.05 else 'black' for n in range(8)]
        plt.bar(br1, prcc_point[m1], color ='cyan', width = barWidth, yerr = np.array(prcc_conf_intv[m1]).T, capsize=8, error_kw = {'label':'95% CI'},
                edgecolor = edge_colors_m1, linewidth = 2, label = legend_labels[0])
        plt.bar(br2, prcc_point[m2], color ='green', width = barWidth, yerr = np.array(prcc_conf_intv[m2]).T, capsize=8,
                edgecolor = edge_colors_m2, linewidth = 2, label = legend_labels[1])

        # Beautify
        plt.ylim(-1,1)
        plt.ylabel('Partial Rank Correlation Coefficient', fontsize = 14)
        plt.xticks([r + barWidth for r in range(len(prcc_point[m1]))],
                [r'$\gamma$', r'$\log_{10}(K)$', r'$\sigma$', r'$c$', r'$\mu$', r'$\log_{10}(w)$', r'$\log_{10}(y)$', r'$b$'], fontsize=14, rotation=45)
        plt.legend(fontsize=15)
        plt.hlines(y=0,xmin=0,xmax=8, color='k')
        plt.savefig(name + '.pdf', bbox_inches='tight')
    
if __name__ == '__main__':
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        main(pool)