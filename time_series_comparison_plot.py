import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from bee_model import simulate

# Define model parameters. 
fig, axs = plt.subplots(1,2, sharex='all', figsize=(15,10))
for n, beta in enumerate([0.2, 0.8]):
    parm = {}
    parm['gamma'] = 0.15; parm['K_expn'] = 4
    parm['c'] = 2.7; parm['sigma'] = 0.25
    parm['mu'] = 0.136
    parm['w_expn'] = -6; parm['y_expn'] = -6
    parm['p'] = 0; parm['beta'] = beta; parm['nu'] = parm['mu']
    parm['b'] = 0.95; parm['a'] = (1 - parm['b'])/(parm['c']*parm['b'])

    # Solve the IVP.
    tsol = np.linspace(0,180,1800)
    Z = simulate(tsol, [3000, 500, 0], parm, system = "autonomous")
    H = Z.y[0,:]; F = Z.y[1,:]; I = Z.y[2,:]
    S = parm['sigma']*(H - parm['c']*F*(1 + parm['p']*I/(F + 10**parm['y_expn'])))
    E = parm['gamma']*(F + parm['p']*I)/(parm['a']*H + F + parm['p']*I + 10**parm['w_expn']) * (1 - F/(10**parm['K_expn']))
    burn = int(0.05*len(tsol))
    axs[n].plot(tsol, H, label='H', color='violet', linewidth=2)
    axs[n].plot(tsol, F, label='F', color='green', linewidth=2)
    axs[n].plot(tsol, I, label='I', color='blue', linewidth=2)
    axs[n].set_xlabel(r'$t$', fontsize=20)
    axs_twin = axs[n].twinx()
    axs_twin.plot(tsol, 1/(S/H), color='red', linestyle='-', linewidth=2)
    axs_twin.set_ylim(6.5,12.5)
    axs_twin.spines['right'].set_color('red')
    axs_twin.tick_params(axis='y', colors='red')

    if n == 0:
        axs[n].set_ylabel('Bees', rotation=0, fontsize=20, labelpad=50)
        axs[n].set_title(r'$\beta = 0.2$', fontsize=20)
        axs[n].plot(np.NaN, np.NaN, '-', color='red', label='AAOF')
        axs[n].legend(fontsize=10)
    else:
        axs[n].set_title(r'$\beta = 0.8$', fontsize=20)
        axs_twin.set_ylabel(r'AAOF', rotation=0, fontsize=20, labelpad=30, color='red')

#plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.savefig("time_series_plot.pdf",bbox_inches='tight')
plt.show()