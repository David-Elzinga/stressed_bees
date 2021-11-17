import bee_classes as bc
import numpy as np
from itertools import product
import multiprocessing

def worker(obj):
    time = np.linspace(0,100,1000)
    soln = obj.solve_odes(time)
    return soln[:,-1]

def main():

    # Define the parameters of the model
    params = {}
    params['gamma'] = 1; params['sigma'] = 0.3; params['c'] = 1
    params['beta'] = 0.6; params['p'] = 0; params['mu'] = 0.3; 
    params['nu'] = 0.4; params['a'] = 0.5; params['K'] = 1000; params['w'] = 0.1

    # Construct a list of systems with different initial conditions. 
    S_ICs = range(1,10,2)
    F_ICs = S_ICs; I_ICs = S_ICs
    systems = [bc.system(params, IC, bc.social_odes) for IC in list(product(S_ICs, F_ICs, I_ICs))]
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    end_vals = pool.map(worker, systems)

    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()