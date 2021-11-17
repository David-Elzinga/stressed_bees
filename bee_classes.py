import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Set I = 1, H, F = 0, then social inhibition pulls out of F even though nothing is there??
def social_odes(t,x,params):
    H = x[0]; F = x[1]; I = x[2]
    E = F + params['p']*I
    dx = np.zeros(3)
    dx[0] = params['gamma']*H*E/(params['a']*H + E + params['w'])*(1-H/params['K']) - params['sigma']*(H-params['c']*E)
    dx[1] = params['sigma']*(H-params['c']*E) - params['beta']*F - params['mu']*F
    dx[2] = params['beta']*F - params['nu']*I
    return dx

class system:
    def __init__(self,params,ICs,fun_odes):
        self.params = params
        self.ICs = ICs
        self.fun_odes = fun_odes

    def solve_odes(self,t):
        ivp = solve_ivp(fun=self.fun_odes, t_span=[t.min(), t.max()], y0=self.ICs, args=(self.params,), dense_output=True)
        return ivp.sol(t)

    def ts_plot(self,t):
        z = self.solve_odes(t)
        plt.plot(t, z[0,:], color='peru', label=r'$H$')
        plt.plot(t, z[1,:], color='green', label=r'$F$')
        plt.plot(t, z[2,:], color='mediumorchid', label=r'$I$')
        plt.xlabel(r'$t$',fontsize=12); plt.ylabel('Bees',fontsize=12); plt.legend(fontsize=12)
        plt.show()

    def silly(self):
        print('my ic is:' + str(self.ICs))