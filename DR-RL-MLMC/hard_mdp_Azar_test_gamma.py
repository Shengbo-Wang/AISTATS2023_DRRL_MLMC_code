import numpy as np
import multiprocessing
from typing import Callable
from abc import ABC, abstractmethod
from  dr_rl_mlmc import DR_RL_MLMC
from generative_model import Hard_MDP_Azar
import matplotlib.pyplot as plt

rep = 300
delta = 0.1
gammas = np.linspace(0.95, 0.7,10)
niters = [500,1000,1500]
#gammas = np.array([0.7])


def err_at_test_iter(parms):
    gamma = parms[0]
    test_iter = parms[1]
    p = (4*gamma-1)/3/gamma
    mdp = Hard_MDP_Azar(p)
    lr = lambda x: 1/(1+(1-gamma)*x)
    drrl_mlmc = DR_RL_MLMC(mdp,delta,gamma)
    target = drrl_mlmc.Q_star
    err = 0
    for i in range(rep):
        Q,_,_ = drrl_mlmc.n_step_sa(lr,test_iter)
        err += max(abs(Q-target))
        drrl_mlmc.reset()
    return err/rep

if __name__ == '__main__':
    parms = []
    pool = multiprocessing.Pool()
    for gamma in gammas:
        for i in niters:
            parms.append((gamma,i))
    output = pool.map(err_at_test_iter, parms)
    err_dict = {}
    xx = np.log10(1-gammas)
    for idx in range(0,len(output)):
        parm = parms[idx]
        err_dict[parm[1]] = err_dict.get(parm[1],[]) + [output[idx]]
    
    plt.rcParams["figure.figsize"] = (6.5,4.5)
    plt.rcParams.update({'font.size': 20})
    plt.rc('legend',fontsize=16)
    c = ['tab:orange','tab:purple','tab:cyan']
    cidx = 0
    slopes = []
    for key in err_dict.keys():
        yy = np.log10(err_dict[key])
        plt.scatter(xx,yy,marker = 'o',c = c[cidx])
        X = np.array([xx,np.ones(len(xx))]).T
        beta = np.linalg.pinv(X.T@X)@X.T@(yy.reshape((len(yy),1)))
        plt.plot(xx,X@beta.flatten(),c = c[cidx],linestyle='dashed',
                 label = 'iter: {}'.format(key))
        cidx +=1 
        slopes.append(np.round(beta.flatten()[0],decimals=4))
    plt.ylabel("lg avg error")
    plt.xlabel("lg(1-γ)")
    plt.legend()
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('Azar_test_gamma_plt_200_rep,slope = {}.png'.format(slopes), dpi=1000)
    
    
    