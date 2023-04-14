import numpy as np
import multiprocessing
from typing import Callable
from abc import ABC, abstractmethod
from  dr_rl_mlmc import DR_RL_MLMC
from generative_model import inventory_MDP
import matplotlib.pyplot as plt

def log_log_error_plot(mdp,gamma,delta,step = 2,start_iter = 20,end_iter = 40 ,n_traj = 10):
    drrl_mlmc = DR_RL_MLMC(mdp,delta,gamma)
    #K = 640*np.e*np.log(5)/(1-drrl_mlmc.gamma)**3
    #eps = 4/(1-drrl_mlmc.gamma)
    #lr = lambda x: eps/(x+K)
    lr = lambda x: 1/(1+(1-gamma)*x)
    target = drrl_mlmc.Q_star
    
    
    errs_avg = None
    
    
    for nt in np.arange(0,n_traj):
      drrl_mlmc.reset()
      print((gamma,nt))
      drrl_mlmc.n_step_sa(lr,start_iter-step)
      errs = np.array([])
      #log_n_samples = np.array([])
      log_n_iter = np.array([])
    
      for iter in np.arange(start_iter,end_iter+1,step):
        Q, n_sample, n_iter = drrl_mlmc.n_step_sa(lr,step)
        errs = np.append(errs,max(abs(Q-target)))
        #log_n_samples = np.append(log_n_samples,np.log(n_sample))
        log_n_iter = np.append(log_n_iter, np.log(n_iter))
      if errs_avg is None:
        errs_avg = errs*0
      errs_avg = errs_avg + errs
      
    errs_avg = errs_avg/n_traj
    
    plt.plot(log_n_iter,np.log(errs_avg),label ='gamma = {}'.format(drrl_mlmc.gamma))
    plt.xlabel("log #iter")
    plt.ylabel("log error")
    plt.legend(loc='upper right')
    plt.savefig('log-log_err-iter, gamma = {}.png'.format(gamma), dpi=1000)
    return (log_n_iter,np.log(errs_avg))

if __name__ == '__main__':
    print('inventory_mdp')
    gammas = [0.9,0.7,0.5]
    mdp = inventory_MDP(10,10,10)
    pool = multiprocessing.Pool()
    output = pool.starmap(log_log_error_plot, [(mdp,0.9,0.9),(mdp,0.7,0.9),(mdp,0.5,0.9)])
    for i in [0,1,2]:
        plt.plot(output[i][0],output[i][1],label ='gamma = {}'.format(gammas[i]))
        plt.xlabel("log #iter")
        plt.ylabel("log error")
        plt.legend(loc='upper right')
    plt.savefig('log-log_err-iter, gammas = {}.png'.format(gammas), dpi=1000)