import numpy as np
import multiprocessing
from typing import Callable
from abc import ABC, abstractmethod
from  dr_rl_mlmc import DR_RL_MLMC
from generative_model import Hard_MDP
import matplotlib.pyplot as plt

def log_log_error_plot(mdp,gamma,delta,step = 2,start_iter = 200,end_iter = 800,n_traj = 400):
    drrl_mlmc = DR_RL_MLMC(mdp,delta,gamma)
    #K = 640*np.e*np.log(5)/(1-drrl_mlmc.gamma)**3
    #eps = 4/(1-drrl_mlmc.gamma)
    #lr = lambda x: eps/(x+K)
    lr = lambda x: 1/(1+(1-gamma)*x)
    #lr = lambda x: 0.05
    target = drrl_mlmc.Q_star
    
    
    errs_avg = None
    
    
    for nt in np.arange(0,n_traj):
        print((gamma,nt))
        drrl_mlmc.reset()
        drrl_mlmc.n_step_sa(lr,start_iter-step)
        errs = np.array([])
        #log_n_samples = np.array([])
        log_n_iter = np.array([])
      
        for iter in np.arange(start_iter,end_iter+1,step):
            Q, n_sample, n_iter = drrl_mlmc.n_step_sa(lr,step)
            errs = np.append(errs,max(abs(Q-target)))
            #log_n_samples = np.append(log_n_samples,np.log(n_sample))
            log_n_iter = np.append(log_n_iter, np.log10(n_iter))
        if errs_avg is None:
            errs_avg = errs*0
        errs_avg = errs_avg + errs
      
    errs_avg = errs_avg/n_traj
    return (log_n_iter,np.log10(errs_avg))

def log_log_error(mdp,gamma,delta,step = 1,start_iter = 1500,end_iter = 1500,n_traj = 500):
    drrl_mlmc = DR_RL_MLMC(mdp,delta,gamma)
    #K = 640*np.e*np.log(5)/(1-drrl_mlmc.gamma)**3
    #eps = 4/(1-drrl_mlmc.gamma)
    #lr = lambda x: eps/(x+K)
    lr = lambda x: 1/(1+(1-gamma)*x)
    target = drrl_mlmc.Q_star
    
    
    errs_avg = None
    
    
    for nt in np.arange(0,n_traj):
        print((gamma,nt))
        drrl_mlmc.reset()
        drrl_mlmc.n_step_sa(lr,start_iter-step)
        errs = np.array([])
        #log_n_samples = np.array([])
        log_n_iter = np.array([])
        
        for iter in np.arange(start_iter,end_iter+1,step):
            Q, n_sample, n_iter = drrl_mlmc.n_step_sa(lr,step)
            errs = np.append(errs,max(abs(Q-target)))
            log_n_iter = np.append(log_n_iter, np.log10(n_iter))
        if errs_avg is None:
            errs_avg = errs*0
        errs_avg = errs_avg + errs
      
    errs_avg = errs_avg/n_traj
    return (log_n_iter,np.log10(errs_avg))

if __name__ == '__main__':
    print('hard_mdp_plot')
    gamma = 0.7
    pgamma = lambda x: (4*x-1)/3/x
    #pgamma = lambda x: 6/7
    deltas = [0.02,0.04,0.08,0.16,0.32]
    loglog_input = []
    for delta in deltas:
        loglog_input.append((Hard_MDP(pgamma(gamma)),gamma,delta))
    pool = multiprocessing.Pool()
    output = pool.starmap(log_log_error_plot, loglog_input)
    plt.rcParams["figure.figsize"] = (6.5,4.5)
    plt.rcParams.update({'font.size': 20})
    plt.rc('legend',fontsize=16)
    c = ['tab:blue','tab:red','tab:green','tab:orange','tab:purple']
    for i in np.arange(len(deltas)):
        plt.plot(output[i][0],output[i][1],c=c[i],label ='δ = {}'.format(deltas[i]))
    a = output[0][0][0]
    yy = -(output[0][0]-a)/2 - 1.5
    #plt.plot(output[0][0],yy,c='k',linestyle='dashed',label ='slope:-1/2')
    plt.xlabel("lg #iter")
    plt.ylabel("lg avg error")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('log-log_err-iter small deltas = {}.png'.format(deltas), dpi=1000)
    
    
    plt.clf()
    
    
    gamma = 0.7
    pgamma = lambda x: (4*x-1)/3/x
    #pgamma = lambda x: 6/7
    deltas = [0.5,1,2,4,8]
    loglog_input = []
    for delta in deltas:
        loglog_input.append((Hard_MDP(pgamma(gamma)),gamma,delta))
    pool = multiprocessing.Pool()
    output = pool.starmap(log_log_error_plot, loglog_input)
    plt.rcParams["figure.figsize"] = (6.5,4.5)
    plt.rcParams.update({'font.size': 20})
    plt.rc('legend',fontsize=16)
    c = ['tab:blue','tab:red','tab:green','tab:orange','tab:purple']
    for i in np.arange(len(deltas)):
        plt.plot(output[i][0],output[i][1],c=c[i],label ='δ = {}'.format(deltas[i]))
    a = output[0][0][0]
    yy = -(output[0][0]-a)/2 - 1.5
    #plt.plot(output[0][0],yy,c='k',linestyle='dashed',label ='slope:-1/2')
    plt.xlabel("lg #iter")
    plt.ylabel("lg avg error")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('log-log_err-iter large deltas = {}.png'.format(deltas), dpi=1000)
    # print('hard_mdp_test_gamma')
    # gammas = [0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
    # pgamma = lambda x: (4*x-1)/3/x
    # loglog_input = []
    # for gamma in gammas:
    #     loglog_input.append((Hard_MDP(pgamma(gamma)),gamma,0.1))
        

    # pool = multiprocessing.Pool()
    # output = pool.starmap(log_log_error, loglog_input)
    # for gamma in gammas: