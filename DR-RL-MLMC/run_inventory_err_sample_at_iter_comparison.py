import numpy as np
import multiprocessing
from typing import Callable
from abc import ABC, abstractmethod
from  dr_rl_mlmc import DR_RL_MLMC
from generative_model import inventory_MDP
import matplotlib.pyplot as plt
import time
import pandas as pd


vD = 7
vS = 7
vA = 7
delta = 0.5
gamma = 0.7
iter_min = 1000
iter_max = 5000
n_traj = 300


mdp = inventory_MDP(vS,vA,vD)
target = DR_RL_MLMC(mdp,delta,gamma).Q_star

iters = np.arange(iter_min,iter_max)
#envs = gen_deployment_envs(n_envs,vD+1,1,unif_d_dist,delta,kl_loss)

def run_one_trajectory(g):
    n_sample_at_iter = []
    err_at_iter = []
    drrl_mlmc = DR_RL_MLMC(mdp,delta,gamma,perform_value_iteration = False,g = g)
    lr = lambda x: 1/(1+(1-gamma)*x)
    Q, n_sample, n_iter = drrl_mlmc.n_step_sa(lr,iter_min)
    for n in iters:
        Q, n_sample, n_iter = drrl_mlmc.n_step_sa(lr,2)
        err_at_iter.append(max(abs(Q-target)))
        n_sample_at_iter.append(n_sample)
    return (n_sample_at_iter, err_at_iter)

if __name__ == '__main__':
    g_new = np.zeros(n_traj)+5/8
    g_old = np.zeros(n_traj)+0.499
    pool = multiprocessing.Pool()
    print('#processers: {}'.format(pool._processes))

    output_new = pool.map_async(run_one_trajectory,g_new)
    output_old = pool.map_async(run_one_trajectory,g_old)
    new = output_new.get()
    old = output_old.get()
    
    new_data_n_sample = []
    new_data_err = []
    old_data_n_sample = []
    old_data_err = []
    
    avg_new_data_n_sample = np.zeros(len(iters))
    avg_new_data_err = np.zeros(len(iters))
    avg_old_data_n_sample = np.zeros(len(iters))
    avg_old_data_err = np.zeros(len(iters))

    lab = 0
    for data in [new,old]:
        for tpl in data:
            if lab == 0:
                new_data_n_sample = new_data_n_sample + tpl[0]
                avg_new_data_n_sample = avg_new_data_n_sample + np.array(tpl[0])
                new_data_err = new_data_err + tpl[1]
                avg_new_data_err = avg_new_data_err + np.array(tpl[1])
            else:
                old_data_n_sample = old_data_n_sample + tpl[0]
                avg_old_data_n_sample = avg_old_data_n_sample + np.array(tpl[0])
                old_data_err = old_data_err + tpl[1]
                avg_old_data_err = avg_old_data_err + np.array(tpl[1])
        lab += 1
    avg_new_data_n_sample /= n_traj
    avg_new_data_err /= n_traj
    avg_old_data_n_sample /= n_traj
    avg_old_data_err /= n_traj
    
    plt.scatter(new_data_n_sample, new_data_err, marker = '.', s=0.12, edgecolor = 'none', c = 'tab:blue')
    plt.scatter(old_data_n_sample, old_data_err, marker = '.', s=0.12, edgecolor = 'none', c = 'tab:orange')
    
    plt.plot(avg_new_data_n_sample,avg_new_data_err,c = 'k', linewidth = 1.5)
    plt.plot(avg_new_data_n_sample,avg_new_data_err,c = 'tab:blue',label = 'g = 5/8', linewidth = 1.0)
    
    plt.plot(avg_old_data_n_sample,avg_old_data_err,c = 'k', linewidth = 1.5)
    plt.plot(avg_old_data_n_sample,avg_old_data_err,c = 'tab:orange',label = 'g = 0.499', linewidth = 1.0)
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("average #sample")
    plt.ylabel("average error")
    plt.legend(loc='upper right')
    plt.savefig('err&sample_at_iter_output/err_#sample_at_iter.png', dpi=1000)
    
    data = {'new_data_n_sample':new_data_n_sample,
            'new_data_err':new_data_err,
            'old_data_n_sample':old_data_n_sample,
            'old_data_err':old_data_err}
    data_avg = {'avg_new_data_n_sample':avg_new_data_n_sample,
                'avg_new_data_err':avg_new_data_err,
                'avg_old_data_n_sample':avg_old_data_n_sample,
                'avg_old_data_err':avg_old_data_err}
    export_data = pd.DataFrame(data=data)
    export_data.to_csv('err&sample_at_iter_output/scatter_data.csv')
    export_data = pd.DataFrame(data=data_avg)
    export_data.to_csv('err&sample_at_iter_output/avg_data.csv')
    
    
    
    
    
    