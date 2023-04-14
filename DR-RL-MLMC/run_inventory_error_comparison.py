import numpy as np
import multiprocessing
from typing import Callable
from abc import ABC, abstractmethod
from  dr_rl_mlmc import DR_RL_MLMC
from generative_model import inventory_MDP
import matplotlib.pyplot as plt
import time


vD = 7
vS = 7
vA = 7
delta = 0.5
gamma = 0.7
error_max = 2
error_min = 0.3
n_traj = 1000


mdp = inventory_MDP(vS,vA,vD)
target = DR_RL_MLMC(mdp,delta,gamma).Q_star

errors = np.linspace(error_max, error_min,num = 50)
#envs = gen_deployment_envs(n_envs,vD+1,1,unif_d_dist,delta,kl_loss)

def run_one_trajectory(g):
    n_sample_at_err = []
    drrl_mlmc = DR_RL_MLMC(mdp,delta,gamma,perform_value_iteration = False,g = g)
    lr = lambda x: 1/(1+(1-gamma)*x)
    curr_err = max(abs(target))
    for err in errors:
        while curr_err > err:
            Q, n_sample, n_iter = drrl_mlmc.n_step_sa(lr,1)
            curr_err = max(abs(Q-target))
        n_sample_at_err.append(n_sample)
    return n_sample_at_err

if __name__ == '__main__':
    g_new = np.zeros(n_traj)+5/8
    g_old = np.zeros(n_traj)+0.499
    pool = multiprocessing.Pool()
    print(pool._processes)
    
    start_time = time.time()
    output_new = pool.map_async(run_one_trajectory,g_new)
    output_new.wait()
    print(time.time() - start_time)
    new = output_new.get()
    
    start_time = time.time()
    output_old = pool.map_async(run_one_trajectory,g_old)
    output_old.wait()
    print(time.time() - start_time)
    old = output_old.get()
    
    ratio = np.mean(np.array(old),0)/np.mean(np.array(new),0)
    plt.plot(errors,ratio)
    plt.xlabel("error")
    plt.ylabel("avg #sample ratio")
    plt.savefig('err_ratio_plt', dpi=1000)
    
    
    
    