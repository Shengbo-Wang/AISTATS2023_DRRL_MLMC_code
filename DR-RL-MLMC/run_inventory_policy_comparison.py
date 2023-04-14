import numpy as np
import multiprocessing
from typing import Callable
from abc import ABC, abstractmethod
from  dr_rl_mlmc import DR_RL_MLMC
from generative_model import inventory_MDP
import matplotlib.pyplot as plt

def gen_deployment_envs(n,d,alpha,ref,delta,loss):
    i = 0
    ad = np.zeros(d) + alpha
    data = []
    while i < n:
        temp = np.random.dirichlet(ad)
        if loss(temp,ref)<= delta:
            i+=1
            data.append(temp)
            print(i)
    return data
def kl_loss(p,q):
    supp = p > 0
    ppos = p[supp]
    qpos = q[supp]
    return np.log(ppos/qpos)@ppos



if __name__ == '__main__':
    vD = 10
    vS = 10
    vA = 10
    n_envs = 10000
    delta = 0.5
    gamma = 0.6
    
    unif_d_dist = np.zeros(vD+1) + 1/(1+vD)
    ref_mdp = inventory_MDP(vS,vA,vD)
    drrl_mlmc = DR_RL_MLMC(ref_mdp,delta,gamma)
    ref_non_dr_Q = DR_RL_MLMC.non_DR_value_iteration(ref_mdp, gamma)
    ref_non_dr_V = drrl_mlmc.value_func_from_Q(Q = ref_non_dr_Q)
    ref_non_dr_policy = drrl_mlmc.policy_from_Q(Q = ref_non_dr_Q)
    ref_dr_policy = drrl_mlmc.policy_from_Q(drrl_mlmc.Q_star)
    #envs = gen_deployment_envs(n_envs,vD+1,1,unif_d_dist,delta,kl_loss)
    
    def compare_dpolicies_on_env(env):
        mdp = inventory_MDP(vS,vA,vD,demand_dist = env)
        v0_non = DR_RL_MLMC.non_DR_dpolicy_evaluation(mdp, gamma, ref_non_dr_policy)[0]
        v0_dr = DR_RL_MLMC.non_DR_dpolicy_evaluation(mdp, gamma, ref_dr_policy)[0]
        return [v0_non,v0_dr]
    def compare_dpolicies_different_demand(delta):
        mdp = inventory_MDP(vS,vA,vD,different_demand=True,delta = delta)
        v0_non = DR_RL_MLMC.non_DR_dpolicy_evaluation(mdp, gamma, ref_non_dr_policy)[0]
        v0_dr = DR_RL_MLMC.non_DR_dpolicy_evaluation(mdp, gamma, ref_dr_policy)[0]
        return [v0_non,v0_dr]
    
    # mdp = inventory_MDP(vS,vA,vD)
    # drrl = DR_RL_MLMC(mdp,delta,gamma)
    # dr_Q = drrl.Q_star
    # dr_V = drrl.value_func_from_Q(Q = dr_Q)
    # dr_V_using_non_dr_pi = drrl.value_func_from_Q(Q = dr_Q,dpolicy = ref_non_dr_policy)
    
    
    # envs = gen_deployment_envs(n_envs,vD+1,1,unif_d_dist,delta,kl_loss)
    # for env in envs:
    #   hist_data.append(compare_dpolicies_on_env(env))
    
    hist_data = []
    for i in range(n_envs):
        print(i)
        hist_data.append(compare_dpolicies_different_demand(delta))
    hist_data = np.array(hist_data)
    plt.hist(hist_data,50,label = ['non DR','DR'])
    plt.legend()
    
    
    
    