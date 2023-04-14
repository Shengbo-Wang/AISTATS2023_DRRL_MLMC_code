import numpy as np
from typing import Callable
from abc import ABC, abstractmethod

class DR_RL_MLMC:
    def __init__(self,generative_model,delta,gamma,perform_value_iteration = True,g = 5/8,Q_0 = None):
        self.delta = delta
        self.gamma = gamma
        self.model = generative_model
        self.r_max = generative_model.r_max
        self.states = generative_model.states
        self.rewards = generative_model.rewards
        self.action_at_state = generative_model.action_at_state
        self.sa_pairs = generative_model.get_sa_pairs()
        self.is_mdp = self.model.is_mdp
        self.Q = Q_0
        self.R = None
        if self.Q is None:
            self.Q = np.zeros(len(self.sa_pairs))
        self.Q_star = None
        if perform_value_iteration:
            self.value_iteration()
        self.step_count = 0
        self.sample_used = 0
        self.g = g

    def one_step_sa(self,lr,disp = False):
        self.step_count += 1
        self.Q = (1-lr)*self.Q + lr*self.apply_MLMC_Bellman()
        if disp:
            print("At step {}, the Q function is {}.".format(self.step_count,self.Q))
        return self.step_count, self.Q

    def n_step_sa(self,lrfunc:Callable,n=1,disp = False):
        for i in range(0,n):
            self.step_count += 1
            self.Q = (1-lrfunc(self.step_count))*self.Q + lrfunc(self.step_count)*self.apply_MLMC_Bellman()
        if disp:
            print("At step {}, the Q function is {}.".format(self.step_count,self.Q))
        return self.Q, self.sample_used, self.step_count

    def apply_MLMC_Bellman(self,Q= None):
        if Q is None:
            Q = self.Q
        N = np.random.geometric(self.g,1)[0]-1
        pN = (1-self.g)**N*self.g
        sample_size = 2**N
        self.sample_used += sample_size*4 + 2
        vf = self.value_func_from_Q()
        TQ = []
        alpha_v_max = max([abs(v) for v in vf.values() ])/self.delta
        alpha_r_max = self.r_max/self.delta
        for sa in self.sa_pairs:
            odd_samples = self.model.generate(sa,sample_size)
            even_samples = self.model.generate(sa,sample_size)
            zero_sample = self.model.generate(sa,1)
            r0_sa = zero_sample[1][0]
            s0_sa = zero_sample[0][0]
            v0_sa = vf[s0_sa]
            Delta_r_rob_sa = self.Delta_est(odd_samples[1],even_samples[1],alpha_r_max)
            odd_v = np.array([vf[s] for s in odd_samples[0]])
            even_v = np.array([vf[s] for s in even_samples[0]])
            Delta_v_rob_sa = self.Delta_est(odd_v,even_v,alpha_v_max)
            TQ_sa = r0_sa + Delta_r_rob_sa/pN + self.gamma*(v0_sa + Delta_v_rob_sa/pN)
            TQ.append(TQ_sa)
        return np.array(TQ)
    
    # For error and policy analysis; return non-DR Qval
    @staticmethod
    def non_DR_value_iteration(mdp,gamma,tol = 1e-8):
        if not mdp.is_mdp:
            raise Exception("Cannot perform value iteration: input generative model is not a MDP.")
        Q_old = np.zeros(len(mdp.get_sa_pairs()))
        flag = True
        iter = 1
        R = None
        def vf(Q):
            value_func = []
            for state in mdp.states:
                value_func.append(max([ Q[mdp.sa_pairs.index(tuple([state,a]))]  for a in mdp.action_at_state[state]]))
            return np.array(value_func)
        
        def non_DR_value_iteration_once(Q):
            val = vf(Q)
            VQ = []
            nonlocal R
            R_computed = not (R is None)
            if not R_computed:
                R_temp = []
            for sa in mdp.sa_pairs:
                states_dist_sa = mdp.transition_map[sa]
                rewards_dist_sa = mdp.reward_map[sa]
                if not R_computed:
                    R_sa = rewards_dist_sa@mdp.rewards
                    R_temp.append(R_sa)
                V_sa = states_dist_sa@val
                VQ.append(V_sa)
            if not R_computed:
                R = np.array(R_temp)
            return R + gamma*np.array(VQ)
            
        while flag and iter < 5000:
            iter += 1
            Q_new = non_DR_value_iteration_once(Q_old)
            flag = max(abs(Q_old-Q_new))>tol
            Q_old= Q_new
        if iter >= 5000:
            print('value_iteration doesnt converge')
        return Q_new
    # For error and policy analysis; return non-DR val of deterinistic policy
    @staticmethod
    def non_DR_dpolicy_evaluation(mdp,gamma,dpolicy,tol = 1e-8):
        if not mdp.is_mdp:
            raise Exception("Cannot perform value iteration: input generative model is not a MDP.")
        flag = True
        iter = 1
        R = None
        vf_old = np.zeros(len(mdp.states))
        def non_DR_dpolicy_evaluation(vf):
            V = []
            nonlocal R
            R_computed = not (R is None)
            if not R_computed:
                R_temp = []
            for s in mdp.states:
                states_dist_s = mdp.transition_map[(s,dpolicy[s])]
                rewards_dist_s = mdp.reward_map[(s,dpolicy[s])]
                if not R_computed:
                    R_s = rewards_dist_s@mdp.rewards
                    R_temp.append(R_s)
                V_s = states_dist_s@vf
                V.append(V_s)
            if not R_computed:
                R = np.array(R_temp)
            return R + gamma*np.array(V)
            
        while flag and iter < 5000:
            iter += 1
            vf_new = non_DR_dpolicy_evaluation(vf_old)
            flag = max(abs(vf_old-vf_new))>tol
            vf_old= vf_new
        if iter >= 5000:
            print('value_iteration doesnt converge')
        vf = {}
        idx = 0
        for s in mdp.states:
            vf[s] = vf_new[idx]
            idx += 1
        return vf

    def value_iteration(self, tol = 1e-8):
        if not self.is_mdp:
            raise Exception("Cannot perform value iteration: input generative model is not a MDP.")
        if self.Q_star is None:
            self.Q_star = self.Q
        Q_old = self.Q_star
        flag = True
        iter = 1
        while flag and iter < 5000:
            iter += 1
            self.value_iteration_once()
            flag = max(abs(Q_old-self.Q_star))>tol
            Q_old= self.Q_star
        if iter >= 5000:
            print('value_iteration doesnt converge')

    def value_iteration_once(self):
        if not self.is_mdp:
            raise Exception("Cannot perform value iteration: input generative model is not a MDP.")
        if self.Q_star is None:
            self.Q_star = self.Q
        vf = self.value_func_from_Q(self.Q_star)
        alpha_v_max = max([abs(v) for v in vf.values() ])/self.delta
        alpha_r_max = self.r_max/self.delta
        VQ = []
        R_computed = not (self.R is None)
        if not R_computed:
            R_temp = []
        for sa in self.sa_pairs:
            states_dist_sa = self.model.transition_map[sa]
            rewards_dist_sa = self.model.reward_map[sa]
            if not R_computed:
                R_sa = self.dual_opt(self.rewards,rewards_dist_sa,alpha_r_max)
                R_temp.append(R_sa)
            V_sa = self.dual_opt(np.array([vf[s] for s in self.states]),states_dist_sa,alpha_v_max)
            VQ.append(V_sa)
        if not R_computed:
            self.R = np.array(R_temp)
        self.Q_star = self.R + self.gamma*np.array(VQ)

    def Delta_est(self,odd,even,alpha_max):
        n = len(odd)
        meas_o = np.ones(n)/n
        meas_oe = np.ones(2*n)/2/n
        return self.dual_opt(np.concatenate((odd,even)),meas_oe,alpha_max) - \
                self.dual_opt(odd,meas_o,alpha_max)/2- self.dual_opt(even,meas_o,alpha_max)/2

    def dual_opt(self,data,meas,alpha_max,opt_tol=1e-8):
        barr = opt_tol/100
        xl = barr
        xr = alpha_max+2*barr
        r = 0.382
        pos_data = data[meas > 0]
        pos_meas = meas[meas > 0]
        essinf = min(pos_data)
        h = lambda x: -x*np.log(np.dot(pos_meas,np.exp((-1)*(pos_data-essinf)/x)))-x*self.delta
        diff = 2*opt_tol

        #check if alpha* = 0
        kappa = sum(pos_meas[pos_data == essinf])
        if kappa >= np.exp(-self.delta):
            return essinf

        while diff > opt_tol or xr-xl > opt_tol:
            hatxl = xl + r*(xr-xl)
            hatxr = xl + (1-r)*(xr-xl)
            if h(hatxl)==np.Inf:
                print("dual_opt has a numerical issue")
                return min(pos_data)
            diff = h(hatxl)-h(hatxr)
            if diff <0:
                xl = hatxl
            else:
                xr = hatxr
        opt_x = xr
        #print(essinf,opt_x,alpha_max)
        return h(opt_x)+essinf


    def value_func_from_Q(self,Q = None,dpolicy = None):
        if dpolicy is None:
            if Q is None:
                Q = self.Q
            value_func = {}
            for state in list(self.action_at_state.keys()):
                value_func[state] = max([ Q[self.sa_pairs.index(tuple([state,a]))]  for a in self.action_at_state[state]])
            return value_func
        else:
            if Q is None:
                Q = self.Q
            value_func = {}
            for state in list(self.action_at_state.keys()):
                value_func[state] = Q[self.sa_pairs.index(tuple([state,dpolicy[state]]))]  
            return value_func
        
    def policy_from_Q(self,Q = None):
        if Q is None:
            Q = self.Q
        value_func = {}
        for state in list(self.action_at_state.keys()):
            value_func[state] = np.argmax([ Q[self.sa_pairs.index(tuple([state,a]))]  for a in self.action_at_state[state]])
        return value_func
    
    def reset(self):
        self.Q = np.zeros(len(self.sa_pairs))
        self.step_count = 0
        self.sample_used = 0
        print('Reset Q function, sample used, step count to 0.')