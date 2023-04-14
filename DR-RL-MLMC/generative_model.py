import numpy as np
from abc import ABC, abstractmethod
from typing import Callable
 
class Generative_Model(ABC):
    @property
    @abstractmethod
    def is_mdp(self):
      pass
    
    @property
    @abstractmethod
    def states(self):
      pass
    
    @property
    @abstractmethod
    def rewards(self):
      pass
    
    @property
    @abstractmethod
    def sa_pairs(self):
      pass

    @property
    @abstractmethod
    def action_at_state(self):
      pass
    
    @property
    def r_max(self):
      return max(self.rewards)
    
    def generate_action_at_state(self):
        action_at_state = {}
        for sa in self.sa_pairs:
            temp = action_at_state.get(sa[0],[])
            temp.append(sa[1])
            action_at_state[sa[0]] = temp
        return action_at_state

    def generate(self,sa,k):
        sdist = self.transition_map[sa]
        rdist = self.reward_map[sa]
        state = np.random.choice(self.states[sdist > 0],k,p = sdist[sdist > 0])
        reward = np.random.choice(self.rewards[rdist > 0],k,p = rdist[rdist > 0])
        return np.array([state,reward])

    def get_sa_pairs(self):
        return self.sa_pairs
    


class Hard_MDP(Generative_Model):
  @property
  def is_mdp(self):
    return True
  @property
  def states(self):
    return np.array([0, 1, 2, 3])
  @property
  def rewards(self):
    return np.array([0, 1])
  @property
  def sa_pairs(self):
    return [(3,1),(2,1),(0,1),(1,1),(1,2)]
  @property
  def action_at_state(self):
    return self.generate_action_at_state()

  def __init__(self,p):
    self.transition_map = {self.sa_pairs[0]: np.array([0,0,0,1]),
                           self.sa_pairs[1]: np.array([1-p,0,p,0]),
                           self.sa_pairs[2]: np.array([1,0,0,0]),
                           self.sa_pairs[3]: np.array([1-p,p,0,0]),
                           self.sa_pairs[4]: np.array([1-p,p,0,0])}
    self.reward_map = {self.sa_pairs[0]: np.array([0,1]),
                       self.sa_pairs[1]: np.array([0,1]),
                       self.sa_pairs[2]: np.array([1,0]),
                       self.sa_pairs[3]: np.array([0,1]),
                       self.sa_pairs[4]: np.array([0,1])}


class Hard_MDP_Azar(Generative_Model):
  @property
  def is_mdp(self):
    return True
  @property
  def states(self):
    return np.array([1, 2, 3,4,5])
  @property
  def rewards(self):
    return np.array([0, 1])
  @property
  def sa_pairs(self):
    return [(1,0),(1,1),(2,0),(3,0),(4,0),(5,0)]
  @property
  def action_at_state(self):
    return self.generate_action_at_state()

  def __init__(self,p):
    self.transition_map = {self.sa_pairs[0]: np.array([0,1,0,0,0]),
                           self.sa_pairs[1]: np.array([0,0,1,0,0]),
                           self.sa_pairs[2]: np.array([0,p,0,1-p,0]),
                           self.sa_pairs[3]: np.array([0,0,p,0,1-p]),
                           self.sa_pairs[4]: np.array([0,0,0,1,0]),
                           self.sa_pairs[5]: np.array([0,0,0,0,1])}
    self.reward_map = {self.sa_pairs[0]: np.array([1,0]),
                       self.sa_pairs[1]: np.array([1,0]),
                       self.sa_pairs[2]: np.array([0,1]),
                       self.sa_pairs[3]: np.array([0,1]),
                       self.sa_pairs[4]: np.array([1,0]),
                       self.sa_pairs[5]: np.array([1,0])}




class inventory_MDP(Generative_Model):
    @property
    def is_mdp(self):
        return True
    @property
    def states(self):
        return self._states
    @states.setter
    def states(self, value):
        self._states = value

    @property
    def rewards(self):
        return self._rewards
    @rewards.setter
    def rewards(self, value):
        self._rewards = value
    
    @property
    def sa_pairs(self):
        return self._sa_pairs
    @sa_pairs.setter
    def sa_pairs(self, value):
        self._sa_pairs = value
    
    @property
    def action_at_state(self):
        return self.generate_action_at_state()

    def __init__(self,vS,vA,vD,k = 3,h = 1,p = 2,demand_dist = None,different_demand = False,delta = 0.1):
        if demand_dist is None:
            demand_dist = np.zeros(vD+1) + 1/(vD+1)
        else:
            assert len(demand_dist) == vD+1
        
        def gen_deployment_envs(loss):
            i = 0
            ad = np.zeros(vD+1) + 1
            while i < 1:
                temp = np.random.dirichlet(ad)
                if loss(temp,demand_dist)<= delta:
                    i+=1
            return np.array(temp)
        def kl_loss(p,q):
            supp = p > 0
            ppos = p[supp]
            qpos = q[supp]
            return np.log(ppos/qpos)@ppos
        
        
        reward_func = lambda s,a,d: -k*(a > 0) - h*(s+a-d)*(s+a-d > 0) - p*(d-s-a)*(d-s-a > 0) + max(h*vS,p*vD) + k
        self.states = np.arange(0,vS+1)
        sa_list = []
        for s in self.states:
            sa_list += [(s,a) for a in np.arange(0,vA-s+1)]
        self.sa_pairs = sa_list
        rewards_list = []
        self.reward_map = {}
        self.transition_map = {}
        for sa in self.sa_pairs:
            rewards_list += [reward_func(sa[0],sa[1],d) for d in np.arange(0,vD+1)]
        self.rewards = np.array(rewards_list)
        for sa_idx in np.arange(len(self.sa_pairs)):
            sa = self.sa_pairs[sa_idx]
            state_dist = np.zeros(vS+1)
            reward_dist = np.zeros((len(self.sa_pairs),vD+1))
            if different_demand:
                demand_dist = gen_deployment_envs(kl_loss)
            for d in np.arange(0,vD+1):
                state_dist[(sa[0]+sa[1]-d > 0)*(sa[0]+sa[1]-d)] += demand_dist[d]
                reward_dist[sa_idx,d] += demand_dist[d]
            self.transition_map[sa] = state_dist
            self.reward_map[sa] = reward_dist.flatten()