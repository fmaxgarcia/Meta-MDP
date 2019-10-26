import numpy as np 
import random

class Sarsa():

    def __init__(self, num_actions, num_features, init_w=None, gamma=0.9, alpha=0.001, lambda_=0.9):
        self.gamma = gamma 
        self.alpha = alpha
        self.lambda_ = lambda_
        self.e = np.zeros( (num_actions, num_features) )

        if init_w is None:
            self.W = np.random.random( (num_actions, num_features) )
        else:
            self.W = init_w

    def select_action(self, state, stochastic=False):
        
        if stochastic:
            z = self.W.dot( state )
            softmax = np.exp(z - np.max(z)) / np.sum( np.exp(z - np.max(z)) )
            rand = random.random()
            p_sum = 0.0
            for idx, a in enumerate(softmax):
                p_sum += a 
                if rand <= p_sum:
                    action = idx
                    break
        else:
            q_values = self.W.dot( state )
            action = q_values.argmax()
        return action


    def update(self, s, a, r, s_n, a_n, etraces):
        Qnext = self.W.dot(s_n)
        Q = self.W.dot(s)
        delta_ = r + (self.gamma * Qnext[a_n]) - Q[a]

        if etraces == False:
            self.W[a] += self.alpha * delta_ * s
        else:
            self.e *= self.lambda_
            self.e[a,:] = 1.0
            self.W[a] += self.alpha * delta_ * s * self.e[a]

    '''
    Sample batch is a list of dictionaries containing keys: {'state', 'action', 'reward', 'state_next', 'acion_next'}
    It allows me to make an average update based on a step over several different domains
    '''
    def update_batch(self, sample_batch):
        W = self.W.copy()
        W_updates = np.zeros(W.shape)
        action_count = [0.0 for i in range(W.shape[0])]
        for update_info in sample_batch:
            if update_info is None:
                continue

            state = update_info["state"]
            action, explore = update_info["action"]
            reward = update_info["reward"]
            state_next = update_info["state_next"]
            action_next, explore_next = update_info["action_next"]
                    
            if explore is True:
                phi = state 
                phi_next = state_next
                Qnext = W.dot(phi_next)
                Q = W.dot(phi)
                delta_ = reward + (self.gamma * Qnext[action_next]) - Q[action]
                W_updates[action] += self.alpha * delta_ * phi
                
                action_count[action] += 1.0
        
        for a in range(W.shape[0]):
            if action_count[a] != 0.0:
                self.W[a] += (W_updates[action] / action_count[a])



