import numpy as np 

class REINFORCE():

    def __init__(self, alpha, beta, gamma, num_actions, num_features, w=None, Theta=None):
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
        self.w = w if w is not None else np.random.random( (num_features, 1) )
        self.Theta = Theta if Theta is not None else np.random.random( (num_actions, num_features) )
        self.action_type = 0


    def _pi_s(self, s):
        z = self.Theta.dot( s )
        softmax = np.exp(z - np.max(z)) / np.sum( np.exp(z - np.max(z)) )
        return softmax

    def select_action(self, state):
        softmax = self._pi_s(state)
        rand = np.random.random()
        p_sum = 0.0
        for idx, a in enumerate(softmax):
            p_sum += a 
            if rand <= p_sum:
                action = idx
                break
        return action

    def _softmax_gradient(self, s, a, theta):
        s = s.reshape((-1,1))
        first_term = np.zeros(self.Theta.shape)
        first_term[a] = s[:,0] 

        theta_s = theta.dot(s)
        second_term = (1.0/np.sum(np.exp( theta_s-np.max(theta_s) ))) * np.exp( theta_s - np.max(theta_s) ).dot(s.T)
        return first_term - second_term


    def update(self, trajectory_samples, other_w=None, other_theta=None, averages=True):
        grad_theta = np.zeros(self.Theta.shape)
        grad_w = np.zeros(self.w.shape)

        w = self.w if other_w is None else other_w.copy()
        theta = self.Theta if other_theta is None else other_theta.copy()

        for trajectory in trajectory_samples:
            for i, t in enumerate(trajectory):
                r = t['reward']
                s = t['state']
                a, explore = t['action']
                delta_ =  r - w.T.dot(s)
                grad_w[:,0] += delta_ * s
                pi_gradient = self._softmax_gradient(s, a, theta)
                grad_theta += (self.gamma**i) * delta_ * pi_gradient
        
        if averages:
            self.w = w + self.beta * (grad_w / len(trajectory_samples))
            self.Theta = theta + self.alpha * (grad_theta / len(trajectory_samples))
        else:
            self.w = w + self.beta * grad_w
            self.Theta = theta + self.alpha * grad_theta


