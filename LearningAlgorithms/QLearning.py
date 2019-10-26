import numpy as np 


class QLearning():

    def __init__(self, num_actions, num_features, gamma=0.9, alpha=0.001):
        self.gamma = gamma 
        self.alpha = alpha
        self.W = np.random.random( (num_actions, num_features))


    def select_action(self, state):
        q_values = self.W.dot( state )
        action = q_values.argmax()
        return action

    def update(self, s, a, r, s_n):
        Qnext = self.W.dot(s_n)
        Q = self.W.dot(s)

        delta_ = r + self.gamma * np.max(Qnext) - Q[a]
        self.W[a] += self.alpha * delta_ * s



