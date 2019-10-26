
from Features import fourier_basis
import numpy as np
import sys


class EnvWrapper:

    def __init__(self, env, basis_order=2, normalization=0):
        self.env = env
        
        self.env.reset()
        self.normalization = normalization
        self.order = basis_order
        (obs, reward, done, info) = self.env.step(self.env.action_space.sample())
        if normalization == 0:
            obs = EnvWrapper.modified_sigmoid(obs)
        else:
            obs = EnvWrapper.normalize_range(obs, env.env_range)
        phi = fourier_basis(obs, order=self.order)
        self.num_features = phi.shape[0]
        self.num_actions = self.env.action_space.n

        self.action_space = self.env.action_space



    @staticmethod
    def normalize_range(x, r):
        return (x - r[:,0]) / (r[:,1] - r[:,0]) 

    @staticmethod
    def modified_sigmoid(x):
        #return x normalized 0-1
        #Find values to normalize with sigmoid in range
        xout = x.copy()
        xout[0] = 1 / (1 + np.exp(-(0.5*x[0])))
        xout[1] = 1 / (1 + np.exp(-(0.002*x[1])))
        xout[2] = 1 / (1 + np.exp(-(5.0*x[2])))
        xout[3] = 1 / (1 + np.exp(-(0.002*x[3])))

        return xout

    def reset(self):
        obs = self.env.reset()
        if self.normalization == 0:
            obs = self.modified_sigmoid(obs)
        else:
            obs = self.normalize_range(obs, self.env.env_range)
        obs = fourier_basis(obs, order=self.order)
        return obs

    def render(self):
        self.env.render()

    def random_action(self):
        return self.env.action_space.sample()

    def step(self, action=None):
        if action is None:
            obs, reward, done, _ = self.env.step( self.env.action_space.sample() )
        else:
            obs, reward, done, _ = self.env.step(action)
        
        if self.normalization == 0:
            obs = self.modified_sigmoid(obs)
        else:
            obs = self.normalize_range(obs, self.env.env_range)
        obs = fourier_basis(obs, order=self.order)
        return obs, reward, done, ""
    
