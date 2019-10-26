

from LinearMeta import MetaPolicy
from Features import fourier_basis
from Environments import EnvWrapper
from linear_agent import LinearAgent
import time
import matplotlib.pyplot as plt
import numpy as np
import gym

basis_order = 1
alpha = 1e-2
beta = 1e-2
setup = {"force" : 20.0, "pole_length" : 1.2, "masscart" : 5.0, "masspole" : 0.1}        
gym_env = gym.make('CartPole-v0')
gym_env.env.force_mag = setup["force"]
gym_env.env.length = setup["pole_length"]
gym_env.env.masscart = setup["masscart"]
gym_env.env.masspole = setup["masspole"]

env = EnvWrapper(gym_env, basis_order=basis_order, normalization=0)

agent = LinearAgent(env, meta_policy=None, alpha=alpha, beta=beta, algo="PPO")
agent.learning_algorithm.t_length = 8
agent.learning_algorithm.update_steps = 16
agent.learning_algorithm.epochs = 4
agent.learning_algorithm.batch_size = 8

rewards = agent.train(num_episodes=500, max_steps=1000, verbose=True, update_meta=False, render=False)

rewards = [ np.mean(rewards[i*10:(i+1)*10]) for i in range(len(rewards)//10) ]
plt.plot(range(len(rewards)), rewards)
plt.show(block=True)
# for _ in range(10000):
#     reward, done, update_info = agent.perform_step(update_meta=False)
#     env.render()
#     time.sleep(1.0)
