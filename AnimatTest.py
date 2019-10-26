

from LinearMeta import MetaPolicy
from Features import fourier_basis
from Environments import EnvWrapper
from CustomEnvironments.AnimatEnv import AnimatEnv 
from linear_agent import LinearAgent
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle

basis_order = 3
alpha = 1e-5
beta = 1e-4

env = AnimatEnv("./CustomEnvironments/maze7.txt")
env.reset()

(obs, reward, done, info) = env.step(env.action_space.sample())
obs = EnvWrapper.normalize_range(obs, env.env_range)
phi = fourier_basis(obs, order=basis_order)

num_features = phi.shape[0]# + len( cartpole_setup[0].keys() )
num_actions = env.action_space.n


# meta = MetaPolicy(num_features=num_features, num_actions=num_actions)

mazes = ["maze5.txt", "maze6.txt", "maze7.txt"]

for m in mazes:
    gym_env = AnimatEnv("./CustomEnvironments/"+m)

    env = EnvWrapper(gym_env, basis_order=basis_order, normalization=1)
    agent = LinearAgent(env, meta_policy=None, alpha=alpha, beta=beta, algo="PPO")
    agent.learning_algorithm.t_length = 8
    agent.learning_algorithm.update_steps = 64
    agent.learning_algorithm.epochs = 4
    agent.learning_algorithm.batch_size = 16

    dir = "./AnimatPPOEvalNoMeta/" + m.split(".")[0] + "/"
    # agent.random_action_prob = 0.0
    rewards = agent.train(num_episodes=500, max_steps=800, verbose=True, update_meta=False, render=False, save_path=dir)
    pickle.dump(rewards, open(dir+"rewards.pkl", "wb"))

# rewards = [ np.mean(rewards[i*10:(i+1)*10]) for i in range(len(rewards)/10) ]
# plt.plot(range(len(rewards)), rewards)
# plt.show(block=True)
env.reset()
for _ in range(10000):
    reward, done, update_info = agent.perform_step()
    env.render()
    time.sleep(0.1)
