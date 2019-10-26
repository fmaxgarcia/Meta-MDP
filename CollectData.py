

from Features import fourier_basis
from Environments import EnvWrapper
from CustomEnvironments.AnimatEnv import AnimatEnv 
import gym
from linear_agent import LinearAgent
import time
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import os
import pickle


def init_env_animat(setup):
    env = AnimatEnv(setup)
    env.reset()

    (obs, reward, done, info) = env.step(env.action_space.sample())
    obs = EnvWrapper.normalize_range(obs, env.env_range)
    phi = fourier_basis(obs, order=basis_order)

    num_features = phi.shape[0]
    num_actions = env.action_space.n

    gym_env = AnimatEnv(setup)

    env = EnvWrapper(gym_env, basis_order=basis_order, normalization=1)
    return env


def init_env_cartpole(setup):
    gym_env = gym.make('CartPole-v0')
    gym_env.env.force_mag = setup["force"]
    gym_env.env.length = setup["pole_length"]
    gym_env.env.masscart = setup["masscart"]
    gym_env.env.masspole = setup["masspole"]

    env = EnvWrapper(gym_env, basis_order=basis_order, normalization=0)
    return env


if __name__ == "__main__":
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-e", "--environment", action="store", help="environment name", type="string", default="animat")
    parser.add_option("-a", "--alpha", action="store", help="learning rate", type="float", default=0.01)
    parser.add_option("-b", "--basis", action="store", help="Number of basis", type="int", default=3)
    parser.add_option("-d", "--save_directory", action="store", help="Save directory", type="string", default="./AnimatData/")
    parser.add_option("--episodes", action="store", help="Num episodes", type="int", default=1000)
    parser.add_option("--steps", action="store", help="Num steps", type="int", default=1000)
    parser.add_option("--algorithm", action="store", help="agent learning algorithm", type="string", default="SARSA")

    (options, args) = parser.parse_args()
    
    save_dir = options.save_directory
    basis_order = options.basis
    alpha = options.alpha
    env_name = options.environment
    episodes = options.episodes 
    steps = options.steps
    algo = options.algorithm

    if env_name.lower() == "animat":
        setups = ["./CustomEnvironments/maze1.txt", "./CustomEnvironments/maze2.txt", 
                  "./CustomEnvironments/maze3.txt", "./CustomEnvironments/maze4.txt" 
                  ]

    elif env_name.lower() == "cartpole":
        setups = [{"force" : 10.0, "pole_length" : 0.5, "masscart" : 1.0, "masspole" : 0.1},            
                    {"force" : 5.0, "pole_length" : 0.25, "masscart" : 2.0, "masspole" : 0.2},
                    {"force" : 20.0, "pole_length" : 1.2, "masscart" : 5.0, "masspole" : 0.1},
                    {"force" : 5.0, "pole_length" : 1.0, "masscart" : 2.0, "masspole" : 0.5},
                    {"force" : 10.0, "pole_length" : 0.5, "masscart" : 5.0, "masspole" : 1.0}
                    ]
    else:
        print("Unrecognized environment: " + env_name)
        assert(False)
        
    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)

    data = []
    for setup in setups:
        if env_name.lower() == "animat":
            env = init_env_animat(setup)
        elif env_name.lower() == "cartpole":
            env = init_env_cartpole(setup)


        agent = LinearAgent(env, meta_policy=None, alpha=alpha, algo=algo)

        rewards = agent.train(num_episodes=episodes, max_steps=steps, verbose=True, update_meta=False, render=False)

        setup_data = {'setup' : setup, 'max_r' : max(rewards), 'episodes' : episodes, 
                        'max_steps' : steps, 'alpha' : alpha, 'order' : basis_order, 'algo' : algo}

        data.append( setup_data )
        print(setup_data)
        time.sleep(2.0)


    pickle.dump(data, open(save_dir+"/"+env_name+"_data.pkl", "wb"))
        
