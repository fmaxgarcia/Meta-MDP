import gym
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import random
from threading import Thread
from linear_agent import LinearAgent
import os
import pickle
from LinearMeta import MetaPolicy
from Features import fourier_basis
from Environments import EnvWrapper
import sys
import copy 


class ExperimentsCartpole:

    RECORDED_DATA = None

    @staticmethod
    def thread_train(d, k, agents, steps, traj_by_domain, max_r, optimize_meta):

        agent = agents[d][k]
        #TODO testing just updating exploration
        # agent.random_action_prob = 1.0
        trajectory = []
        states, next_states, rewards, actions, next_actions, dones = [], [], [], [], [], []
        for step in range(steps):
            reward, done, update_info = agent.perform_step()
            if agent.algo == "SARSA" or agent.algo == "PPO":
                agent.update_policy(update_info=update_info, update_meta=False)
            
            if agent.meta_policy is not None and agent.meta_policy.algo == "PPO":
                states.append( update_info['state'] )
                next_states.append( update_info['state_next'] )
                actions.append( update_info['action'][0] )
                rewards.append( update_info['reward'] / max_r ) 
                next_actions.append( update_info['action_next'] )
                dones.append( update_info['done'] )

                if len(states) == agent.meta_policy.learning_algorithm.t_length:
                    # print("Updating meta...")
                    agent.meta_policy.lock.acquire()
                    # print("Lock D:%d K:%d" % (d, k))
                    agent.meta_policy.ppo_update_agent_batch(states, actions, rewards, next_states, next_actions, dones)
                    states, next_states, rewards, actions, next_actions, dones = [], [], [], [], [], []
                    # print("Release D:%d K:%d" %(d,k) )
                    agent.meta_policy.lock.release()

                if optimize_meta:
                    if d==0 and k==0 and step != 0 and step % agent.meta_policy.learning_algorithm.update_steps == 0:
                        agent.meta_policy.ppo_optimize()


            trajectory.append( update_info )
            if done == True:
                if agent.meta_policy is not None and agent.meta_policy.algo == "PPO":
                    agent.meta_policy.lock.acquire()
                    agent.meta_policy.ppo_update_agent_batch(states, actions, rewards, next_states, next_actions, dones)
                    agent.meta_policy.lock.release()
                    if len(states) > 0 and d == 0 and k == 0:
                        # print("Updating meta...")
                        if optimize_meta:
                            agent.meta_policy.ppo_optimize()
                break

        agent.update_random_action_prob()
        traj_by_domain[d][k] = trajectory
        

    @staticmethod
    def _run_episode(domain_agents, num_steps=1000, r_maxs=None, optimize_meta=True):
        
        traj_by_domain = {}
        for i in range(len(domain_agents)):
            traj_by_domain[i] = [list()] * len(domain_agents[i])


        threads = []
        for d, agent_group in enumerate(domain_agents):
            for k, agent in enumerate(agent_group):
                agent.reset()
                max_r = r_maxs[d] if r_maxs is not None else 1.0
                thread = Thread(target=ExperimentsCartpole.thread_train, args=(d, k, domain_agents, num_steps, traj_by_domain, max_r, optimize_meta))
                threads.append( thread )

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        while len(threads) > 0:
            del threads[0]

        for i in range(len(traj_by_domain[0])):
            string = "Sample: " + str(i)
            for k in traj_by_domain.keys():
                string += " A" + str(k) + ": " + str(len(traj_by_domain[k][i]))
            print(string)

        for d, agent_group in enumerate(domain_agents):
            for k, agent in enumerate(agent_group):
                if agent.algo == "REINFORCE":
                    agent.learning_algorithm.update( [traj_by_domain[d][k]] )

        return traj_by_domain

    @staticmethod
    def _run_episode_single_thread(domain_agents, num_steps=100):


        traj_by_domain = {}
        for i in range(len(domain_agents)):
            traj_by_domain[i] = [None] * len(domain_agents[i])

        for d, agent_group in enumerate(domain_agents):
            for k, agent in enumerate(agent_group):
                agent.env.reset()
                ExperimentsCartpole.thread_train(d, k, domain_agents, num_steps, traj_by_domain)

        for i in range(len(traj_by_domain[0])):
            string = "Sample: " + str(i)
            for k in traj_by_domain.keys():
                string += " A" + str(k) + ": " + str(len(traj_by_domain[k][i]))
            print(string)

        return traj_by_domain

    @staticmethod
    def experiment_train_meta(save_directory, meta_alpha, meta_beta):    

        gym_env = gym.make('CartPole-v0')
        gym_env.reset()
        basis_order = ExperimentsCartpole.RECORDED_DATA[0]['order']

        (obs, reward, done, info) = gym_env.step(gym_env.env.action_space.sample())
        obs = EnvWrapper.modified_sigmoid(obs)
        phi = fourier_basis(obs, order=basis_order)

        num_features = phi.shape[0]
        num_actions = gym_env.env.action_space.n

        env = EnvWrapper(gym_env, basis_order=basis_order, normalization=0)

        meta = MetaPolicy(num_features=num_features, num_actions=num_actions, algo="PPO", alpha=meta_alpha, beta=meta_beta, env=env)

        if os.path.isdir(save_directory) == False:
            os.mkdir(save_directory)

        meta_train_episodes = 500
        num_samples = 3
        for k in range(meta_train_episodes):
            agents = []

            print("Loading environments...")
            r_maxs = []
            trial_by_domain = {}
            for i, d in enumerate(ExperimentsCartpole.RECORDED_DATA):
                print("Setup: " + str(d['setup']))
                setup = d['setup']
                max_r = d['max_r']
                episodes = d['episodes']
                steps = d['max_steps']
                basis_order = d['order']
                # alpha = d['alpha']	        
                alpha = 0.0001
                
                domain_agents = []
                for _ in range(num_samples):
                    gym_env = gym.make('CartPole-v0')
                    gym_env.env.force_mag = setup["force"]
                    gym_env.env.length = setup["pole_length"]
                    gym_env.env.masscart = setup["masscart"]
                    gym_env.env.masspole = setup["masspole"]

                    env = EnvWrapper(gym_env, basis_order=basis_order, normalization=0)
                    agent = LinearAgent(env, meta_policy=meta, alpha=alpha, algo="PPO")

                    domain_agents.append( agent )

                agents.append( domain_agents )
                r_maxs.append( max_r )

                trial_by_domain[i] = [ list() for _ in range(num_samples) ]
            print("Done loading...")

            
            domain_rewards_by_episode = {}
            for ep in range(episodes):
                
                trajectories_by_domain = ExperimentsCartpole._run_episode(domain_agents=agents, num_steps=steps, r_maxs=r_maxs)
                
                domain_samples = []
                for i in trajectories_by_domain.keys():
                    sample_rewards = []
                    for j in range(len(trajectories_by_domain[i])):
                        t_rewards = []
                        for t in trajectories_by_domain[i][j]:
                            t_rewards.append( t['reward'] )
                            t['reward'] = t['reward'] / r_maxs[i] 
                            trial_by_domain[i][j].append( t )
                    
                        sample_rewards.append( t_rewards )
                    domain_samples.append( sample_rewards )

                print("Episode %d - Trial %d" %(ep, k))
                domain_rewards_by_episode[ep] = domain_samples

            trajectories = []
            for key in trial_by_domain.keys():
                for traj in trial_by_domain[key]:
                    trajectories.append( traj )

            if meta.algo == "REINFORCE":
                print("Updating meta....")
                meta.montecarlo_update(trajectories)

            meta.learning_algorithm.save_model(save_directory+"/", k)
            # pickle.dump(meta, open(save_directory+"/meta_iter_"+str(k)+".pkl", "wb"))
            pickle.dump(domain_rewards_by_episode, open(save_directory+"/trajectory_iter_"+str(k)+".pkl", "wb"))



    @staticmethod
    def experiment_random_baseline(save_directory):    

        env = gym.make('CartPole-v0')
        env.reset()
        basis_order = ExperimentsCartpole.RECORDED_DATA[0]['order']

        (obs, reward, done, info) = env.step(env.action_space.sample())
        obs = EnvWrapper.modified_sigmoid(obs)
        phi = fourier_basis(obs, order=basis_order)

        num_features = phi.shape[0]
        num_actions = env.action_space.n

        
        if os.path.isdir(save_directory) == False:
            os.mkdir(save_directory)

        meta_train_episodes = 100
        num_samples = 5
        for k in range(meta_train_episodes):
            agents = []

            print("Loading environments...")
            r_maxs = []
            trial_by_domain = {}
            for i, d in enumerate(ExperimentsCartpole.RECORDED_DATA):
                print("Setup: " + str(d['setup']))
                setup = d['setup']
                max_r = d['max_r']
                episodes = d['episodes']
                steps = d['max_steps']
                basis_order = d['order']
                # alpha = d['alpha']	        
                alpha = 0.0001
                
                domain_agents = []
                for _ in range(num_samples):
                    gym_env = gym.make('CartPole-v0')
                    gym_env.env.force_mag = setup["force"]
                    gym_env.env.length = setup["pole_length"]
                    gym_env.env.masscart = setup["masscart"]
                    gym_env.env.masspole = setup["masspole"]

                    env = EnvWrapper(gym_env, basis_order=basis_order, normalization=0)
                    agent = LinearAgent(env, meta_policy=None, alpha=alpha, algo="SARSA")

                    domain_agents.append( agent )

                agents.append( domain_agents )
                r_maxs.append( max_r )

                trial_by_domain[i] = [ list() for _ in range(num_samples) ]
            print("Done loading...")

            
            domain_rewards_by_episode = {}
            for ep in range(episodes):
                
                trajectories_by_domain = ExperimentsCartpole._run_episode(domain_agents=agents, num_steps=steps)
                
                domain_samples = []
                for i in trajectories_by_domain.keys():
                    sample_rewards = []
                    for j in range(len(trajectories_by_domain[i])):
                        t_rewards = []
                        for t in trajectories_by_domain[i][j]:
                            t_rewards.append( t['reward'] )
                    
                        sample_rewards.append( t_rewards )
                    domain_samples.append( sample_rewards )

                print("Episode %d - Trial %d" %(ep, k))
                domain_rewards_by_episode[ep] = domain_samples

            pickle.dump(domain_rewards_by_episode, open(save_directory+"/trajectory_iter_"+str(k)+".pkl", "wb"))



    @staticmethod
    def experiment_meta_vs_random(meta_actor, meta_critic, save_directory, setups, episodes, steps):    

        alpha = 0.001
        basis_order = 3

        env = gym.make('CartPole-v0')
        env.reset()

        (obs, reward, done, info) = env.step(env.action_space.sample())
        obs = EnvWrapper.modified_sigmoid(obs)
        phi = fourier_basis(obs, order=basis_order)

        num_features = phi.shape[0]
        num_actions = env.action_space.n

        if os.path.isdir(save_directory) == False:
            os.mkdir(save_directory)

        num_samples = 5

        for k in range(2):
            if k == 0:
                meta = MetaPolicy(num_features=num_features, num_actions=num_actions, algo="PPO", alpha=1e-3, beta=1e-3, env=env)
                meta.learning_algorithm.load_model(meta_actor, meta_critic)
            else:
                meta = None

            agents = []
            for setup in setups:
                domain_agents = []
                for _ in range(num_samples):
                    gym_env = gym.make('CartPole-v0')
                    gym_env.env.force_mag = setup["force"]
                    gym_env.env.length = setup["pole_length"]
                    gym_env.env.masscart = setup["masscart"]
                    gym_env.env.masspole = setup["masspole"]

                    env = EnvWrapper(gym_env, basis_order=basis_order, normalization=0)
                    agent = LinearAgent(env, meta_policy=meta, alpha=alpha, algo="PPO")

                    if meta is None:
                        agent.random_action_prob = 0.0

                    domain_agents.append( agent )

                agents.append( domain_agents )

            domain_rewards_by_episode = {}
            for ep in range(episodes):
                
                trajectories_by_domain = ExperimentsCartpole._run_episode(domain_agents=agents, num_steps=steps, optimize_meta=False)
                
                domain_samples = []
                for i in trajectories_by_domain.keys():
                    sample_rewards = []
                    for j in range(len(trajectories_by_domain[i])):
                        t_rewards = []
                        for t in trajectories_by_domain[i][j]:
                            t_rewards.append( t['reward'] )
                    
                        sample_rewards.append( t_rewards )
                    domain_samples.append( sample_rewards )

                print("Episode %d" %(ep))
                domain_rewards_by_episode[ep] = domain_samples

                if ep % 100 == 0:
                    filename = "meta_test"+str(ep)+".pkl" if k == 0 else "no_meta_test_"+str(ep)+".pkl"
                    pickle.dump(domain_rewards_by_episode, open(save_directory+"/"+filename, "wb"))
                


    @staticmethod
    def experiment_explore_vs_exploit(meta_path, save_directory, setups, episodes, steps):    

        alpha = 0.0001
        basis_order = 3

        env = gym.make('CartPole-v0')
        env.reset()

        (obs, reward, done, info) = env.step(env.action_space.sample())
        obs = EnvWrapper.modified_sigmoid(obs)
        phi = fourier_basis(obs, order=basis_order)

        num_features = phi.shape[0]
        num_actions = env.action_space.n

        if os.path.isdir(save_directory) == False:
            os.mkdir(save_directory)

        num_samples = 5

        meta = pickle.load(open(meta_path, "rb")) 

        agents = []
        for setup in setups:
            gym_env = gym.make('CartPole-v0')
            gym_env.env.force_mag = setup["force"]
            gym_env.env.length = setup["pole_length"]
            gym_env.env.masscart = setup["masscart"]
            gym_env.env.masspole = setup["masspole"]
            
            env = EnvWrapper(gym_env, basis_order=basis_order, normalization=0)
            agent = LinearAgent(env, meta_policy=meta, alpha=alpha, algo="SARSA")
            

            agents.append( agent )

        policies = []
        for agent in agents:
            rewards = agent.train(num_episodes=episodes, max_steps=steps, verbose=True, update_meta=False, render=False)
            policies.append( copy.deepcopy(agent.learning_algorithm) )


        rewards = []        
        for i, agent in enumerate(agents):
            agent.learning_algorithm = policies[i]
            agent.random_action_prob = 0.0
            agent.RANDOM_ACTION_DECAY = 1.0
            exploit_rewards = agent.train(num_episodes=episodes, max_steps=steps, verbose=True, update_meta=False, render=False)

            agent.random_action_prob = 1.0
            explore_rewards = agent.train(num_episodes=episodes, max_steps=steps, verbose=True, update_meta=False, render=False)


            rewards.append( {"explore" : explore_rewards, "exploit" : exploit_rewards} )

        pickle.dump(rewards, open(save_directory+"/explore_exploit.pkl", "wb"))

