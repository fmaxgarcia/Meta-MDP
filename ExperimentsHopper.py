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

import sys
import copy

import gym 
import roboschool

class ExperimentsHopper:


    @staticmethod
    def thread_train(d, k, agents, steps, traj_by_domain, max_r, optimize_meta=True):

        agent = agents[d][k]
        trajectory = []
        states, next_states, rewards, actions, next_actions, dones = [], [], [], [], [], []
        for step in range(steps):
            reward, done, update_info = agent.perform_step()
            if agent.algo == "PPO":
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
                    agent.meta_policy.ppo_update_agent_batch(states, actions, rewards, next_states, next_actions, dones)
                    states, next_states, rewards, actions, next_actions, dones = [], [], [], [], [], []
                    agent.meta_policy.lock.release()

                if optimize_meta:
                    if d==0 and k==0 and step != 0 and step % agent.meta_policy.learning_algorithm.update_steps == 0:
                        agent.meta_policy.ppo_optimize()


            trajectory.append( update_info['reward'] )
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
    def _run_episode(domain_agents, num_steps=100, r_maxs=None, optimize_meta=True):


        traj_by_domain = {}
        for i in range(len(domain_agents)):
            traj_by_domain[i] = [None] * len(domain_agents[i])

        threads = []
        for d, agent_group in enumerate(domain_agents):
            for k, agent in enumerate(agent_group):
                agent.reset()
                max_r = r_maxs[d] if r_maxs is not None else 1.0
                thread = Thread(target=ExperimentsHopper.thread_train, args=(d, k, domain_agents, num_steps, traj_by_domain, max_r, optimize_meta))
                threads.append( thread ) 

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        for i in range(len(traj_by_domain[0])):
            string = "Sample: " + str(i)
            for k in traj_by_domain.keys():
                string += " A" + str(k) + ": " + str(sum(traj_by_domain[k][i]))
            print(string)

        return traj_by_domain

    @staticmethod
    def experiment_train_meta(save_directory, meta_alpha, meta_beta, xml_models): 

        alpha = 1e-4
        beta = 1e-3  

        env = gym.make("RoboschoolHopper-v1")
        env.reset()

        (obs, reward, done, info) = env.step(env.action_space.sample())
        
        num_features = obs.shape[0]
        num_actions = env.action_space.low.shape[0]
        
        
        meta = MetaPolicy(num_features=num_features, num_actions=num_actions, algo="PPO", alpha=meta_alpha, beta=meta_beta, env=env)
        meta.learning_algorithm.t_length = 128
        meta.learning_algorithm.update_steps = 256

        if os.path.isdir(save_directory) == False:
            os.mkdir(save_directory)

        meta_train_episodes = 30
        num_samples = 4
        for k in range(meta_train_episodes):
            agents = []

            print("Loading environments...")
            trial_by_domain = {}
            for i, model in enumerate(xml_models):
                
                domain_agents = []
                for _ in range(num_samples):
                    env = gym.make("RoboschoolHopper-v1")
                    env.env.model_xml = model
                    
                    agent = LinearAgent(env, meta_policy=meta, algo="PPO", alpha=alpha, beta=beta)

                    domain_agents.append( agent )
                
                    agent.learning_algorithm.t_length = 128
                    agent.learning_algorithm.update_steps = 128
                    agent.learning_algorithm.epochs = 8
                    

                agents.append( domain_agents )

                trial_by_domain[i] = [ list() for _ in range(num_samples) ]
            print("Done loading...")

            episodes = 1000
            steps = 500
            domain_rewards_by_episode = {}
            for ep in range(episodes):
                
                trajectories_by_domain = ExperimentsHopper._run_episode(domain_agents=agents, num_steps=steps, r_maxs=None)
                
                domain_samples = []
                for i in trajectories_by_domain.keys():
                    sample_returns = []
                    for j in range(len(trajectories_by_domain[i])):
                        
                        sample_returns.append( sum(trajectories_by_domain[i][j]) )

                    domain_samples.append( sample_returns )

                print("Episode %d - Trial %d" %(ep, k))
                domain_rewards_by_episode[ep] = domain_samples

                if ep % 100 == 0:
                    val = (k*episodes)+ep
                    meta.learning_algorithm.save_model(save_directory+"/", val)

                    # pickle.dump(meta, open(save_directory+"/meta_iter_"+str(k)+".pkl", "wb"))
                    pickle.dump(domain_rewards_by_episode, open(save_directory+"/trajectory_iter_"+str(val)+".pkl", "wb"))


   
    @staticmethod
    def experiment_meta_vs_random(meta_actor, meta_critic, save_directory, xml_models, episodes, steps):    


        alpha = 1e-4
        beta = 1e-3  

        env = gym.make("RoboschoolHopper-v1")
        env.reset()

        (obs, reward, done, info) = env.step(env.action_space.sample())

        num_features = obs.shape[0]
        num_actions = env.action_space.low.shape[0]

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
            for model in xml_models:
                domain_agents = []

                for _ in range(num_samples):
                    env = gym.make("RoboschoolHopper-v1")
                    env.env.model_xml = model

                    agent = LinearAgent(env, meta_policy=meta, algo="PPO", alpha=alpha, beta=beta)
                    agent.learning_algorithm.t_length = 128
                    agent.learning_algorithm.update_steps = 128
                    agent.learning_algorithm.epochs = 8

                    if meta is None:
                        agent.random_action_prob = 0.0

                    domain_agents.append( agent )

                agents.append( domain_agents )

            domain_rewards_by_episode = {}
            for ep in range(episodes):
                null_actions = {}
                trajectories_by_domain = ExperimentsHopper._run_episode(domain_agents=agents, num_steps=steps, optimize_meta=False)
                
                domain_samples = []
                for i in trajectories_by_domain.keys():
                    sample_returns = []
                    for j in range(len(trajectories_by_domain[i])):
                        
                        sample_returns.append( sum(trajectories_by_domain[i][j]) )

                    domain_samples.append( sample_returns )

                print("Episode %d / %d" %(ep, episodes))
                domain_rewards_by_episode[ep] = domain_samples

                if ep % 100 == 0:
                    filename = "meta_test_"+str(ep)+".pkl" if k == 0 else "no_meta_test_"+str(ep)+".pkl"
                    pickle.dump(domain_rewards_by_episode, open(save_directory+"/"+filename, "wb"))
                    for ai, a in enumerate(agents):
                        type_ = "meta_" if k == 0 else "no_meta_"
                        type_ += str(ai) + "_"
                        a[0].learning_algorithm.save_model(save_directory+"/"+type_, ep)
