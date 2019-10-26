import numpy as np 
import random
import math 
from sys import stdout

from LinearMeta import MetaPolicy
from LearningAlgorithms.Sarsa import Sarsa
from LearningAlgorithms.QLearning import QLearning
from LearningAlgorithms.REINFORCE import REINFORCE
from LearningAlgorithms.PPO import PPO

from PIL import Image

import time

class LinearAgent:
    RANDOM_ACTION_DECAY = 0.99
    MIN_RANDOM_ACTION_PROB = 0.0
    FREEZE_POLICY = False
    SAMPLES_PER_EPISODES = 1

    random_action_prob = 0.8
    one_step_algos = ["SARSA", "QLEARNING", "PPO"]
    montecarlo_algos = ["REINFORCE"]
   

    def __init__(self, env, meta_policy=None, alpha=0.001, beta=0.01, algo="SARSA", prevent_actions=None):
        self.algo = algo
        self.prevent_actions = prevent_actions
        self.update_type = "ONE_STEP" if algo in self.one_step_algos else "MC"
        self.reset_variables(env=env, alpha=alpha, beta=beta, meta_policy=meta_policy)
      

    def perform_step(self):

        if self.action is None or self.state is None:
            self.state = self.env.reset()
            self.action, self.explore = self.select_action(self.state)

        if self.learning_algorithm.action_type == 1: #Continuous
            if self.learning_algorithm.distribution == "beta":
                a = (self.action * (self.env.action_space.high - self.env.action_space.low) + self.env.action_space.low)
            else:
                a = np.clip(self.action, np.zeros(self.action.shape), np.ones(self.action.shape))
                a = (a * (self.env.action_space.high - self.env.action_space.low) + self.env.action_space.low)
        else:
            a = self.action

        obs, reward, done, _ = self.env.step( a )
        state_next = obs

        action_next, explore_next = self.select_action(state_next)

        assert(self.state is not None)
        assert(self.action is not None)
        assert(state_next is not None)

        update_info = {"state" : self.state.copy(), "action" : (self.action, self.explore), "reward" : reward,
                        "state_next" : state_next.copy(), "action_next" : (action_next, explore_next), "done" : done} 
        
        self.state = state_next
        self.action = action_next
        self.explore = explore_next
        
        return reward, done, update_info


    def update_policy(self, update_info, update_meta):
        s = update_info['state']
        a, e = update_info['action']
        r = update_info['reward']
        sn = update_info['state_next']
        an, e = update_info['action_next']
        done = update_info["done"]

        if self.FREEZE_POLICY == False and self.update_type == "ONE_STEP":
            if self.algo == "SARSA":
                self.learning_algorithm.update(s, a, r, sn, an, etraces=True)
            elif self.algo == "PPO":
                self.learning_algorithm.update(s, a, r, sn, an, done)
    
        if update_meta is True and self.meta_policy.algo in MetaPolicy.one_step_algos:
            self.meta_policy.one_step_update(s, a, r, sn, an)
    

    def update_montecarlo(self, trajectories):
        self.learning_algorithm.update( trajectories )


    def train(self, num_episodes=500, max_steps=1000, verbose=False, update_meta=False, render=False, save_path=None):

        total_steps = 0
        step_counts = []

        all_rewards = []

        for episode in range(num_episodes):     
            sampled_trajectories = []
            episode_rewards = []
            self.update_random_action_prob()
            for _ in range(self.SAMPLES_PER_EPISODES):
                trajectory = []        
                self.state = self.env.reset()

                steps = 0
                self.action, self.explore = self.select_action(self.state)
            
                for step in range(max_steps):
                    # stdout.write("\rStep: %d - Episode: %d " %(step, episode))
                    # stdout.flush()
                    if render == True:
                        self.env.render()

                    reward, done, update_info = self.perform_step()
                    self.update_policy(update_info, update_meta)

                    
                    steps += 1
                    total_steps += 1
                    episode_rewards.append(reward)
                    trajectory.append( update_info )
                    if done:
                        break 

                sampled_trajectories.append( trajectory )

            if self.update_type == "MC":
                self.update_montecarlo( sampled_trajectories )

            if self.algo == "PPO":
                self.learning_algorithm.reset()

            if save_path is not None:
                self.learning_algorithm.save_model(save_path, episode)

            all_rewards.append( sum(episode_rewards) / self.SAMPLES_PER_EPISODES )
            # mean_rewards = np.mean(all_rewards[-100:])
            mean_rewards = all_rewards[-1]
            if verbose:
                print("Training episode = {}, Total steps = {}, Last-100 mean reward = {}, Basis = {}, Alpha = {}"
                                                         .format(episode, total_steps, mean_rewards, self.env.order, self.learning_algorithm.alpha))


        return all_rewards

    
    def select_action(self, state):
        explore = False
        if random.random() < self.random_action_prob:
            if self.meta_policy is None:
                # action = self.env.random_action()
                action = self.env.action_space.sample()
            else:
                action = self.meta_policy.predict(state)

                #Hacky way of running some tests for animat domain. Remove this when done.
                # ################################
                # if self.prevent_actions is not None:
                    
                #     while action in self.prevent_actions:
                #         action = (action+1) % self.learning_algorithm.Theta.shape[0]
                ################################

            explore = True
        else:
            action = self.learning_algorithm.select_action(state)
            
        return action, explore

    def update_random_action_prob(self):
        self.random_action_prob *= self.RANDOM_ACTION_DECAY
        if self.random_action_prob < self.MIN_RANDOM_ACTION_PROB:
            self.random_action_prob = self.MIN_RANDOM_ACTION_PROB
        
    def reset(self):
        self.state = self.env.reset()
        self.action = None

    def reset_variables(self, env, alpha, beta=0.1, meta_policy=None):
        self.env = env
        self.env.reset()
        self.random_action_prob = 0.8
        self.meta_policy = meta_policy
        self.action = None

        if self.algo == "REINFORCE":
            self.learning_algorithm = REINFORCE(alpha=alpha, beta=beta, gamma=0.99, 
                                    num_actions=self.env.num_actions, num_features=self.env.num_features)
        elif self.algo == "SARSA":
            self.learning_algorithm = Sarsa(self.env.num_actions, self.env.num_features, alpha=alpha, gamma=0.9)
        elif self.algo == "PPO":
            self.learning_algorithm = PPO(self.env, alpha=alpha, beta=beta, gamma=0.99, action_type=0, update_steps=64, t_length=16)
        else:
            assert("Unrecognized algo")
    

    def play(self, max_steps=200, delay=0.1, save_path=None):
        self.state = self.env.reset()
        done = False
        steps = 0
        self.action, self.explore = self.select_action(self.state)
        while not done and steps < max_steps:
            if save_path is not None:
                img = self.env.render("rgb_array")
                im = Image.fromarray(img)
                if steps < 10:
                    im_name = "im_000%d.png" %(steps)
                elif steps < 100:
                    im_name = "im_00%d.png" %(steps)
                elif steps < 1000:
                    im_name = "im_0%d.png" %(steps)
                else:
                    im_name = "im_%d.png" %(steps)

                im.save(save_path+im_name)
            else:
                # self.env.render("human")
                self.env.render()
                if steps == 0:
                    time.sleep( 10.0 )
                    
            time.sleep( delay )
            action = self.select_action(self.state)
            reward, done, update_info = self.perform_step()

            steps += 1
        return steps
