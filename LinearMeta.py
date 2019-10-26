import numpy as np
from LearningAlgorithms.Sarsa import Sarsa
from LearningAlgorithms.REINFORCE import REINFORCE
from LearningAlgorithms.PPO import PPO
import time
import threading

class MetaPolicy:

    one_step_algos = ["SARSA", "PPO"]
    montecarlo_algos = ["REINFORCE"]

    lock = threading.Lock()


    def __init__(self, num_features, num_actions, algo="SARSA", alpha=1e-5, beta=1e-7, env=None):
        self.algo = algo
        if algo == "SARSA":
            W = np.ones( (num_actions, num_features) )
            self.learning_algorithm = Sarsa(num_actions, num_features, init_w=W, gamma=1.0, lambda_=0.9, alpha=alpha)
        elif algo == "REINFORCE":
            w = np.zeros( (num_features, 1) )
            Theta = np.zeros( (num_actions, num_features) )
            self.learning_algorithm = REINFORCE(alpha=alpha, beta=beta, gamma=1.0, num_actions=num_actions, num_features=num_features, w=w, Theta=Theta)
        elif algo == "PPO":
            self.learning_algorithm = PPO(env, alpha, beta, gamma=0.99, action_type=0, t_length=16, update_steps=128, buff_size=1000)
            self.updating = False
            self.optimizing = False


    def predict(self, state, domain_state=None):
        if self.algo == "REINFORCE":
            return self.learning_algorithm.select_action(state)
        elif self.algo == "SARSA":
            return self.learning_algorithm.select_action(state, stochastic=True)
        elif self.algo == "PPO":
            return self.learning_algorithm.select_action(state)
        
        print("Action selection not implemented")
        assert("FALSE")

    def one_step_update_single(self, state, action, reward, state_next, action_next, domain_state=None):
        assert(self.algo in self.one_step_algos)
        if self.algo == "PPO":
            self.learning_algorithm.update(state, action, reward, state_next, action_next, done=False)
        else:
            self.learning_algorithm.update(state, action, reward, state_next, action_next, etraces=True)

    def ppo_update_agent_batch(self, states, actions, rewards, next_states, next_actions, dones):

        assert(self.algo == "PPO")

        for i in range(len(states)):
            self.learning_algorithm.update(states[i], actions[i], rewards[i], next_states[i], next_actions[i], dones[i], optimize=False)


    def ppo_optimize(self):
        # print("optimizing")
        while self.optimizing == True:
            # print("Thread waiting...")
            time.sleep(0.1)
        
        self.optimizing = True

        self.learning_algorithm.optimize( verbose=False ) 
        # print("done optimizing")
        self.optimizing = False


    def one_step_update(self, sample_updates):
        assert(self.algo in self.one_step_algos)
        self.learning_algorithm.update_batch(sample_updates)
            
    def montecarlo_update(self, trajectory_samples, averages=True):
        self.learning_algorithm.update(trajectory_samples, other_w=None, other_theta=None, averages=averages)
