import gym 
import roboschool 
import sys 
sys.path.append("../")
from linear_agent import LinearAgent
from MamlAgent import MamlAgent
from Environments import EnvWrapper
from CustomEnvironments.AnimatEnv import AnimatEnv 

from optparse import OptionParser
import time

from PIL import Image

if __name__ == "__main__":
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-e", "--environment", action="store", help="environment name", type="string", default="animat")
    parser.add_option("-m", "--model", action="store", help="env model", type="string")
    parser.add_option("-a", "--actor", action="store", help="agent actor", type="string")
    parser.add_option("-c", "--critic", action="store", help="agent critic", type="string")
    parser.add_option("-s", "--save_path", action="store", help="path for saving vid", type="string", default=None)

    (options, args) = parser.parse_args()

    env = options.environment

    if env != "animat":
        env = gym.make( env )
        env.env.model_xml = options.model
    else:
        gym_env = AnimatEnv(options.model)
        env = EnvWrapper(gym_env, basis_order=3, normalization=1)

    agent = LinearAgent(env, meta_policy=None, algo="PPO")
    agent.random_action_prob = 0.0

    agent.learning_algorithm.load_model(options.actor, options.critic)
    agent.play( max_steps=10000, delay=0.01, save_path=options.save_path)

    time.sleep(0.5)