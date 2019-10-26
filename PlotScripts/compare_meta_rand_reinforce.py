import pickle
import sys 
import os 
import sys 
import matplotlib.pyplot as plt
import numpy as np
import math

sys.path.append("../")
from LinearMeta import MetaPolicy


def get_learning_curves(files):

    learning_curves = []
    for k, f in enumerate(files):
        f = f[0]
        if k % 10 != 0:
            continue

        print(f)
        data = pickle.load(open(directory+f, "rb"))

        rewards = []
        stds = []
        keys = sorted(data.keys())
        for episode in keys:
            rewards_by_domain = data[episode]

            episode_avg_rewards = []
            for domain_samples in rewards_by_domain:
                
                sample_rewards = []
                for sample in domain_samples:
                    sample_rewards.append( np.sum(sample) )

                domain_mean_reward = np.mean(sample_rewards)

                episode_avg_rewards.append( domain_mean_reward )
                
            rewards.append( np.mean(episode_avg_rewards) )
            # stds.append( np.std(episode_rewards) )

        rewards = [np.mean(rewards[i*10:(i+1)*10]) for i in range(len(rewards)/10)]
        learning_curves.append( np.array(rewards) )

    return np.asarray( learning_curves )




directory = sys.argv[1]
files = []

for f in os.listdir(directory):
    if f.startswith("traj"):
        iteration = int(f.split("_")[-1].split(".")[0])
        files.append( (f, iteration) )


files.sort(key=lambda x: x[1])


meta_learning_curves = get_learning_curves(files[400:])


directory = sys.argv[2]
files = []

for f in os.listdir(directory):
    if f.startswith("traj"):
        iteration = int(f.split("_")[-1].split(".")[0])
        files.append( (f, iteration) )


files.sort(key=lambda x: x[1])

rand_learning_curves = get_learning_curves(files)


fig = plt.figure()
ax = plt.subplot(211)

for i in range(len(meta_learning_curves)):
    ax.plot( meta_learning_curves[i], label="Iteration: %d-%d R:%f" %(i*50, (i+1)*50, np.sum(meta_learning_curves[i])))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


ax = plt.subplot(212)

for i in range(len(rand_learning_curves)):
    ax.plot( rand_learning_curves[i], label="Iteration: %d-%d R:%f" %(i*50, (i+1)*50, np.sum(rand_learning_curves[i])))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))



plt.ylabel("Average Return")
plt.xlabel("Training Iteration")
plt.title("Progression of Learning Curves")


plt.show()

