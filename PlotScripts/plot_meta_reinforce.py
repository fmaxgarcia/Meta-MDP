import pickle
import sys 
import os 
import sys 
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
from LinearMeta import MetaPolicy

directory = sys.argv[1]
files = []

for f in os.listdir(directory):
    if f.startswith("traj"):
        iteration = int(f.split("_")[-1].split(".")[0])
        files.append( (f, iteration) )


files.sort(key=lambda x: x[1])

metas, learning_curves = [], []
for k, f in enumerate(files):
    f = f[0]
    # if k % 3 != 0: # or (k > 50 and k < 450):
    #     continue

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

    # rewards = [np.mean(rewards[i:(i+10)]) for i in range(len(rewards)-10)]

    rewards = [np.mean(rewards[i*10:(i+1)*10]) for i in range(len(rewards)/10)]
    learning_curves.append( np.array(rewards) )
    metas.append( (f, np.sum(rewards)) )

learning_curves = np.asarray( learning_curves )

# plots = [ learning_curves[0,:] ]
# plots.extend( [np.mean(learning_curves[i*3:(i+1)*3], axis=0) for i in range(learning_curves.shape[0]/3)] )

fig = plt.figure()
ax = plt.subplot(111)

for i in range(len(learning_curves)):
    ax.plot( learning_curves[i], label="Iter: %d-%d R:%d" %(i*50, (i+1)*50, np.sum(learning_curves[i])))

# ax.plot( range(0, 1000, 10), learning_curves[0], label="Iter: 0-50 R:%d" %(np.sum(learning_curves[0])), marker="*", markevery=2)
# ax.plot( range(0, 1000, 10), learning_curves[1], label="Iter: 450-500 R:%d" %(np.sum(learning_curves[1])), marker="^", markevery=2)
# plt.xticks(range(0, 1000, 10))

box = ax.get_position()
ax.legend(loc=2)
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.ylabel("Average Return")
plt.xlabel("Training Iteration")
plt.title("Progression of Learning Curves")


plt.show()

