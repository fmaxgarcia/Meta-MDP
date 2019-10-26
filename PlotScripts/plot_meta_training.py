import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
from matplotlib import markers, colors

sys.path.append("../")
from LinearMeta import MetaPolicy

marks = markers.MarkerStyle.markers.keys()
cs = colors.cnames.keys()

# training_directory = "/nfs/nemo/u3/fmgarcia/ExplorationSanityCheck/MetaTraining"
# save_directory = "/nfs/nemo/u3/fmgarcia/ExplorationSanityCheck/MetaTraining"
training_directory = sys.argv[1]
# save_directory = sys.argv[2]

files = os.listdir(training_directory)
files.sort()

fig = plt.figure()
ax = fig.add_subplot(111)
count = 0
for k, f in enumerate(files):
    if f.endswith(".pkl"):
        print(training_directory)
        print(f)
        data = pickle.load(open(training_directory+"/"+f, "rb"))
        steps_per_episode = data["training_by_episode"]
        # steps_per_episode = pickle.load(open(training_directory+"/"+f, "rb"))
        rewards = []
        for episode in steps_per_episode.keys():
            episode_rewards = np.zeros( (len(steps_per_episode[episode][0]),) )

            for step_reward in steps_per_episode[episode]:
                episode_rewards += np.asarray(step_reward)

            mean_reward = np.mean( episode_rewards )
            rewards.append(mean_reward)

        # rewards = np.asarray(rewards)

        w_size = 10
        stds = np.asarray([np.std(rewards[i*w_size:(i+1)*w_size]) for i in range(len(rewards) / w_size)])
        rewards = np.asarray([np.mean(rewards[i*w_size:(i+1)*w_size]) for i in range(len(rewards) / w_size)])

        ax.plot(range(rewards.shape[0]), rewards, color=cs[count], label="Iteration " + str(count))
        count += 1
        ax.fill_between(range(rewards.shape[0]), rewards+stds,
                            rewards-stds, facecolor=cs[k], edgecolor=cs[k], alpha=0.2, interpolate=True)


plt.legend(loc=2)
plt.show()
# fig.savefig(save_directory+"/meta_training.png")
# plt.close('all')





