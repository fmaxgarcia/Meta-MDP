import pickle
import sys 
import os 
import sys 
import matplotlib.pyplot as plt
import numpy as np
import math

sys.path.append("../")
from LinearMeta import MetaPolicy

directories = []
for i in range(len(sys.argv)-1):
    directories.append( sys.argv[i+1] )


all_progress = []
for directory in directories[:-1]:
    files = []

    for f in os.listdir(directory):
        if f.startswith("traj"):
            iteration = int(f.split("_")[-1].split(".")[0])
            files.append( (f, iteration) )


    files.sort(key=lambda x: x[1])

    progress = []
    for k, f in enumerate(files):
        f = f[0]
        # if k % 5 != 0:
        #     continue

        print(f)
        data = pickle.load(open(directory+f, "rb"))

        rewards = []
        stds = []
        for episode in data.keys():
            rewards_by_domain = data[episode]
            
            episode_avg_reward = 0.0
            for domain_samples in rewards_by_domain:
                
                domain_mean_reward = 0.0
                for sample in domain_samples:
                    domain_mean_reward += sum(sample)

                domain_mean_reward /= len(domain_samples) 

                episode_avg_reward += domain_mean_reward
                
            episode_avg_reward /= len(rewards_by_domain)

            rewards.append( episode_avg_reward )
            # stds.append( np.std(episode_rewards) )
        
        # rewards = [np.mean(rewards[i*5:(i+1)*5]) for i in range(len(rewards)/10)]
        progress.append( np.sum(rewards) )

    progress = [np.mean(progress[i*5:(i+1)*5]) for i in range(len(progress)/5)]    
    all_progress.append( np.asarray(progress) )


all_progress = np.asarray( all_progress )
mean_progress = np.mean(all_progress, axis=0)
std_progress = np.std( all_progress, axis=0 ) / math.sqrt(all_progress.shape[0])

rand_dir = directories[-1]
files = []

for f in os.listdir(rand_dir):
    if f.startswith("traj"):
        iteration = int(f.split("_")[-1].split(".")[0])
        files.append( (f, iteration) )


files.sort(key=lambda x: x[1])

rand_progress = []
for k, f in enumerate(files):
    f = f[0]

    print(f)
    data = pickle.load(open(rand_dir+f, "rb"))

    rewards = []
    stds = []
    for episode in data.keys():
        rewards_by_domain = data[episode]
        
        episode_avg_reward = 0.0
        for domain_samples in rewards_by_domain:
            
            domain_mean_reward = 0.0
            for sample in domain_samples:
                domain_mean_reward += sum(sample)

            domain_mean_reward /= len(domain_samples) 

            episode_avg_reward += domain_mean_reward
            
        episode_avg_reward /= len(rewards_by_domain)

        rewards.append( episode_avg_reward )
        # stds.append( np.std(episode_rewards) )
    
    # rewards = [np.mean(rewards[i*10:(i+1)*10]) for i in range(len(rewards)/10)]
    rand_progress.append( np.sum(rewards) )

print("================")



fig = plt.figure()
ax = fig.add_subplot(111)

mean_rand_progress = np.array([np.mean(rand_progress) for i in range(len(rand_progress))])
diff_rand_progress = np.array([np.std(rand_progress) / math.sqrt(len(rand_progress)) for i in range(len(rand_progress))])

x_axis = range(0,500,5)
ax.plot( x_axis, mean_rand_progress, label="Random Exploration", color="red")
ax.fill_between(x_axis, mean_rand_progress+diff_rand_progress,
                            mean_rand_progress-diff_rand_progress, alpha=0.2, interpolate=True, color="red")

ax.plot( x_axis, mean_progress, label="Advisor Policy", color="blue")
ax.fill_between(x_axis, mean_progress+std_progress,
                            mean_progress-std_progress, alpha=0.2, interpolate=True, color="blue")

plt.title("Exploration training progress")
plt.ylabel("Sum of Returns")
plt.xlabel("Training Iteration")
plt.legend(loc=2)

plt.show()

