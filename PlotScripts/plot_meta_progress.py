import pickle
import sys 
import os 
import sys 
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
from LinearMeta import MetaPolicy

directories = []
for i in range(len(sys.argv)-1):
    directories.append( sys.argv[i+1] )


all_progress = []
for directory in directories:
    files = []

    for f in os.listdir(directory):
        if f.startswith("traj"):
            iteration = int(f.split("_")[-1].split(".")[0])
            files.append( (f, iteration) )


    files.sort(key=lambda x: x[1])

    progress = []
    for k, f in enumerate(files):
        f = f[0]
        # if k > 50:
        #     break

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
        
        # rewards = [np.mean(rewards[i*10:(i+1)*10]) for i in range(len(rewards)/10)]
        progress.append( np.sum(rewards) )

    # progress = [np.mean(progress[i*5:(i+1)*5]) for i in range(len(progress)/5)]    
    progress = [np.mean(progress[i:(i+5)]) for i in range(len(progress)-5)]    
    all_progress.append( np.asarray(progress) )
    print(progress)
    print("================")

all_progress = np.asarray( all_progress )
mean_progress = np.mean(all_progress, axis=0)
std_progress = np.std( all_progress, axis=0 )

fig = plt.figure()
ax = fig.add_subplot(111)

print("Plotting")
ax.plot( range(mean_progress.shape[0]), mean_progress, label="Meta Learning")
ax.fill_between(range(mean_progress.shape[0]), mean_progress+std_progress,
                            mean_progress-std_progress, alpha=0.2, interpolate=True)
plt.title("Exploration training progress")
plt.legend(loc=2)
plt.show()

