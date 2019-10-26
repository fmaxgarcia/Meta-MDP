import numpy as np 
import sys 
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

directory = sys.argv[1]


action_frequency = {"meta" : list(),"no meta" : list()}


meta_data = pickle.load(open(directory+"/meta_test.pkl", "rb"))
meta_null = pickle.load(open(directory+"/null_actions_meta.pkl", "rb"))

total_steps = 0
steps_per_episodes = []
for episode in meta_data.keys():
    rewards_by_domain = meta_data[episode]
    episode_steps = []
    for d, domain_samples in enumerate(rewards_by_domain):
        for sample in domain_samples:
            total_steps += len(sample)
            episode_steps.append( len(sample) )

    steps_per_episodes.append( episode_steps )

freq_meta = {}
freq_meta_episodes = []
for episode in meta_null.keys():
    total_fraction = 0.0
    for action in meta_null[episode].keys():
        count = meta_null[episode][action]
        f = float(count) / np.sum(steps_per_episodes[episode])
        total_fraction += f

    freq_meta_episodes.append( total_fraction )


no_meta_data = pickle.load(open(directory+"/no_meta_test.pkl", "rb"))
no_meta_null = pickle.load(open(directory+"/null_actions_no_meta.pkl", "rb"))

total_steps = 0
steps_per_episodes = []
for episode in no_meta_data.keys():
    rewards_by_domain = meta_data[episode]
    episode_steps = []
    for d, domain_samples in enumerate(rewards_by_domain):
        for sample in domain_samples:
            total_steps += len(sample)
            episode_steps.append(len(sample))

    steps_per_episodes.append(episode_steps)


freq_no_meta = {}
freq_no_meta_episodes = []
for episode in no_meta_null.keys():
    total_fraction = 0.0
    for action in no_meta_null[episode].keys():
        count = no_meta_null[episode][action]
        f = float(count) / np.sum(steps_per_episodes[episode])
        total_fraction += f
    freq_no_meta_episodes.append( total_fraction )


freq_meta_episodes = np.asarray(freq_meta_episodes)
freq_no_meta_episodes = np.asarray(freq_no_meta_episodes)

freq_meta_episodes_avg = np.asarray([ np.mean(freq_meta_episodes[i-20:i]) for i in range(20, len(freq_meta_episodes))])
freq_no_meta_episodes_avg = np.asarray([ np.mean(freq_no_meta_episodes[i-20:i]) for i in range(20, len(freq_no_meta_episodes))])

freq_meta_episodes_std = np.asarray([ np.std(freq_meta_episodes[i-20:i]) for i in range(20, len(freq_meta_episodes))])
freq_no_meta_episodes_std = np.asarray([ np.std(freq_no_meta_episodes[i-20:i]) for i in range(20, len(freq_no_meta_episodes))])


plt.plot(range(freq_no_meta_episodes_avg.shape[0]), freq_no_meta_episodes_avg, color="red", label="Random")
plt.fill_between(range(freq_no_meta_episodes_avg.shape[0]), freq_no_meta_episodes_avg-freq_no_meta_episodes_std,
                 freq_no_meta_episodes_avg+freq_no_meta_episodes_std, alpha=0.5, color="red")

plt.plot(range(freq_meta_episodes_avg.shape[0]), freq_meta_episodes_avg, color="blue", label="Advisor")
plt.fill_between(range(freq_meta_episodes_avg.shape[0]), freq_meta_episodes_avg-freq_meta_episodes_std,
                 freq_meta_episodes_avg+freq_meta_episodes_std, alpha=0.5, color="blue")


plt.xlabel('Action')
plt.ylabel('Frequency')
plt.title('Action frequency comparison')
plt.legend()
 
plt.show()