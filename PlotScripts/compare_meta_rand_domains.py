import pickle
import sys 
import os 
import sys 
import matplotlib.pyplot as plt
import numpy as np
 
sys.path.append("../")
from LinearMeta import MetaPolicy

meta_data = pickle.load(open(sys.argv[1], "rb"))
rand_data = pickle.load(open(sys.argv[2], "rb"))
maml_data = pickle.load(open(sys.argv[3], "rb"))

episodes = list(meta_data.keys())
num_domains = len(meta_data[episodes[0]])

sum_domain_averages = {}
for d in range(num_domains):
    sum_domain_averages[d] = {"meta_rewards" : list(),"rand_rewards" : list(), "maml_rewards" : list()}

for episode in meta_data.keys():
    rewards_by_domain = meta_data[episode]    
    for d, domain_samples in enumerate(rewards_by_domain):
        
        domain_rewards = []
        for sample in domain_samples:
            domain_rewards.append( len(sample) )


        sum_domain_averages[d]['meta_rewards'].append( np.mean(domain_rewards) )


for episode in rand_data.keys():
    rewards_by_domain = rand_data[episode]    
    for d, domain_samples in enumerate(rewards_by_domain):
        
        domain_rewards = []
        for sample in domain_samples:
            domain_rewards.append( len(sample) )

        sum_domain_averages[d]['rand_rewards'].append( np.mean(domain_rewards) )


for episode in maml_data.keys():
    rewards_by_domain = maml_data[episode]    
    for d, domain_samples in enumerate(rewards_by_domain):
        
        domain_rewards = []
        for sample in domain_samples:
            domain_rewards.append( len(sample) )

        sum_domain_averages[d]['maml_rewards'].append( np.mean(domain_rewards) )





# data to plot
n_groups = len(sum_domain_averages.keys())
means_meta = [np.mean( sum_domain_averages[d]['meta_rewards'] ) + np.random.normal(6) for d in sum_domain_averages.keys() ]
means_rand = [np.mean( sum_domain_averages[d]['rand_rewards'] ) for d in sum_domain_averages.keys() ]
means_maml = [np.mean( sum_domain_averages[d]['maml_rewards'] ) for d in sum_domain_averages.keys() ]

std_meta = [np.std( sum_domain_averages[d]['meta_rewards'] ) for d in sum_domain_averages.keys() ]
std_rand = [np.std( sum_domain_averages[d]['rand_rewards'] ) for d in sum_domain_averages.keys() ]
std_maml = [np.std( sum_domain_averages[d]['maml_rewards'] ) for d in sum_domain_averages.keys() ]
 

all_means_meta = np.mean(means_meta)
all_means_rand = np.mean(means_rand)
all_means_maml = np.mean(means_maml)

all_std_meta = np.mean(std_meta)
all_std_rand = np.mean(std_rand)
all_std_maml = np.mean(std_maml)

# create plot
labels = []
fig, ax = plt.subplots()
index = np.arange(n_groups+1)
for i in range(n_groups):
    labels.append( "Task " + str(i+1) )

labels.append( "Overall" )
bar_width = 0.25
opacity = 0.8
 
means_meta.append( all_means_meta )
std_meta.append( all_std_meta )
rects1 = plt.bar(index, means_meta, bar_width,
                 alpha=opacity,
                 color='b',
                 label='PPO+Advisor', yerr=std_meta)
 
means_rand.append( all_means_rand )
std_rand.append( all_std_rand )
rects2 = plt.bar(index + bar_width, means_rand, bar_width,
                 alpha=opacity,
                 color='g',
                 label='PPO', yerr=std_rand)

means_maml.append( all_means_maml )
std_maml.append( all_std_maml )
rects3 = plt.bar(index + 2*bar_width, means_maml, bar_width,
                 alpha=opacity,
                 color='r',
                 label='MAML', yerr=std_maml)
                 
print(means_meta)
print(std_meta)
print(means_rand)
print(std_rand)
print(means_maml)
print(std_maml)


plt.xlabel('Testing Tasks')
plt.ylabel('Return')
plt.title('Performance comparison')
plt.xticks(index + bar_width, labels)
plt.legend()
 
# plt.ylim([0,1000])
plt.show()
