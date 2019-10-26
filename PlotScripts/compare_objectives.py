import numpy as np
import pickle

import matplotlib.pyplot as plt

no_meta_rewards = pickle.load(open("./no_meta_/rewards.pkl", "rb"))

meta_directories = [
    "./meta_1_/", "./meta_2_/", "./meta_3_/",
    "./meta_4_/", "./meta_5_/", "./meta_6_/",
    "./meta_7_/", "./meta_8_/", "./optimize/"
]

best_performances = []
ideal_value = 0.0
for i in range(len(no_meta_rewards)):
    mean_rewards = np.mean(no_meta_rewards[i], axis=0)
    best_performances.append( np.max(mean_rewards) )

    ideal_value += 1.0 * mean_rewards.shape[0]



objectives = []
for dir in meta_directories:
    rewards = pickle.load(open(dir+"rewards.pkl", "rb"))

    objective = 0.0
    print(dir)
    for i in range(len(rewards)):
        mean_rewards = np.mean(rewards[i], axis=0)
        ratios = mean_rewards / best_performances[i]

        objective += np.sum(ratios)
    objectives.append( objective )    

labels = meta_directories
labels.append("ideal")

objectives.append( ideal_value )

    


ind = np.arange(len(objectives))  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, objectives, width, color='r')#, yerr=men_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Objective')
ax.set_title('Objective value by explore policy')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(labels)
ax.set_ylim([0, np.max(objectives)*1.1])



def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
plt.show()

