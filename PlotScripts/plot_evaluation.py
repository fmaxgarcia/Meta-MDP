import matplotlib.pyplot as plt
import numpy as np 
import sys 
import pickle 
import os

mean_returns = []
std_returns = []
rewards = pickle.load(open(sys.argv[1], "rb"))

keys = sorted(rewards.keys())
for episode in keys:
    domain_rewards = rewards[episode]
    ep_mean_return = []
    for d in domain_rewards:
        domain_returns = []
        for r in d:    
            domain_returns.append( np.sum(r) )
        ep_mean_return.append( np.mean(domain_returns) )
    mean_returns.append( np.mean(ep_mean_return) )
    std_returns.append( np.std(ep_mean_return) )

print(np.mean(mean_returns[-5:]))
print(np.mean(std_returns[-5:]))

fig = plt.figure()
ax = fig.add_subplot(111)
# ax.set_ylim([-500, 100])

mov_avg = [np.mean(mean_returns[i-2:i]) for i in range(2,len(mean_returns))]
ax.plot(range(len(mov_avg)), mov_avg, color="blue")


plt.show()


# for i, domain_rewards in enumerate(rewards):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_ylim([0, 100])
#     mean_rewards = np.mean(domain_rewards, axis=0)
#     ax.plot(range(len(mean_rewards)), mean_rewards, color="blue")
#     ax.fill_between(range(len(mean_rewards)), mean_rewards+np.std(domain_rewards, axis=0),
#                         mean_rewards-np.std(domain_rewards, axis=0), facecolor="blue", edgecolor="blue", alpha=0.4, interpolate=True)

#     plt.show()

    # fig.savefig(directory+"/domain_"+str(i)+".png")
    # plt.close('all')
