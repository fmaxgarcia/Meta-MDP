import matplotlib.pyplot as plt
import numpy as np 
import sys 
import pickle 

rewards = pickle.load(open(sys.argv[1],"rb"))
# directory = sys.argv[2]


fig = plt.figure()
ax = fig.add_subplot(111)

keys = sorted(rewards.keys())
avg, std = [], []
for key in keys:
    domain_rewards = rewards[key]

    m_r = []
    s_r = []
    for r in domain_rewards:
        m_r.append( np.mean(r) )
        s_r.append( np.std(r) )

    
    avg.append( np.mean(m_r) )
    std.append( np.mean(s_r) )
    

ax.plot(range(len(avg)), avg)
plt.show()

    # fig.savefig(directory+"/domain_"+str(i)+".png")
    # plt.close('all')
