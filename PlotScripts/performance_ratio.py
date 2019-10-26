import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt

first = pickle.load(open(sys.argv[1], "rb"))
second = pickle.load(open(sys.argv[2], "rb"))

save_directory = sys.argv[3]


ratios = []
for i in range(len(first)):
    
    first_data = np.mean(first[i], axis=0)
    second_data = np.mean(second[i], axis=0)

    ratio = first_data / second_data
    ratios.append(ratio)

ratios = np.asarray(ratios)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([0, 2])
ax.plot(range(ratios[0].shape[0]), [1.0] * ratios[0].shape[0], color="red", label="Standard training")
ax.plot(range(ratios[0].shape[0]), np.mean(ratios, axis=0), color="blue", alpha=0.6, label="Exploration policy")


plt.legend(loc=2)
fig.savefig(save_directory+"/ratios.png")
plt.close('all')





