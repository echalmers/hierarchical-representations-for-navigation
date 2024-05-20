import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nmdar_block import experiments


with open('nmdar_block.pkl', 'rb') as f:
    results = pickle.load(f)

times = pd.DataFrame()
for resultset in results:
    for result in resultset:
        times = pd.concat((times, pd.DataFrame(result)), ignore_index=True)
times['nmdar_block'].replace({True: 'simulated NMDAR block', False: "control"}, inplace=True)


fig = plt.figure(constrained_layout=True)
subfigs = fig.subfigures(2, 1)

subfigs[0].suptitle('mice', x=0.01, ha='left', weight='extra bold', color='b', fontsize='x-large')
subfigs[0].set_facecolor([0.9, 0.9, 1])

subfigs[1].suptitle('artificial agents', x=0.01, ha='left', weight='extra bold', color='g', fontsize='x-large')
subfigs[1].set_facecolor([0.9, 1, 0.9])

axs = subfigs[0].subplots(1, 4)

# Figure 6 A (2005)
plt.sca(axs[0])
plt.ylabel("time to goal (sec)")
plt.xlabel("trial")
plt.xticks([1,2,3,4])
plt.yticks([0, 10, 20, 30, 40, 50, 60])
plt.title(list(experiments)[0])
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(2))
plt.ylim(0, 60)
plt.xlim(0, 5)
plt.plot([1,2,3,4], np.array([[34.5,14,11,7], [32,12,10,6]]).mean(axis=0), color='grey', linewidth=1, markersize=4, label='PRE-CPP')
# plt.plot([1,2,3,4], [32,12,10,6], linewidth=1, markersize=4, label='PRE-Control')
plt.legend(['all agents'], loc='upper right')
plt.xlim([0.9, 4.1])

# Figure 6 B (2005)
plt.sca(axs[1])
plt.xlabel("trial")
plt.xticks([2, 4, 6, 8])
plt.yticks([0, 10, 20, 30, 40, 50, 60])
plt.title(list(experiments)[1])
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(2))
plt.ylim(0, 60)
plt.xlim(0, 9)
plt.plot([1,2,3,4,5,6,7,8], [45,13.5,10,8,6,7,8.5,5.5], color='orange', linewidth=1, markersize=4, label='NMDAR block')
plt.plot([1,2,3,4,5,6,7,8], [38,8,8.2,8.2,8.2,6,5.5,6], color='blue', linewidth=1, markersize=4, label='control')
plt.legend(loc='upper right')
plt.xlim([0.7, 8.3])

# Figure 6 C (2005)
plt.sca(axs[2])
plt.xlabel("trial")
plt.xticks([2, 4, 6, 8])
plt.yticks([0, 5, 10, 15, 20, 25])
plt.title(list(experiments)[2])
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(1))
plt.ylim(0, 25)
plt.xlim(0, 9)
plt.plot([1,2,3,4,5,6,7,8], [7.5, 11, 7.5, 6, 4, 8, 6, 5], color='orange', linewidth=1, markersize=4, label='NMDAR block')
plt.plot([1,2,3,4,5,6,7,8], [16, 14, 5, 3.5, 8, 6.5, 5, 3], color='blue', linewidth=1, markersize=4, label='control')
# plt.legend(loc='upper right')
plt.xlim([0.7, 8.3])

# Figure 6.A (2019)
plt.sca(axs[3])
plt.xlabel("trial")
plt.xticks([2, 4, 6, 8])
plt.yticks([0, 10, 20, 30, 40, 50, 60])
plt.title(list(experiments)[3])
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(1))
plt.ylim(0, 55)
plt.xlim(0, 9)
plt.plot([1,2,3,4,5,6,7,8], [28, 40, 33, 28, 26, 14, 24, 25], color='orange', linewidth=1, markersize=4, label='NMDAR block')
plt.plot([1,2,3,4,5,6,7,8], [24, 12, 10, 8, 5.5, 6, 5, 4], color='blue', linewidth=1, markersize=4, label='control')
# plt.legend(loc='upper right')
plt.xlim([0.7, 8.3])


axs = subfigs[1].subplots(1, 4)
for i in range(len(experiments)):
    plt.sca(axs[i])
    s = sns.lineplot(times[times['experiment'] == list(experiments)[i]], x='trial', y='time_to_goal', hue='nmdar_block', palette=['grey'] if i==0 else ['orange', 'blue'], legend=False if i>=2 else 'brief')
    plt.yscale('log')
    plt.xticks([1,2,3,4,5])
    if i == 0:
        plt.legend(['all agents'])
    elif i==1:
        plt.gca().legend().set_title('')
    plt.ylabel('')

axs[0].set_ylabel('time to goal (steps)')
plt.show()
