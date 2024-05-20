import matplotlib.pyplot as plt
import pickle
import numpy as np
from state_abstraction_rl.environments.grid_world import GridWorldEnv


env = GridWorldEnv('4_rooms_wide_door.bmp', render_mode='rgb_array')
im = env.img

with open('inactivation.pkl', 'rb') as f:
    results = pickle.load(f)


a = plt.figure(figsize=(10,2.5), dpi=200, layout='constrained').subplot_mosaic(
    """
    ab.ef.ij.mn
    cd.gh.kl.op
    """,
    width_ratios=[1,1,0.25,1,1,0.25,1,1,0.25,1,1,]
)

for pane in 'abcdefghijklmnop':
    a[pane].imshow(im)
    a[pane].set_xticks([])
    a[pane].set_yticks([])


def plot(axs, route):
    for i in range(len(route)-1):
        axs.plot([route[i][1], route[i+1][1]], [route[i][0], route[i+1][0]], c='b')
    axs.scatter([2], [2], marker='s', color=[0,1,0], s=20, zorder=2)
    axs.scatter([13], [13], marker='o', color=[0, 0, 0], s=20, zorder=2)


a['a'].set_title('0% inactivation', loc='left')
plot(a['a'], results[0][0])
plot(a['b'], results[0][1])
plot(a['c'], results[0][2])
plot(a['d'], results[0][3])

a['e'].set_title('30% inactivation', loc='left')
plot(a['e'], results[0.3][0])
plot(a['f'], results[0.3][1])
plot(a['g'], results[0.3][2])
plot(a['h'], results[0.3][3])

a['i'].set_title('50% inactivation', loc='left')
plot(a['i'], results[0.5][0])
plot(a['j'], results[0.5][1])
plot(a['k'], results[0.5][2])
plot(a['l'], results[0.5][3])

a['m'].set_title('80% inactivation', loc='left')
plot(a['m'], results[0.8][0])
plot(a['n'], results[0.8][1])
plot(a['o'], results[0.8][2])
plot(a['p'], results[0.8][3])

plt.show()
