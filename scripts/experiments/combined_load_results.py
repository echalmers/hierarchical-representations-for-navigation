import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from making_new_plan_results import plot_results as plot_planning_load
from cognitive_load_results import plot_writes_by_trial, plot_time_by_trial
from state_abstraction_rl.environments.grid_world import GridWorldEnv

import random
import numpy as np
random.seed(1)
np.random.seed(1)

mosaic = """
a.bc
d.ee
"""
fig, a = plt.subplot_mosaic(mosaic, figsize=(10,5), dpi=200, layout='constrained', width_ratios=[2, 0.25, 1, 1])

a['a'].get_shared_x_axes().join(a['a'], a['d'])

# plot initial learning curve
plot_time_by_trial(a['a'])
a['a'].set_title('initial learning curve')
a['a'].axes.xaxis.set_ticklabels([])

# plot writes during initial learning
plot_writes_by_trial(a['d'])
a['d'].set_title('mental effort of initial learning')

# plot maps
env, env_inversed = GridWorldEnv.get_random_gridworld(size=15, n_walls=5, n_doors=1, return_inversed=True, render_mode='rgb_array')
env.reset()
env_inversed.reset()
a['b'].imshow(env.render())
a['b'].set_title('initial task')
a['b'].set_xticks([])
a['b'].set_yticks([])
a['c'].imshow(env_inversed.render())
a['c'].set_title('new task')
a['c'].set_xticks([])
a['c'].set_yticks([])

# plot new planning load
plot_planning_load(a['e'])
a['e'].set_title('mental effort of planning in new task')

# add pane labels
for label, ax in a.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', weight='bold', va='bottom', fontfamily='serif')

# plt.savefig('combined_load_results.png', dpi=300)
plt.show()