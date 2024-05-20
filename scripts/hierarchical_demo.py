import time

from state_abstraction_rl.environments.grid_world import GridWorldEnv
from state_abstraction_rl.agents.hierarchical.hierarchical_agents import HMBRL
from state_abstraction_rl.utils.tables import StateActionTable

import matplotlib.pyplot as plt


def plot(axes_agent, figure_abs, env, agent, current_location, show_abstractions=True):
    """
    This function is kind of a mess of matplotlib trickery
    """

    axes_agent[0].cla()
    axes_agent[0].imshow(env.render())
    axes_agent[0].set_xticks([])
    axes_agent[0].set_yticks([])
    axes_agent[0].set_title('grid world')

    axes_agent[1].cla()
    value_map_0 = env.get_value_map(agent.Q.get_best_values_dict())
    a = axes_agent[1].imshow(value_map_0, cmap='copper')
    axes_agent[1].set_xticks([])
    axes_agent[1].set_yticks([])
    axes_agent[1].set_title(f'world-level value map')

    axes_agent[3].cla()
    axes_agent[3].imshow(env.heatmap, cmap='hot')
    axes_agent[3].set_xticks([])
    axes_agent[3].set_yticks([])
    axes_agent[3].set_title('heatmap of states visited')
    if len(agent.planners) > 0:
        intermediate_value_map = value_map_0.copy()

        if agent.planners[1].current_intermediate_goal is not None:
            intermediate_Q = agent.planners[1].plans[agent.planners[1].current_intermediate_goal]
            intermediate_value_map = StateActionTable(default_value=0)
            for s1 in intermediate_Q.Q.get_all_states():
                for a in list(range(env.action_space.n)):
                    for s0 in agent.abs_mdp.get_lower_states(s1, 1, 0):
                        intermediate_value_map[s0, a] = intermediate_Q.Q.get_best_value(s1)
            intermediate_value_map = env.get_value_map(intermediate_value_map.get_best_values_dict())

        elif agent.planners[0].current_intermediate_goal is not None:
            intermediate_value_map = env.get_value_map(
                agent.planners[0].plans[agent.planners[0].current_intermediate_goal].Q.get_best_values_dict()
            )

        axes_agent[2].cla()
        axes_agent[2].imshow(intermediate_value_map, cmap='copper')
        axes_agent[2].scatter(current_location[0], current_location[1], color=[0, 1, 1])
        axes_agent[2].set_title('current intermediate goal')
        axes_agent[2].set_xticks([])
        axes_agent[2].set_yticks([])

    if agent.abs_mdp is not None and show_abstractions:
        figure_abs = agent.abs_mdp.plot_simple(current_position=tuple(current_location), f=figure_abs)

    plt.pause(0.00001)
    return figure_abs


if __name__ == '__main__':

    import numpy as np
    import random
    seed = 5
    np.random.seed(seed)
    random.seed(seed)

    # Create environment object
    env = GridWorldEnv('4_rooms.bmp', render_mode='rgb_array')
    # env = GridWorldEnv.get_random_gridworld(size=23, n_walls=5, n_doors=1, render_mode='rgb_array')

    # create agent
    agent = HMBRL(
        actions=list(range(env.action_space.n)) if not hasattr(env, 'get_available_actions') else env.get_available_actions,
        discount_factor=0.6,
        theta_threshold=0.001,
        max_value_iterations=1000,
        q_default=env.max_reward * 10,
        abstraction_size_reduction_factor=9,
        random_walk_scale_max=1.1,
        random_walk_scale_min=0.5
    )

    # create figure windows
    plt.ion()
    figure_abs = None
    _, axes_agent = plt.subplots(1, 4, figsize=(16, 5))
    plt.pause(0.1)

    start_time = time.time()
    for episode in range(26):
        reward_this_episode = 0
        state, _ = env.reset()

        for step in range(5000):
            # get action
            action = agent.choose_action(tuple(state))
            # action = int(input('action: '))

            # execute action
            s_prime, reward, terminated, truncated, info = env.step(action)
            reward_this_episode += reward

            # update agent
            agent.update(tuple(state), action, tuple(s_prime), float(reward))

            # reset environment if necessary
            if reward_this_episode >= 50:
                break

            # plot
            if agent.abs_mdp is not None and episode >= 3:
                figure_abs = plot(axes_agent, figure_abs, env, agent, s_prime)

            # prepare for next iteration
            state = s_prime

    print('total time:', time.time() - start_time)

    # save agent's tables
    with open('sample_agent_tables.pkl', 'wb') as f:
        import pickle
        pickle.dump({'T': agent.T, 'C': agent.C}, f)
        print('agent saved')

    # plot abstractions
    f = plt.figure()
    agent.abs_mdp.plot_simple()
    plt.pause(3600)

    input('enter to quit')
    env.close()



