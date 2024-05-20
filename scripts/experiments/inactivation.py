import multiprocessing
import time

from state_abstraction_rl.environments.grid_world import GridWorldEnv
from state_abstraction_rl.agents.hierarchical.hierarchical_agents import HMBRL2, MBRL
from state_abstraction_rl.state_abstraction.hippocluster_abstraction import AbstractedMDP

import matplotlib.pyplot as plt

import numpy as np
import random
import itertools

class InactivatedHMBRL(HMBRL2):

    def __init__(self, inactive_idx, **kwargs):
        self.inactive_idx = inactive_idx
        super().__init__(**kwargs)

    def update(self, s, a, s_prime, r, terminated=False):

        if s in self.inactive_idx or s_prime in self.inactive_idx:
            self.Q[s, a] = -1
            return
        super().update(s, a, s_prime, r, terminated)


def choose_inactive_places(env, fraction):
    agent = MBRL(actions=[0, 1, 2, 3])
    for episode in range(10):
        state, _ = env.reset()

        for step in range(5000):
            # get action
            action = agent.choose_action(tuple(state))

            # execute action
            s_prime, reward, terminated, truncated, info = env.step(action)

            # update agent
            agent.update(tuple(state), action, tuple(s_prime), float(reward))

            # reset environment if necessary
            if reward >= 1:
                break

            # prepare for next iteration
            state = s_prime

    abs_mdp = AbstractedMDP(agent, size_reduction_factor=9)
    states_1 = abs_mdp.t_tables[1].get_all_states()
    inactive_1 = set(random.sample(states_1, k=int(len(states_1) * fraction)))
    # inactive_1 = set(random.sample(states_1, k=int(len(states_1) * fraction))) - {abs_mdp.get_higher_state(tuple(env.target_location), 0, 1)}
    inactive_0 = set(itertools.chain.from_iterable([abs_mdp.get_lower_states(s, 1, 0) for s in inactive_1]))
    return inactive_0


def run(inactive_fraction):

    # Create environment object
    env = GridWorldEnv('4_rooms_wide_door.bmp', render_mode='rgb_array')
    # env = GridWorldEnv.get_random_gridworld(size=23, n_walls=5, n_doors=4, render_mode='rgb_array')

    # setup inactive indexes
    inactive_idx = choose_inactive_places(env, fraction=inactive_fraction)

    # create agent
    agent = InactivatedHMBRL(
        inactive_idx=inactive_idx,
        actions=list(range(env.action_space.n)) if not hasattr(env,
                                                                'get_available_actions') else env.get_available_actions,
        discount_factor=0.6,
        theta_threshold=0.001,
        max_value_iterations=1000,
        q_default=env.max_reward * 1,
        abstraction_size_reduction_factor=9,
        random_walk_scale_max=1.1,
        random_walk_scale_min=0.5
    )

    start_time = time.time()
    for episode in range(20):
        reward_this_episode = 0
        state, _ = env.reset()

        if agent.abs_mdp is not None:
            agent.create_abstractions()

        route = [state]

        for step in range(5000):
            # get action
            action = agent.choose_action(tuple(state))
            # action = int(input('action: '))

            # execute action
            s_prime, reward, terminated, truncated, info = env.step(action)
            reward_this_episode += reward
            route.append(s_prime)

            # update agent
            agent.update(tuple(state), action, tuple(s_prime), float(reward))

            # reset environment if necessary
            if reward > 1:
                break

            # prepare for next iteration
            state = s_prime

    print('total time:', time.time() - start_time)
    return route


if __name__ == '__main__':
    import pickle
    from multiprocessing import Pool, cpu_count


    REPS = 4
    runs = []

    for rep in range(REPS):
        for inactive in [0.0, 0.3, 0.5, 0.8]:
            runs.append(inactive)
    pool = multiprocessing.Pool(cpu_count() - 1)
    runs = list(pool.map(run, runs))

    results = {0.0: [], 0.3: [], 0.5: [], 0.8: []}
    for rep in range(REPS):
        for inactive in [0.0, 0.3, 0.5, 0.8]:
            results[inactive].append(runs.pop(0))

    with open('inactivation.pkl', 'wb') as f:
        pickle.dump(results, f)
