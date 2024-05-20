from state_abstraction_rl.environments.grid_world import GridWorldEnv
from state_abstraction_rl.agents.hierarchical.hierarchical_agents import HMBRL
from state_abstraction_rl.agents.mbrl import MBRL
import state_abstraction_rl.utils as utils
import time
import pickle


def run_agent(args):

    agent_type, discount_factor, theta_threshold, max_value_iterations, env = args

    if agent_type == 'h':
        agent = HMBRL(
            actions=list(range(env.action_space.n)) if not hasattr(env, 'get_available_actions') else env.get_available_actions,
            discount_factor=discount_factor,
            theta_threshold=theta_threshold,
            max_value_iterations=max_value_iterations,
            q_default=env.max_reward * 10,
            abstraction_size_reduction_factor=9,
            random_walk_scale_max=1.1,
            random_walk_scale_min=0.5,
        )
    elif agent_type == 'n':
        agent = MBRL(
            actions=list(range(env.action_space.n)) if not hasattr(env, 'get_available_actions') else env.get_available_actions,
            discount_factor=discount_factor,
            theta_threshold=theta_threshold,
            max_value_iterations=max_value_iterations,
            q_default=env.max_reward * 10
        )

    utils.tables.total_Q_writes = 0

    writes_history = []
    reward_history = []
    time_to_goal = []
    writes_at_episode = []

    start_time = time.time()
    for episode in range(5):
        reward_this_episode = 0

        state, _ = env.reset()

        for step in range(4000):
            # get action
            action = agent.choose_action(tuple(state))

            # execute action
            s_prime, reward, terminated, truncated, info = env.step(action)
            reward_this_episode += reward

            # update agent
            agent.update(tuple(state), action, tuple(s_prime), float(reward))

            writes_history.append(utils.tables.total_Q_writes)
            reward_history.append(reward)

            # reset environment if necessary
            if reward_this_episode >= (100 if episode == 0 else 10):
                break

            # prepare for next iteration
            state = s_prime
        time_to_goal.append(step)
        writes_at_episode.append(utils.tables.total_Q_writes)

    print('total time:', time.time() - start_time, 'Q writes:', utils.tables.total_Q_writes)
    # return writes_history, reward_history
    return {
        'agent_type': agent_type,
        'discount_factor': discount_factor,
        'theta_threshold': theta_threshold,
        'max_value_iterations': max_value_iterations,
        'cumulative_reward': sum(reward_history),
        'Q_writes': writes_history[-1],
        'reward_history': reward_history,
        'writes_history': writes_history,
        'time_to_goal': time_to_goal,
        'writes_at_episode': writes_at_episode,
        'total_time': time.time() - start_time,
    }


if __name__ == '__main__':
    import multiprocessing
    import pandas as pd

    experiments = []

    for env in [GridWorldEnv.get_random_gridworld(size=23, n_walls=5, n_doors=1, render_mode=None) for _ in range(20)]:
        for discount_factor in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for theta_threshold in [0.01, 0.001]:
                for max_value_iterations in [100, 1000]:
                    experiments.append(('h', discount_factor, theta_threshold, max_value_iterations, env))
                    experiments.append(('n', discount_factor, theta_threshold, max_value_iterations, env))

    pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    results = pd.DataFrame(pool.map(run_agent, experiments))
    pool.close()
    print(results)

    # get worst performance within each parameter set
    worst = results.groupby(['agent_type', 'discount_factor', 'theta_threshold', 'max_value_iterations'])['cumulative_reward'].min().reset_index()

    # remove parameter sets that failed at least once
    worst = worst[worst['cumulative_reward'] > 100][['agent_type', 'discount_factor', 'theta_threshold', 'max_value_iterations']]
    results = pd.merge(worst, results, on=['agent_type', 'discount_factor', 'theta_threshold', 'max_value_iterations'], how='left')

    # identify params that give lowest writes for each agent
    best_writes = results.groupby(['agent_type', 'discount_factor', 'theta_threshold', 'max_value_iterations'])['Q_writes'].min().reset_index()
    best_writes = best_writes.loc[best_writes.groupby('agent_type')['Q_writes'].idxmin()]

    # reduce to final dataset
    results = pd.merge(best_writes, results, on=['agent_type', 'discount_factor', 'theta_threshold', 'max_value_iterations'], how='left')

    # save
    with open(f'cognitive_load.pkl', 'wb') as f:
        pickle.dump(results, f)
