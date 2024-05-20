import pandas as pd
pd.options.display.width = 0
import pickle
from state_abstraction_rl.environments.grid_world import GridWorldEnv
from state_abstraction_rl.agents.hierarchical.hierarchical_agents import HMBRL
from state_abstraction_rl.agents.mbrl import MBRL
from state_abstraction_rl.utils import tables
from state_abstraction_rl.agents.hierarchical.planner import PlanningAgent
import multiprocessing


def run_agent(args):

    agent_type, discount_factor, theta_threshold = args

    # create environemnt
    env, env_inversed = GridWorldEnv.get_random_gridworld(size=23, n_walls=5, n_doors=1, return_inversed=True, render_mode=None)

    # get the initial learning params from the cognitive load experiment
    with open(f'cognitive_load.pkl', 'rb') as f:
        params: pd.DataFrame = pickle.load(f)
    params = params[['agent_type', 'discount_factor', 'theta_threshold', 'max_value_iterations']].drop_duplicates().set_index('agent_type')

    # create hierarchical and normal agents
    if agent_type == 'h':
        agent = HMBRL(
            actions=list(range(env.action_space.n)) if not hasattr(env, 'get_available_actions') else env.get_available_actions,
            discount_factor=params.loc['h', 'discount_factor'],
            theta_threshold=params.loc['h', 'theta_threshold'],
            max_value_iterations=params.loc['h', 'max_value_iterations'],
            q_default=env.max_reward * 10,
            abstraction_size_reduction_factor=9,
        )
    elif agent_type == 'n':
        agent = MBRL(
            actions=list(range(env.action_space.n)) if not hasattr(env, 'get_available_actions') else env.get_available_actions,
            discount_factor=params.loc['n', 'discount_factor'],
            theta_threshold=params.loc['n', 'theta_threshold'],
            max_value_iterations=params.loc['n', 'max_value_iterations'],
            q_default=env.max_reward * 10
        )

    # learn the environment
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

            # reset environment if necessary
            if reward_this_episode >= 100:
                break

            # prepare for next iteration
            state = s_prime

    # switch to new env (same world, different task)
    env = env_inversed
    state, _ = env.reset()

    tables.total_Q_writes = 0
    writes_history = [0]

    # prepare the agent for the new task
    if agent_type == 'h':
        agent.discount_factor = discount_factor
        agent.theta_threshold = theta_threshold
        agent.create_abstractions()
    elif agent_type == 'n':
        agent = PlanningAgent(
            actions=list(range(env.action_space.n)) if not hasattr(env, 'get_available_actions') else env.get_available_actions,
            t_table=agent.T, c_table=agent.C,
            discount_factor=discount_factor,
            theta_threshold=theta_threshold,
            # max_value_iterations=max_value_iterations
        )
        agent.plan(goal_state=tuple(env.target_location))

    # run through the new task
    # plt.ion()
    state_history = []
    for step in range(4000):
        # get action
        if agent_type == 'n':
            action = agent.choose_action(tuple(state))
        else:
            state_history.append(tuple(state))
            if len(state_history) >= 10 and len(set(state_history[-8:])) < 5:  # agent regenerates model if stuck
                agent.create_abstractions()
                state_history.clear()
            action = agent.choose_action(tuple(state), goal_state=tuple(env.target_location))

        # execute action
        s_prime, reward, terminated, truncated, info = env.step(action)

        # record writes
        writes_history.append(tables.total_Q_writes)

        # reset environment if necessary
        if reward > 0:
            break

        # prepare for next iteration
        state = s_prime

    return {
        'agent_type': agent_type,
        'discount_factor': discount_factor,
        'theta_threshold': theta_threshold,
        'writes_history': writes_history,
        'total_steps': step,
        'total_writes': writes_history[-1],
    }


if __name__ == '__main__':

    experiments = []
    for agent_type in ['n', 'h']:
        for discount_factor in [0.5, 0.7, 0.9]:
            for theta_threshold in [0.01, 0.001]:
                for reps in range(20):
                    experiments.append((agent_type, discount_factor, theta_threshold))

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    results = pd.DataFrame(pool.map(run_agent, experiments))
    pool.close()
    print(results)

    # get worst performance within each parameter set
    worst = results.groupby(['agent_type', 'discount_factor', 'theta_threshold'])['total_steps'].max().reset_index()

    # remove parameter sets that failed at least once
    worst = worst[worst['total_steps'] < 3500][['agent_type', 'discount_factor', 'theta_threshold']]
    results = pd.merge(worst, results, on=['agent_type', 'discount_factor', 'theta_threshold'], how='left')

    # identify params that give lowest writes for each agent
    best_writes = results.groupby(['agent_type', 'discount_factor', 'theta_threshold'])['total_writes'].mean().reset_index()
    best_writes = best_writes.loc[best_writes.groupby('agent_type')['total_writes'].idxmin()]

    # reduce to final dataset
    results = pd.merge(best_writes, results, on=['agent_type', 'discount_factor', 'theta_threshold'], how='left')

    # save
    with open(f'making_new_plan.pkl', 'wb') as f:
        pickle.dump(results, f)
