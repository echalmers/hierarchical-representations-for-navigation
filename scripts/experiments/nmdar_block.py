from state_abstraction_rl.environments.grid_world import GridWorldEnv
from state_abstraction_rl.agents.hierarchical.hierarchical_agents import HMBRL

# Create environment objects
env1 = GridWorldEnv('4_rooms.bmp', render_mode='rgb_array')

env2 = GridWorldEnv('4_rooms_c.bmp', render_mode='rgb_array')

env3 = GridWorldEnv('4_rooms_d.bmp', render_mode='rgb_array')

experiments = {
    'initial training': env1,
    'new task in\nsame environment': env2,
    'return to\noriginal task': env1,
    'new environment': env3
}

def run(nmdar_block: bool):


    # Create agent objects
    agent = HMBRL(
        actions=list(range(env1.action_space.n)) if not hasattr(env1, 'get_available_actions') else env1.get_available_actions,
        discount_factor=0.6,
        theta_threshold=0.01,
        max_value_iterations=1000,
        q_default=env1.max_reward * 10,
        abstraction_size_reduction_factor=9,
        detect_changes=True
    )

    results = []

    for experiment, env in experiments.items():

        # if this is not the first run, apply NMDAR block if specified
        agent = agent.rest()
        if len(results) > 0:
            agent.NMDAR_block = nmdar_block

        time_to_goal = []

        for episode in range(5):

            reward_this_episode = 0
            state, _ = env.reset()

            for step in range(4000):
                # get action
                action = agent.choose_action(
                    tuple(state),
                    # goal_state=tuple(env.target_location) if episode >= 5 else None
                )

                # execute action
                s_prime, reward, terminated, truncated, info = env.step(action)
                reward_this_episode += reward

                # update agent
                agent.update(tuple(state), action, tuple(s_prime), float(reward))

                # reset environment if necessary
                if reward_this_episode >= (100 if episode == 0 else 10):
                    break

                # prepare for next iteration
                state = s_prime
            time_to_goal.append(step)

        results.append({
            'experiment': experiment,
            'nmdar_block': agent.NMDAR_block,
            'trial': list(range(1, 6)),
            'time_to_goal': time_to_goal
        })

    return results


if __name__ == '__main__':
    import multiprocessing
    import pickle

    pool = multiprocessing.Pool(multiprocessing.cpu_count()-2)
    reps = 25
    results = list(pool.map(run, [True]*reps + [False]*reps))
    pool.close()

    with open('nmdar_block.pkl', 'wb') as f:
        pickle.dump(results, f)
