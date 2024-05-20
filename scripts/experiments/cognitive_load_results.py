import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


with open(f'cognitive_load.pkl', 'rb') as f:
    results: pd.DataFrame = pickle.load(f)
results['agent_type'] = results['agent_type'].replace({'h': 'hierarchical', 'n': 'normal'})

df_steps = pd.DataFrame()

for idx, row in results.iterrows():
    df_steps = pd.concat((
        df_steps,
        pd.DataFrame({
            'agent_type': row['agent_type'],
            'discount_factor': row['discount_factor'],
            'theta_threshold': row['theta_threshold'],
            'max_value_iterations': row['max_value_iterations'],
            'reward_history': np.array(row['reward_history']).cumsum(),
            'writes_history': row['writes_history'],
            'time': 'time',
            'step': list(range(len(row['reward_history'])))
        })
    ),
        ignore_index=True
    )
df_steps = df_steps[df_steps['step'] % 20 == 0]

df_episodes = pd.DataFrame()
for idx, row in results.iterrows():
    df_episodes = pd.concat((
        df_episodes,
        pd.DataFrame({
            'agent_type': row['agent_type'],
            'discount_factor': row['discount_factor'],
            'theta_threshold': row['theta_threshold'],
            'max_value_iterations': row['max_value_iterations'],
            'time_to_goal': row['time_to_goal'],
            'writes_at_trial': row['writes_at_episode'],
            'trial': np.array(list(range(len(row['writes_at_episode'])))) + 1,
        })
    ),
        ignore_index=True
    )


def plot_reward_history(axis):
    plt.sca(axis)

    f = plt.subplot(2, 1, 1)
    sns.lineplot(df_steps, x='step', y='reward_history', hue='agent_type')
    plt.grid(axis='x')
    plt.xlabel('')


def plot_writes_history(axis):
    plt.sca(axis)

    plt.subplot(2, 1, 2, sharex=f)
    sns.lineplot(df_steps, x='step', y='writes_history', hue='agent_type')
    plt.grid(axis='x')


def plot_time_by_trial(axis):
    plt.sca(axis)

    sns.lineplot(df_episodes, x='trial', y='time_to_goal', hue='agent_type', palette=['blue', 'grey'])
    plt.yscale('log')
    plt.ylabel('time-to-goal (steps)')
    plt.grid(axis='x')
    plt.xlabel('')
    plt.legend('', frameon=False)
    plt.xticks([1, 2, 3, 4, 5])
    plt.text(2, 60, 'flat', c='grey', fontweight='extra bold')
    plt.text(2, 400, 'hierarchical', c='blue', fontweight='extra bold')


def plot_writes_by_trial(axis):
    plt.sca(axis)

    df = df_episodes.copy()
    df['writes_at_trial'] /= 1000
    sns.lineplot(df, x='trial', y='writes_at_trial', hue='agent_type', palette=['blue', 'grey'])

    plt.grid(axis='x')
    plt.ylabel('cognitive load\n(cumulative value calculations)\nx1000')
    plt.xlabel('trial')
    plt.xticks([1,2,3,4,5])
    plt.text(2, 80, 'flat', c='grey', fontweight='extra bold')
    plt.text(2.5, 60, 'hierarchical', c='blue', fontweight='extra bold')
    plt.legend('', frameon=False)


if __name__ == '__main__':
    plt.subplot(2, 1, 1)
    plot_time_by_trial(plt.gca())
    plt.subplot(2, 1, 2)
    plot_writes_by_trial(plt.gca())
    plt.show()