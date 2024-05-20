import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_results(axis):
    plt.sca(axis)

    with open(f'making_new_plan.pkl', 'rb') as f:
        results: pd.DataFrame = pickle.load(f)
    results['agent_type'] = results['agent_type'].replace({'h': 'hierarchical', 'n': 'normal'})

    for idx, row in results.groupby('agent_type').head(20).iterrows():
        plt.plot(np.array(row['writes_history'])/1000, c='blue' if row['agent_type'] == 'hierarchical' else 'grey')
    for idx, row in results.groupby('agent_type').head(20).iterrows():
        plt.scatter(len(row['writes_history'])-1, row['writes_history'][-1]/1000, marker='s', c=[0,0.9,0], zorder=2)
    plt.scatter(0, 0, marker='o', c=(0,0,0,1), zorder=2)

    plt.legend('', frameon=False)
    plt.text(20, 34, 'flat', c='grey', fontweight='extra bold')
    plt.text(10, 11, 'hierarchical', c='blue', fontweight='extra bold')
    plt.ylabel('cognitive load\n(cumulative value calculations)\nx1000')
    plt.xlabel('step in environment')


if __name__ == '__main__':
    plt.figure()
    plot_results(plt.gca())
    plt.show()
