from collections import namedtuple
import itertools
from state_abstraction_rl.utils.tables import TTable, StateActionTable
from state_abstraction_rl.agents.mbrl import MBRL
from hippocluster.algorithms.hippocluster import Hippocluster
from state_abstraction_rl.utils.structs import StateAction
import networkx as nx
import numpy as np
import distinctipy


class AbstractedMDP:

    def __init__(self, agent: MBRL, size_reduction_factor: int, aggregate_transitions=False, min_walk_scale=0.75, max_walk_scale=1.25):
        """
        create a hierarchical abstraction of an MDP - or rather, of an agent's model of an MDP
        :param agent: model-based agent who has encountered at least most of an environment
        :param size_reduction_factor: Each level of the hierarchy will have roughly this-x fewer states
        :param aggregate_transitions: if True, abstract MDP will aggregate transitions as well as states. In this case
            the aggregate transition represents the max probability of transitioning between the macro states at the
            level below, and the max Q value of the transitions at the level below, and the "action" for that transition
            is equal to the new macro state. If False, the abstract MDP will retain one transition for each possible
            transition between the macro states at the level below. The "action" for each of those transitions is a
            tuple of lower-level-state, lower-level-action
        :param min_walk_scale: min walk length is N/k * this
        :param max_walk_scale: max walk length is N/k * this
        """
        self.t_tables = [agent.T]
        self.c_tables = [agent.C]
        self.r_tables = [agent.R]
        self.assignments = []
        self.reverse_assignments = [None]
        self.aggregate_transitions = aggregate_transitions

        while len(agent.T.forward_map) // size_reduction_factor > 1:
            print('abstracting...')

            n_clusters = len(agent.T.forward_map) // size_reduction_factor
            hippocluster = Hippocluster(
                n_clusters=n_clusters,
                drop_threshold=0,
                n_walks=len(agent.T.forward_map)*10,
                batch_size=n_clusters * 10,
                min_len=int(len(agent.T.forward_map) / n_clusters * min_walk_scale),
                max_len=int(len(agent.T.forward_map) / n_clusters * max_walk_scale)
            )
            agent.nodes = agent.T.get_all_states()
            assignments = hippocluster.fit(agent)['assignments']
            reverse_assignments = dict()
            for k, v in assignments.items():
                reverse_assignments[v] = reverse_assignments.get(v, []) + [k]
            self.assignments.append(assignments)
            self.reverse_assignments.append(reverse_assignments)

            newT = TTable()
            newC = StateActionTable(default_value=0)
            newR = StateActionTable(default_value=0)
            for state in assignments:  #agent.T.forward_map:
                abstract_state = assignments[state]

                for action in agent.T.get_known_actions_from_state(state):
                    next_states = agent.get_dist_over_next_states(state, action)

                    abstract_next_state_p = dict()
                    for next_state, probability in next_states.items():
                        abstract_next_state = assignments.get(next_state, abstract_state) # if next state not in the clustering, assume next state belongs to same cluster as current
                        abstract_next_state_p[abstract_next_state] = abstract_next_state_p.get(abstract_next_state, 0) + probability

                    for abstract_next_state, probability in abstract_next_state_p.items():
                        if abstract_state != abstract_next_state:
                            if aggregate_transitions:
                                newT[abstract_state, abstract_next_state, abstract_next_state] = max(
                                    newT[abstract_state, abstract_next_state, abstract_next_state],
                                    probability
                                )
                                newC[abstract_state, abstract_next_state] = 1
                                newR[abstract_state, abstract_next_state] = max(newR[abstract_state, abstract_next_state], agent.Q[state, action])
                            else:
                                newT[abstract_state, StateAction(state, action), abstract_next_state] = probability
                                newC[abstract_state, StateAction(state, action)] = 1
                                newR[abstract_state, StateAction(state, action)] = agent.Q[state, action]

            self.t_tables.append(newT)
            self.c_tables.append(newC)
            self.r_tables.append(newR)
            agent = MBRL(actions=newT.get_known_actions_from_state, t_table=newT, c_table=newC)

        # attributes used for plotting
        self.graphs = [t.to_simple_graph() for t in self.t_tables]
        self.graphs[0].remove_edges_from(nx.selfloop_edges(self.graphs[0]))
        self.pos = [None] + [nx.drawing.spring_layout(g) for g in self.graphs[1:]]
        self.colors = np.random.random(size=(self.graphs[0].order(), 3))
        # self.colors = distinctipy.get_colors(self.graphs[0].order())

    @property
    def n_levels(self):
        return len(self.t_tables)

    def get_higher_state(self, state, current_level, to_level) -> int:
        high_state = state
        for level in range(current_level, to_level):
            high_state = self.assignments[level][high_state]
        return high_state

    def get_lower_states(self, state, current_level, to_level) -> list:

        low_states = [state]
        for level in range(current_level, to_level, -1):
            low_states = itertools.chain.from_iterable([self.reverse_assignments[level][s] for s in low_states])
        return list(low_states)

    def state_action_to_goal(self, state, action, level):
        if self.aggregate_transitions:
            raise Exception('logic with aggregate transitions not implemented')

        if level == 0:
            return state, action
        else:
            return action[0], action[1]

    def plot_simple(self, f=None, current_position=None):
        import matplotlib.pyplot as plt

        if f is None:
            f = plt.subplots(2, self.n_levels - 1)
            f = (f[0], f[1].reshape(2, self.n_levels - 1))
        if f[1].shape[1] < self.n_levels - 1:
            plt.close(f[0])
            f = plt.subplots(2, self.n_levels - 1)
            f = (f[0], f[1].reshape(2, self.n_levels - 1))

        g0 = self.graphs[0]

        for level in range(1, self.n_levels):

            g_level = self.graphs[level]
            pos = self.pos[level]

            f[1][0, level-1].cla()
            for edge in g_level.edges:
                f[1][0, level-1].plot([pos[edge[0]][0], pos[edge[1]][0]], [pos[edge[0]][1], pos[edge[1]][1]], c='k', zorder=1, linewidth=0.5)
                # f[1][0, level - 1].annotate('', xy=[pos[edge[1]][0], pos[edge[1]][1]], xytext=[pos[edge[0]][0], pos[edge[0]][1]], arrowprops=dict(arrowstyle="->"))

            try:
                state_this_level = self.get_higher_state(current_position, current_level=0, to_level=level) if current_position is not None else None
                linewidths = [4 if state_this_level == node else 0 for node in g_level]
            except KeyError as ex:
                linewidths = 0
            f[1][0, level-1].scatter([pos[node][0] for node in g_level], [pos[node][1] for node in g_level], c=[self.colors[node] for node in g_level], zorder=2, edgecolors=None, linewidths=linewidths)
            f[1][0, level-1].set_xticks([])
            f[1][0, level-1].set_yticks([])
            f[1][0, level - 1].set_title(f'level {level} abstraction')

            low_level_colors = []
            for node in g0.nodes:
                try:
                    low_level_colors.append(self.colors[self.get_higher_state(node, 0, level)])
                except:
                    low_level_colors.append([0, 0, 0])

            f[1][1, level - 1].cla()
            linewidths = [4 if current_position == node else 0 for node in g0]
            f[1][1, level - 1].scatter([node[0] for node in g0], [-node[1] for node in g0], c=low_level_colors, edgecolors=None, linewidths=linewidths)
            f[1][1, level - 1].set_xticks([])
            f[1][1, level - 1].set_yticks([])

        return f



if __name__ == '__main__':
    import pickle
    from matplotlib import pyplot as plt
    import time

    with open(r'sample_agent_tables.pkl', 'rb') as f:
        tables = pickle.load(f)
        T: TTable = tables['T']
        C = tables['C']

    a = AbstractedMDP(MBRL(actions=T.get_known_actions_from_state, t_table=T, c_table=C), size_reduction_factor=9)

    f = a.plot_simple()
    plt.show()


