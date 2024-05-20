import sys
import tempfile
import os
import random

from state_abstraction_rl.utils.structs import StateAction
from state_abstraction_rl.state_abstraction.hippocluster_abstraction import AbstractedMDP
from state_abstraction_rl.agents.mbrl import MBRL
from state_abstraction_rl.agents.hierarchical import planner
from state_abstraction_rl.utils.tables import StateActionTable, TTable

import numpy as np


class HMBRL(MBRL):

    def __init__(
        self,
        actions,
        epsilon=0.1,
        discount_factor=0.9,
        theta_threshold=0,
        max_value_iterations=sys.maxsize,
        q_default=100,
        r_default=0,
        c_default=0,
        abstraction_size_reduction_factor=10,
        random_walk_scale_min=0.75,
        random_walk_scale_max=1.25,
        NMDAR_block=False,
        detect_changes=False,
    ):
        super().__init__(
            actions=actions,
            epsilon=epsilon,
            discount_factor=discount_factor,
            theta_threshold=theta_threshold,
            max_value_iterations=max_value_iterations,
            q_default=q_default,
            r_default=r_default,
            c_default=c_default
        )
        self.size_reduction_factor = abstraction_size_reduction_factor
        self.planners = []
        self.abs_mdp = None
        self.NMDAR_block = NMDAR_block
        self.detect_changes = detect_changes
        self.save_filename = os.path.join(tempfile.mkdtemp(), str(id(self)))
        self.random_walk_scale_min = random_walk_scale_min
        self.random_walk_scale_max = random_walk_scale_max

    def choose_action(self, s, goal_state=None):
        """
        Chooses an action from a specific state
        """
        print(f'****\ncurrent state: {s}')

        # create an abstract model of the environment if necessary (and allowed)
        if self.abs_mdp is None and not self.NMDAR_block and self.Q.get_best_value(s, actions=self.actions(s)) <= 0:
            self.create_abstractions()

        # if there's a good Q value, follow it
        if goal_state is None and (self.Q.get_best_value(s, actions=self.actions(s)) > 0.1 or self.abs_mdp is None):
            self.clear_goals()
            action = super().choose_action(s)

        # otherwise use the model of the env to make a plan
        else:
            if goal_state is None:
                goal_state = max(self.Q.get_all_states(), key=lambda s: self.Q.get_best_value(s, actions=self.actions(s)))
            action = self.planners[0].get_plan(s, StateAction(goal_state, None))
            if action is None:

                # refresh abstractions if necessary
                if len(self.abs_mdp.assignments) > 0 and len(self.T.forward_map) - len(self.abs_mdp.assignments[0]) > 1:
                    self.create_abstractions()

                action = random.choice(self.actions(s))  #  super().choose_action(s)
                self.clear_goals()

        return action

    def update(self, s, a, s_prime, r, terminated=False):

        if self.detect_changes:
            if self.C.contains(s, a) and \
                    self.T.get_state_probabilities_from_state_action(s, a, as_dict=True).get(s_prime, 0) == 0:
                self.Q = StateActionTable(default_value=self.Q.default_value)
                self.R = StateActionTable(default_value=self.R.default_value)
                self.C = StateActionTable(default_value=self.C.default_value)
                self.T = TTable()
                self.planners = []
                self.abs_mdp = None

            elif self.R.contains(s, a) and abs(r - self.R[s, a]) / self.R[s, a] > 0.25:
                self.Q = StateActionTable(default_value=self.q_default)
                self.R = StateActionTable(default_value=self.r_default)
                self.C = StateActionTable(default_value=self.c_default)
                self.T = TTable()

        super().update(s, a, s_prime, r, terminated)

    def create_abstractions(self):
        # Create planners based off of number of abstractions

        self.abs_mdp = AbstractedMDP(agent=self,
                                     size_reduction_factor=self.size_reduction_factor,
                                     aggregate_transitions=False,
                                     max_walk_scale=self.random_walk_scale_max,
                                     min_walk_scale=self.random_walk_scale_min,
                                     )

        self.planners.clear()
        for i in range(self.abs_mdp.n_levels - 1, -1, -1):
            self.planners.insert(0,
                                 planner.ValueIterationPlanner(
                                     level=i,
                                     abs_mdp=self.abs_mdp,
                                     parent=self.planners[0] if i < self.abs_mdp.n_levels - 1 else None,
                                     discount_factor=self.discount_factor,
                                     theta_threshold=self.theta_threshold,
                                     max_iterations=self.max_value_iterations
                                 )
            )

    def clear_goals(self):
        for planner in self.planners:
            planner.current_intermediate_goal = None

    def rest(self):

        if not self.NMDAR_block:
            self.save(self.save_filename)

        return self.load(self.save_filename, self.actions)


