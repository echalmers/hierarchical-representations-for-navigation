import sys
from typing import Union
from abc import abstractmethod, ABC

from state_abstraction_rl.state_abstraction.hippocluster_abstraction import AbstractedMDP
from state_abstraction_rl.agents.mbrl import MBRL
from state_abstraction_rl.utils import tables
from state_abstraction_rl.utils.structs import StateAction


class ValueIterationPlanner:

    def __init__(self, level: int, abs_mdp: AbstractedMDP, parent, discount_factor, theta_threshold, max_iterations):
        self.abs_mdp = abs_mdp
        self.level = level
        self.plans: dict[StateAction, MBRL] = dict()
        self.parent = parent
        self.discount_factor = discount_factor
        self.theta_threshold = theta_threshold
        self.max_iterations = max_iterations
        self.current_intermediate_goal = None

    def get_plan(self, state, goal: StateAction) -> Union[StateAction, None]:
        try:
            state_this_level = self.abs_mdp.get_higher_state(state, 0, self.level)
            goal_this_level = StateAction(self.abs_mdp.get_higher_state(goal.state, 0, self.level), None)
        except KeyError:
            return None

        # if goal state already reached at this level, return nothing
        if state_this_level == goal_this_level.state:
            return None

        # get intermediate goal if none exists (but default to goal at this level)
        self.current_intermediate_goal = self.current_intermediate_goal or \
                                         (self.parent.get_plan(state, goal) if self.parent is not None else None) or \
                                         goal_this_level

        # generate a plan to the intermediate goal (if there isn't one already)
        if self.current_intermediate_goal not in self.plans:
            self.plans[self.current_intermediate_goal] = self._generate_plan_this_level(
                state_this_level,
                self.current_intermediate_goal
            )

        # use plan to choose an action
        if state_this_level not in self.plans[self.current_intermediate_goal].Q.table:
            self.clear_all_goals()
            return None
        action = self.plans[self.current_intermediate_goal].choose_action(state_this_level)

        # clear current intermediate goal if this action satisfies current plan
        if StateAction(state_this_level, action) == self.current_intermediate_goal:
            self.current_intermediate_goal = None

        return action

    def clear_all_goals(self):
        self.current_intermediate_goal = None
        if self.parent is not None:
            self.parent.clear_all_goals()

    def _generate_plan_this_level(self, state_this_level, goal_this_level: StateAction) -> MBRL:
        agent = PlanningAgent(
            actions=self.abs_mdp.t_tables[self.level].get_known_actions_from_state,
            t_table=self.abs_mdp.t_tables[self.level],
            c_table=self.abs_mdp.c_tables[self.level],
            discount_factor=self.discount_factor,
            theta_threshold=self.theta_threshold,
            max_value_iterations=self.max_iterations
        )
        agent.plan(goal_state=goal_this_level.state, goal_action=goal_this_level.action)
        return agent


class PlanningAgent(MBRL):
    # todo: the planning should consider environment rewards too (e.g. so you don't plan to run over a lava trap)
    def __init__(
        self,
        actions,
        t_table,
        c_table,
        discount_factor=0.9,
        theta_threshold=0,
        max_value_iterations=sys.maxsize,
    ):
        super().__init__(
            actions=actions,
            epsilon=0,
            discount_factor=discount_factor,
            theta_threshold=theta_threshold,
            max_value_iterations=max_value_iterations,
            q_default=0,
            r_default=0,
            c_default=0,
            t_table=t_table,
            c_table=c_table
        )
        self.goal_state = None
        self.goal_action = None

    def update(self, s, a, s_prime, r, terminated=False):
        r = 0
        if self.goal_action is not None:
            if self.goal_state == s and self.goal_action == a:
                r = 1
        else:
            if self.goal_state == s_prime:
                r = 1
        super().update(s, a, s_prime, r, terminated)

    def plan(self, goal_state, goal_action=None, n_updates=None):
        self.goal_state = goal_state
        self.goal_action = goal_action

        self.Q = tables.StateActionTable(default_value=0)
        self.R = tables.StateActionTable(default_value=0)

        if goal_action is None:
            for predecessor, act in self.T.get_state_actions_with_access_to(goal_state):
                self.R[predecessor, act] = 1
                self.PQueue.insert((predecessor, act), float('inf'))
        else:
            self.R[goal_state, goal_action] = 1
            self.PQueue.insert((goal_state, goal_action), float('inf'))

        self.process_priority_queue(n_updates or self.max_value_iterations)
