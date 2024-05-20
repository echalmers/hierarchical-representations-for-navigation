from abc import  abstractmethod
from typing import TypeVar, Generic


state_type = TypeVar("state_type")
action_type = TypeVar("action_type")


class Agent(Generic[state_type, action_type]):

    @abstractmethod
    def choose_action(self, s: state_type):
        """
        Chooses an action given a state
        """
        raise NotImplementedError()

    def update(self, s: state_type, a: action_type, s_prime: state_type, r: float, terminated: bool = False):
        """
        update the agent's learning system given a transition
        :param s: original state
        :param a: action performed
        :param s_prime: new state
        :param r: reward
        :param terminated: episode terminated flag
        :return: None
        """