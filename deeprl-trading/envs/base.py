from abc import ABC, abstractmethod


class Environment(ABC):
    @property
    @abstractmethod
    def observation_space(self):
        """provide the shape of observations"""
        pass

    @property
    @abstractmethod
    def action_space(self):
        """provide the shape of actions"""
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def test(self):
        pass
