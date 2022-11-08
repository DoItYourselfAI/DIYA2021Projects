from abc import ABC, abstractmethod
from utils.logger import Logger
import envs


class Agent(ABC):
    def __init__(self, args=None, name=None):
        self.args = args

        # initialize logger
        self.logger = Logger(name, args)
        self.logger.log("Initialized {} agent for {}, using {} architecture".format(name, args.env, args.arch))

        # initialize environment
        self.env = getattr(envs, args.env)(args=args)
        self.state = self.env.reset()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass
