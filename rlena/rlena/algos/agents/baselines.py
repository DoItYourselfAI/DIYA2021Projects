from rlena.envs.playground.pommerman import characters
from rlena.envs.playground.pommerman.agents import BaseAgent, SimpleAgent

class StoppedAgent(BaseAgent):
    def __init__(self, character=characters.Bomber):
        super(StoppedAgent, self).__init__(character)

    def act(self, obs, action_space):
        return 0 # stop action

class NoBombSimpleAgent(SimpleAgent):
    def act(self, obs, action_space):
        action = super().act(obs, action_space)
        if action == 5: # Bomb action
            action = 0
        return action