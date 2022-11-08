import gym
import pommerman

from copy import deepcopy
from easydict import EasyDict
from rl2.agents.utils import LinearDecay
from algos.agents.coma import COMAPGModel, COMAgent
from algos.workers import ComaWorker
from ..envs.customPomme import TwoVsTwoPomme
from algos.utils import Logger

env = TwoVsTwoPomme()

# get partial obs
obs = env.reset()
# get global obs
g_obs_shape = env.get_global_obs_shape()

observation_shape = env.observation_shape
ac_shape = (env.action_space.n,)
n_agents = len(env._agents)
jac_shape = tuple([ac_shape for i in range(n_agents)])

config = EasyDict({
    'n_agents': 4,
    'n_env': 1,
    'buffer_size': 128,
    'batch_size': 30,
    'num_epochs': 1,
    'update_interval': 128,
    'train_interval': 30,
    'init_collect': 1,
    'log_interval': 10,
    'lr_ac': 1e-4,
    'lr_cr': 1e-3,
    'gamma': 0.95,
    'lamda': 0.99,
    'polyak': 0.99,
    'tag': 'LongDecay',
})
eps = LinearDecay(start=0.5, end=0.02, decay_step=2000)

if __name__ == '__main__':
    logger = Logger(name='COMA', args=config)

    team1 = COMAPGModel(observation_shape, g_obs_shape, ac_shape, n_agents=2)
    team2 = deepcopy(team1)

    magent = COMAgent([team1, team2], eps=eps, **config)

    worker = ComaWorker(
        env,
        n_env=1,
        agent=magent,
        n_agents=4,
        max_episodes=20000,
        training=True,
        logger=logger,
        log_interval=config.log_interval,
        render=True,
        render_interval=10,
        is_save=True,
    )

    worker.run()
