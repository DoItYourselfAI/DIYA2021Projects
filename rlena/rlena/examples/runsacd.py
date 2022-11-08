import gym
import pommerman

import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
import json
from easydict import EasyDict


from rlena.algos.agents.sac_discrete import SACAgentDISC, SACModelDISC
from rlena.algos.workers import SACDworker
from rlena.algos.utils import Logger

from rlena.envs.customPomme import ConservativeEnvWrapper
from pommerman.configs import one_vs_one_env
import pommerman.envs as envs
from pommerman.agents import SimpleAgent
from pommerman.characters import Bomber

from rl2.buffers.base import ReplayBuffer
from rl2.examples.temp_logger import Logger

import torch

buffer_kwargs = {
    'size': 1e6,
    'elements': {
        'obs': ((5, 9, 9), (8, ), np.float32),
        'action': ((6, ), np.float32),
        'reward': ((1, ), np.float32),
        'done': ((1, ), np.float32),
        'obs_': ((5, 9, 9), (8, ), np.float32)
    }
}


def obs_handler(obs, keys=['locational', 'additional']):
    if isinstance(obs, dict):
        loc, add = [obs[key] for key in keys]
    else:
        loc = []
        add = []
        for o in obs:
            loc.append(o[0]['locational'])
            add.append(o[0]['additional'])
        loc = np.stack(loc, axis=0)
        add = np.stack(add, axis=0)
    return loc, add


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='SACD Single agent')
    argparser.add_argument('--cuda_device', type=int, default=0)
    argparser.add_argument('--train_interval', type=int, default=1)
    argparser.add_argument('--update_interval', type=int, default=5)
    argparser.add_argument('--max_steps', type=int, default=1000)
    argparser.add_argument('--dir_name', type=str, default=None)
    argparser.add_argument('--empty_map', type=bool, default=False)
    argparser.add_argument('--rand_until', type=int, default=1028)
    argparser.add_argument('--eps', type=float, default=0.5)
    args = argparser.parse_args()

    args.device = 'cuda:{}'.format(args.cuda_device) if \
                  torch.cuda.is_available() else 'cpu'

    if not args.dir_name:
        dir_name = datetime.now().strftime('%Y%b%d_%H_%M_%S')
    else:
        dir_name = args.dir_name

    config = EasyDict({
        'gamma': 0.99,
        'n_agents': 2,
        'n_env': 1,
        'train_interval': args.train_interval,
        'train_after': args.rand_until,
        'update_interval': args.update_interval,
        'update_after': args.rand_until,
        'rand_until': args.rand_until,
        'save_interval': 1000,
        'batch_size': 32,
        'max_steps': args.max_steps,
        'device': args.device,
        'render': True,
        'render_interval': 10,
        'log_interval': 10,
        'eps': args.eps,  # for decaying epsilon greedy
        'log_dir': f'./SACDexp/train/log/{dir_name}',
        'save_dir': f'./SACDexp/train/ckpt/{dir_name}',
        'lr_ac': 1e-4,
        'lr_cr': 1e-4,
    })

    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    # save experiment conditions
    with open(config.log_dir+'/readme.txt', 'w') as f:
        f.write(json.dumps(config))
    logger = Logger(name='TestSACD', args=config, log_dir=config.log_dir)

    # set environment

    env_config = one_vs_one_env()
    if args.empty_map:
        env_config['env_kwargs']['num_rigid'] = 0
        env_config['env_kwargs']['num_wood'] = 0
        env_config['env_kwargs']['num_items'] = 0
    env_config['env_kwargs']['agent_view_size'] = 4
    env_config['env_kwargs']['max_step'] = args.max_steps
    env = ConservativeEnvWrapper(env_config)

    action_shape = env.action_space.n if hasattr(env.action_space,
                                                 'n') else env.action_space.shape
    observation_shape, additional_shape = env.observation_shape


    model = SACModelDISC(observation_shape, (action_shape, ),
                         discrete=True,
                         injection_shape=additional_shape,
                         preprocessor=obs_handler,
                         lr_ac=config.lr_ac,
                         lr_cr=config.lr_cr,
                         is_save=True,
                         device=config.device)

    # observation: tuple, action_shape: int
    trainee_agent = SACAgentDISC(model,
                                 batch_size=config.batch_size,
                                 train_interval=config.train_interval,
                                 train_after=config.train_after,
                                 update_interval=config.update_interval,
                                 update_after=config.update_after,
                                 render_interval=config.render_interval,
                                 save_interval=config.save_interval,
                                 buffer_cls=ReplayBuffer,
                                 buffer_kwargs=buffer_kwargs,
                                 save_dir=config.save_dir,
                                 eps = config.eps,
                                 rand_until=config.rand_until,
                                 # log_dir='sac_discrete/train/log',
                                 character=Bomber(0, env_config["game_type"]))

    agents = {
        0: trainee_agent,
        1: SimpleAgent(env_config['agent'](1, env_config["game_type"])),
    }
    env.set_agents(list(agents.values()))
    env.set_training_agents(0)
    env.seed(44)
    worker = SACDworker(env,
                        agents=[trainee_agent],
                        n_agents=config.n_agents,
                        n_env=config.n_env,
                        max_episodes=3e4,
                        training=True,
                        logger=logger,
                        log_interval=config.log_interval,
                        render=config.render,
                        render_interval=config.render_interval,
                        is_save= True,
                        random_until=config.rand_until)

    worker.run()
