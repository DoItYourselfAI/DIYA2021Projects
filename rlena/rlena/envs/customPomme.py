import os
import imageio
import numpy as np

from pathlib import Path
from typing import Dict, List
from pommerman.agents import BaseAgent
from pommerman.envs.v0 import Pomme
from pommerman.configs import team_competition_env

from gym import spaces


class CustomAgent(BaseAgent):
    def act(self, *args):
        pass


class CustomEnvWrapper(Pomme):
    def __init__(self, config) -> None:
        super().__init__(**config['env_kwargs'])
        self.seed(0)
        agents = {}
        for agent_id in range(4):
            agents[agent_id] = CustomAgent(
                config["agent"](agent_id, config["game_type"]))
        self.set_agents(list(agents.values()))
        self.set_init_game_state(None)
        view_range = 2 * self._agent_view_size + 1
        locational_shape = (5, view_range, view_range)
        additional_shape = (8,)
        self.observation_shape = (locational_shape, additional_shape)

    def reset(self):
        obs = super().reset()
        obs = self._preprocessing(obs)

        return obs

    def step(self, acs):
        obs, reward, done, info = super().step(acs)
        info['original_obs'] = obs
        obs = self._preprocessing(obs)

        return obs, reward, done, info

    def get_global_obs(self):
        obs = self.model.get_observations(curr_board=self._board,
                                          agents=self._agents,
                                          bombs=self._bombs,
                                          flames=self._flames,
                                          is_partially_observable=False,
                                          agent_view_size=self._agent_view_size,
                                          game_type=self._game_type,
                                          game_env=self._env)
        obs = self._preprocessing(obs)

        return obs

    def _preprocessing(self, obs: List[Dict], **kwargs) -> List[Dict]:
        out = []
        for d in obs:
            custom_obs = {}
            keys = ['alive', 'game_type', 'game_env']
            _ = list(map(d.pop, keys))  # remove useless obs

            # Change enums into int
            d.update({'teammate': d.get('teammate').value})
            enemies = list(map(lambda x: x.value, d.get('enemies')))
            enemies.remove(9)  # Remove dummy agent from enemies list
            d.update({'enemies': enemies})

            # Gather infos
            locational = []
            additional = []
            for k, v in d.items():
                if hasattr(v, 'shape'):
                    # Make border walls for locational obs
                    # obs['board'] borders are represented as 1(= Rigid wall)
                    # else borders are filled with 0 values.
                    if k != 'board':
                        v = np.insert(v, (0, v.shape[0]), 0, axis=0)
                        v = np.insert(v, (0, v.shape[1]), 0, axis=1)

                        if not kwargs.setdefault('global_obs', False):
                            for _ in range(self._agent_view_size - 1):
                                v = np.insert(v, (0, v.shape[0]), 0, axis=0)
                                v = np.insert(v, (0, v.shape[1]), 0, axis=1)

                    else:
                        v = np.insert(v, (0, v.shape[0]), 1, axis=0)
                        v = np.insert(v, (0, v.shape[1]), 1, axis=1)

                        if not kwargs.get('global_obs'):
                            for _ in range(self._agent_view_size - 1):
                                v = np.insert(v, (0, v.shape[0]), 1, axis=0)
                                v = np.insert(v, (0, v.shape[1]), 1, axis=1)

                    # Cut views by centering the agent for localized observation
                    if not kwargs.get('global_obs'):
                        pos = np.array(d.get('position'))
                        view_range = 2 * self._agent_view_size + 1
                        v = v[pos[0]:pos[0] + view_range,
                              pos[1]:pos[1] + view_range]
                    locational.append(v)

                    if k == 'board' and kwargs.get('onehot', False):
                        locational.pop()
                        # one hot vectorized board observation
                        for i in range(14):
                            onehot = np.asarray(v == i, dtype=np.int)
                            locational.append(onehot)

                else:
                    if hasattr(v, '__iter__'):
                        additional += v
                    else:
                        additional.append(v)

            custom_obs.update({
                'locational': np.stack(locational),
                'additional': np.array(additional, dtype='float64')
            })

            out.append(custom_obs)

        return out

class ConservativeEnvWrapper(CustomEnvWrapper):
    """
    A very similar wrapper to Eunki's CustomEnvWrapper. Made change in or added
      - Option for OneVsOne env(SAC training)
      - functionality for multiple training agents
      - maintain original observations for non-custom agents(SimpleAgent, RandomAgent)
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        if 'OneVsOne-v0' in config['env_id']:
            self.one_vs_one = True
        else:
            self.one_vs_one = False

        self._training_agents = []

    def set_training_agents(self, *args):
        self._training_agents = list(args)

    def act(self, obs):
        agents = [
            agent for agent in self._agents
            if agent.agent_id not in self._training_agents
        ]

        # the env model uses the agent's sepcific observation
        # indexed by agent_id
        actions = self.model.act(agents, obs, self.action_space)
        for i, action in enumerate(actions):
            if i in self._training_agents:
                actions.insert(i, None)
        return actions

    def _preprocessing(self, obs: List[Dict], **kwargs) -> List[Dict]:
        out = []
        for i, d in enumerate(obs):
            if i not in self._training_agents:
                out.append(d)
                continue
            custom_obs = {}
            keys = ['alive', 'game_type', 'game_env']
            _ = list(map(d.pop, keys))  # remove useless obs

            # Change enums into int
            d.update({'teammate': d.get('teammate').value})
            enemies = list(map(lambda x: x.value, d.get('enemies')))
            if not self.one_vs_one:
                enemies.remove(9)  # Remove dummy agent from enemies list
            d.update({'enemies': enemies})

            # Gather infos
            locational = []
            additional = []
            for k, v in d.items():
                if hasattr(v, 'shape'):
                    # Make border walls for locational obs
                    # obs['board'] borders are represented as 2(= Rigid wall)
                    # else borders are filled with 0 values.
                    if k != 'board':
                        for _ in range(self._agent_view_size):
                            v = np.insert(v, (0, v.shape[0]), 0, axis=0)
                            v = np.insert(v, (0, v.shape[1]), 0, axis=1)
                    else:
                        for _ in range(self._agent_view_size):
                            v = np.insert(v, (0, v.shape[0]), 1, axis=0)
                            v = np.insert(v, (0, v.shape[1]), 1, axis=1)

                    # Cut views by centering the agent for localized observation
                    if not kwargs.setdefault('global', False):
                        pos = np.array(d.get('position'))
                        view_range = 2 * self._agent_view_size + 1
                        v = v[pos[0]:pos[0] + view_range,
                              pos[1]:pos[1] + view_range]

                    locational.append(v)

                else:
                    if hasattr(v, '__iter__'):
                        additional += v
                    else:
                        additional.append(v)

            custom_obs.update({
                'locational':
                np.stack(locational),
                'additional':
                np.array(additional, dtype='float64')
            })

            out.append(custom_obs)

        return out


class TwoVsTwoPomme(CustomEnvWrapper):
    def __init__(self, **kwargs):
        config = team_competition_env()
        config['env_kwargs'].update({
            'max_steps': kwargs.get('max_steps'),
        })
        super().__init__(config)
        # Reorder outputs from env.reset & env.step
        self.order = np.array([0, 2, 1, 3])
        self.random_num_wall = kwargs.get('random_num_wall', True)
        if self.random_num_wall:
            self.max_rigid = kwargs.get('max_rigid', 18)
            self.max_wood = kwargs.get('max_wood', 8)
        self.remove_stop = int(kwargs.get('remove_stop', False))
        self.onehot = kwargs.get('onehot', False)
        if self.onehot:
            self.observation_shape = ((18, 9, 9), (8,))

    def step(self, acs):
        acs = acs.copy() + self.remove_stop
        obs, reward, done, info = Pomme.step(self, acs)
        info.update({'original_obs': obs,
                     'done': done})
        obs = self._preprocessing(obs, onehot=self.onehot)
        dones = list(map(lambda x: not x.is_alive, self._agents))

        obs, reward, dones = self._reorder(obs, reward, dones)
        dead_agents = np.where(self.old_dones != dones)[0]
        if len(dead_agents) > 0:
            reward = np.asarray(reward, dtype=np.float)
            reward[dead_agents] -= 0.5
            reward = list(reward)
        self.old_dones = np.asarray(dones)
        dones.append(done)
        obs.append(self.get_global_obs())

        if done:
            if self._step_count > self._max_steps:
                reward = list(np.asarray(reward, dtype=np.float) - 1.0)
            dones = [True] * 5
        return obs, reward, dones, info

    def _reorder(self, *args):
        out = []
        for a in args:
            a = list(np.array(a)[self.order])
            out.append(a)

        return out

    def get_global_obs_shape(self):
        g_obs = self.get_global_obs().values()
        self.g_obs_shape = [o.shape for o in g_obs]

        return self.g_obs_shape

    def get_global_obs(self):
        obs = self.model.get_observations(curr_board=self._board,
                                          agents=self._agents,
                                          bombs=self._bombs,
                                          flames=self._flames,
                                          is_partially_observable=False,
                                          agent_view_size=self._agent_view_size,
                                          game_type=self._game_type,
                                          game_env=self._env)

        obs = self._preprocessing(obs, global_obs=True, onehot=self.onehot)
        # locational obs only uses the first obs
        loc = obs[0].get('locational')
        # additional obs concatenate all values
        adds = list(map(lambda x: x.get('additional'), obs))
        add = np.concatenate(adds)

        out = {'locational': loc,
               'additional': add}

        return out

    def reset(self):
        # original = 36,36
        # wood min = 20; probably due to the num_items
        if self.random_num_wall:
            self._num_rigid = np.random.randint(0, self.max_rigid) * 2
            self._num_wood = 20 + (np.random.randint(0, self.max_wood) * 2)
        obs = Pomme.reset(self)
        obs = self._preprocessing(obs, onehot=self.onehot)
        obs.append(self.get_global_obs())

        self.old_dones = np.asarray([False] * 4)

        return obs

    def render(self, *args, **kwargs):
        if args[0] == 'human':
            path = './tmp/'
            Path(path).mkdir(parents=True, exist_ok=True)
            Pomme.render(self, record_pngs_dir=path, *args, **kwargs)
            filename = os.listdir(path)[-1]
            rgb_array = imageio.imread(
                os.path.join(path, filename), pilmode='RGB')
            os.remove(os.path.join(path, filename))
            Path(path).rmdir()

            return rgb_array
        else:
            return Pomme.render(self, *args, **kwargs)



class SkynetEnvWrapper(Pomme):
    """
    One-hot coded feature env
    """

    def __init__(self, config, agent_list) -> None:
        super().__init__(**config['env_kwargs'])
        self.seed(0)
        self.config = config
        agents, self.isCustum = [], []

        for id_, (custom,agent) in enumerate(agent_list):
            # NOTE: This is IMPORTANT so that the agent character is initialized
            self.isCustum.append(custom)
            agent.init_agent(id_, self.config['game_type'])
            agents.append(agent)
        
        self.set_agents(agents)
        self.set_init_game_state(None)

        obs_shape = (18, self._board_size, self._board_size)
        self.observation_space = spaces.Box(low = 0, high = np.inf, shape = obs_shape)

        self.global_state = None
        self._pos_queue = [[] for _ in range(len(agent_list))]

    def reset(self):
        self.global_state = super().reset()
        obs = self._preprocessing(self.global_state)
        self._pos_queue = [[] for _ in range(len(self.isCustum))]

        return obs

    def step(self, acs):
        state, reward, done, info = super().step(acs)
        reward = self._custum_get_rewards(state, self.global_state)
        obs = self._preprocessing(state)   
        self.global_state = state
        self._pos_queue_add()

        return obs, reward, done, info

    def _pos_queue_add(self,):
        """Reward에 사용하기 위해 history position 기록"""
        for i, state in enumerate(self.global_state):
            self._pos_queue[i].append(state['position'])
            while len(self._pos_queue[i]) > self._board_size * self._board_size:
                self._pos_queue[i].pop(0)

    def _custum_get_rewards(self, new_states, old_states):
        original_rewards = super()._get_rewards()
        rewards = []
        for i, original_reward in enumerate(original_rewards):
            rewards.append(self._get_reward_helper(new_states[i], 
                                                   old_states[i], 
                                                   self._pos_queue[i], 
                                                   original_reward,
                                                   i+10))
        return rewards

    def _get_reward_helper(self, new_state, old_state, pos_queue, original_reward, agent_id):
        reward = 0

        # 1. get kick --> 0.02 pts
        reward += (new_state['can_kick'] - old_state['can_kick']) * 0.02 
        # 2. get ammo --> 0.01 pts
        reward += (new_state['ammo'] - old_state['ammo']) * 0.01
        # 3. get blast length --> 0.01 pts
        reward += (new_state['blast_strength'] - old_state['blast_strength']) * 0.01
        # 4. Exploration --> 0.001
        reward += 0.001 if not new_state['position'] in pos_queue else 0
        # 5. dead agent in winning team gets 0.5
        if original_reward == 1:
            reward += 1 if agent_id in new_state['alive'] else 0.5
        # 6. draw gives 0
        if original_reward == -1:
            reward += 0 if agent_id in new_state['alive'] else -1
        # alive change
        teammate = new_state.get('teammate').value # change enum to dict
        enemies = [enemy.value for enemy in new_state['enemies']] # change enum to dict
        try: 
            enemies.remove(9)  # Remove dummy agent from enemies list
        except:
            pass
        for agent_id in old_state['alive']:
            if not agent_id in new_state['alive']:
                # 7. teammate death -0.5
                if agent_id == teammate:
                    reward += -0.5
                # 8. enemy death  0.5
                elif agent_id in enemies:
                    reward += 0.5
    
        return reward

    def get_global_obs(self):
        obs = []
        for i, state in enumerate(self.global_state):
            state = self._preprocessing_helper(state, agent_id=i+10)
            obs.append(state[:8]) # only personal obs
        obs.append(state[8:]) # common obs
        
        return np.concatenate(obs,axis=0) # shape[0] == 42

    def _preprocessing(self, states: List[Dict], **kwargs) -> List[Dict]:
        obs = []
        for i, state in enumerate(states):
            if self.isCustum[i]:
                obs.append(self._preprocessing_helper(state, agent_id=i+10))
            else:
                obs.append(state)
        return obs

    def _preprocessing_helper(self, state: Dict, agent_id, **kwargs) -> Dict:
        d = state.copy()
        obs = []
        map_shape = (self._board_size, self._board_size)

        # 1~7 : observation of each agent
        # 1. ego location
        alive = agent_id in d['alive']
        loc = np.zeros(shape=map_shape)
        pos = d['position']
        loc[pos[0], pos[1]] = 1 if alive else 0
        obs.append(loc)
        # 2. number of ammo
        ammo = np.ones(map_shape) * d['ammo']
        obs.append(ammo)
        # 3. blast strength
        blast_strength = np.ones(map_shape) * d['blast_strength']
        obs.append(blast_strength)
        # 4. can kick or not
        kick = np.ones(map_shape) if d['can_kick'] else np.zeros(map_shape)
        obs.append(kick)
        # 5. teamate or not
        d.update({'teammate': d.get('teammate').value}) # change enum to dict
        single = 9 == d['teammate'] # if it has 9, its single game
        issingle = np.zeros(map_shape) if single else np.ones(map_shape)
        obs.append(issingle)
        # 6. teammate location
        team_loc = np.zeros(map_shape)
        if not single:
            team_id = d['teammate']-10
            team_pos = self.global_state[team_id]['position']
            team_loc[team_pos[0],team_pos[1]] = 1 if team_id in d['alive'] else 0
        obs.append(team_loc)
        # 7. enemy location
        enemies = [enemy.value-10 for enemy in d['enemies']] # change enum to dict
        try: 
            enemies.remove(-1)  # Remove dummy agent from enemies list
        except:
            pass
        enemy_loc = np.zeros(map_shape)
        for enemy_id in enemies:
            enemy_pos = self.global_state[enemy_id]['position']
            enemy_loc[enemy_pos[0],enemy_pos[1]] = 1 if enemy_id in d['alive'] else 0
        obs.append(enemy_loc)

        # 8~18 : Common observation
        # 8. blast strength
        obs.append(d['bomb_blast_strength']) 
        # 9. bomb life
        obs.append(d['bomb_life'])
        # 10. passage
        board = d['board']
        passage = (board == 0).astype(int)
        obs.append(passage)
        # 11. rigid wall
        rigid_wall = (board == 1).astype(int)
        obs.append(rigid_wall)
        # 12. wodden walls
        wooden_wall = (board == 2).astype(int)
        obs.append(wooden_wall)
        # 13. bomb
        bomb = (board == 3).astype(int)
        obs.append(bomb)
        # 14. flames
        flames = (board == 4).astype(int)
        obs.append(flames)
        # 15. extra bomb power up
        extra_bomb_power_up = (board == 6).astype(int)
        obs.append(extra_bomb_power_up)
        # 16. blast strngth power up
        blast_strength_power_up = (board == 7).astype(int)
        obs.append(blast_strength_power_up)
        # 17. kicking ability power up
        kick_power_up = (board == 8).astype(int)
        obs.append(kick_power_up)
        # 18. total time step
        time_step = np.ones(map_shape) * (self._step_count / self._max_steps)
        obs.append(time_step)

        return np.stack(obs, axis=0)