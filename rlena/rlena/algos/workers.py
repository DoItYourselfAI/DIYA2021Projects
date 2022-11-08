from collections import deque
from typing import Iterable
from rl2.workers.base import EpisodicWorker

import numpy as np
from tqdm import trange


class ComaWorker(EpisodicWorker):
    def __init__(self,
                 env,
                 n_env,
                 agent,
                 max_episodes: int,
                 log_interval: int,
                 logger,
                 **kwargs):
        super().__init__(env, n_env, agent, max_episodes=max_episodes,
                         log_interval=log_interval, logger=logger, **kwargs)
        self.done = False
        self.scores = deque(maxlen=log_interval)
        self.winner = deque(maxlen=log_interval)
        self.save_gif = kwargs.get('save_gif', False)

    def rollout(self):
        ac = self.agent.act(self.obs)
        if len(ac.shape) == 2 and ac.shape[0] == 1:
            ac = ac.squeeze(0)
        obs, rew, done, info = self.env.step(ac)
        self.done = done[-1]
        if self.training:
            info_a = self.agent.step(self.obs, ac, rew, done, obs)
            if info:
                if isinstance(info, dict):
                    info = {**info, **info_a}
                elif isinstance(info, Iterable):
                    info = {**info_a}
            else:
                info = {**info_a}
        self.num_steps += self.n_env
        self.episode_score = self.episode_score + np.array(rew)
        if not self.training:
            done.pop()
        steps = ~np.asarray(done)
        self.ep_steps += steps.astype(np.int)

        results = None
        if self.done:
            self.num_episodes += 1
            obs = self.env.reset()
            self.scores.append(self.episode_score)
            self.episode_score = np.zeros_like(self.episode_score, np.float)
            self.ep_length.append(self.ep_steps)
            self.ep_steps = np.zeros_like(self.ep_steps)
            self.winner.append(info.get('winners', [-1])[0])
        self.obs = obs

        return self.done, info, results

    def run(self):
        while self.num_episodes < self.max_episodes:
            prev_num_ep = self.num_episodes
            done, info_r, _ = self.rollout()

            if info_r.get('is_loss', False):
                rm_keys = ['result', 'winners', 'original_obs',
                           'done', 'is_loss']
                for key in rm_keys:
                    info_r.pop(key, None)
                self.info.update(info_r)

            if self.render and self.start_log_image:
                image = self.env.render(self.render_mode)
                self.logger.store_rgb(image)

            log_cond = done if np.asarray(done).size == 1 else any(done)
            if log_cond:
                if self.start_log_image:
                    print('save video')
                    self.logger.video_summary(tag='playback',
                                              step=self.num_steps,
                                              save_gif=self.save_gif)
                    self.start_log_image = False
                if self.render:
                    if (prev_num_ep // self.render_interval !=
                       self.num_episodes // self.render_interval):
                        self.start_log_image = True

                if (prev_num_ep // self.log_interval != self.num_episodes // self.log_interval):
                    ep_lengths = np.asarray(self.ep_length)
                    results = np.asarray(self.winner)
                    winrate = results[np.where(
                        results != -1)].sum()/len(results)
                    tierate = results[np.where(
                        results == -1)].sum()/len(results)
                    info = {
                        'Counts/num_steps': self.num_steps,
                        'Counts/num_episodes': self.num_episodes,
                        'Counts/eps': self.agent.eps(self.agent.curr_ep),
                        'Episodic/ep_length': ep_lengths.max(-1).mean(),
                        'Episodic/avg_winrate': winrate,
                        'Episodic/avg_tierate': np.abs(tierate),
                    }
                    ep_lengths = np.split(ep_lengths, 2, axis=-1)
                    scores = np.split(np.asarray(self.scores), 2, axis=-1)
                    for team, score in enumerate(scores):
                        info[f'team{team}/rews_avg'] = np.mean(score)
                        for i, ep_len in enumerate(ep_lengths[team].mean(0)):
                            info[f'team{team}/agent{i}_ep_length'] = ep_len

                    self.info.update(info)

                    self.logger.scalar_summary(self.info, self.num_steps)


class QmixWorker:
    def __init__(self,
                 env,
                 agent,
                 critic,
                 config,
                 logger):
        self.done = True
        self.env = env
        self.agent1 = agent[0]
        self.agent2 = agent[1]
        self.critic = critic

        self.config = config
        self.logger = logger

        self.init_config()

    def init_config(self):
        if self.config['load_model']:
            self.agent1.load(1)
            self.agent2.load(2)
        if self.config['mode'] == 'train':
            self.agent1.train()
            self.agent2.train()
        else:
            self.agent1.eval()
            self.agent2.eval()

    def rollout(self):
        actions = self.env.act(self.state)
        global_state = self.env.get_global_obs()
        gru_hidden = [self.agent1.gru_hidden.detach().cpu().numpy(), self.agent2.gru_hidden.detach().cpu().numpy()]
        state_prime, reward, self.done, info = self.env.step(actions)

        # add data in memory
        if self.config['mode'] == "train":
            self.critic.mem_append([self.state, 
                            gru_hidden, 
                            global_state, 
                            actions, 
                            reward, 
                            self.done, 
                            state_prime])
        
        self.state = state_prime

        return reward

    def run(self):
        episode, r_episode_1, r_episode_2 = 0, 0, 0
        loss = 0
        for step in trange(int(self.config['max_step'])):
            
            # Episode done
            if self.done:
                episode += 1
                self.state= self.env.reset()

                if self.config['mode'] != 'train':
                    print("{} - episode done".format(episode-1))
                    if self.config['max_episode'] < episode:
                        return
                
                if episode % self.config['tensorboard_frequency'] == 0:
                    # reward writing
                    self.logger.scalar_summary({"agent1 reward" :r_episode_1 / self.config['tensorboard_frequency'],
                                        "agent2 reward" :r_episode_2 / self.config['tensorboard_frequency']}, 
                                    step)
                    r_episode_1 = 0
                    r_episode_2 = 0

                    # model save
                    if self.config['mode'] == 'train':
                        self.agent1.save(1)
                        self.agent2.save(2)
                        self.critic.save()
                    
                        

            if self.config['render']:
                self.env.render()
            
            reward = self.rollout()

            # reward summation
            r_episode_1 += reward[0]
            r_episode_2 += reward[2]

            if (self.config['mode']=='train') and (self.critic.memory.size() >= self.config['learn_threshold']):
                if step % self.config['target_frequency'] == 0 :
                    self.critic.target_update()
                if step % self.config['learn_frequency'] == 0:
                    loss = self.critic.learn()
                    self.logger.scalar_summary({"critict loss" :loss.detach().item()}, step)
                
                # epsilon greedy decaying
                self.agent1.e_decay()
                self.agent2.e_decay()
                
        self.env.close()


# Worker for sac_discrete agetn
class SACDworker(EpisodicWorker):
    def __init__(self, env, n_env, agents, n_agents, max_episodes: int,
                 log_interval: int, logger, render, render_interval, random_until, **kwargs):
        super().__init__(env,
                         n_env,
                         agents,
                         max_episodes=max_episodes,
                         log_interval=log_interval,
                         logger=logger,
                         **kwargs)
        self.render = render
        self.render_interval = render_interval
        self.done = False
        self.winner = deque(maxlen=10)
        self.trainee_agents = agents
        self.env = env
        self.obs = self.env.reset()
        self.random_until = random_until
        self.agents = agents
        self.n_agents = n_agents
        self.render_mode = 'rgb_array'

    def rollout(self):
        action_dist = [None] * len(self.env._agents)
        actions = self.env.act(self.obs)  # (None:trainee_agents, action: simple_agent)

        for agent in self.trainee_agents:
            i = agent.agent_id
            actions[i] = agent.act(self.obs[i])
            #print('actions: {}'.format(actions))

            # the buffer should save the action discributions
            dist, _, _ = agent.model(self.obs[i])
            action_dist[i] = dist.probs.detach()[0].cpu().numpy()
            #print(i, dist.probs)

        obs_, rewards, done, info = self.env.step(actions)
        #if rewards[0] == 0:
        #    rewards[0] = 0.3 
        #if rewards[0] == -1:
        #    rewards[0] = -5 
        self.done = done if np.asarray(done).size == 1 else any(done)

        if self.training:
            for agent in self.trainee_agents:
                i = agent.agent_id
                info_a = agent.step(self.obs[i], action_dist[i], rewards[i],
                                    done, obs_[i])
                for k, v in info_a.items():
                    if isinstance(v, tuple):
                        info[k] = sum(info_a[k])
                    else:
                        info[k] = info_a[k]
            self.num_steps += self.n_env
            #self.logger.scalar_summary(info_a, self.num_steps)
            self.episode_score = self.episode_score + np.array(rewards)
            steps = ~np.asarray(done)
            self.ep_steps += steps.astype(np.int)

        if self.done:
            # if rewards[0] ==1:
            #    print('win')
            # else:
            #    print('loose')
            self.num_episodes += 1
            obs = self.env.reset()
            self.scores.append(self.episode_score)
            self.episode_score = np.zeros_like(self.episode_score, np.float)
            self.ep_length.append(self.ep_steps)
            self.ep_steps = np.zeros_like(self.ep_steps)
            self.winner.append(info.pop('result').value)
        self.obs = obs_
        results = None
        return self.done, info, results

    def run(self):
        while self.num_episodes < self.max_episodes:
            prev_num_ep = self.num_episodes
            done, info_r, results = self.rollout()

            if self.render and self.start_log_image:
                image = self.env.render(self.render_mode)
                self.logger.store_rgb(image)

            log_cond = done if np.asarray(done).size == 1 else any(done)
            if log_cond:
                if self.start_log_image:
                    print('save_video')
                    self.logger.video_summary(tag='playback',
                                              step=self.num_steps)
                    self.start_log_image = False
                if self.render:
                    if (prev_num_ep // self.render_interval !=
                            self.num_episodes // self.render_interval):
                        self.start_log_image = True
                ep_lengths = np.asarray(self.ep_length)

                info = {
                    'Counts/num_steps': self.num_steps,
                    'Counts/num_episodes': self.num_episodes,
                    'Episodic/ep_length': ep_lengths.max(-1).mean(),
                    'Episodic/avg_winrate': np.asarray(self.winner).mean()
                }
                if self.n_agents == 4:
                    ep_lengths = np.split(ep_lengths, 2, axis=-1)
                scores = np.split(np.asarray(self.scores), 2, axis=-1)

                for team, score in enumerate(scores):
                    info[f'team{team}/rews_avg'] = np.mean(score)
                    if self.n_agents == 4:
                        for i, ep_len in enumerate(ep_lengths[team].mean(0)):
                            info[f'team{team}/agent{i}_ep_length'] = ep_len
                    else:
                        info[
                            f'team{team}/agent{team}_ep_length'] = ep_lengths.mean(
                            )

                self.info.update(info)

                if info_r.get('entropies', False):
                    rm_keys = ['winners', 'original_obs', 'done', 'is_loss']
                    for key in rm_keys:
                        info_r.pop(key, None)
                    self.info.update(info_r)
                if (prev_num_ep // self.log_interval) != (self.num_episodes //
                                                          self.log_interval):
                    self.logger.scalar_summary(self.info, self.num_steps)
