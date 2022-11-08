import time
from copy import deepcopy
from utils.summary import EvaluationMetrics
import envs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.base import Agent
from agents.ddpg.buffer import ReplayBuffer
from agents.ddpg.model import CNNActor, CNNCritic
from agents.ddpg.model import TransformerActor, TransformerCritic


class OrnsteinUhlenbeck:
    def __init__(self, action_space, sigma=0.1, theta=0.15, dt=1e-2, device=None):
        self._mu = torch.zeros(*action_space, device=device)
        self._sigma = torch.ones(*action_space, device=device) * sigma
        self._theta = theta
        self._dt = dt

        self.eps = None
        self.reset()

    def sample(self):
        eps = self.eps
        eps += self._theta * (self._mu - self.eps) * self._dt
        eps += self._sigma * np.sqrt(self._dt) * torch.randn_like(self._mu)
        self.eps = eps
        return eps

    def reset(self):
        self.eps = torch.zeros_like(self._mu)


class DDPG(Agent):
    def __init__(self, args=None, name='DDPG'):
        super().__init__(args=args, name=name)
        # initialize models
        if args.arch == 'cnn':
            self.model = nn.ModuleDict({
                'actor': CNNActor(
                    self.env.observation_space,
                    self.env.action_space,
                ),
                'critic': CNNCritic(
                    self.env.observation_space,
                    self.env.action_space,
                )
            })
        elif args.arch == 'transformer':
            self.model = nn.ModuleDict({
                'actor': TransformerActor(
                    self.env.observation_space,
                    self.env.action_space,
                ),
                'critic': TransformerCritic(
                    self.env.observation_space,
                    self.env.action_space,
                )
            })
        else:
            raise NotImplementedError

        # load checkpoint if available
        if args.checkpoint is not None:
            self.logger.log(
                "Loading model checkpoint from {}".format(args.checkpoint))
            self.model.load_state_dict(torch.load(args.checkpoint))
        self.target = deepcopy(self.model)
        self.model.to(args.device)
        self.target.to(args.device)

        # initialize random process for action noise
        self.action_noise = OrnsteinUhlenbeck(
            self.env.action_space,
            sigma=args.sigma,
            device=args.device
        )

        # initialize replay buffer
        self.buffer = ReplayBuffer(
            self.env.observation_space,
            self.env.action_space,
            size=args.buffer_size
        )

        # intialize optimizers
        self.actor_optim = optim.RAdam(
            self.model.actor.parameters(),
            lr=args.lr_actor
        )
        self.critic_optim = optim.RAdam(
            self.model.critic.parameters(),
            lr=args.lr_critic
        )

        # initialize statistics
        self.info = EvaluationMetrics([
            'Time/Step',
            'Loss/Critic',
            'Loss/Actor',
            'Values/Reward',
            'Values/QValue',
            'Scores/Train',
            'Scores/Val',
        ])
        self.step = 0

        # Current eval score for checkpointing max score
        self.eval_score =  0.

    def update_target(self):
        tau = self.args.polyak
        # update actor params
        actor = self.model.actor.parameters()
        target_actor = self.target.actor.parameters()
        for param, target_param in zip(actor, target_actor):
            target_param.data.copy_(
                tau * target_param.data + (1 - tau) * param.data
            )
        # update critic params
        critic = self.model.critic.parameters()
        target_critic = self.target.critic.parameters()
        for param, target_param in zip(critic, target_critic):
            target_param.data.copy_(
                tau * target_param.data + (1 - tau) * param.data
            )

    def train(self):
        st = time.time()
        self.step += 1

        self.model.train()
        self.target.eval()
        self.env.train()

        # collect transition
        self.state = torch.FloatTensor(self.state).to(self.args.device)
        with torch.no_grad():
            action = self.model.actor(self.state.unsqueeze(0))
            action = torch.tanh(action + self.action_noise.sample())
        action = action.squeeze(0).cpu().numpy()
        state, reward, done, epinfo = self.env.step(action)

        self.state = self.state.squeeze(0).cpu().numpy()
        self.buffer.push(self.state, action, reward, done, state)
        self.state = state

        self.info.update('Values/Reward', reward)
        if done:
            self.info.update('Scores/Train', epinfo['profit'])

        # check if warming up
        if self.step >= self.args.warmup:
            # sample from replay buffer
            s, a, r, d, s_next = tuple(
                map(lambda x: torch.FloatTensor(x).to(self.args.device), 
                    self.buffer.sample(self.args.batch_size))
            )
            r = r.unsqueeze(1)
            d = d.unsqueeze(1)

            # update critic
            with torch.no_grad():
                a_next = torch.tanh(self.target.actor(s_next))
                q_next = self.target.critic(s_next, a_next)
            q_trg = r + self.args.gamma * q_next * (1 - d)
            loss_critic = (q_trg - self.model.critic(s, a)).pow(2).mean()
            self.info.update('Loss/Critic', loss_critic.item())

            self.critic_optim.zero_grad()
            loss_critic.backward()
            if self.args.grad_clip is not None:
                nn.utils.clip_grad_norm_(
                    self.model.critic.parameters(),
                    self.args.grad_clip
                )
            self.critic_optim.step()

            # update actor
            a = torch.tanh(self.model.actor(s))
            q_val = self.model.critic(s, a).mean()
            loss_actor = -q_val
            self.info.update('Values/QValue', q_val.item())
            self.info.update('Loss/Actor', loss_actor.item())

            self.actor_optim.zero_grad()
            loss_actor.backward()
            if self.args.grad_clip is not None:
                nn.utils.clip_grad_norm_(
                    self.model.actor.parameters(),
                    self.args.grad_clip
                )
            self.actor_optim.step()

            # update target network
            self.update_target()

        # log training statistics
        elapsed = time.time() - st
        self.info.update('Time/Step', elapsed)
        if self.step % self.args.log_step == 0:
            self.logger.log("Training statistics for step {}".format(self.step))
            self.logger.scalar_summary(self.info.avg, self.step)
            self.info.reset()

    @torch.no_grad()
    def eval(self):
        self.model.eval()

        # create new environment
        env = getattr(envs, self.args.env)(args=self.args)
        env.eval()
        state = env.reset()

        # run until terminal
        done = False
        while not done:
            state = torch.FloatTensor(state).to(self.args.device)
            action = torch.tanh(self.model.actor(state.unsqueeze(0)))
            action = action.squeeze(0).cpu().numpy()
            state, _, done, epinfo = env.step(action)
        self.info.update('Scores/Val', epinfo['profit'])
        self.eval_score = epinfo['profit']

    @torch.no_grad()
    def test(self):
        self.logger.log("Begin test run from {}".format(self.args.start_test))
        self.model.eval()

        # create new environment
        env = getattr(envs, self.args.env)(args=self.args)
        env.test()
        state = env.reset()

        # run until terminal
        done = False
        pnl = []
        returns = []
        while not done:
            state = torch.FloatTensor(state).to(self.args.device)
            action = torch.tanh(self.model.actor(state.unsqueeze(0)))
            action = action.squeeze(0).cpu().numpy()
            state, _, done, epinfo = env.step(action)
            pnl.append(epinfo['profit'])
            returns.append(epinfo['return'])

        # log test result
        self.logger.log("Test run complete")
        self.logger.log("PnL: {}".format(epinfo['profit']))
        return pnl, returns
