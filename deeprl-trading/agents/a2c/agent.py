import time
from utils.summary import EvaluationMetrics
import envs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from agents.base import Agent
from agents.a2c.model import CNNActorCritic
from agents.a2c.model import TransformerActorCritic


class A2C(Agent):
    def __init__(self, args=None, name='A2C'):
        super().__init__(args=args, name=name)
        # initialize models
        if args.arch == 'cnn':
            self.model = CNNActorCritic(
                self.env.observation_space,
                self.env.action_space,
                min_var=args.sigma
            )
        elif args.arch == 'transformer':
            self.model = TransformerActorCritic(
                self.env.observation_space,
                self.env.action_space,
                min_var=args.sigma
            )
        else:
            raise NotImplementedError

        # load checkpoint if available
        if args.checkpoint is not None:
            self.logger.log(
                "Loading model checkpoint from {}".format(args.checkpoint))
            self.model.load_state_dict(torch.load(args.checkpoint))
        self.model.to(args.device)

        # initialize buffer
        self.keys = ['states', 'actions', 'values', 'nlls', 'rewards', 'dones']
        self.buffer = {k: [] for k in self.keys}
        self.done = False

        # intialize optimizers
        self.optim = optim.RMSprop(
            self.model.parameters(),
            lr=args.lr_actor
        )

        # initialize statistics
        self.info = EvaluationMetrics([
            'Time/Step',
            'Loss/Critic',
            'Loss/Actor',
            'Values/Reward',
            'Values/Value',
            'Values/Entropy',
            'Scores/Train',
            'Scores/Val',
        ])
        self.step = 0
        
        # Current eval score for checkpointing max score
        self.eval_score =  0.

    def compute_loss(self, idx):
        states = self.buffer['states'][idx]
        actions = self.buffer['actions'][idx]

        # compute log probability and entropy
        mus, sigs, values = self.model(states)
        self.info.update('Values/Value', values.mean().item())
        dists = Normal(mus, sigs)
        nlls = -dists.log_prob(actions).sum(dim=-1)
        entropy = dists.entropy().sum(dim=-1).mean()
        self.info.update('Values/Entropy', entropy.item())

        # critic loss
        advs = self.buffer['advs'][idx]
        returns = self.buffer['values'][idx] + advs
        loss_critic = (returns.detach() - values).pow(2).mean()
        self.info.update('Loss/Critic', loss_critic.item())

        # actor loss
        loss_actor = (advs.detach() * nlls.unsqueeze(1)).mean()
        self.info.update('Loss/Actor', loss_actor.item())

        # total loss with entropy bonus
        loss = loss_actor + self.args.cr_coef * loss_critic
        loss -= self.args.ent_coef * entropy
        return loss

    def train(self):
        st = time.time()
        self.step += 1

        self.model.train()
        self.env.train()

        # collect transition
        self.state = torch.FloatTensor(self.state).to(self.args.device)
        with torch.no_grad():
            mu, sig, value = self.model(self.state.unsqueeze(0))
            dist = Normal(mu, sig)
            action = dist.sample()
            nll = -dist.log_prob(action).sum(dim=-1)
        s, a, v, nll = map(
            lambda x: x.squeeze(0).cpu().numpy(),
            [self.state, action, value, nll]
        )
        self.buffer['states'].append(s)
        self.buffer['actions'].append(a)
        self.buffer['values'].append(v)
        self.buffer['nlls'].append(nll)
        self.buffer['dones'].append(np.asarray([self.done], dtype=float))

        a = np.tanh(a)
        self.state, reward, self.done, epinfo = self.env.step(a)
        self.buffer['rewards'].append(reward)
        self.info.update('Values/Reward', reward)
        if self.done:
            self.info.update('Scores/Train', epinfo['profit'])

        # compute Generalized Advantage Estimate
        if self.step % self.args.update_every == 0:
            for k, v in self.buffer.items():
                v = torch.from_numpy(np.asarray(v))
                self.buffer[k] = v.float().to(self.args.device)

            gae = 0.0
            gam = self.args.gamma
            lam = getattr(self.args, 'lambda')
            advs = torch.zeros_like(self.buffer['values'])

            # check latest value
            with torch.no_grad():
                state = torch.FloatTensor(self.state).to(self.args.device)
                _, _, value = self.model(state.unsqueeze(0))
            done = torch.FloatTensor([self.done]).to(self.args.device)

            # reversed update
            for t in reversed(range(self.args.update_every)):
                if t == self.args.update_every - 1:
                    _value = value.squeeze(0)
                    _nonterminal = 1.0 - done
                else:
                    _value = self.buffer['values'][t + 1]
                    _nonterminal = 1.0 - self.buffer['dones'][t + 1]

                value = self.buffer['values'][t]
                reward = self.buffer['rewards'][t]
                delta = reward + _nonterminal * gam * _value - value
                gae = delta + _nonterminal * gam * lam * gae
                advs[t] = gae
            self.buffer['advs'] = advs

            # update actor critic
            for _ in range(self.args.update_epoch):
                # shuffle batch
                idx = np.arange(len(advs))
                np.random.shuffle(idx)

                # train from batch
                start = 0
                batch_size = self.args.batch_size
                for _ in range(len(advs) // batch_size):
                    end = start + batch_size
                    _idx = idx[start:end]
                    start = end

                    loss = self.compute_loss(_idx)
                    self.optim.zero_grad()
                    loss.backward()
                    if self.args.grad_clip is not None:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.args.grad_clip
                        )
                    self.optim.step()

            # clear buffer
            self.buffer = {k: [] for k in self.keys}

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
            mu, _, _ = self.model(state.unsqueeze(0))
            action = torch.tanh(mu)
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
            mu, _, _ = self.model(state.unsqueeze(0))
            action = torch.tanh(mu)
            action = action.squeeze(0).cpu().numpy()
            state, _, done, epinfo = env.step(action)
            pnl.append(epinfo['profit'])
            returns.append(epinfo['return'])

        # log test result
        self.logger.log("Test run complete")
        self.logger.log("PnL: {}".format(epinfo['profit']))
        return pnl, returns
