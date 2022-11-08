from typing import Dict, Iterable, Callable
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import random

from rlena.algos.utils import IBMWithNormalization

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from rl2.models.torch.base import BaseEncoder, TorchModel, InjectiveBranchModel
from rl2.networks.torch.networks import ConvEnc, MLP
from rl2.distributions.torch.distributions import CategoricalHead
from rl2.agents.base import Agent
from rl2.buffers.base import ReplayBuffer

from pommerman.agents.base_agent import BaseAgent

__all__ = ['SACAgentDISC', 'SACModelDISC']

training_cnt = 0


# batch of states, actions, rewards, next_states
def loss_func_ac(data, model, **kwargs):
    obs, _, _, _, _ = data

    # (Log of) probabilities to calculate expectations of
    # Q and action entropies.
    act_dist, log_act_probs, _ = model(obs)

    loc, add = model.preprocessor(obs)
    loc, add = tuple(
        map(lambda x: T.from_numpy(x).float().to(model.device), [loc, add]))
    with T.no_grad():
        q1 = model.q1(loc, add)
        q2 = model.q2(loc, add)
        q = T.min(q1, q2)

    # Expectation of entropies
    entropies = -T.sum(act_dist.probs * log_act_probs, dim=1, keepdim=True)

    # Expecdtation of q values
    q = T.sum(act_dist.probs * q, dim=1, keepdim=True)

    #pi_loss = model.alpha * (-entropies) - q
    #pi_loss[pi_loss == 0.] = model.eps

    pi_loss = (model.alpha * (-entropies) - q).mean()
    #if not pi_loss:
    #    pi_loss = model.eps

    return pi_loss, entropies.mean().detach()


def loss_func_cr(data, model, **kwargs):
    obs, actions, rewards, dones, obs_ = data

    data = [rewards, dones]
    rewards, dones = tuple(
        map(lambda x: T.from_numpy(x).float().to(model.device), data))

    next_act_dist, next_log_act_probs, _ = model(obs_)

    next_loc, next_add = model.preprocessor(obs_)
    next_loc, next_add = map(
        lambda x: T.from_numpy(x).float().to(model.device),
        [next_loc, next_add])

    next_q1 = model.q1.forward_trg(next_loc, next_add)
    next_q2 = model.q2.forward_trg(next_loc, next_add)

    next_state_val = (
        next_act_dist.probs *
        (T.min(next_q1, next_q2) - model.alpha * next_log_act_probs)).sum(
            dim=1, keepdim=True)
    soft_q_func = rewards + (1.0 - dones) * model.gamma * next_state_val

    loc, add = model.preprocessor(obs)
    loc, add = tuple(
        map(lambda x: T.from_numpy(x).float().to(model.device), [loc, add]))

    curr_q1 = model.q1(loc, add)
    curr_q2 = model.q2(loc, add)

    #soft_bellman_residual1 = curr_q1 - soft_q_func
    #soft_bellman_residual2 = curr_q2 - soft_q_func

    #soft_bellman_residual1[soft_bellman_residual1 == 0.] = model.eps
    #soft_bellman_residual2[soft_bellman_residual2 == 0.] = model.eps
    q1_loss = F.mse_loss(curr_q1, soft_q_func)
    q2_loss = F.mse_loss(curr_q2, soft_q_func)

    #q1_loss = (.5 * soft_bellman_residual1**2).mean()
    #q2_loss = (.5 * soft_bellman_residual2**2).mean()

    return q1_loss, q2_loss, curr_q1.mean().detach(), curr_q2.mean().detach()


def loss_func_alpha(entropies, model, **kwargs):
    l_alpha = (model.log_alpha * (model.target_entropy - entropies)).mean()
    return l_alpha


class SACModelDISC(TorchModel):
    def __init__(
            self,
            obs_shape,
            action_shape,
            actor: T.nn.Module = None,
            critic: T.nn.Module = None,
            encoder: T.nn.Module = None,
            encoded_dim: int = 64,
            head: T.nn.Module = None,
            optim_ac: str = 'torch.optim.Adam',
            optim_cr: str = 'torch.optim.Adam',
            gamma: float = 0.99,
            lr_ac: float = 1e-4,
            lr_cr: float = 1e-4,
            grad_clip: float = 1e-2,
            polyak: float = 0.995,
            discrete: bool = True,
            deterministic: bool = False,
            flatten: bool = False,  # True if you don't need CNN in the encoder
            reorder: bool = False,  # Flag for (C, H, W)
            additional: bool = False,
            preprocessor: Callable = None,
            **kwargs):

        super().__init__(obs_shape, (action_shape, ), **kwargs)
        assert 'injection_shape' in kwargs, "You need injection_shape for \
        additional_shape"

        self.injection_shape = kwargs.get('injection_shape', 8)
        del kwargs['injection_shape']

        self.gamma = gamma
        self.encoded_dim = encoded_dim
        self.action_shape = action_shape

        self.lr_ac = lr_ac
        self.lr_cr = lr_cr

        self.grad_clip = grad_clip
        self.polyak = polyak

        self.is_save = kwargs.get('is_save', False)
        self.eps = np.finfo(np.float32).eps.item()
        self.use_automatic_entropy_tuning = kwargs.get(
            'use_automatic_entropy_tuning', True)

        self.preprocessor = preprocessor

        # will be using encoded_dim of 256
        self.encoder = nn.Sequential(ConvEnc(obs_shape, 256, high=12),
                                     nn.Linear(256, encoded_dim), nn.ReLU())

        # input shape to head
        head_shape = encoded_dim + self.injection_shape[0]

        # CategoricalHead is a simple 1 layer MLP with categorical dist
        # at the end via a softmax function
        self.pi_head = CategoricalHead(head_shape, action_shape[0], depth=0)

        # 1 layer MLP with 128 hidden logits
        self.q1_head = MLP(head_shape, 1, hidden=[128], activ='ReLU')
        self.q2_head = MLP(head_shape, 1, hidden=[128], activ='ReLU')

        # stochastic policy network
        self.pi = IBMWithNormalization(obs_shape,
                                       action_shape,
                                       injection_shape=self.injection_shape,
                                       encoder=self.encoder,
                                       head=self.pi_head,
                                       optimizer=optim_ac,
                                       lr=lr_ac,
                                       discrete=discrete,
                                       deterministic=deterministic,
                                       **kwargs)
        self.q1 = IBMWithNormalization(obs_shape,
                                       action_shape,
                                       injection_shape=self.injection_shape,
                                       encoder=self.encoder,
                                       head=self.q1_head,
                                       optimizer=optim_cr,
                                       lr=lr_cr,
                                       discrete=discrete,
                                       deterministic=deterministic,
                                       make_target=True,
                                       **kwargs)
        self.q2 = IBMWithNormalization(obs_shape,
                                       action_shape,
                                       injection_shape=self.injection_shape,
                                       encoder=self.encoder,
                                       head=self.q2_head,
                                       optimizer=optim_cr,
                                       lr=lr_cr,
                                       make_target=True,
                                       **kwargs)

        # set Alpha tuning
        assert isinstance(self.action_shape, Iterable)
        self.target_entropy = -np.log(1 / self.action_shape[0]) * .98

        self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

        # optimizing log_alpha instead of alpha
        self.alpha_optim = T.optim.Adam([self.log_alpha],
                                        lr=lr_ac,
                                        eps=self.eps)

        self.init_params(self.pi)
        self.init_params(self.q1)
        self.init_params(self.q2)

    def forward(self, obs, **kwargs):
        loc, inj = self.preprocessor(obs)
        loc, inj = map(lambda x: T.from_numpy(x).float().to(self.device),
                       [loc, inj])

        # without batch
        if len(inj.shape) == 1:
            inj = inj.unsqueeze(0)

        act_dist = self.pi(loc, inj)
        act_log_probs = T.log(act_dist.probs.detach()[0],
                              out=T.tensor(self.eps).to(self.device))

        max_act = T.argmax(act_dist.probs)

        return act_dist, act_log_probs, max_act

    def act(self, obs):
        act_probs, _, _ = self.forward(obs)
        action = act_probs.sample()

        return action.item()

    def update_trg(self):
        self.q1.update_trg()
        self.q2.update_trg()

    def save(self, save_dir):
        model = {
            'pi': self.pi.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'log_alpha': self.log_alpha
        }
        T.save(model, save_dir)
        print('model saved in {}'.format(save_dir))

    def load(self, save_dir):
        networks = T.load(save_dir, map_location=self.device)
        self.pi.load_state_dict(networks['pi'])
        self.q1.load_state_dict(networks['q1'])
        self.q2.load_state_dict(networks['q2'])

        self.log_alpha = networks['log_alpha']  # tensor
        self.alpha = self.log_alpha.exp()
        print('model loaded from {}'.format(save_dir))


class SACAgentDISC(Agent, BaseAgent):
    def __init__(self,
                 model: TorchModel,
                 buffer_cls: ReplayBuffer,
                 buffer_size: int = int(1e6),
                 buffer_kwargs: dict = None,
                 batch_size: int = None,
                 num_epochs: int = None,
                 update_interval: int = 30,
                 update_after: int = 256,
                 train_after: int = 256,
                 train_interval: int = 10,
                 save_interval: int = 1000,
                 save_dir: str = None,
                 eps: float = 0.5,
                 rand_until: int = 1024,
                 **kwargs):

        # for pommerman, you need to initialiize the base agent
        if 'character' in kwargs:
            self._character = kwargs.get('character')

        assert isinstance(buffer_kwargs, dict)

        self.update_interval = update_interval
        self.update_after = update_after
        self.train_interval = train_interval
        self.train_after = train_after
        self.save_interval = save_interval
        self.batch_size = batch_size
        self.eps = eps
        self.rand_until = rand_until

        super().__init__(model, train_interval, num_epochs, buffer_cls,
                         buffer_kwargs)

        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        self.now = datetime.now()
        # self.summary = SummaryWriter(
        #    os.path.join(log_dir, self.now.strftime("%Y%b%d_%H_%M_%S")))

    def act(self, obs):
        if self.curr_step < self.rand_until:
            action = np.float(random.randint(0, 5))
        else:
            # decaying epsilon greedy
            # eps / log(t+1e6)
            p = np.random.random()
            if p < self.eps/np.log((self.curr_step - self.rand_until) + 1e-6):
                action = np.float(random.randint(0, 5))
            else:  
                action = self.model.act(obs)

        return action

    def collect(self, s, a, r, d, s_):
        self.buffer.push(obs=s, action=a, reward=r, done=d, obs_=s_)

    # one step in the environment
    def step(self, s, a, r, d, s_):
        self.curr_step += 1

        self.collect(s, a, r, d, s_)
        info = {}
        if (self.curr_step % self.train_interval == 0
                and self.curr_step > self.train_after):
            info = self.train()

        if (self.curr_step % self.update_interval == 0
                and self.curr_step > self.update_after):
            self.model.update_trg()

        # TODO Save model
        if (self.curr_step % self.save_interval == 0 and self.model.is_save):
            self.model.save(self.save_dir + f'/{int(self.curr_step/1000)}k')

        return info

    def train(self):
        global training_cnt
        training_cnt += 1
        batch = self.buffer.sample(self.batch_size)
        l_q1, l_q2, m_q1, m_q2 = loss_func_cr(batch, self.model)
        l_pi, entropies = loss_func_ac(batch, self.model)
        l_alpha = loss_func_alpha(entropies, self.model)

        info = {
            'q_loss': (l_q1.detach(), l_q2.detach()),
            'q_value': (m_q1, m_q2),
            'pi_loss': l_pi.detach(),
            'alpha_loss': l_alpha.detach(),
            'entropies': entropies
        }

        l_q1.backward(retain_graph=True)
        l_q2.backward(retain_graph=True)
        l_pi.backward(retain_graph=True)

        nn.utils.clip_grad_norm_(self.model.q1.parameters(),
                                 self.model.grad_clip)
        nn.utils.clip_grad_norm_(self.model.q2.parameters(),
                                 self.model.grad_clip)
        nn.utils.clip_grad_norm_(self.model.pi.parameters(),
                                 self.model.grad_clip)

        self.model.q1.optimizer.step()
        self.model.q2.optimizer.step()
        self.model.pi.optimizer.step()

        self.model.alpha_optim.zero_grad()
        l_alpha.backward()
        self.model.alpha_optim.step()

        return info


class SACDAgentLOGGER(SACAgentDISC):
    def __init__(self, *arg, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    pass
    # testing this code
