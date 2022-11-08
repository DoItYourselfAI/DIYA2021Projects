import os
import operator
import numpy as np
import torch
import torch.nn.functional as F

from copy import deepcopy
from pathlib import Path
from functools import reduce
from easydict import EasyDict
from typing import Callable, Iterable, List, Union

from rl2.models.torch.base import InjectiveBranchModel, TorchModel
from rl2.agents.base import MAgent
from rl2.agents.utils import LinearDecay
from rl2.buffers.base import ReplayBuffer

from rlena.algos.buffers import EpisodicBuffer
from rlena.algos.utils import mask_action

"""
Implementation of Counterfactual Multi-Agent(COMA) Policy Gradients
from Foerster et al. (https://arxiv.org/pdf/1705.08926.pdf)
"""


def loss_func_ac(data, model, **kwargs):
    loc, add, h, a, r, d, v, g_loc, g_add = map(
        lambda x: torch.from_numpy(x).float().to(model.device), data)
    hiddens = np.split(h.cpu(), 2, axis=1)
    hiddens = tuple(map(lambda x: x.squeeze(1).to(model.device), hiddens))
    losses = []
    for actor, policy in enumerate(model.policies):
        # For each actors
        taken_actions = a[:, actor].long()
        qs = v[:, actor]
        q = qs.gather(1, taken_actions.view(-1, 1))  # Q(state, joint action)

        hidden = tuple(map(lambda x: (x[:, actor]), hiddens))
        ac_dist, _ = policy(loc[:, actor], add[:, actor],
                            hidden=hidden, mask=d[:, actor])
        pi = ac_dist.probs
        counterfactual_baseline = torch.sum(qs * pi, dim=-1, keepdim=True)
        counterfactual_adv = q - counterfactual_baseline
        nlps = -ac_dist.log_prob(taken_actions)
        loss = -nlps * counterfactual_adv.squeeze()
        losses.append(loss.mean())

    return losses


def loss_func_cr(data, model, **kwargs):
    """
    loss function for centralized critic
    """
    td_trg = np.vstack(kwargs.get("td_trg"))
    td_trg = torch.from_numpy(td_trg).float().to(model.device)

    data = data[3:]
    a, _, _, _, g_loc, g_add = map(
        lambda x: torch.from_numpy(x).float().to(model.device), data)

    masked_acs = [mask_action(a, actor) for actor in range(2)]
    injections = [torch.cat([g_add, acs], dim=-1) for acs in masked_acs]
    idx = masked_acs[-1][:, 0].long()

    qs = [model.value(g_loc, inj).mean.gather(1, idx.view(-1, 1))
          for inj in injections]
    qs = torch.cat(qs, dim=-1)

    loss = F.smooth_l1_loss(td_trg, qs)

    return loss


def loss_func(data, model, **kwargs):
    critic_loss = loss_func_cr(data, model, **kwargs)
    actor_loss = loss_func_ac(data, model, **kwargs)

    return critic_loss, actor_loss


class COMAPGModel(TorchModel):
    def __init__(self,
                 observation_shape: Union[Iterable[int], Iterable[Iterable]],
                 g_observation_shape,
                 action_shape,
                 n_agents: int = 2,
                 encoder: torch.nn.Module = None,
                 encoded_dim: int = 64,
                 optimizer='torch.optim.Adam',
                 lr_ac=1e-4,
                 lr_cr=1e-5,
                 discrete: bool = True,
                 deterministic: bool = False,
                 flatten: bool = False,  # True if you don't want CNN enc
                 reorder: bool = False,  # flag for (C, H, W)
                 recurrent: bool = True,
                 additional: bool = False,
                 **kwargs):

        self.additional = additional
        # Presetting for handling additional observation.
        if isinstance(observation_shape[0], Iterable):
            # Unpack the tuple of shapes
            observation_shape, _ = observation_shape
            # First shape of the observation assumed to be a locational info
            # and Thus will be treated as a main obs and Assummed to be (C,H,W)
            self.additional = True

        if isinstance(g_observation_shape[0], Iterable):
            g_observation_shape, g_additional_shape = g_observation_shape
            inj_shape_cr = (g_additional_shape[0] + n_agents,)
            self.g_additional = True

        super().__init__(observation_shape, action_shape, **kwargs)
        if hasattr(encoder, 'output_shape'):
            encoded_dim = encoder.output_shape

        self.encoded_dim = encoded_dim
        self.recurrent = recurrent
        self.n_agents = n_agents
        # TODO: Currently recurrent unit uses LSTM -> change to GRU later
        # Actor uses inputs of locational obs, additional obs, and old action.
        # Policy network uses locational obs as main obs, and other inputs are
        # injected together after passing the CNN enc.
        self.policy = InjectiveBranchModel(observation_shape, action_shape,
                                           injection_shape=(9,),
                                           encoded_dim=encoded_dim,
                                           optimizer=optimizer,
                                           lr=lr_ac,
                                           discrete=discrete,
                                           deterministic=deterministic,
                                           flatten=flatten,
                                           reorder=False,
                                           recurrent=True,  # True
                                           **kwargs)
        # Decentralized Actors
        self.policies = [deepcopy(self.policy) for i in range(self.n_agents)]

        # Centralized Critic
        self.value = InjectiveBranchModel(g_observation_shape,
                                          action_shape,
                                          inj_shape_cr,  # n_acs?
                                          encoder=encoder,
                                          encoded_dim=encoded_dim,
                                          optimizer=optimizer,
                                          lr=lr_cr,
                                          discrete=True,
                                          deterministic=True,
                                          flatten=flatten,
                                          reorder=reorder,
                                          recurrent=False,
                                          make_target=True,
                                          **kwargs)

        # Initialize params
        list(map(self.init_params, self.policies))
        self.init_params(self.value)

    def forward(self, obs: torch.tensor, *args, **kwargs):
        obs = obs.to(self.device)
        args = [a.to(self.device) for a in args]
        action_dist = self.policy(obs, **kwargs)
        value_dist = self.value(obs, **kwargs)
        if self.recurrent:
            action_dist = action_dist[0]
            value_dist = value_dist[0]

        return action_dist, value_dist

    def act(self, agent: int, obs: np.ndarray, *args) -> np.ndarray:
        # *args contains the injections i.e. (additional obs, old_action)
        inj = reduce(operator.add, args)
        policy = self.policies[agent]
        if self.recurrent:
            hidden = tuple(
                map(lambda x: (x[:, agent].unsqueeze(0)), self.hidden))
            action_dist, hidden = self._infer_from_numpy(
                policy, obs, inj, hidden=hidden)
        else:
            action_dist = self.infer_from_numpy(policy, obs, inj)

        action = action_dist.sample().squeeze()
        action = action.detach().cpu().numpy()

        info = {}
        if self.recurrent:
            info['hidden'] = hidden

        return action, info

    def val(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        # Value network calculates the value of state with global obs
        # *args contains the injections i.e. (additional obs, old_action)
        inj = np.concatenate(args, axis=-1)
        self.recurrent = False
        if kwargs.get('forward_trg'):
            val_dist, _ = self._infer_from_numpy(
                self.value.forward_trg, state, inj)
        else:
            val_dist, _ = self._infer_from_numpy(self.value, state, inj)

        self.recurrent = True
        value = val_dist.mean.squeeze()
        value = value.detach().cpu().numpy()

        info = {}

        return value, info

    def save(self, save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_dir = os.path.join(save_dir, type(self).__name__ + '.pt')
        torch.save(self.state_dict(), save_dir)
        print(f'model saved in {save_dir}')

    def load(self, load_dir):
        load_dir = os.path.join(load_dir, type(self).__name__ + '.pt')
        ckpt = torch.load(load_dir, map_location=self.device)
        self.load_state_dict(ckpt)


class COMAgent(MAgent):
    def __init__(self,
                 models: List[TorchModel],
                 n_env=1,
                 n_agents: int = 4,
                 buffer_cls: ReplayBuffer = EpisodicBuffer,
                 buffer_kwargs: dict = None,
                 buffer_size: int = 30,
                 loss_func: Callable = loss_func,
                 init_actions: Iterable = np.array([0, 0, 0, 0]),
                 init_collect: int = 1,
                 train_interval: int = 30,
                 update_interval: int = 150,
                 num_epochs: int = 1,
                 explore: bool = True,
                 eps: Union[float, LinearDecay] = 0.02,
                 polyak: float = 0.1,
                 gamma: float = 0.99,
                 lamda: float = 0.99,
                 **kwargs,
                 ):
        # Handle config dict;
        # If config exists in kwargs, config dict overrides the init arguments
        self.config = EasyDict(kwargs.setdefault(
            'config', {'n_env': n_env,
                       'n_agents': n_agents,
                       'buffer_size': buffer_size,
                       'init_actions': init_actions,
                       'init_collect': init_collect,
                       'train_interval': train_interval,
                       'update_interval': update_interval,
                       'num_epochs': num_epochs,
                       'polyak': polyak,
                       'gamma': gamma,
                       'lamda': lamda,
                       }))
        self.config.update(kwargs)

        # Set attributes with values from config
        self.n_env = self.config.n_env
        self.init_collect = self.config.init_collect
        self.buffer_size = self.config.buffer_size
        self.train_interval = self.config.train_interval
        self.update_interval = self.config.update_interval
        self.polyak = self.config.polyak
        self.gamma = self.config.gamma
        self.lamda = self.config.lamda

        self.explore = explore
        self.eps = LinearDecay(start=0.5, end=eps, decay_step=750)
        if isinstance(eps, LinearDecay):
            self.eps = eps
        # just for Pomme
        self.order = np.array([0, 2, 1, 3])
        self.save_dir = kwargs.get('ckpt_dir', 'logs/ckpt')
        self.save_interval = kwargs.get('save_interval', int(1e3))

        if buffer_kwargs is None:
            buffer_kwargs = {'size': 800,
                             'max_episodes': self.config.buffer_size}

        super().__init__(models, train_interval, num_epochs,
                         buffer_cls, buffer_kwargs)
        self.obs = None

        self.n_agents = n_agents
        if n_env == 1:
            self.dones = [[False, False], [False, False]]
        else:
            raise NotImplementedError

        self.old_actions = init_actions
        self.done = all(self.dones)
        self.curr_ep = 1
        self.loss_func = loss_func

        self.n_actors = []
        self.hiddens = []
        self.pre_hiddens = []
        dones = [self.dones[:n] for n in (2, 2)]

        for model, done in zip(self.models, dones):
            model._init_hidden(done)
            self.hiddens.append(model.hidden)
            self.pre_hiddens.append(model.hidden)
            self.n_actors.append(len(model.policies))

        if kwargs['load_dir']:
            for team, model in enumerate(self.models):
                path = os.path.join(kwargs.get('load_dir'), f'team{team}')
                model.load(path)

    def act(self, obss: Iterable, old_acs: Iterable = None) -> np.ndarray:
        # Handle global obs for centralized critic; Assumes the last elem in obss is a global obs
        g_obs, g_add = obss.pop().values()
        self.g_state = (g_obs, g_add)
        # Handle additional observation
        if old_acs is None:
            old_acs = self.old_actions
        # Split inputs by team
        obss = [obss[:n] for n in self.n_actors]
        old_acs = [old_acs[:n] for n in self.n_actors]
        actions = []

        for team, (model, team_obss, team_acs) in enumerate(zip(self.models, obss, old_acs)):
            obs, adds, infos = [], [], []
            if model.additional:
                for o in team_obss:
                    obs.append(o.get('locational'))
                    adds.append(o.get('additional'))
                inputs = list(zip(obs, adds, team_acs))
            else:
                inputs = list(zip(obs, team_acs))

            for agent, input in enumerate(inputs):
                action, info = model.act(agent, *input)
                if self.explore and np.random.random() < self.eps(self.curr_ep):
                    action = np.random.randint(model.action_shape[0], size=())
                if model.recurrent:
                    for e, e_ in zip(self.hiddens[team], info.get('hidden')):
                        e[:, agent] = e_
                actions.append(action)

        actions = np.asarray(actions)
        # Reorder actions
        actions = actions[self.order]

        return actions

    def step(self, s, a, r, d, s_):
        self.curr_step += 1
        # Store data to buffer
        self.collect(s, a, r, d)

        if self.done:
            # If an episode ends, store trajectory and init buffer
            self.curr_ep += 1
            for buffer in self.buffers:
                buffer.reset()

        for team, (model, d) in enumerate(zip(self.models, self.dones)):
            if model.recurrent:
                model._update_hidden(d, self.hiddens[team])
            if self.curr_step % self.save_interval == 0:
                path = os.path.join(
                    self.save_dir, str(self.curr_step)+'step', f'team{team}')
                model.save(path)

        info = {'done': self.done}

        if self.curr_ep % self.train_interval == 0:
            # Train models
            for team, (model, buffer) in enumerate(zip(self.models, self.buffers)):
                td_trg = self._calculate_td_trg(buffer.episodes, model)
                info_t = self.train(buffer.sample(), model, td_trg)
                if model.recurrent:
                    self.pre_hiddens[team] = model.hidden

                info[f'team{team}/critic_loss'] = info_t['critic_loss']
                for i, loss in enumerate(info_t['actors_loss']):
                    info[f'team{team}/actor{i}_loss'] = loss

                if self.curr_ep % self.update_interval == 0:
                    model.value.update_trg(alpha=self.polyak)

            info['is_loss'] = True

        return info

    def collect(self, s, a, r, d):
        # Unpack datas for collect
        loc = np.array(list(map(lambda x: x['locational'], s)))
        add = np.array(list(map(lambda x: x['additional'], s)))

        g_loc, g_add = self.g_state
        g_loc, g_add = (np.expand_dims(x, 0) for x in [g_loc, g_add])
        r = np.asarray(r)

        data = [loc, add, a, r]
        data = [np.split(e, 2) for e in data]

        loc = tuple(map(lambda x: np.expand_dims(x, 0), data[0]))
        add = tuple(map(lambda x: np.expand_dims(x, 0), data[1]))
        _, _, acs, rew = data

        # Pop global done
        self.done = d.pop()
        # Seperate dones for agents by team
        self.dones = np.split(np.asarray(d), 2)

        # Collect data from a transition
        for team, model in enumerate(self.models):
            # Convert hidden states in to np array
            hiddens = list(map(lambda x: x.detach().cpu().numpy(),
                               self.hiddens[team]))
            hiddens = np.stack(hiddens, axis=1)

            values = []
            masked_acs = [mask_action(acs[team], actor) for actor in range(2)]
            for ac in masked_acs:
                inj = np.concatenate([g_add, np.expand_dims(ac, 0)], axis=-1)
                qs, _ = model.val(g_loc, inj)
                values.append(qs)
            values = np.expand_dims(np.array(values), 0)

            self.buffers[team].push(loc[team], add[team], hiddens,
                                    acs[team], rew[team], self.dones[team],
                                    values, g_loc, g_add)

    def _calculate_td_trg(self, trajectories, model):
        # Calculate td traget values
        td_trgs = []
        for traj in trajectories:
            rews = traj['rewards']
            td_trg = np.zeros(rews.shape)
            T = len(td_trg)
            keys = ['g_loc', 'g_add', 'actions']
            g_loc, g_add, acs = map(lambda key: traj[key], keys)
            for t in range(T):
                for n in range(1, T-t):
                    masked_acs = [mask_action(
                        acs[t+n], actor) for actor in range(2)]
                    g_t = self._calculate_G_t(model, t, n, rews, masked_acs,
                                              g_loc[t+n], g_add[t+n])
                    td_trg[t] += self.lamda ** n-1 * g_t

            td_trg *= 1 - self.lamda
            td_trgs.append(td_trg)

        return td_trgs

    def _calculate_G_t(self, critic, t, n_step, rews, masked_acs, *args):
        out = np.zeros_like(rews[0], dtype='float64')

        for l in range(1, n_step+1):
            out += self.gamma ** (l-1) * rews[t + l]

        trg_qs = []
        # Calculate bootstraped value
        for i, actions in enumerate(masked_acs):
            inputs = list(args)
            inputs.append(actions)
            trg_q, _ = critic.val(*inputs, forward_trg=True)
            ac = masked_acs[i-1][i]
            trg_qs.append(trg_q[ac])
        out += self.gamma ** n_step * np.array(trg_qs)

        return out

    def train(self, data, model, td_trg):
        loss_cr, loss_ac = self.loss_func(data, model, td_trg=td_trg)
        model.value.step(loss_cr)
        for policy, loss in zip(model.policies, loss_ac):
            policy.step(loss)

        info = {}
        info['actors_loss'] = list(map(lambda x: x.item(), loss_ac))
        info['critic_loss'] = loss_cr.item()

        return info
