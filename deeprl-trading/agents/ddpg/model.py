import torch
import torch.nn as nn
import numpy as np
from agents.transformer import TransformerBlock


class CNNActor(nn.Module):
    def __init__(self, ob_space, ac_space):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(ob_space[0], 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
        )
        sample = torch.zeros(1, *ob_space)
        size = np.prod(self.features(sample).size())

        self.ac_head = nn.Linear(size, np.prod(ac_space))
        self.ac_head.weight.data.uniform_(-3e-3, 3e-3)
        self.ac_head.bias.data.uniform_(-3e-3, 3e-3)
        self.ac_space = ac_space

    def forward(self, obs):
        z = self.features(obs).view(obs.size(0), -1)
        out = self.ac_head(z)
        out = out.view(obs.size(0), *self.ac_space)
        return out


class CNNCritic(nn.Module):
    def __init__(self, ob_space, ac_space):
        super().__init__()
        self.obs_features = nn.Sequential(
            nn.Conv1d(ob_space[0], 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
        )
        sample = torch.zeros(1, *ob_space)
        size = np.prod(self.obs_features(sample).size())

        self.acs_features = nn.Sequential(
            nn.Linear(np.prod(ac_space), 400),
            nn.ReLU(),
            nn.Linear(400, size),
            nn.ReLU(),
        )
        self.val_head = nn.Linear(2 * size, 1)
        self.val_head.weight.data.uniform_(-3e-3, 3e-3)
        self.val_head.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs, acs):
        z_obs = self.obs_features(obs).view(obs.size(0), -1)
        z_acs = self.acs_features(acs.view(acs.size(0), -1))
        z = torch.cat([z_obs, z_acs], dim=-1)
        out = self.val_head(z)
        return out


class TransformerActor(nn.Module):
    def __init__(self, ob_space, ac_space):
        super().__init__()
        # Stack of 6 Transformer blocks as in original implementation
        self.transformer_encoder = nn.Sequential(
            TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
        )
        # To go from shape [b, l, d] to [b, l, 1]
        self.linearhead = nn.Linear(ob_space[1], 1)

        self.ac_head = nn.Linear(ob_space[0], np.prod(ac_space))
        self.ac_head.weight.data.uniform_(-3e-3, 3e-3)
        self.ac_head.bias.data.uniform_(-3e-3, 3e-3)
        self.ac_space = ac_space

    def forward(self, obs):
        out = self.transformer_encoder(obs)
        out = self.linearhead(out)
        out = out.squeeze(dim=-1)
        out = self.ac_head(out)
        out = out.view(obs.size(0), *self.ac_space)
        # nn.functional.softmax(out, dim=-1)
        return out


class TransformerCritic(nn.Module):
    def __init__(self, ob_space, ac_space):
        super().__init__()
        # Stack of 6 Transformer blocks as in original implementation
        self.transformer_encoder = nn.Sequential(
            TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
        )
        # self.transformerblock = TransformerBlock(d=ob_space[1])
        # self.flatten = nn.Flatten()
        self.linearhead = nn.Linear(ob_space[1], 300)

        self.acs_features = nn.Sequential(
            nn.Linear(np.prod(ac_space), 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
        )

        # connect transformer output (state) and ffn output (action)
        self.val_head = nn.Linear(300 + 300, 1)
        self.val_head.weight.data.uniform_(-3e-3, 3e-3)
        self.val_head.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs, acs):
        obs = self.transformer_encoder(obs)
        # Either flatten:
        # obs = self.flatten(obs)
        # or take mean of sequences (less params.)
        obs = obs.mean(dim=1)
        obs = self.linearhead(obs)
        acs = self.acs_features(acs)
        out = torch.cat([obs, acs], dim=-1)
        out = self.val_head(out)
        return out
