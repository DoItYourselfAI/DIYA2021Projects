import torch
import torch.optim as optim
from torch.distributions import Normal
from agents.a2c.agent import A2C


class PPO(A2C):
    def __init__(self, args=None, name='PPO'):
        super().__init__(args=args, name=name)
        # intialize optimizers
        self.optim = optim.RAdam(
            self.model.parameters(),
            lr=args.lr_actor
        )

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
        clip = self.args.cliprange
        values_old = self.buffer['values'][idx]
        values_clipped = (values_old
                          + torch.clamp(values - values_old, -clip, clip))
        advs = self.buffer['advs'][idx]
        returns = self.buffer['values'][idx] + advs
        loss_critic_clipped = (returns.detach() - values_clipped).pow(2)
        loss_critic = (returns.detach() - values).pow(2)
        loss_critic = torch.max(loss_critic, loss_critic_clipped).mean()
        self.info.update('Loss/Critic', loss_critic.item())

        # actor loss
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        advs = advs.squeeze(-1).detach()
        ratio = torch.exp(self.buffer['nlls'][idx] - nlls)
        loss_actor = -advs * ratio
        ratio = torch.clamp(ratio, 1 - clip, 1 + clip)
        loss_actor_clipped = -advs * ratio
        loss_actor = torch.max(loss_actor, loss_actor_clipped).mean()
        self.info.update('Loss/Actor', loss_actor.item())

        # total loss with entropy bonus
        loss = loss_actor + self.args.cr_coef * loss_critic
        loss -= self.args.ent_coef * entropy
        return loss
