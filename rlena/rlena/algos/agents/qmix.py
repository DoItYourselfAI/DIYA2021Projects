import os
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym

# re-register issues
try:
    from rlena.envs.playground.pommerman import characters
    from rlena.envs.playground.pommerman.agents import BaseAgent, SimpleAgent

except gym.error.Error as e:
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if 'Pomme' in env or 'OneVsOne' in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]
    
    from rlena.envs.playground.pommerman import characters
    from rlena.envs.playground.pommerman.agents import BaseAgent, SimpleAgent

from rlena.algos.buffers import PriorityBuffer
from rlena.algos.utils import flatten


###########################################
########### QMIX agent ####################
###########################################

class ConvGRU_model(nn.Module):
    def __init__(self, configs):
        super(ConvGRU_model, self).__init__()
        self.action_n = 6
        self.input_shape = configs['input_shape']

        # convolution 
        conv_filters = configs['conv_filters']
        conv_layer = []
        for conv_filter in conv_filters:
            conv_layer += [nn.Conv2d(*conv_filter),
                           nn.BatchNorm2d(conv_filter[1]),
                           nn.ReLU()]
        self.conv_layer = nn.Sequential(*conv_layer)
        self.h_dim = self._cal_dim()

        # FC 
        fc_hidden = configs['fc_hidden'].copy()
        fc_hidden.insert(0, self.h_dim)
        fc = []
        for i in range(len(fc_hidden)-1):
            fc += [nn.Linear(fc_hidden[i], fc_hidden[i+1]),
                   nn.ReLU()]
        self.fc = nn.Sequential(*fc)

        # GRU
        lstm_cell_size = configs['lstm_cell_size']
        dp = configs['dropout']
        self.GRUCell = nn.GRUCell(fc_hidden[-1], lstm_cell_size)

        # FC
        self.fc_Q = nn.Linear(lstm_cell_size, self.action_n)

    def _cal_dim(self):
        with torch.no_grad():
            x = torch.zeros(self.input_shape).unsqueeze(dim=0)
            x = self.conv_layer(x).view(1,-1)
        return x.shape[1]


    def forward(self, obs, gru_hidden = None):
        # conv layer
        if len(obs.shape) == 3:
            x = self.conv_layer(obs.unsqueeze(dim=0))
        else:
            x = self.conv_layer(obs)
        x = torch.flatten(x, start_dim=1)

        # FC
        x = self.fc(x)
        
        # GRU
        if gru_hidden is not None:
            gru_hidden = gru_hidden.to(x.get_device())
            h_RNN = self.GRUCell(x, gru_hidden)
        else:
            h_RNN = self.GRUCell(x)

        # FC last layer
        Q_val = self.fc_Q(h_RNN)

        return Q_val, h_RNN


class QMIXAgent(BaseAgent):
    def __init__(self, configs, character=characters.Bomber):
        super(QMIXAgent, self).__init__(character)
        self.online_model = ConvGRU_model(configs['model_config'])
        self.target_model = ConvGRU_model(configs['model_config'])
        self.target_model.requires_grad_ = False

        self.optim = optim.Adam(self.online_model.parameters(),configs['learning_rate'])
        self.epsilon = configs['epsilon']
        self.min_epsilon = configs['min_epsilon']
        self.epsilon_decay = configs['epsilon_decay']

        self.device = configs['device']
        self.online_model.to(self.device)
        self.target_model.to(self.device)
        
        self.save_path = configs['save_agent_path']

        # save values for Q networks
        self.lstm_cell_size = configs['model_config']['lstm_cell_size']
        self.gru_hidden = torch.zeros(size=(1,self.lstm_cell_size))

    def act(self, obs, action_space):
        
        obs = torch.Tensor(obs).to(self.device)
            
        with torch.no_grad():
            Q_all, self.gru_hidden = self.online_model(obs, self.gru_hidden)
        if random.random() < self.epsilon: # training 일 때에만
            action = action_space.sample()
        else:
            action = torch.argmax(Q_all).detach().item()

        del Q_all, obs
        torch.cuda.empty_cache()
        
        return action
    
    def getQ(self, obs, action, hidden = None):
        Q_all, _ = self.online_model(obs, hidden)

        if action == None:
            return torch.argmax(Q_all, dim=1)

        return torch.gather(Q_all, dim=1, index=action)

    def getQ_target(self, obs, hidden = None):
        with torch.no_grad():
            Q_all, _ = self.target_model(obs, hidden)

        return Q_all
    
    def zero_grad(self):
        self.optim.zero_grad()

    def step(self):
        self.optim.step()

    def target_update(self, rate):
        for target_param, param in zip(self.target_model.parameters(), self.online_model.parameters()):
            target_param.data.copy_(target_param.data * (1-rate) + param.data * rate)

    def mem_append(self, sample):
        default_error = 1 # 쌓이는 데이터는 error 계산이 안되었으므로 그냥 1 넣어주자 (나중에 바뀔 수 있음)
        self.memory.add(default_error, sample)

    def mem_update(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            self.memory.update(idx, error)

    def episode_end(self, reward):
        """This is called at the end of the episode to let the agent know that
        the episode has ended and what is the reward.

        Args:
          reward: The single reward scalar to this agent.
        """
        self.gru_hidden = torch.zeros(size=(1,self.lstm_cell_size))

    def train(self):
        self.online_model.train()

    def eval(self):
        self.online_model.eval()

    def e_decay(self):
        self.epsilon = self.epsilon*self.epsilon_decay if self.epsilon > self.min_epsilon else self.epsilon

    def save(self, num):
        """agent 끼리도 다르기 때문에 num으로 구분한다."""
        path = self.save_path.split('.')
        path = path[0]+"_"+str(num)+'.'+path[1]
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.online_model.state_dict(), path)

    def load(self, num):
        path = self.save_path.split('.')
        path = path[0]+"_"+str(num)+'.'+path[1]
        if not os.path.isdir(os.path.dirname(path)):
            assert "There is no trained model parameters. Please train model first"
        self.online_model.load_state_dict(torch.load(path, map_location=self.device))
        self.target_model.load_state_dict(self.online_model.state_dict())


###########################################
########### QMIX critic ###################
###########################################

class hypernet(nn.Module):
    def __init__(self, input_dim, shape, hidden_dim, bias, final):
        super(hypernet, self).__init__()
        self.shape = shape
        self.bias = bias
        self.hidden_dim = hidden_dim

        """Gain 으로 initialization 조절 했는데, 잘 한 건지는 모르겠다"""

        if not final:
            self.fc = nn.Linear(input_dim, self.shape[0]*self.shape[1])
            torch.nn.init.xavier_normal_(self.fc.weight, gain=1/(self.shape[0]*self.shape[1])) 
        else:
            self.fc = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.shape[0]*self.shape[1])
            )
            torch.nn.init.xavier_normal_(self.fc[0].weight, gain=1/np.sqrt(self.shape[0]*self.shape[1]))
            torch.nn.init.xavier_normal_(self.fc[2].weight, gain=1/np.sqrt(self.shape[0]*self.shape[1]))

    def forward(self, x):
        x = self.fc(x)
        if not self.bias:
            x = torch.abs(x)
        x = x.view(x.shape[0], self.shape[0], self.shape[1])

        return x


class MixingNet(nn.Module):
    def __init__(self, configs):
        super(MixingNet, self).__init__()
        self.device = 'cpu'
        self.agent_num = 2 # args.agent_num
        self.input_shape = configs['input_shape']
        self.hidden_dim = configs['fc_hidden']
        
        # conv layer
        conv_filters = configs['conv_filters']
        conv_layer = []
        for conv_filter in conv_filters[:-1]:
            conv_layer += [nn.Conv2d(*conv_filter),
                           nn.BatchNorm2d(conv_filter[1]),
                           nn.ReLU()]
        conv_layer += [nn.Conv2d(*conv_filters[-1]),
                       nn.ReLU()]
        self.conv_layer = nn.Sequential(*conv_layer)
        self.input_dim = self._cal_dim()

        # hypernet
        self.w_1 = hypernet(input_dim=self.input_dim,
                            shape=(self.agent_num, self.hidden_dim), 
                            hidden_dim=self.hidden_dim, 
                            bias=False, final=False) # Agent 는 2개로 고정인 상황
        self.b_1 = hypernet(input_dim=self.input_dim, 
                            shape=(1,self.hidden_dim), 
                            hidden_dim=self.hidden_dim, 
                            bias=True, final=False)
        self.w_2 = hypernet(input_dim=self.input_dim, 
                            shape=(self.hidden_dim,1), 
                            hidden_dim=self.hidden_dim, 
                            bias=False, final=False)
        self.b_2 = hypernet(input_dim=self.input_dim, 
                            shape=(1,1), 
                            hidden_dim=self.hidden_dim, 
                            bias=True, final=True) # value function 역할

    def _cal_dim(self):
        with torch.no_grad():
            x = torch.zeros(self.input_shape).unsqueeze(dim=0)
            x = self.conv_layer(x).view(1,-1)
        return x.shape[1]

    def forward(self, qvals, global_state): # 여기부터 할 것
        if len(global_state.shape) == 3:
            global_state = global_state.unsqeeze(dim=0)
        global_hidden = self.conv_layer(global_state)
        global_hidden = torch.flatten(global_hidden, start_dim=1)
        
        qvals = qvals.view(-1,1,self.agent_num)
        
        x = torch.bmm(qvals, self.w_1(global_hidden))
        x = F.elu(x + self.b_1(global_hidden))
        x = torch.bmm(x, self.w_2(global_hidden)) # 여기까지 Advantage라고 보면 됨
        x = x + self.b_2(global_hidden) # Dueling의 일종이라고 봐도 될 듯

        return x

    def to(self, device):
        self.device = device
        self.conv_layer.to(self.device)
        self.w_1.to(self.device)
        self.w_2.to(self.device)
        self.b_1.to(self.device)
        self.b_2.to(self.device)


class QMIXCritic:
    def __init__(self, agents, configs):
        self.online_net = MixingNet(configs['model_config'])
        self.target_net = MixingNet(configs['model_config'])
        self.device = configs['device']
        self.online_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.requires_grad_ = False

        self.memory = PriorityBuffer(configs['memory_config'])
        self.agents = agents

        self.gamma = configs['gamma']
        self.batch_size = configs['batchsize']

        self.optim = optim.Adam(self.online_net.parameters(),configs['learning_rate'])
        self.target_rate = configs['soft_target_rate']

        self.save_path = configs['save_critic_path']

    def learn(self):
        batch = self.memory.sample(self.batch_size)

        batch_state = torch.tensor(batch[0], device=self.device).float()  
        batch_hidden = torch.tensor(batch[1], device=self.device).squeeze().float()
        batch_global_state = torch.tensor(batch[2], device=self.device).float()
        batch_action = torch.tensor(batch[3], device=self.device)
        batch_reward = torch.tensor(batch[4], device=self.device).sum(dim=1).float()
        batch_done = torch.tensor(batch[5], device=self.device).squeeze().float()
        batch_state_prime = torch.tensor(batch[6], device=self.device).float()

        # calculate target Q value
        with torch.no_grad():
            Qval_1 = self.agents[0].getQ_target(batch_state_prime[:,0], hidden=batch_hidden[:,0]) # 32X6
            Qval_2 = self.agents[1].getQ_target(batch_state_prime[:,1], hidden=batch_hidden[:,1]) # 32X6
            
            Q_max = None
            for i in range(Qval_1.shape[1]):
                for j in range(Qval_2.shape[1]):
                    q1 = Qval_1[:,i:i+1] # 32
                    q2 = Qval_2[:,j:j+1] # 32
                    Qtotal_target = self.target_net(torch.cat((q1,q2), dim=1), batch_global_state)
                    if Q_max is None:
                        Q_max = Qtotal_target
                    else:
                        for k in range(self.batch_size):
                            if Q_max[k] < Qtotal_target[k]:
                                Q_max[k] = Qtotal_target[k]
            
            Q_max = Q_max.squeeze() # 32

            target = batch_reward + self.gamma * Q_max * batch_done # 32

        # Q value caculate
        Qval = torch.cat((self.agents[0].getQ(batch_state[:,0], batch_action[:,0:1], hidden= batch_hidden[:,0]), 
                            self.agents[1].getQ(batch_state[:,1], batch_action[:,1:2], hidden= batch_hidden[:,1])), dim = 1) # 32X2
        Qtotal = self.online_net(Qval, batch_global_state).squeeze()

        loss = F.mse_loss(target,Qtotal)
        self.zero_grad()
        loss.backward()
        self.step()

        # deleting cache
        del batch_state, batch_hidden, batch_global_state, batch_action, batch_reward, batch_state_prime, batch_done, Q_max, Qtotal, Qtotal_target, target
        del Qval_1, Qval_2, q1, q2
        torch.cuda.empty_cache()

        return loss.detach().sum() / self.batch_size

    def target_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(target_param.data * (1-self.target_rate) + param.data * self.target_rate)
        for agent in self.agents:
            agent.target_update(self.target_rate)

    def zero_grad(self):
        self.optim.zero_grad()
        for agent in self.agents:
            agent.zero_grad()

    def step(self):
        self.optim.step()
        for agent in self.agents:
            agent.step()

    def mem_append(self, sample):
        error = 1 # 쌓이는 데이터는 error 계산이 안되었으므로 그냥 1 넣어주자 (나중에 바뀔 수 있음)

        state, gru_hidden, global_state, actions, reward, done, state_prime = sample

        state = state[0::2]
        actions = actions[0::2]
        reward = reward[0::2]
        done = [1] if done else [0]
        state_prime = state_prime[0::2]

        sample = [state, gru_hidden, global_state,  actions, reward, done, state_prime]
        
        self.memory.add(error, sample)

    def mem_update(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            self.memory.update(idx, error)

    def save_memory(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)

    def load_memory(self, path):
        with open(path, 'rb') as f:
            self.memory = pickle.load(f)

    def save(self):
        if not os.path.isdir(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))
        torch.save(self.online_net.state_dict(), self.save_path)

    def load(self):
        if not os.path.isdir(os.path.dirname(self.save_path)):
            assert "There is no trained model parameters. Please train model first"
        self.online_net.load_state_dict(torch.load(self.save_path))
        self.target_net.load_state_dict(self.online_net.state_dict())