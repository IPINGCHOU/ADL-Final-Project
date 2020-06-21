import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
import sys
import cv2
from typing import Dict, List, Tuple
from torch.distributions import Normal

sys.path.insert(1, '../game')

from collections import namedtuple
from game_controller import ReplaySaver
use_cuda = torch.cuda.is_available()

from visualize import *#
_saliencies = []

# h-para
DEVICE = 'cuda:0'
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
REWARD_MULTI = 1
RESIZE = True
RESIZE_SIZE = (80,80)

# model settings
MEMORY_CAPACITY = 20000
TRAINING_START = 2048
BATCH_SIZE = 1024
#TARGET_UPDATE_FREQ = 10
#LEARNING_RATE = 1e-02

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


class Actor(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        """Initialize."""
        super(Actor, self).__init__()
        
        # set the log std range
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.conv1 = nn.Conv2d(in_dim, 32, kernel_size=8, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)
        
        # set log_std layer
        self.log_std_layer = nn.Linear(128, out_dim)
        self.log_std_layer = init_layer_uniform(self.log_std_layer)

        # set mean layer
        self.mu_layer = nn.Linear(128, out_dim)
        self.mu_layer = init_layer_uniform(self.mu_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = self.relu(self.bn1(self.conv1(state)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = self.lrelu(self.fc1(x.view(x.size(0), -1)))
        x = self.lrelu(self.fc2(x))
        x = self.lrelu(self.fc3(x))
        
        
        # get mean
        mu = self.mu_layer(x).tanh()
        
        # get std
        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        std = torch.exp(log_std)
        
        # sample actions
        dist = Normal(mu, std)
        z = dist.rsample()
        
        # normalize action and log_prob
        # see appendix C of [2]
        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob
    
    
class CriticQ(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(CriticQ, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, 32, kernel_size=8, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)
        
        
        self.out = nn.Linear(128, 1)
        self.out = init_layer_uniform(self.out)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Forward method implementation."""
        #state = state.float()
        action = action.float()
        action = action.unsqueeze(2).unsqueeze(3).expand(-1,state.size(1),state.size(2),-1)
        
        x = torch.cat((state, action), dim=-1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = self.lrelu(self.fc1(x.view(x.size(0), -1)))
        x = self.lrelu(self.fc2(x))
        x = self.lrelu(self.fc3(x))
        
        value = self.out(x)
        
        return value
    
    
class CriticV(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(CriticV, self).__init__()
        
        self.conv1 = nn.Conv2d(in_dim, 32, kernel_size=8, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)

        self.out = nn.Linear(128, 1)
        self.out = init_layer_uniform(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = self.relu(self.bn1(self.conv1(state)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = self.lrelu(self.fc1(x.view(x.size(0), -1)))
        x = self.lrelu(self.fc2(x))
        x = self.lrelu(self.fc3(x))
        value = self.out(x)
        
        return value

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        # save a transtion
        if len(self.memory) < self.capacity: # if still has capacity, init memory[position]
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args) # push
        self.position = (self.position + 1) % self.capacity # make position stays in [0, capacity]
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class AgentSAC:
    def __init__(self, env, MODE, load_path = './'):
        self.env = env
        self.obs_dim = 3
        self.action_dim = 5

        # automatic entropy tuning
        self.target_entropy = -np.prod((self.action_dim,)).item()  # heuristic
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        # actor
        self.actor = Actor(self.obs_dim, self.action_dim).to(DEVICE)
        
        # v function
        self.vf = CriticV(self.obs_dim).to(DEVICE)
        self.vf_target = CriticV(self.obs_dim).to(DEVICE)
        self.vf_target.load_state_dict(self.vf.state_dict())
        
        # q function
        #self.qf_1 = CriticQ(self.obs_dim + self.action_dim).to(DEVICE)
        #self.qf_2 = CriticQ(self.obs_dim + self.action_dim).to(DEVICE)
        self.qf_1 = CriticQ(self.obs_dim).to(DEVICE)
        self.qf_2 = CriticQ(self.obs_dim).to(DEVICE)
        

        if MODE=='test':
            self.load(load_path)

        # discounted reward
        self.gamma = 0.99

        # training hyperparameters
        self.train_freq = 1 # frequency to train the online network
        self.learning_start = TRAINING_START # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = BATCH_SIZE
        self.num_timesteps = 3000000 # total training steps
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 200000 # frequency to save the model
        self.buffer_size = MEMORY_CAPACITY # max size of replay buffer

        self.tau = 5e-3
        self.policy_update_freq = 2

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=3e-4)
        self.qf_1_optimizer = optim.Adam(self.qf_1.parameters(), lr=3e-4)
        self.qf_2_optimizer = optim.Adam(self.qf_2.parameters(), lr=3e-4)

        self.steps = 0 # num. of passed steps

        # TODO: initialize your replay buffer
        self.memory = ReplayMemory(self.buffer_size)

    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.actor.state_dict(), save_path + '_actor.cpt')
        torch.save(self.vf.state_dict(), save_path + '_vf.cpt')
        torch.save(self.qf_1.state_dict(), save_path + '_qf_1.cpt')
        torch.save(self.qf_2.state_dict(), save_path + '_qf_2.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.actor.load_state_dict(torch.load(load_path + 'sac_actor.cpt', map_location = DEVICE))
            self.vf.load_state_dict(torch.load(load_path + 'sac_vf.cpt', map_location = DEVICE))
            self.qf_1.load_state_dict(torch.load(load_path + 'sac_qf_1.cpt', map_location = DEVICE))
            self.qf_2.load_state_dict(torch.load(load_path + 'sac_qf_2.cpt', map_location = DEVICE))
        else:
            
            self.actor.load_state_dict(torch.load(load_path + 'sac_actor.cpt', map_location = lambda storage, loc: storage))
            self.vf.load_state_dict(torch.load(load_path + 'sac_vf.cpt', map_location = lambda storage, loc: storage))
            self.qf_1.load_state_dict(torch.load(load_path + 'sac_qf_1.cpt', map_location = lambda storage, loc: storage))
            self.qf_2.load_state_dict(torch.load(load_path + 'sac_qf_2.cpt', map_location = lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting
        pass

    def make_action(self, state, test=False):
        # TODO:
        # Implement epsilon-greedy to decide whether you want to randomly select
        # an action or not.
        # HINT: You may need to use and self.steps
        sample = random.random()  # get random number from [0.0, 1.0)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps / EPS_DECAY)
        
        
        if test == True:
            eps_threshold = 0

        if sample > eps_threshold:
            with torch.no_grad():
                # if True, return the move with highest reward
                #print(self.actor(state)[0].size())
                action = self.actor(state)[0].max(1)[1].view(1,1)

        else:
            # if False, randomly select a move
            
            action = torch.tensor([[random.randrange(self.action_dim)]], dtype = torch.long).to(DEVICE)

        if test == True:
            return action.item()
        else:
            return action


    def update(self):
        
        if len(self.memory) < self.batch_size:
            return # return if not enough memory to update
        trans = self.memory.sample(self.batch_size)
        batch = self.memory.transition(*zip(*trans))

        # non-final mask, concatenate batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s : s is not None, batch.next_state)), device = DEVICE, dtype = torch.bool)
        
        non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])

        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        new_action, log_prob = self.actor(state)
        new_action = new_action.max(1)[1].view(-1,1)


        #print(state.size())
        #print(new_action)
        #print(reward.size())
        #torch.autograd.set_detect_anomaly(True)
        # train alpha (dual problem)
        alpha_loss = (
            -self.log_alpha.exp() * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()  # used for the actor loss calculation

        # q function loss
        mask = int(non_final_mask ==  'true')
        q_1_pred = self.qf_1(state, action)
        q_2_pred = self.qf_2(state, action)
        v_target = self.vf_target(non_final_next_state)
        q_target = reward.unsqueeze(1) + self.gamma * v_target *mask
        qf_1_loss = F.mse_loss(q_1_pred, q_target.detach())
        qf_2_loss = F.mse_loss(q_2_pred, q_target.detach())

        qf_loss = qf_1_loss + qf_2_loss

        # v function loss
        v_pred = self.vf(state)
      
        q_pred = torch.min(
            self.qf_1(state, new_action), self.qf_2(state, new_action)
        )

        v_target = q_pred - alpha * log_prob
        vf_loss = F.mse_loss(v_pred, v_target.detach())

        

        if self.steps % self.policy_update_freq == 0:
            # actor loss
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()
            
            
            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            
            self.actor_optimizer.step()
        
            # target update (vf)
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)

        # train Q functions
        self.qf_1_optimizer.zero_grad()
        qf_1_loss.backward()
        self.qf_1_optimizer.step()

        self.qf_2_optimizer.zero_grad()
        qf_2_loss.backward()
        self.qf_2_optimizer.step()
        
        

        # train V function
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        

        return actor_loss.data, qf_loss.data, vf_loss.data, alpha_loss.data

    def train(self):
        record_reward = []
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        losses = (0,0,0,0)
        actor_losses, qf_losses, vf_losses, alpha_losses = [], [], [], []
        best_avg = 0
        while(True):
            state = self.env.reset(resize = RESIZE, size = RESIZE_SIZE)
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).type(torch.cuda.FloatTensor).to(DEVICE)
            done = False
            episode_reward = 0
            while(not done):
                # select and perform action
                action = self.make_action(state).to(DEVICE)
                
                next_state, reward, done, _ = self.env.step(action.item(), resize = RESIZE, size = RESIZE_SIZE)
                total_reward += (reward * REWARD_MULTI)
                episode_reward += (reward * REWARD_MULTI)
                print('\r Now reward: {}'.format(episode_reward), end = '\r')

                reward = torch.tensor([reward]).type(torch.cuda.FloatTensor).to(DEVICE)

                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).type(torch.cuda.FloatTensor).unsqueeze(0)
                next_state = next_state.to(DEVICE)
                

                # TODO: store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # move to the next state
                state = next_state
                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    losses = self.update()
                    
                    actor_losses.append(losses[0])
                    qf_losses.append(losses[1])
                    vf_losses.append(losses[2])
                    alpha_losses.append(losses[3])

                

                

                self.steps += 1
                record_reward.append(reward.item())

            if episodes_done_num % self.display_freq == 0:
                avg_reward = total_reward / self.display_freq
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, avg_reward, losses[2]))
                total_reward = 0
                if avg_reward > best_avg:
                    self.save('sac')
                    best_avg = avg_reward
                np.save('sac_reward', np.array(record_reward))

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                self.save('sac_final')
                break
        self.save('sac')

    def test(self, episodes = 10, saving_path = './test.mp4', size = (750,750), fps = 30):
        saver = ReplaySaver()
        rewards = []
        best_reward = 0

        
        for i in range(episodes):
            done = False
            state = self.env.reset(resize = RESIZE, size = RESIZE_SIZE)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).type(torch.cuda.FloatTensor).to(DEVICE)
            saver.get_current_frame()
            episode_reward = 0.0

            # play one game
            while(not done):
                action = self.make_action(state, test=True)
                state, reward, done, end = self.env.step(action, resize= RESIZE, size = RESIZE_SIZE)
                state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).type(torch.cuda.FloatTensor).to(DEVICE)

                #self.render_saliency(self.env, state, self.actor, action)##

                saver.get_current_frame()
                episode_reward += reward
                print('\r episode: {} | Now reward: {} '.format(i+1,episode_reward), end = '\r')
            
            while(end):
                _,_,_, end = self.env.step(action)
                saver.get_current_frame()
            
            if episode_reward <= best_reward:
                saver.reset()
            else:
                best_reward = episode_reward
                saver.save_best()
                saver.reset()

            rewards.append(episode_reward)
        print('Run %d episodes'%(episodes))
        print('Mean:', np.mean(rewards))
        print('Median:', np.median(rewards))
        print('Saving best reward video')
        saver.make_video(path = saving_path, size = size, fps = fps)
        print('Video saved :)')

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau
        
        for t_param, l_param in zip(
            self.vf_target.parameters(), self.vf.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def render_saliency(self, env, state, policy, action):#

        # get frame
        
        I = env.step(action)[0].astype(np.float32)
        
        

        # compute saliency
        saliency, sign = saliency_map(state, policy, I.shape[:2])

        # red and blue masks
        blue_mask = np.ones_like(I)
        blue_mask[:, :, :] = (0, 0, 255)
        red_mask = np.ones_like(I)
        red_mask[:, :, :] = (255, 0, 0)

        # post processing + normalize
        saliency = (saliency.squeeze().data).cpu().numpy()[:, :, None]
        sign = (sign.squeeze().data).cpu().numpy()[:, :, None]

        global _saliencies
        _saliencies += [np.max(saliency)]
        if len(_saliencies) > 1000:
            _saliencies.pop(0)

        saliency = np.sqrt(saliency / np.percentile(_saliencies, 75))

        thresh = 0.0
        saliency *= 3.0
        saliency = 0.7 * (saliency.clip(thresh, 1.0) - thresh) / (1 - thresh)

        # apply masks
        I = np.where(sign > 0, saliency * red_mask + (1. - saliency) * I, saliency * blue_mask + (1. - saliency) * I)

        # render
        I = I.clip(1, 255).astype('uint8')
        #print(np.shape(I))
        
        I = cv2.resize(I,(750,750))
        import pygame
        surf = pygame.surfarray.make_surface(I)

        env.window.blit(surf, (0,0))

        pygame.display.update()
            




