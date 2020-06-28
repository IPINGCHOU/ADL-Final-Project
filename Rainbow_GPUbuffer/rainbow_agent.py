#%%
import math
import os
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from segment_tree import MinSegmentTree, SumSegmentTree
from utils import *
import sys
sys.path.insert(1, '../game')
from game_config import *
from game_controller import ReplaySaver

#%%
# h-para
DEVICE = 'cuda:0'
REWARD_MULTI = 1
RESIZE = True
print(RESIZE_SIZE)

# model settings
NUM_TIMESTEPS = 10000000000
DISPLAY_FREQ = 1
SAVE_FREQ = 1
MEMORY_CAPACITY = 20000
BATCH_SIZE = 512
TARGET_UPDATE_FREQ = 2000
LEARNING_RATE = 1e-05
N_STEP_LEARNING = 0
MAX_FRACTION_FRAME = 500000

EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 200000
IS_EPS = True

INVINCIBLE = True
VALID_INVINCIBLE = True
EPISODE_MAX_T = 1000
VALID_MAX_T = 500

#%%
# def double_conv(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, padding=1),
#         nn.ReLU(inplace=True),
#         nn.BatchNorm2d(out_channels),
#         nn.Conv2d(out_channels, out_channels, 3, padding=1),
#         nn.ReLU(inplace=True),
#         nn.BatchNorm2d(out_channels)
#     ) 

# def dilation_conv(in_channels, out_channels, dilation):
#     return nn.Sequential(
#         nn.ZeroPad2d(dilation),
#         nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0, dilation = dilation),
#         nn.ReLU(inplace = True)
#     )

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = conv3x3(in_channels, out_channels, stride)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(out_channels, out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample

#         def forward(self, x):
#             residual = x
#             out = self.conv1(x)
#             out = self.bn1(out)
#             out = self.relu(out)
#             out = self.conv2(out)
#             out = self.bn2(out)
#             if self.downsample:
#                 residual = self.downsample(x)
#             out += residual
#             out = self.relu(out)
#             return out

class Network(nn.Module):
    def __init__(
        self, 
        channels: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor,
        NoisyNet
    ):
        """Initialization."""
        super(Network, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size
        
        # set advantage layer
        if IS_EPS == False:
            self.advantage_hidden_layer = NoisyNet(128, 128)
            self.advantage_layer = NoisyNet(128, out_dim * atom_size)

            # set value layer
            self.value_hidden_layer = NoisyNet(128, 128)
            self.value_layer = NoisyNet(128, atom_size)
        else:
            self.advantage_hidden_layer = nn.Linear(128,128)
            self.advantage_layer = nn.Linear(128, out_dim * atom_size)
            self.value_hidden_layer = nn.Linear(128,128)
            self.value_layer = nn.Linear(128, atom_size)

        # # for features
        self.output_kernel = 3
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=7, stride=3, padding = 0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding = 0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding = 0)
        self.top_layer = nn.Conv2d(128, self.output_kernel, kernel_size = 1, stride = 1)

        self.latlayer3 = nn.Conv2d(128,self.output_kernel,1,1)
        self.latlayer2 = nn.Conv2d(64,self.output_kernel,1,1)
        self.latlayer1 = nn.Conv2d(32,self.output_kernel,1,1)

        self.smooth3 = nn.Conv2d(3,3, kernel_size = 3, stride = 1, padding = 1)
        self.smooth2 = nn.Conv2d(3,3, kernel_size = 3, stride = 1, padding = 1)
        self.smooth1 = nn.Conv2d(3,3, kernel_size = 3, stride = 1, padding = 1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # for fcn
        self.upconv1 = nn.Conv2d(64, 128, kernel_size = 1, stride = 1)

        self.predict_f3 = nn.Linear(12*12*self.output_kernel, 128)
        self.predict_f2 = nn.Linear(29*29*self.output_kernel, 128)
        self.predict_f1 = nn.Linear(65*65*self.output_kernel, 128)
        self.predict_top = nn.Linear(12*12*self.output_kernel, 128)

        self.predict_output = nn.Linear(128*4, 128)
        

        # for U-net dilation features
        # u down
        # self.dconv_down1 = double_conv(3, 8)
        # self.dconv_down2 = double_conv(8, 16)
        # self.dconv_down3 = double_conv(16, 32)
        # self.dconv_down4 = double_conv(128, 256)     

        # self.maxpool = nn.MaxPool2d(2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)   
        # u up
        # self.dconv_up3 = double_conv(32 + 32, 32)
        # self.dconv_up2 = double_conv(16 + 16, 8)
        # self.dconv_up1 = double_conv(8 + 8, 4)
        # parallel dilated conv
        # self.dilate1 = dilation_conv(16,16,1)
        # self.dilate2 = dilation_conv(16,16,2)
        # self.dilate3 = dilation_conv(16,16,4)
        # self.dilate4 = dilation_conv(32,32,8)
        # self.dilate5 = dilation_conv(128,128,16)
        # self.dilate6 = dilation_conv(128,128,32)

        # self.conv_out1 = nn.Conv2d(4, 3, 8, stride = 2)
        # self.conv_out2 = nn.Conv2d(3, 2, 6, stride = 2)
        # self.conv_out3 = nn.Conv2d(2, 1, 1)
        # self.conv_out4 = nn.Conv2d(4, 1, 1)

        self.fc1 = nn.Linear(2116, 256)
        self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(256, 128)

        self.relu = nn.ReLU(inplace = True)
        self.lrelu = nn.LeakyReLU(0.01)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        # upsample is deprecated
        return F.interpolate(x, size=(H,W), mode='bilinear') + y

    def fpn_feature_layer(self,x):
        f1 = self.relu(self.bn1(self.conv1(x)))
        f2 = self.relu(self.bn2(self.conv2(self.maxpool(f1))))
        f3 = self.relu(self.bn3(self.conv3(self.maxpool(f2))))
        top = self.top_layer(f3)

        f3 = self._upsample_add(top, self.latlayer3(f3))
        f2 = self._upsample_add(f3, self.latlayer2(f2))
        f1 = self._upsample_add(f2, self.latlayer1(f1))

        f3 = self.smooth3(f3)
        f2 = self.smooth2(f2)
        f1 = self.smooth1(f1)

        # print(f1.shape)
        # print(f2.shape)
        # print(f3.shape)
        # print(top.shape)

        top = self.relu(self.predict_top(top.view(top.size(0), -1)))
        f1 = self.relu(self.predict_f1(f1.view(f1.size(0), -1)))
        f2 = self.relu(self.predict_f2(f2.view(f2.size(0), -1)))
        f3 = self.relu(self.predict_f3(f3.view(f3.size(0), -1)))

        x = self.predict_output(torch.cat((top,f1,f2,f3), dim = 1))
        # print(x)
        return x

    # def U_feature_layer(self,x):
    #     conv1 = self.dconv_down1(x)
    #     x = self.maxpool(conv1)
    #     conv2 = self.dconv_down2(x)
    #     x = self.maxpool(conv2)

    #     # parallel dilated conv
    #     d_1 = self.dilate1(x)
    #     d_2 = self.dilate2(d_1)
    #     d_3 = self.dilate3(d_2)
    #     d_sum  = d_1 + d_2 + d_3

    #     x = self.upsample(d_sum)
    #     x = torch.cat([x, conv2], dim = 1)
    #     x = self.dconv_up2(x)
    #     x = self.upsample(x)
    #     x = torch.cat([x, conv1], dim = 1)
    #     x = self.dconv_up1(x)

    #     x = self.conv_out1(x)
    #     x = self.conv_out2(x)
    #     x = self.conv_out3(x)
    #     x = self.lrelu(self.fc1(x.view(x.size(0), -1)))
    #     x = self.lrelu(self.fc2(x))

    #     return x

    # def feature_layer(self, x):
    #     x = self.relu(self.bn1(self.conv1(x)))
    #     x = self.relu(self.bn2(self.conv2(x)))
    #     x = self.relu(self.bn3(self.conv3(x)))
    #     x = self.relu(self.bn4(self.conv4(x)))
    #     x = self.relu(self.bn5(self.conv5(x)))
    #     #  print(x.shape)
    #     x = self.lrelu(self.fc1(x.view(x.size(0), -1)))
    #     x = self.lrelu(self.fc2(x))
    #     x = self.lrelu(self.fc3(x))
    #     # x = self.lrelu(self.fc4(x))

    #     return x

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        # feature = self.feature_layer(x)
        # feature = self.U_feature_layer(x)
        feature = self.fpn_feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )

        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

#%%
class RainbowAgent:
    def __init__(
        self, 
        env,
        memory_size: int = MEMORY_CAPACITY,
        batch_size: int = BATCH_SIZE,
        target_update: int = TARGET_UPDATE_FREQ,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = N_STEP_LEARNING,
        # training
        num_timesteps: int = NUM_TIMESTEPS,
        display_freq: int = DISPLAY_FREQ,
        save_freq: int = SAVE_FREQ
    ):

        self.valid_reward = []
        self.valid_hit_list = []
        channels = 3
        obs_dim = env.obs_resize_shape
        self.action_dim = env.action_space
        self.num_timesteps = num_timesteps
        self.display_freq = display_freq
        self.save_freq = save_freq
        self.fraction_max_frame = MAX_FRACTION_FRAME
        self.checkpoint_n = 0
        
        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        # NoisyNet: All attributes related to epsilon are removed
        
        # device: cpu / gpu
        self.device = DEVICE
        self.invincible = INVINCIBLE
        
        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps

        self.input_dim = obs_dim
        get = self.input_dim.pop(2)
        self.input_dim.insert(0, get)


        self.memory = PrioritizedReplayBuffer(
            self.input_dim, memory_size, self.device, batch_size, alpha=alpha
        )
        
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                self.input_dim, memory_size, self.device, batch_size = batch_size, n_step=n_step, gamma=gamma
            )
            
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(
            channels, self.action_dim, self.atom_size, self.support, NoisyLinear
        ).to(self.device)
        self.dqn_target = Network(
            channels, self.action_dim, self.atom_size, self.support, NoisyLinear
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.RMSprop(self.dqn.parameters(), lr = LEARNING_RATE)

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        
        temp_trans = [state]
        if IS_EPS == False:
            with torch.no_grad():
                selected_action = self.dqn(state).argmax()
                # selected_action = selected_action.detach().cpu().numpy()
                selected_action = selected_action.detach()
        else:
            sample = random.random()  # get random number from [0.0, 1.0)
            
            if self.is_test == True:
                eps_threshold = 0
            else:
                eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps / EPS_DECAY)
            if sample > eps_threshold:
                with torch.no_grad():
                    # print(self.dqn(state))
                    selected_action = self.dqn(state).argmax()
                    selected_action = selected_action.detach()
            
            else:
                selected_action = torch.tensor([[random.randrange(self.action_dim)]], dtype = torch.long)


        temp_trans.append(selected_action)
        if not self.is_test:
            self.transition = temp_trans
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, end = self.env.step(action, resize=RESIZE, size = RESIZE_SIZE, invincible = self.invincible)
        # print(type(next_state))
        if done:
            if self.invincible == True and self.t <= EPISODE_MAX_T:
                done = False
            else:
                done = True

        next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0).type(torch.FloatTensor).to(self.device)
        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done, end

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        # loss.retain_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        if IS_EPS == False:
            self.dqn.reset_noise()
            self.dqn_target.reset_noise()

        # return loss.item(), torch.sum(torch.abs(self.dqn.advantage_layer.weight_sigma.grad)).item()
        return loss.item()
    
    def save(self, save_path):
        print('Save model to', save_path)
        torch.save(self.dqn.state_dict(), save_path + '_online.cpt')
        torch.save(self.dqn_target.state_dict(), save_path + '_target.cpt')
    
    def load(self, load_path):
        print('Load model from', load_path)
        self.dqn.load_state_dict(torch.load(load_path + '_online.cpt', map_location = DEVICE))
        self.dqn_target.load_state_dict(torch.load(load_path + '_target.cpt', map_location = DEVICE))

    def train(self):
        """Train the agent."""
        self.steps = 0
        self.is_test = False
        update_cnt = 0
        total_reward = 0
        episodes_done_num = 0
        best_avg = 0
        record_reward = []
        losses = []

        while(True):
            state = self.env.reset(resize = RESIZE, size = RESIZE_SIZE)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).type(torch.FloatTensor).to(self.device)
            done = False
            episode_reward = 0
            frame_idx = 0
            loss = 0
            self.t = 0
            check = 0
            while(not done):
                self.t += 1
                frame_idx +=1
                self.steps += 1
                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                # print(reward, done, self.invincible)

                state = next_state
                total_reward += reward
                episode_reward += reward
                print('\r Now reward: {} | frame: {}'.format(episode_reward, self.t), end = '\r')
                
                # NoisyNet: removed decrease of epsilon
                
                # PER: increase beta
                fraction = min(frame_idx / self.fraction_max_frame, 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)

                # if training is ready
                if len(self.memory) >= self.batch_size:
                    loss = self.update_model()
                    update_cnt += 1
                    losses.append(loss)
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

                record_reward.append(reward)

            self.t = 0
            if episodes_done_num % self.display_freq == 0:
                avg_reward = total_reward / self.display_freq
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, avg_reward, loss))
                total_reward = 0
                if avg_reward > best_avg and (len(self.memory) >= self.batch_size):
                    self.save('rainbow')
                    best_avg = avg_reward
                np.save('./checkpoints/rainbow_reward', np.array(record_reward))
                np.save('./checkpoints/rainbow_losses', np.array(losses))

                if episodes_done_num % SAVE_FREQ == 0:
                    if episodes_done_num != 0:
                        self.save('rainbow_checkpoint')
                    self.validation()
                    self.checkpoint_n += 1


            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                self.save('rainbow_final')
                break


    def validation(self):
        self.is_valid = True
        self.is_test = True
        saver = ReplaySaver()
        state = self.env.reset(resize = RESIZE, size = RESIZE_SIZE)
        state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).type(torch.FloatTensor).to(self.device)
        done = False
        total_reward = 0

        if VALID_INVINCIBLE == True:
            self.invincible = True
        else:
            self.invincible = False
        
        self.valid_hit = 0
        vaild_t = 0
        if VALID_INVINCIBLE == False:
            while not done:
                action = self.select_action(state)
                state, reward, done, end = self.step(action)
                total_reward += reward
                print('\r Now rewards: {}'.format(total_reward), end = '\r')
                saver.get_current_frame()
                
        else:
            while (not done and vaild_t <= VALID_MAX_T):
                vaild_t += 1
                action = self.select_action(state)
                state, reward, done, end = self.step(action)
                total_reward += reward
                print('\r Now rewards: {}'.format(total_reward), end = '\r')
                saver.get_current_frame()
                self.valid_hit += self.env.is_collision()
            self.valid_hit_list.append(self.valid_hit)
            np.save('./checkpoints/valid_hits', np.array(self.valid_hit_list))
        
        self.valid_reward.append(total_reward)
        print('Valid reward: {} | Hit get: {}'.format(total_reward, self.valid_hit))
        np.save('./checkpoints/valid_reward', np.array(self.valid_reward))
        saver.save_best()
        print('Making video...')
        saver.make_video('./checkpoints/checkpoint_{}.mp4'.format(self.checkpoint_n))
        print('Validation video made!!')
        
        self.is_test = False
        self.is_valid = False
        if INVINCIBLE == True:
            self.invincible = True
        else:
            self.invincible = False

    def test(self, episodes = 10, saving_path = './test.mp4', size = (750,750), fps = 30, invincible_mode = False, test_max_t = 500):
        saver = ReplaySaver()
        rewards = []
        best_reward = -np.inf
        self.is_test = True
        if invincible_mode == True:
            self.invincible = True
        else:
            self.invincible = False

        for i in range(episodes):
            done = False
            state = self.env.reset(resize = RESIZE, size = RESIZE_SIZE)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).type(torch.cuda.FloatTensor).to(DEVICE)
            saver.get_current_frame()
            episode_reward = 0.0

            # play one game
            if self.invincible == False:
                while(not done):
                    action = self.select_action(state)
                    state, reward, done, end = self.step(action)
                    
                    saver.get_current_frame()
                    episode_reward += reward
                    print('\r episode: {} | Now reward: {} '.format(i+1,episode_reward), end = '\r')
            else:
                self.t = 0
                while(self.t <= test_max_t):
                    self.t+=1
                    action = self.select_action(state)
                    state, reward, done, end = self.step(action)
                    
                    saver.get_current_frame()
                    episode_reward += reward
                    print('\r episode: {} | Now reward: {} '.format(i+1,episode_reward), end = '\r')
            
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
        np.save('./checkpoints/test_reward', rewards)
        print('Video saved :)')

    def test_random(self, episodes = 10, saving_path = './test_random.mp4', size = (750,750), fps = 30, invincible_mode = False, test_max_t = 500):
        saver = ReplaySaver()
        rewards = []
        best_reward = -np.inf
        self.is_test = True

        if invincible_mode == True:
            self.invincible = True
        else:
            self.invincible = False

        for i in range(episodes):
            done = False
            state = self.env.reset(resize = RESIZE, size = RESIZE_SIZE)
            saver.get_current_frame()
            episode_reward = 0.0

            # play one game
            if self.invincible == False:
                while(not done):
                    action = np.random.choice(5, 1)[0]
                    state, reward, done, end = self.env.step(action, resize=RESIZE, size = RESIZE_SIZE, invincible = self.invincible)
                    
                    saver.get_current_frame()
                    episode_reward += reward
                    print('\r episode: {} | Now reward: {} '.format(i+1,episode_reward), end = '\r')
            
            else:
                t = 0
                while(t <= test_max_t):
                    t+=1
                    action = np.random.choice(5, 1)[0]
                    state, reward, done, end = self.env.step(action, resize=RESIZE, size = RESIZE_SIZE, invincible = self.invincible)
                    
                    saver.get_current_frame()
                    episode_reward += reward
                    print('\r episode: {} | Now reward: {} '.format(i+1,episode_reward), end = '\r')
            
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
        np.save('./checkpoints/test_random_reward', rewards)
        print('Video saved :)')

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        # state = torch.FloatTensor(samples["obs"])
        # next_state = torch.FloatTensor(samples["next_obs"])
        # action = torch.LongTensor(samples["acts"])
        # reward = torch.FloatTensor(samples["rews"].reshape(-1, 1))
        # done = torch.FloatTensor(samples["done"].reshape(-1, 1))

        state = samples["obs"]
        next_state = samples["next_obs"]
        action = samples["acts"]
        reward = samples["rews"].view(-1, 1)
        done = samples["done"].view(-1, 1)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action.type(torch.cuda.LongTensor)])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                


# %%
