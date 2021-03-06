import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from game_controller import ReplaySaver

total_epoch = 10000000
lr = 1e-5
display_freq = 10
gamma = 0.99
batch_size = 4
rw_path = "rw.npy"
loss_path = "loss.npy"
model_path = "best_model"
is_resize = True
resize_size = (100, 100)
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_start = 0

INVINCIBLE = True
EPISODE_MAX_T = 1000

ppo_clip = 0.2
ppo_steps = 5

class MLP(nn.Module):
    def __init__(self, channels, num_actions):
        super(MLP, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_actions)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = self.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))

        x = self.out(x)
        
        return x

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        
        self.actor = actor
        self.critic = critic
        
    def forward(self, state):
        
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        
        return action_pred, value_pred

class AgentPG:
    def __init__(self, env, mode):
        self.env = env

        actor = MLP(channels=3, num_actions=5)
        critic = MLP(channels=3, num_actions=1)

        self.model = ActorCritic(actor, critic).to(device)
        self.model_path = model_path
        self.rw_path = rw_path
        self.loss_path = loss_path
        self.device = device
        self.learning_start = learning_start
        self.invincible = INVINCIBLE

        # ppo parameters
        self.ppo_clip = ppo_clip
        self.ppo_steps = ppo_steps
        
        if mode == "test":
            self.load(self.model_path)
            
        # discounted reward
        self.gamma = gamma

        # training hyperparameters
        self.total_epoch = total_epoch # total training episodes (actually too large...)
        self.display_freq = display_freq # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)

        # saved rewards and actions
        self.rewards, self.saved_log_probs, self.value_list, self.actions, self.states = [], [], [], [], []


    def save(self, save_path):
        print('save model to', save_path)
        save_path = save_path + '.cpt'
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        load_path = load_path + '.cpt'
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_log_probs, self.value_list, self.actions, self.states = [], [], [], [], []

    def make_action(self, state, test=False):
        # action = self.env.action_space.sample() # TODO: Replace this line!
        # Use your model to output distribution over actions and sample from it.
        # HINT: torch.distributions.Categorical
        # state = torch.from_numpy(state).float().unsqueeze(0)

        action_pred, value_pred = self.model(state)
        action_prob = F.softmax(action_pred, dim = 1)
        m = Categorical(action_prob)
        action = m.sample()

        self.states.append(state)
        self.actions.append(action)
        self.saved_log_probs.append(m.log_prob(action))
        self.value_list.append(value_pred)

        
        return action.item()
        

    def update(self):
        # discount reward
        # R_i = r_i + GAMMA * R_{i+1}
        r_list = []
        R = 0
        eps = np.finfo(np.float32).eps.item()
        norm = lambda a: (a - a.mean()) / (a.std() + eps)
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            r_list.append(R)
        r_list = r_list[::-1]
        r_list = torch.tensor(r_list)
        r_list = norm(r_list).to(self.device)

        states = torch.cat(self.states)
        actions = torch.cat(self.actions).to(self.device)
        saved_log_probs = torch.cat(self.saved_log_probs).to(self.device)
        v_list = torch.cat(self.value_list).squeeze(-1)
        advantage = r_list - v_list
        advantage = norm(advantage)

        # compute PG loss
        # loss = sum(-R_i * log(action_prob))
        states = states.detach()
        actions = actions.detach()
        log_prob_actions = saved_log_probs.detach()
        advantage = advantage.detach()
        r_list = r_list.detach()
        
        loss = 0
        for _ in range(self.ppo_steps):
            #get new log prob of actions for all input states
            action_pred, value_pred = self.model(states)
            value_pred = value_pred.squeeze(-1)
            action_prob = F.softmax(action_pred, dim = -1)
            dist = Categorical(action_prob)

            #new log prob using old actions
            new_log_prob_actions = dist.log_prob(actions)

            policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
            policy_loss_1 = policy_ratio * advantage
            policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - self.ppo_clip, max = 1.0 + self.ppo_clip) * advantage

            policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()
            value_loss = F.smooth_l1_loss(r_list, value_pred).mean()
            loss += policy_loss.item() + value_loss.item()

            self.optimizer.zero_grad()
            
            policy_loss.backward()
            value_loss.backward()

            self.optimizer.step()
        loss /= self.ppo_steps
        return loss
        
    def step(self, action):
        """Take an action and return the response of the env."""
        next_state, reward, done, end = self.env.step(action, resize=is_resize, size = resize_size, invincible = self.invincible)

        if done:
            if self.invincible == True and self.t <= EPISODE_MAX_T:
                done = False
            else:
                done = True
    
        return next_state, reward, done, end
        
    def train(self):
        st_time = datetime.now()
        bst_reward = -1e9
        rw_list = []
        loss_list = []
        
        trange = tqdm(range(self.total_epoch), total = self.total_epoch)

        for epoch in trange:
            avg_reward = 0
            state = self.env.reset(resize = is_resize, size = resize_size)

            self.init_game_setting()
            done = False
            self.t = 0
            while(not done):
                self.t += 1
                state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).type(torch.cuda.FloatTensor).to(self.device)
                action = self.make_action(state)
                state, reward, done, _ = self.step(action)
                #state, reward, done, _ = self.env.step(action)

                self.rewards.append(reward)

            self.t = 0
            # update model
            loss = self.update()
            loss_list.append(loss)

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward += last_reward 
            rw_list.append(last_reward)

            if epoch > self.learning_start and epoch % self.display_freq == 0:
                avg_reward /= self.display_freq

                if avg_reward > bst_reward:
                    bst_reward = avg_reward
                    self.save(self.model_path)
                    
                    print("Model saved!!")
                
                trange.set_postfix(
                    Avg_reward = avg_reward,
                    Bst_reward = bst_reward,
                )

                np.save(self.rw_path, rw_list)
                np.save(self.loss_path, loss_list)
                avg_reward = 0

        
        print(f"Cost time: {datetime.now()-st_time}")
        
    def test(self, episodes = 10, saving_path = './test.mp4', size = (750,750), fps = 60):
        saver = ReplaySaver()
        rewards = []
        best_reward = -1234567890

        
        for i in range(episodes):
            done = False
            state = self.env.reset(resize = is_resize, size = resize_size)
            
            saver.get_current_frame()
            episode_reward = 0.0

            # play one game
            while(not done):
                state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).type(torch.cuda.FloatTensor).to(self.device)
                action = self.make_action(state, test=True)
                state, reward, done, end = self.env.step(action, resize= is_resize, size = resize_size)

                saver.get_current_frame()
                episode_reward += reward
                print('\r episode: {} | Now reward: {} '.format(i+1,episode_reward), end = '\r')
            
            while(end):
                _, _ , _, end = self.env.step(action)
                saver.get_current_frame()
            
            if episode_reward <= best_reward:
                saver.reset()
            else:
                best_reward = episode_reward
                saver.save_best()
                saver.reset()

            rewards.append(episode_reward)
        print('\nRun %d episodes'%(episodes))
        print('Mean:', np.mean(rewards))
        print('Median:', np.median(rewards))
        print('Saving best reward video')
        saver.make_video(path = saving_path, size = size, fps = fps)
        print('Video saved :)')
