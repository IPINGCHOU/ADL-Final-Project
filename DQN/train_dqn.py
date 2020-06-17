#%%
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
import sys
sys.path.insert(1, '../game')
from collections import namedtuple

#%%
MODE = 'train'
EXPOLDE_MODE = False
SCREEN_SHOW = False

if SCREEN_SHOW == False:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

from game_controller import GameManager
from dqn_agent import *

env = GameManager(False)
dqn = AgentDQN(env, MODE)
dqn.train()

#%%