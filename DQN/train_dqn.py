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
BULLET_MODE = 'random'
EXPOLDE_MODE = False
PLANE_SHOW = False
SCORE_SHOW = False
SCREEN_SHOW = False

if SCREEN_SHOW == False:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

from game_controller import GameManager
from dqn_agent import *

env = GameManager(bullet_mode = BULLET_MODE, explode_mode=EXPOLDE_MODE, plane_show=PLANE_SHOW, score_show=SCORE_SHOW)
dqn = AgentDQN(env, MODE)
dqn.train()
