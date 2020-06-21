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
BULLET_MODE = 'random'
EXPOLDE_MODE = True
PLANE_SHOW = False
SCORE_SHOW = False
SCREEN_SHOW = True

# test
TEST_EPISODES = 3


if SCREEN_SHOW == False:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

from game_controller import GameManager
from sac_agent import *

MODE = 'test'
env = GameManager(bullet_mode = BULLET_MODE, explode_mode=EXPOLDE_MODE, plane_show=PLANE_SHOW, score_show=SCORE_SHOW)
sac = AgentSAC(env, MODE, load_path='./')
sac.test(episodes=TEST_EPISODES)