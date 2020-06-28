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
TEST_MODE = sys.argv[1]
TEST_INVINCIBLE = bool(sys.argv[2])
TEST_MAX_T = int(sys.argv[3])
BULLET_MODE = 'random'
EXPOLDE_MODE = False
PLANE_SHOW = False
SCORE_SHOW = False
SCREEN_SHOW = False

# test
TEST_EPISODES = 100


if SCREEN_SHOW == False:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

from game_controller import GameManager
from rainbow_agent import *

MODE = 'test'
env = GameManager(bullet_mode = BULLET_MODE, explode_mode=EXPOLDE_MODE, plane_show=PLANE_SHOW, score_show=SCORE_SHOW)
if TEST_MODE == 'test':
    print('now testing...')
    print((TEST_MODE, TEST_INVINCIBLE, TEST_MAX_T, TEST_EPISODES))
    rainbow = RainbowAgent(env)
    rainbow.load('./rainbow_checkpoint')
    rainbow.test(episodes=TEST_EPISODES, invincible_mode=TEST_INVINCIBLE, test_max_t=TEST_MAX_T)
elif TEST_MODE == 'random':
    print('now random...')
    print((TEST_MODE, TEST_INVINCIBLE, TEST_MAX_T, TEST_EPISODES))
    rainbow = RainbowAgent(env)
    rainbow.test_random(episodes=TEST_EPISODES, invincible_mode=TEST_INVINCIBLE, test_max_t=TEST_MAX_T)