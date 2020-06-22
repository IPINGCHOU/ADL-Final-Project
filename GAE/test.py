import os
import sys
sys.path.insert(1, '../game')

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODE = 'test'
BULLET_MODE = 'random'
EXPOLDE_MODE = False
PLANE_SHOW = False
SCORE_SHOW = False
SCREEN_SHOW = False

# test
TEST_EPISODES = 10

if SCREEN_SHOW == False:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

from game_controller import GameManager
from gae_agent import *

env = GameManager(bullet_mode = BULLET_MODE, explode_mode=EXPOLDE_MODE, plane_show=PLANE_SHOW, score_show=SCORE_SHOW)
agent = AgentPG(env, MODE)
agent.test(episodes=TEST_EPISODES)
