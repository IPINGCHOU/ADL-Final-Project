from agent import *
import os
import sys
sys.path.insert(1, '../game')


# parameters
num_frames = 10000000
memory_size = 1000
batch_size = 32
target_update = 100

MODE = 'train'
BULLET_MODE = 'random'
EXPOLDE_MODE = False
PLANE_SHOW = False
SCORE_SHOW = False
SCREEN_SHOW = False

if not SCREEN_SHOW:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

from game_controller import GameManager
# train
env = GameManager(bullet_mode = BULLET_MODE, explode_mode=EXPOLDE_MODE, plane_show=PLANE_SHOW, score_show=SCORE_SHOW)
agent = DQNAgent(env, memory_size, batch_size, target_update)

agent.train(num_frames)