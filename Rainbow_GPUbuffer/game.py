import os
# If you don't want to display the view, command it.
# os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
pygame.init()
import sys
BULLET_MODE = 'random'
EXPOLDE_MODE = True
PLANE_SHOW = True
SCORE_SHOW = True
SCREEN_SHOW = True

if SCREEN_SHOW == False:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

from rainbow_agent import *
from game_controller import GameManager

env = GameManager(bullet_mode = BULLET_MODE, explode_mode=EXPOLDE_MODE, plane_show=PLANE_SHOW, score_show=SCORE_SHOW)

run = True
while run:
    # action = [random.choice((0,1,2,3,4))]
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        action = 0
    elif keys[pygame.K_RIGHT]:
        action = 1
    elif keys[pygame.K_UP]:
        action = 2
    elif keys[pygame.K_DOWN]:
        action = 3
    else:
        action = 4
    img, score, collision, run = env.step(action)
    # print((counter, np.array(img).shape, score, collision, run))
    # time.sleep(5)
    
