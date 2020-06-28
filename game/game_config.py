import os, shutil
import pygame

# game settings
GAME_FOLDER = os.path.abspath('..')

# game window
WINDOW_WIDTH, WINDOW_HEIGHT = 750, 750
BORDER_WIDTH, BORDER_HEIGHT = 600,600
BORDER_LEN = 3
RESIZE_SIZE = (200,200)
FPS = 1000

# Plane
PLANE_WIDTH, PLANE_HEIGHT = 30,60
PLANE_VEL = 5
PLANE_HITBOX_RADIUS = 6
PLANE_WARNING_RADIUS = 20
PLANE_WARNING_CIRCLE_WIDTH = 3
IS_WARNING = False

PLANE_SIZE = (PLANE_WIDTH, PLANE_HEIGHT)
PLANE_LEFT_IMAGE  = pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER, 'sprites', 'plane', '{}.png'.format(2))), PLANE_SIZE)
PLANE_STAND_IMAGE = pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER, 'sprites', 'plane', '{}.png'.format(3))), PLANE_SIZE)
PLANE_RIGHT_IMAGE = pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER, 'sprites', 'plane', '{}.png'.format(4))), PLANE_SIZE)

# Explode
EXPLODE_WIDTH, EXPLODE_HEIGHT = 90,90
EXPLODE_LATENCY = 2
EXPLODE_SIZE = (EXPLODE_WIDTH, EXPLODE_HEIGHT)
EXPLODE = []
for i in range(15):
    EXPLODE.append(pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER,'sprites','explode','tile{:03}.png'.format(i))), EXPLODE_SIZE))

# Bullets
BULLET_RADIUS = 5
BULLET_VEL = 5
MAX_BULLETS = 100
COLL_TOLERANCE = 2
MAX_ADDIN_BULLETS = 1

# Score
DEAD_PUNISH = -100
WARNING_PUNISH = -1
SURVIVE_SCORE = 0.1

# Colors
WHITE  = (255, 255, 255)
BLACK  = (  0,   0,   0)
RED    = (255,   0,   0)
GREEN  = (  0, 255,   0)
BLUE   = (  0,   0, 255)
YELLOW = (255, 255,  50)
