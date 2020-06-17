import os, shutil
import pygame
import argparse


# game settings
GAME_FOLDER = os.path.abspath('.')

# game window
WINOW_WIDTH, WINOW_HEIGHT = 500, 500
FPS = 60

# Plane
PLANE_WIDTH, PLANE_HEIGHT = 26, 50
PLANE_VEL = 3
PLANE_HITBOX_RADIUS = 5
PLANE_SIZE = (PLANE_WIDTH, PLANE_HEIGHT)
PLANE_LEFT_IMAGE  = pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER, 'sprites', 'plane', '{}.png'.format(2))), PLANE_SIZE)
PLANE_STAND_IMAGE = pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER, 'sprites', 'plane', '{}.png'.format(3))), PLANE_SIZE)
PLANE_RIGHT_IMAGE = pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER, 'sprites', 'plane', '{}.png'.format(4))), PLANE_SIZE)

# Explode
EXPLODE_WIDTH, EXPLODE_HEIGHT = 30,30
EXPLODE_LATENCY = 1
EXPLODE_SIZE = (EXPLODE_WIDTH, EXPLODE_HEIGHT)
EXPLODE = []
for i in range(15):
    EXPLODE.append(pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER,'sprites','explode','tile{:03}.png'.format(i))), EXPLODE_SIZE))

# Bullets
BULLET_RADIUS = 2
BULLET_VEL = 3
MAX_BULLETS = 80

# Colors
WHITE  = (255, 255, 255)
BLACK  = (  0,   0,   0)
RED    = (255,   0,   0)
GREEN  = (  0, 255,   0)
BLUE   = (  0,   0, 255)
YELLOW = (255, 255, 255)
