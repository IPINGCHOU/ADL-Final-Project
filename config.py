import os
import pygame

# game settings
# game window
WIN_WIDTH, WIN_HEIGHT = 500, 500
FRAME_RATE = 60
# Plane
PLANE_WIDTH, PLANE_HEIGHT = 26, 50
PLANE_VEL = 3
PLANE_HITBOX_RADIUS = 5
# Explode
EXPLODE_WIDTH, EXPLODE_HEIGHT = 30,30
EXPLODE_LATENCY = 5
# Bullets
BULLET_RADIUS = 2
BULLET_VEL = 3
MAX_BULLETS = 80
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# read plane images
GAME_FOLDER = os.path.abspath('.')
plane_size = (PLANE_WIDTH, PLANE_HEIGHT)
explode_size = (EXPLODE_WIDTH, EXPLODE_HEIGHT)
PLANE_LEFT, PLANE_RIGHT, PLANE_STAND = [], [], []
PLANE_LEFT.append(pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER,'sprites','plane','{}.png'.format(2))), plane_size))
PLANE_LEFT.append(pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER,'sprites','plane','{}.png'.format(1))), plane_size))
PLANE_RIGHT.append(pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER,'sprites','plane','{}.png'.format(4))), plane_size))
PLANE_RIGHT.append(pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER,'sprites','plane','{}.png'.format(5))), plane_size))
PLANE_STAND.append(pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER,'sprites','plane','{}.png'.format(3))), plane_size))
EXPLODE = []
for i in range(15):
    EXPLODE.append(pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER,'sprites','explode','tile{:03}.png'.format(i))), explode_size))

# window = 0
clock = pygame.time.Clock()
def env_init(screen_on = False):
    global window

    if screen_on == False:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()

    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption('bullet hell drill')
    # pygame.time.set_timer(pygame.USEREVENT, 100)
    return window