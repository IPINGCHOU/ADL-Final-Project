#%%
import os
import random 
import pygame
import numpy as np
import math
pygame.init()

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
YELLOW = (255, 255, 255)

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

window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption('bullet hell drill')
clock = pygame.time.Clock()

class Plane(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = PLANE_WIDTH
        self.height = PLANE_HEIGHT
        self.vel = PLANE_VEL
        self.left = False
        self.right = False
        self.horizontal_move_count = 0
        self.hitbox = (self.x, self.y)
        self.explode = 0
    
    def render(self, window):
        x_render = self.x-PLANE_WIDTH/2
        y_render = self.y-PLANE_HEIGHT/2
        if self.left:
            if self.horizontal_move <= 5:
                window.blit(PLANE_LEFT[0], (x_render,y_render))
            else:
                window.blit(PLANE_LEFT[1], (x_render,y_render))
            self.horizontal_move += 1
        elif self.right:
            if self.horizontal_move <= 5:
                window.blit(PLANE_RIGHT[0], (x_render,y_render))
            else:
                window.blit(PLANE_RIGHT[1], (x_render,y_render))
            self.horizontal_move += 1
        else:
            window.blit(PLANE_STAND[0], (x_render,y_render))
            horizontal_move = 0
        
        # draw hitbox
        self.hitbox = (self.x, self.y)
        pygame.draw.circle(window, RED, self.hitbox, PLANE_HITBOX_RADIUS, 2)

class Bullet(object):
    def __init__(self, color, plane_x, plane_y):
        self.color = color
        self.radius = BULLET_RADIUS
        self.vel = BULLET_VEL

        if random.choice((0,1)) == 0: # bullet come from left or right
            self.x = random.choice((0,WIN_WIDTH))
            self.y = random.uniform(WIN_HEIGHT, 0)
        else: # bullet come from top or buttom
            self.x = random.uniform(WIN_WIDTH, 0)
            self.y = random.choice((0, WIN_HEIGHT))

        x_diff = plane_x - self.x
        y_diff = plane_y - self.y
        angle = math.atan2(y_diff, x_diff)
        self.change_x = math.cos(angle) * BULLET_VEL
        self.change_y = math.sin(angle) * BULLET_VEL
    
    def move(self):
        self.x += self.change_x
        self.y += self.change_y
    
    def render(self, window):
        pygame.draw.circle(window, self.color, (int(self.x), int(self.y)), self.radius)

class Explode(object):
    def __init__(self, x, y):
        self.x = x - PLANE_WIDTH/2
        self.y = y - PLANE_HEIGHT/2
        self.tick_counter = 0
        self.ani_counter = 0
        self.length = len(EXPLODE)

    def render(self, window):
        window.blit(EXPLODE[self.ani_counter], (self.x, self.y))
        self.tick_counter += 1
        # self.ani_counter += 1

        if self.tick_counter % EXPLODE_LATENCY == 0:
            self.ani_counter += 1


def WindowRender():
    window.fill((0,0,0))
    plane.render(window)
    for bullet in bullets:
        bullet.render(window)
    window.blit(font.render('Score(ms): ' + str(score),1, WHITE), (190,10))

    pygame.display.update()

def CheckCollision(plane, bullets):
    for bullet in bullets:
        distance = math.hypot(plane.x - bullet.x, plane.y - bullet.y)
        if distance < (PLANE_HITBOX_RADIUS + BULLET_RADIUS):
            return True
    return False

def GameOverWindowRender(bullets):
    window.fill((0,0,0))
    explode.render(window)
    pygame.draw.circle(window, RED, plane.hitbox, PLANE_HITBOX_RADIUS, 2)
    for bullet in bullets:
        bullet.render(window)
    window.blit(font.render('Score(ms): ' + str(score),1, WHITE), (190,10))
    pygame.display.update()
    if explode.ani_counter < len(EXPLODE):
        return True
    else:
        return False
            
# init objects
plane = Plane(250,250) # init starting point of the plane
bullets = []
run = True
collision = False
explode_made = False
# set survive timer
pygame.time.set_timer(pygame.USEREVENT, 100)
score = 0
font = pygame.font.SysFont("comicsans", 30, True)

while run:
    clock.tick(FRAME_RATE)

    if collision == False: # plane still alive
        for event in pygame.event.get():  # This will loop through a list of any keyboard or mouse events.
            if event.type == pygame.QUIT: # Checks if the red button in the corner of the window is clicked
                run = False  # Ends the game loop
            if event.type == pygame.USEREVENT:
                score += 1

        while len(bullets) < MAX_BULLETS:
            bullets.append(Bullet(WHITE, plane.x, plane.y))
        
        for bullet in bullets:
            bullet_exist = False
            if 0 <= bullet.x and bullet.x <= WIN_WIDTH :
                if 0 <= bullet.y and bullet.y <= WIN_HEIGHT:
                    bullet.move()
                    bullet_exist = True

            if bullet_exist == False:
                bullets.pop(bullets.index(bullet))

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and plane.x > PLANE_VEL + PLANE_HITBOX_RADIUS: 
            plane.x -= PLANE_VEL
            plane.left = True
            plane.right = False
        elif keys[pygame.K_RIGHT] and plane.x < 500 - PLANE_VEL - PLANE_HITBOX_RADIUS:
            plane.x += PLANE_VEL
            plane.left = False
            plane.right = True
        else:
            plane.left, plane.right = False, False
            plane.horizontal_move = 0

        if keys[pygame.K_UP] and plane.y > 0 + PLANE_HITBOX_RADIUS:
            plane.y -= PLANE_VEL
        if keys[pygame.K_DOWN] and plane.y < 500 - PLANE_HITBOX_RADIUS:
            plane.y += PLANE_VEL

        WindowRender()

    else: # plane crashed
        if explode_made == False:
            explode = Explode(plane.x, plane.y)
            explode_made = True
        
        run = GameOverWindowRender(bullets)

    collision = CheckCollision(plane, bullets)
    
pygame.quit()