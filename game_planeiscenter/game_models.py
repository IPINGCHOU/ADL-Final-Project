from game_config import *
import os, shutil
import pygame
import random 
import numpy as np
import math
import time

class Plane(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = PLANE_WIDTH
        self.height = PLANE_HEIGHT
        self.velocity = PLANE_VEL
        self.hitbox = (self.x, self.y)
        self.explode = 0
        self.image = PLANE_STAND_IMAGE
    
    def move(self, action):

        # action = action[0]
        # choose image and move
        if action == 0: 
            self.x -= PLANE_VEL
            self.image = PLANE_LEFT_IMAGE
        elif action == 1:
            self.x += PLANE_VEL
            self.image = PLANE_RIGHT_IMAGE
        elif action == 2:
            self.y -= PLANE_VEL
            self.image = PLANE_STAND_IMAGE
        elif action == 3:
            self.y += PLANE_VEL
            self.image = PLANE_STAND_IMAGE
        elif action == 4:
            self.image = PLANE_STAND_IMAGE

        # boundary limit
        border_x = int(WINDOW_WIDTH/2 - BORDER_WIDTH/2)
        border_y = int(WINDOW_HEIGHT/2 - BORDER_HEIGHT/2)
        if self.x - (PLANE_HITBOX_RADIUS + border_x) < 0:
            self.x = PLANE_HITBOX_RADIUS + border_x
        if self.x + PLANE_HITBOX_RADIUS > WINDOW_WIDTH - border_x:
            self.x = WINDOW_WIDTH - PLANE_HITBOX_RADIUS - border_x

        if self.y - (PLANE_HITBOX_RADIUS + border_y) < 0:
            self.y = PLANE_HITBOX_RADIUS + border_y
        if self.y + PLANE_HITBOX_RADIUS > WINDOW_HEIGHT - border_y:
            self.y = WINDOW_HEIGHT - PLANE_HITBOX_RADIUS - border_y
        
        self.hitbox = (self.x, self.y)

    def render(self, window, collision=False, plane_show = True, is_warning = True):
        if not collision:
            x_render = self.x - self.width  / 2
            y_render = self.y - self.height / 2
            if plane_show == True:
                window.blit(self.image, (x_render, y_render))

        # draw hitbox
        pygame.draw.circle(window, RED, self.hitbox, PLANE_HITBOX_RADIUS, PLANE_HITBOX_RADIUS)
        # draw warning circle
        if is_warning == True:
            pygame.draw.circle(window, BLUE, self.hitbox, PLANE_WARNING_RADIUS, PLANE_WARNING_CIRCLE_WIDTH)

class Bullet(object):
    def __init__(self, color, plane_x, plane_y):
        self.color = color
        self.radius = BULLET_RADIUS
        self.velocity = BULLET_VEL

        if random.choice((0,1)) == 0: # bullet come from left or right
            self.x = random.choice((0,WINDOW_WIDTH))
            self.y = random.uniform(WINDOW_HEIGHT, 0)
        else: # bullet come from top or buttom
            self.x = random.uniform(WINDOW_WIDTH, 0)
            self.y = random.choice((0, WINDOW_HEIGHT))

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

class Bullet_2(object):
    def __init__(self, color):
        self.color = color
        self.radius = BULLET_RADIUS
        self.x, self.y = 0,0
        self.angle = 0
        choice = random.randint(0, 3)
        if choice == 0:
            self.x = 1
            self.y = 1
            self.angle = random.randint(270, 360)
        elif choice == 1:
            self.x = WINDOW_WIDTH -1
            self.y = 1
            self.angle = random.randint(180, 270)
        elif choice == 2:
            self.x = 1
            self.y = WINDOW_HEIGHT -1
            self.angle = random.randint(0, 90)
        elif choice == 3:
            self.x = WINDOW_WIDTH -1
            self.y = WINDOW_HEIGHT -1
            self.angle = random.randint(90, 180)
        #random speed
        self.speed = random.randint(3, 5)

    def move(self):
        self.x = self.x + math.cos(math.radians(self.angle)) * self.speed
        self.y = self.y - math.sin(math.radians(self.angle)) * self.speed
    
    def render(self, window):
        pygame.draw.circle(window, self.color, (int(self.x), int(self.y)), self.radius)

class Explode(object):
    def __init__(self, x, y):
        self.x = x - PLANE_WIDTH
        self.y = y - PLANE_HEIGHT
        self.tick_counter = 0
        self.ani_counter = 0
        self.length = len(EXPLODE)
    
    def is_stop(self):
        return self.ani_counter >= self.length

    def render(self, window):
        window.blit(EXPLODE[self.ani_counter], (self.x, self.y))
        self.tick_counter += 1

        if self.tick_counter % EXPLODE_LATENCY == 0:
            self.ani_counter += 1