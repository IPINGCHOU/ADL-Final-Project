from config import *
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

        action = action[0]
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
        if self.x - PLANE_HITBOX_RADIUS < 0:
            self.x = PLANE_HITBOX_RADIUS
        if self.x + PLANE_HITBOX_RADIUS > WINOW_WIDTH:
            self.x = WINOW_WIDTH - PLANE_HITBOX_RADIUS

        if self.y - PLANE_HITBOX_RADIUS < 0:
            self.y = PLANE_HITBOX_RADIUS
        if self.y + PLANE_HITBOX_RADIUS > WINOW_HEIGHT:
            self.y = WINOW_HEIGHT - PLANE_HITBOX_RADIUS
        
        self.hitbox = (self.x, self.y)

    def render(self, window):
        x_render = self.x - self.width  / 2
        y_render = self.y - self.height / 2

        window.blit(self.image, (x_render, y_render))
        
        # draw hitbox
        pygame.draw.circle(window, RED, self.hitbox, PLANE_HITBOX_RADIUS, 2)

class Bullet(object):
    def __init__(self, color, plane_x, plane_y):
        self.color = color
        self.radius = BULLET_RADIUS
        self.velocity = BULLET_VEL

        if random.choice((0,1)) == 0: # bullet come from left or right
            self.x = random.choice((0,WINOW_WIDTH))
            self.y = random.uniform(WINOW_HEIGHT, 0)
        else: # bullet come from top or buttom
            self.x = random.uniform(WINOW_WIDTH, 0)
            self.y = random.choice((0, WINOW_HEIGHT))

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
        self.x = x - PLANE_WIDTH / 2
        self.y = y - PLANE_HEIGHT / 2
        self.tick_counter = 0
        self.ani_counter = 0
        self.length = len(EXPLODE)

    def render(self, window):
        window.blit(EXPLODE[self.ani_counter], (self.x, self.y))
        self.tick_counter += 1
        # self.ani_counter += 1

        if self.tick_counter % EXPLODE_LATENCY == 0:
            self.ani_counter += 1