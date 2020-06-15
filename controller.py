import pygame
pygame.init()
import os
import random 
import numpy as np
import math
import time
from config import *
from models import *

class GameMamager:
    def __init__(self):
        
        self.window = pygame.display.set_mode((WINOW_WIDTH, WINOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.score = 0
        self.run = True
        self.plane = Plane(250,250)
        self.bullets = []
        self.collision = False
        self.font = pygame.font.SysFont("comicsans", 30, True)
        
        # set title 
        pygame.display.set_caption('bullet hell drill')

    def render(self):
        self.window.fill((0, 0, 0))
        self.plane.render(self.window)
        for bullet in self.bullets:
            bullet.render(self.window)
        
        pygame.display.update()

    def step(self, actions):
        self.clock.tick(FPS)

        for event in pygame.event.get():  # This will loop through a list of any keyboard or mouse events.
            if event.type == pygame.QUIT: # Checks if the red button in the corner of the self.window is clicked
                self.run = False  # Ends the game loop        
        
        # check collision
        # plane move
        self.plane.move(action)
        # bullets move
        for bullet in self.bullets:
            bullet_exist = False
            if 0 <= bullet.x and bullet.x <= WINOW_WIDTH and 0 <= bullet.y and bullet.y <= WINOW_HEIGHT:
                bullet.move()
                bullet_exist = True

            if bullet_exist == False:
                self.bullets.pop(self.bullets.index(bullet))
        
        while len(self.bullets) < MAX_BULLETS:
            self.bullets.append(Bullet(WHITE, self.plane.x, self.plane.y))
        
        # update frame
        self.render()

        # return 
        screen_shot = pygame.surfarray.array3d(pygame.display.get_surface())
        return screen_shot, self.score, self.collision, self.run

if __name__ == '__main__':
    run = True
    env = GameMamager()
    while run:
        # action = [random.choice((0,1,2,3,4))]
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = [0]
        elif keys[pygame.K_RIGHT]:
            action = [1]
        elif keys[pygame.K_UP]:
            action = [2]
        elif keys[pygame.K_DOWN]:
            action = [3]
        else:
            action = [4]
        # input("!#!@")
        # img, score, collision, run = env.step(action)
        _ = env.step(action)
        # print((counter, np.array(img).shape, score, collision, run))
        # time.sleep(5)
    