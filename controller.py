import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
pygame.init()
import random 
import numpy as np
import math
import time
from config import *
from models import *


print(EXPLODE_MODE)

class GameMamager:
    def __init__(self):
        
        self.window = pygame.display.set_mode((WINOW_WIDTH, WINOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.score = 0
        self.run = True
        self.plane = Plane(250,250)
        self.bullets = []
        self.collision = False
        self.explosion = None
        self.font = pygame.font.SysFont("comicsans", 30, True)
        
        # set title 
        pygame.display.set_caption('bullet hell drill')

    def render(self):
        # reset 
        self.window.fill((0, 0, 0))

        # plane
        self.plane.render(self.window, self.collision)
        
        # bullet
        for bullet in self.bullets:
            bullet.render(self.window)
        
        # explosion
        if EXPLODE_MODE and self.collision:
            self.explosion.render(self.window)

        # score 
        self.window.blit(self.font.render(f'Score(ms): {self.score}', 1, WHITE), (190, 10))
        
        # update
        pygame.display.update()

    def is_collision(self):
        boundary = PLANE_HITBOX_RADIUS + BULLET_RADIUS
        for bullet in self.bullets:
            distance = math.hypot(self.plane.x - bullet.x, self.plane.y - bullet.y)
            if distance < boundary:
                return True
        return False
    
    def score_update(self, start_tick):
        self.score += pygame.time.get_ticks()-start_tick

    def step(self, actions):
        self.clock.tick(FPS)
        start_tick = pygame.time.get_ticks()

        for event in pygame.event.get():  # This will loop through a list of any keyboard or mouse events.
            if event.type == pygame.QUIT: # Checks if the red button in the corner of the self.window is clicked
                self.run = False  # Ends the game loop        
        
        # check explosion
        if not self.collision:
            # plane move
            self.plane.move(actions)
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
            
            # update score
            self.score_update(start_tick)

            # check collision
            if self.is_collision():
                self.collision = True
                if EXPLODE_MODE:
                    self.explosion = Explode(self.plane.x, self.plane.y)
                else:
                    self.run = False
        else:
            # end game
            if self.explosion.is_stop():
                self.run = False
        
        # update frame
        if self.run:
            self.render()

        # return 
        screen_shot = pygame.surfarray.array3d(pygame.display.get_surface())
        return screen_shot, self.score, self.collision, self.run