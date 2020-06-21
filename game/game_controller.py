import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
pygame.init()
import random 
import numpy as np
import math
import time
import cv2
from game_config import *
from game_models import *

class GameManager:
    def __init__(self, bullet_mode, explode_mode, plane_show, score_show):
        
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.score = 0
        self.run = True

        self.plane = Plane(WINDOW_WIDTH//2, WINDOW_HEIGHT//2)
        self.bullets = []
        self.collision = False
        self.explosion = None
        self.font = pygame.font.SysFont("comicsans", 15, True)

        self.bullet_mode = bullet_mode
        self.explode_mode = explode_mode
        self.plane_show = plane_show     
        self.score_show = score_show

        self.dead = False
        self.action_space = 5
        self.obs_resize_shape = [RESIZE_SIZE[0], RESIZE_SIZE[1], 3]
        self.obs_shape = [WINDOW_WIDTH, WINDOW_HEIGHT, 3]

        # set title 
        pygame.display.set_caption('bullet hell drill')
    
    def resize_state(self, size):
        arr = pygame.surfarray.array3d(pygame.display.get_surface())
        image = cv2.resize(arr, size)
        image = np.ascontiguousarray(image, dtype = np.float32) /255
        return image

    def reset(self, resize = True, size = (80,80)):
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.score = 0
        self.run = True
        self.plane = Plane(WINDOW_WIDTH//2, WINDOW_HEIGHT//2)
        self.bullets = []
        self.collision = False
        self.explosion = None
        self.font = pygame.font.SysFont("comicsans", 30, True)
        self.dead = False

        if resize == False:
            return pygame.surfarray.array3d(pygame.display.get_surface())
        else:
            return self.resize_state(size)

    def render(self):
        # reset 
        self.window.fill((0, 0, 0))

        # plane
        self.plane.render(self.window, self.collision, self.plane_show)
        
        # bullet
        for bullet in self.bullets:
            bullet.render(self.window)
        
        # explosion
        if self.explode_mode and self.collision:
            self.explosion.render(self.window)

        # score 
        if self.score_show == True:
            self.window.blit(self.font.render(f'Score(s): {self.score}', 1, WHITE), (50, 10))
        
        # update
        pygame.display.update()

    def is_collision(self):
        boundary = PLANE_HITBOX_RADIUS + BULLET_RADIUS
        for bullet in self.bullets:
            distance = math.hypot(self.plane.x - bullet.x, self.plane.y - bullet.y)
            if distance < (boundary-COLL_TOLERANCE):
                return True
        return False
    
    def in_warning_zone(self):
        boundary = PLANE_WARNING_RADIUS + BULLET_RADIUS
        in_count = 0
        for bullet in self.bullets:
            distance = math.hypot(self.plane.x - bullet.x, self.plane.y - bullet.y)
            if distance < (boundary-COLL_TOLERANCE):
                in_count += 1
        return in_count

    
    def score_update(self, invincible):
        # self.score += time.time()-start_tick
        # in_warning_count = self.in_warning_zone()

        self.score = SURVIVE_SCORE
        # self.score += in_warning_count*WARNING_PUNISH

        if self.run == False or self.dead == True:
            self.score = 0

        if self.collision == True and self.dead == False:
            self.score = DEAD_PUNISH
            self.dead = True
        
        if invincible:
            self.dead = False

    def step(self, actions, resize = True, size = (80,80), invincible = False):
        self.clock.tick(FPS)

        for event in pygame.event.get():  # This will loop through a list of any keyboard or mouse events.
            if event.type == pygame.QUIT: # Checks if the red button in the corner of the self.window is clicked
                self.run = False  # Ends the game loop        
        
        if invincible == True:
            self.collision = False
        
        # check explosion
        if not self.collision:
            # plane move
            self.plane.move(actions)
            # bullets move
            for bullet in self.bullets:
                bullet_exist = False
                if 0 <= bullet.x and bullet.x <= WINDOW_WIDTH and 0 <= bullet.y and bullet.y <= WINDOW_HEIGHT:
                    bullet.move()
                    bullet_exist = True

                if bullet_exist == False:
                    self.bullets.pop(self.bullets.index(bullet))
            
            addin_bullets = 0
            while len(self.bullets) < MAX_BULLETS and addin_bullets <= MAX_ADDIN_BULLETS:
                if self.bullet_mode == 'random':
                    self.bullets.append(Bullet_2(YELLOW))
                elif self.bullet_mode == 'aim':
                    self.bullets.append(Bullet(WHITE, self.plane.x, self.plane.y))
                addin_bullets += 1
            addin_bullets = 0
            
            # check collision
            if self.is_collision():
                self.collision = True
                if self.explode_mode:
                    self.explosion = Explode(self.plane.x, self.plane.y)
                else:
                    if invincible == False:
                        self.run = False
        else:
            # end game
            if self.explode_mode:
                if self.explosion.is_stop():
                    self.run = False
            else:
                self.run = False
        
        # update frame
        if self.run:
            self.render()

        # return 
        if resize == False:
            screen_shot = pygame.surfarray.array3d(pygame.display.get_surface())
        else:
            screen_shot = self.resize_state(size)


        self.score_update(invincible)
        return screen_shot, self.score, self.collision, self.run

class ReplaySaver:
    def __init__(self):
        self.frame_array = []
        self.best_frame_array = []
    
    def reset(self):
        self.frame_array = []
    
    def save_best(self):
        self.best_frame_array = self.frame_array
    
    def get_current_frame(self):
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        self.frame_array.append(frame)
    
    def make_video(self, path, size = (750,750), fps = 60):
        out = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        print('Vid length: {}'.format(len(self.best_frame_array)))
        for i in self.best_frame_array:
            i = np.rot90(i,3)
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            out.write(i)
            
        print('Writing video...')
        out.release()
