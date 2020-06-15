import os
import random 
import pygame
import numpy as np
import math
import time
from config import *

class Plane(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = PLANE_WIDTH
        self.height = PLANE_HEIGHT
        self.vel = PLANE_VEL
        self.left = False
        self.right = False
        self.horizontal_move = 0
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
            self.horizontal_move = 0
        
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

class GameEnvironment:
    def __init__(self):
        self.score = 0
        self.run = True
        self.plane = Plane(250,250)
        self.bullets = []
        self.collision = False
        self.explode_made = False

        self.font = pygame.font.SysFont("comicsans", 30, True)
        
    def WindowRender(self):
        window.fill((0,0,0))
        self.plane.render(window)
        for bullet in self.bullets:
            bullet.render(window)
        
        pygame.display.update()

    def CheckCollision(self):
        for bullet in self.bullets:
            distance = math.hypot(self.plane.x - bullet.x, self.plane.y - bullet.y)
            if distance < (PLANE_HITBOX_RADIUS + BULLET_RADIUS):
                return True
        return False

    def GameOverWindowRender(self):
        window.fill((0,0,0))
        self.explode.render(window)
        pygame.draw.circle(window, RED, self.plane.hitbox, PLANE_HITBOX_RADIUS, 2)
        for bullet in self.bullets:
            bullet.render(window)
        window.blit(self.font.render('Score(ms): ' + str(self.score),1, WHITE), (190,10))
        pygame.display.update()
        if self.explode.ani_counter < len(EXPLODE):
            return True
        else:
            return False
    
    def ScoreUpdateRender(self, start_tick):
        if self.collision == False:
            self.score += (pygame.time.get_ticks()-start_tick)
        window.blit(self.font.render('Score(ms): ' + str(self.score),1, WHITE), (190,10))
        pygame.display.update()

    def step(self, input_action):
        # pygame.event.pump()
        clock.tick(FRAME_RATE)
        start_tick = pygame.time.get_ticks()

        if len(input_action) > 1:
            raise ValueError('Mupltiple input actions ?!')
        input_action = input_action[0]
            
        if self.collision == False: # plane alive
            for event in pygame.event.get():  # This will loop through a list of any keyboard or mouse events.
                if event.type == pygame.QUIT: # Checks if the red button in the corner of the window is clicked
                    self.run = False  # Ends the game loop
                # if event.type == pygame.USEREVENT:
                #     self.score += 1
            
            while len(self.bullets) < MAX_BULLETS:
                self.bullets.append(Bullet(WHITE, self.plane.x, self.plane.y))
        
            for bullet in self.bullets:
                bullet_exist = False
                if 0 <= bullet.x and bullet.x <= WIN_WIDTH :
                    if 0 <= bullet.y and bullet.y <= WIN_HEIGHT:
                        bullet.move()
                        bullet_exist = True

                if bullet_exist == False:
                    self.bullets.pop(self.bullets.index(bullet))
        
            # check inputs 
            # 0: LEFT 1:RIGHT 2:UP 3:DOWN
            if input_action==0 and self.plane.x > PLANE_VEL + PLANE_HITBOX_RADIUS: 
                self.plane.x -= PLANE_VEL
                self.plane.left = True
                self.plane.right = False
            elif input_action==1 and self.plane.x < 500 - PLANE_VEL - PLANE_HITBOX_RADIUS:
                self.plane.x += PLANE_VEL
                self.plane.left = False
                self.plane.right = True
            else:
                self.plane.left, self.plane.right = False, False
                self.plane.horizontal_move = 0

            if input_action==2 and self.plane.y > 0 + PLANE_HITBOX_RADIUS:
                self.plane.y -= PLANE_VEL
            if input_action==3 and self.plane.y < 500 - PLANE_HITBOX_RADIUS:
                self.plane.y += PLANE_VEL

            self.WindowRender()
        else: # plane crashed
            if self.explode_made == False:
                self.score += (pygame.time.get_ticks()-start_tick)
                self.explode = Explode(self.plane.x, self.plane.y)
                self.explode_made = True
            self.run = self.GameOverWindowRender()
        
        self.collision = self.CheckCollision()
        screen_shot = pygame.surfarray.array3d(pygame.display.get_surface())
        self.ScoreUpdateRender(start_tick)

        return screen_shot, self.score, self.collision, self.run
        # screen_shot 500 500 3
        # score = 10*ms

if __name__ == '__main__':
    test_epoch = 3

    for i in range(test_epoch):
        run = True
        # 宣告環境
        window = env_init(screen_on=True)
        env = GameEnvironment()
        counter = 1
        while run:
            action = [random.choice((0,1,2,3))]
            # 調用 step
            img, score, collision, run = env.step(action)
            print((counter, np.array(img).shape, score, collision, run))
            counter += 1
    