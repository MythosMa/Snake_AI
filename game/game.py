import pygame
import numpy as np
from .config import *
from .snake import *
from .score import *
from .food import *
from .dead import *

class Game:
    # 初始化游戏，设置屏幕大小，单个tile的宽度，高度，以及屏幕
    def __init__(self, tileCountX, tileCountY):
        pygame.init()

        self.tileCountX = tileCountX
        self.tileCountY = tileCountY
        self.singleTileWidth = 10
        self.singleTileHeight = 10

        self.screen = pygame.display.set_mode((self.tileCountX * self.singleTileWidth,self. tileCountY * self.singleTileHeight))
        self.clock  = pygame.time.Clock()
        self.gameSpeed = 10
        
        self.snake = None
        self.score = None
        self.food = None
        self.dead = None

    # 初始化游戏，设置蛇，分数，食物，死亡图片
    def initGame(self):
        self.snake = Snake(3, self.singleTileWidth, self.singleTileHeight, self.tileCountX, self.tileCountY, self.screen)
        self.score = Score(self.screen)
        self.food = Food(self.singleTileWidth, self.singleTileHeight)
        self.dead = Dead(self.screen)
        
    # 开始游戏，监听事件，更新蛇，食物，分数，死亡图片，并显示
    def updateGame(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.screen.fill((0, 0, 0))

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.snake.changeDirection(SnakeDirection.UP)
        if keys[pygame.K_DOWN]:
            self.snake.changeDirection(SnakeDirection.DOWN)
        if keys[pygame.K_LEFT]:
            self.snake.changeDirection(SnakeDirection.LEFT)
        if keys[pygame.K_RIGHT]:
            self.snake.changeDirection(SnakeDirection.RIGHT)
                
        self.food.update(self.screen, self.snake.getSnakeBody(), self.tileCountX, self.tileCountY)
        reward, isDead = self.snake.update(self.food, self.score)

        self.score.draw()
        pygame.display.flip()
        self.clock.tick(self.gameSpeed)

        return isDead

