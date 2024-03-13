from .config import *
from .snakeBlock import SnakeBlock
import pygame

class Snake:
    def __init__(self, initLength, tileWidth, tileHeight, tileCountX, tileCountY, screen) -> None:
        self.screen = screen
        self.snakeLength = initLength
        self.tileCountX = tileCountX
        self.tileCountY = tileCountY
        self.tileWidth = tileWidth
        self.tileHeight = tileHeight
        self.bodySpriteGroup = pygame.sprite.Group()
        self.bodyInfo = []
        self.isInputDirection = False
        for i in range(self.snakeLength):
            self.bodyInfo.append([(self.snakeLength - i), 0, SnakeBlock((self.snakeLength - i), 0, tileWidth, tileHeight, "green" if i == 0 else "white")])
            self.bodySpriteGroup.add(self.bodyInfo[i][2])

        self.direction = SnakeDirection.RIGHT

    def update(self, food, score):
        reward = 0
        # b保存尾巴坐标，用来新增吃到食物后的身体
        newBodyInfo = [self.bodyInfo[self.snakeLength - 1][0], self.bodyInfo[self.snakeLength - 1][1]]

        # 移动身体
        for i in range(self.snakeLength - 1, 0, -1):
            self.bodyInfo[i][0] = self.bodyInfo[i - 1][0]
            self.bodyInfo[i][1] = self.bodyInfo[i - 1][1]
            self.bodyInfo[i][2].setPosition( self.bodyInfo[i][0], self.bodyInfo[i][1])

        # 移动头部
        if self.direction == SnakeDirection.RIGHT:
            self.bodyInfo[0][0] += 1
        if self.direction == SnakeDirection.LEFT:
            self.bodyInfo[0][0] -= 1
        if self.direction == SnakeDirection.UP:
            self.bodyInfo[0][1] -= 1
        if self.direction == SnakeDirection.DOWN:
            self.bodyInfo[0][1] += 1
        
        self.isInputDirection = False

        self.bodyInfo[0][2].setPosition( self.bodyInfo[0][0], self.bodyInfo[0][1])

        if self.bodyInfo[0][0] < 0 or self.bodyInfo[0][0] > self.tileCountX - 1 or self.bodyInfo[0][1] < 0 or self.bodyInfo[0][1] > self.tileCountY - 1:
            reward = -2
            return reward, True

        for i in range(self.snakeLength - 1, 0, -1):
            if self.bodyInfo[i][0] == self.bodyInfo[0][0] and self.bodyInfo[i][1] == self.bodyInfo[0][1]:
                reward = -2
                return reward, True
            
        foodPosition = food.getPosition()
        if self.bodyInfo[0][0] == foodPosition[0] and self.bodyInfo[0][1] == foodPosition[1]:
            reward = 2
            score.addScore()
            food.eat()
            newBodyInfo.append(SnakeBlock(newBodyInfo[0], newBodyInfo[1], self.tileWidth, self.tileHeight, "white"))
            self.bodySpriteGroup.add(newBodyInfo[2])
            self.bodyInfo.append(newBodyInfo)
            self.snakeLength = len(self.bodyInfo)
            
        self.bodySpriteGroup.draw(self.screen)

        return reward, False

    def changeDirection(self, direction):
        if self.isInputDirection:
            return
        if direction == SnakeDirection.LEFT and self.direction == SnakeDirection.RIGHT:
            return
        if direction == SnakeDirection.RIGHT and self.direction == SnakeDirection.LEFT:
            return
        if direction == SnakeDirection.UP and self.direction == SnakeDirection.DOWN:
            return
        if direction == SnakeDirection.DOWN and self.direction == SnakeDirection.UP:
            return
        
        self.direction = direction
        self.isInputDirection = True

    def getDirection(self):
        return self.direction

    def getSnakeBody(self):
        return self.bodyInfo
        