import pygame
from .config import *
from .snake import *
from .score import *
from .food import *
from .dead import *

# 定义一个Game类，用于控制游戏
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
        self.running = True
        self.dt = 0

        self.snake = None
        self.score = None
        self.food = None
        self.dead = None

    # 初始化游戏，设置蛇，分数，食物，死亡图片
    def initGame(self):
        self.snake = Snake(self.singleTileWidth, self.singleTileHeight, self.tileCountX, self.tileCountY, self.screen)
        self.score = Score(self.screen)
        self.food = Food(self.singleTileWidth, self.singleTileHeight)
        self.dead = Dead(self.screen)


    def checkIsGameStartRunning(self):
        return self.isGameStartRunning
    

    def generateTrainingData(self):
        snakeBodyPositions = self.snake.getSnakeBodyPositions()
        foodPosition = self.food.getPosition()
        isGameOver = self.snake.checkIsDead()
        snakeDirection = self.snake.getDirection()

        datas = []

        for i in range(0, len(snakeBodyPositions)):
            datas.append(snakeBodyPositions[i][0])
            datas.append(snakeBodyPositions[i][1])
        
        datas.append(foodPosition[0])
        datas.append(foodPosition[1])
        datas.append(0 if isGameOver else 1)
        datas.append(snakeDirection.value)
        
        controlMapping = [SnakeDirection.UP.value, SnakeDirection.DOWN.value, SnakeDirection.LEFT.value, SnakeDirection.RIGHT.value, SnakeDirection.RESET.value]

        return {'status': datas, "controlMapping": controlMapping}
    
        # 开始游戏，监听事件，更新蛇，食物，分数，死亡图片，并显示
    def startGame(self, pushDataFunc):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            if self.running:
                self.screen.fill((0, 0, 0))

                if self.snake.checkIsDead():
                    self.dead.draw()
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_ESCAPE]:
                        self.initGame()
                else:
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
                    self.snake.update(self.dt, self.food, self.score)
                    self.score.draw()

                pygame.display.flip()
                self.dt = self.clock.tick(60) / 1000

                pushDataFunc(self.generateTrainingData())

    # 退出游戏，关闭pygame
    def gameQuit(self):
        pygame.quit()