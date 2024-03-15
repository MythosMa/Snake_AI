import time
import pygame
import numpy as np
from .config import *
from .snake import *
from .score import *
from .food import *
from .dead import *

# 定义一个Game类，用于控制游戏

# 贪吃蛇的游戏状态，经过梳理可以如下安排（所有的条件都使用bool），假设下面这个图， #是蛇头 -是蛇身子，这条蛇可以看出来是在向左移动, $是食物
#
#  0 0 0 0 0 0 0 0 
#  # - - 0 0 0 0 0
#  0 0 0 0 0 0 0 0 
#  0 0 0 0 0 0 0 0 
#  0 0 0 0 0 $ 0 0 
#  0 0 0 0 0 0 0 0 
#  0 0 0 0 0 0 0 0 
#  0 0 0 0 0 0 0 0 
#
# 关于移动后的情况，贪吃蛇的移动，只有三种情况，向前，向左拐，向右拐，
# 所以预测这三个方向是否会导致死亡即可，这个方向是相对于蛇的移动方向的，图例中蛇前方要撞到边框，则向前会导致死亡
#
# [
#  1, 向前是否会导致死亡
#  0, 向左是否会导致死亡
#  0, 向右是否会导致死亡
# ]
#
# 关于当前的移动方向，相对于地图
#
# [
#  1, 向左移动
#  0, 向右移动
#  0, 向上移动
#  0, 向下移动
# ]
#
# 关于食物相对的位置（不是相对蛇的移动方向，而是相对于蛇在地图中的位置，所以食物在蛇的右下）
# 
# [
#  0, 在左边
#  1, 在右边
#  0, 在上边
#  1, 在下边
# ]
#
# 因此我们可以得到一个一维张量，来描述决策，即告诉AI的数据，这个张量的长度即为11
# 对于AI决策后，会给出一个操作，即保持向前移动，还是左转或者右转，这也是一个一维张量，如上述示例图，向左转是个最优方案，即
# 
# [
#  0, 向前走
#  1, 向左拐
#  0, 向右拐
# ]
#
# 对于AI的决策，可能给出的并不是上述0, 1, 0的返回值，而是类似于[0.2, 4.6, 1.7]这样的权重方案，我们只需要将最大的那一位设置为true(1)，其他为false(0)
# 如此，我们就建立了一个状态信息，利用这个状态信息去建立AI模型，让AI去决策行动

# 2024.03.13 关于持续性游戏的思考
# 在之前的版本中，实现了AI的普通玩法，经过200轮左右的迭代，已经可以在 30 * 30 的地图上，吃到50+食物
# 但是在此之后，由于蛇自身的身体已经很长，会出现两种情况，1.为了避免撞上身体产生的绕路，导致超过给定时间而死；2.在寻找食物时卷到自己身体围成的死胡同里，导致撞上自己而死
# 对于问题1，可以使用更宽松的时间限制条件解决。对于问题2，如何让AI实现路径规划，是一个难点。
# 在现在这个时候，我想能否将整个地图数据传递给训练模型，看AI能否从宏观方向上了解整体布局，自我规划。
# 依然以矩阵为例，4代表蛇头在地图上的位置 3代表蛇身在地图上的位置 9代表食物在地图上的位置
#  0 0 0 0 0 0 0 0 
#  4 3 3 0 0 0 0 0
#  0 0 0 0 0 0 0 0 
#  0 0 0 0 0 0 0 0 
#  0 0 0 0 0 9 0 0 
#  0 0 0 0 0 0 0 0 
#  0 0 0 0 0 0 0 0 
#  0 0 0 0 0 0 0 0 

class AIGame:
    # 初始化游戏，设置屏幕大小，单个tile的宽度，高度，以及屏幕
    def __init__(self, tileCountX, tileCountY):
        pygame.init()

        self.tileCountX = tileCountX
        self.tileCountY = tileCountY
        self.singleTileWidth = 10
        self.singleTileHeight = 10

        self.screen = pygame.display.set_mode((self.tileCountX * self.singleTileWidth,self. tileCountY * self.singleTileHeight))
        self.clock  = pygame.time.Clock()
        self.gameSpeed = 60
        self.initSnakeLength = 3
        
        self.snake = None
        self.score = None
        self.food = None
        self.dead = None
        self.notEatTime = 0 # 这个参数是用来阻止AI长时间乱转不吃东西的，每吃过一食物给100步的限额
        self.notEatLimit = 100
        self.notEatLimitUpRate = 1.1

    # 初始化游戏，设置蛇，分数，食物，死亡图片
    def initGame(self):
        self.snake = Snake(self.initSnakeLength, self.singleTileWidth, self.singleTileHeight, self.tileCountX, self.tileCountY, self.screen)
        self.score = Score(self.screen)
        self.food = Food(self.singleTileWidth, self.singleTileHeight)
        self.dead = Dead(self.screen)
        self.notEatTime = 0
        
    # 开始游戏，监听事件，更新蛇，食物，分数，死亡图片，并显示
    def updateGame(self, action):
        self.notEatTime += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.screen.fill((0, 0, 0))
        directions = [SnakeDirection.UP, SnakeDirection.RIGHT, SnakeDirection.DOWN, SnakeDirection.LEFT]
        currentDirectionIndex = directions.index(self.snake.getDirection())
        if np.array_equal(action, [1, 0, 0]):
            self.snake.changeDirection(directions[currentDirectionIndex])
        elif np.array_equal(action, [0, 1, 0]):
            newIndex = (currentDirectionIndex - 1) % len(directions)
            self.snake.changeDirection(directions[newIndex])
        elif np.array_equal(action, [0, 0, 1]):
            newIndex = (currentDirectionIndex + 1) % len(directions)
            self.snake.changeDirection(directions[newIndex])

        self.food.update(self.screen, self.snake.getSnakeBody(), self.tileCountX, self.tileCountY)

        if self.notEatTime >= self.notEatLimit * len(self.snake.getSnakeBody()) * self.notEatLimitUpRate:
            return -10, True, self.score.getScore()
        else:
            reward, isDead = self.snake.update(self.food, self.score)

        self.score.draw()
        pygame.display.flip()
        self.clock.tick(self.gameSpeed)

        return reward, isDead, self.score.getScore()
    
    def getGameState(self):
        # 判断各个方向是否为墙
        snakeHead = self.snake.getSnakeBody()[0]
        isSnakeToRight = self.snake.getDirection() == SnakeDirection.RIGHT
        isSnakeToLeft = self.snake.getDirection() == SnakeDirection.LEFT
        isSnakeToUp = self.snake.getDirection() == SnakeDirection.UP
        isSnakeToDown = self.snake.getDirection() == SnakeDirection.DOWN
        
        snakeLength = len(self.snake.getSnakeBody())
        body = self.snake.getSnakeBody()
        
        def checkCollisionBody(target):
            for i in range(snakeLength - 1, 0, -1):
                if body[i][0] == target[0] and body[i][1] == target[1]:
                    return True
            return False
        
        # 创建地图信息
        mapInfo = [[0 for _ in range(self.tileCountX)] for _ in range(self.tileCountY)]

        # 在地图中添加蛇的位置信息
        for i in range(0, snakeLength, 1):
            if(body[i][1] < len(mapInfo) and body[i][0] < len(mapInfo[body[i][1]])):
                if i == 0:
                    mapInfo[body[i][1]][body[i][0]] = 4
                else:
                    mapInfo[body[i][1]][body[i][0]] = 3
        
        # 在地图中添加食物的位置信息
        mapInfo[self.food.getPosition()[1]][self.food.getPosition()[0]] = 9

        state = []

        # 将地图信息追加到state 中 
        for sublist in mapInfo:
            state.extend(sublist)

        info = [
            # 信息分隔符
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            # 蛇头代表的值
            4,
            # 蛇身代表的值
            3,
            # 食物代表的值
            9,
            # 向前是否会导致死亡
            (isSnakeToLeft and snakeHead[0] == 0) or 
            (isSnakeToLeft and checkCollisionBody([snakeHead[0] - 1, snakeHead[1]])) or
            (isSnakeToDown and snakeHead[1] == self.tileCountY - 1) or 
            (isSnakeToDown and checkCollisionBody([snakeHead[0], snakeHead[1] + 1])) or
            (isSnakeToRight and snakeHead[0] == self.tileCountX - 1) or 
            (isSnakeToRight and checkCollisionBody([snakeHead[0] + 1, snakeHead[1]])) or
            (isSnakeToUp and snakeHead[1] == 0) or
            (isSnakeToUp and checkCollisionBody([snakeHead[0], snakeHead[1] - 1])),

            # 向左是否会导致死亡
            (isSnakeToLeft and snakeHead[1] == self.tileCountY - 1) or
            (isSnakeToLeft and checkCollisionBody([snakeHead[0], snakeHead[1] + 1])) or
            (isSnakeToDown and snakeHead[0] == self.tileCountX - 1) or
            (isSnakeToDown and checkCollisionBody([snakeHead[0] + 1, snakeHead[1]])) or
            (isSnakeToRight and snakeHead[1] == 0) or
            (isSnakeToRight and checkCollisionBody([snakeHead[0], snakeHead[1] - 1])) or
            (isSnakeToUp and snakeHead[0] == 0) or
            (isSnakeToUp and checkCollisionBody([snakeHead[0] - 1, snakeHead[1]])),

            # 向右是否会导致死亡
            (isSnakeToLeft and snakeHead[1] == 0) or
            (isSnakeToLeft and checkCollisionBody([snakeHead[0], snakeHead[1] - 1])) or
            (isSnakeToDown and snakeHead[0] == 0) or
            (isSnakeToDown and checkCollisionBody([snakeHead[0] - 1, snakeHead[1]])) or
            (isSnakeToRight and snakeHead[1] == self.tileCountY - 1) or
            (isSnakeToRight and checkCollisionBody([snakeHead[0], snakeHead[1] + 1])) or
            (isSnakeToUp and snakeHead[0] == self.tileCountX - 1) or
            (isSnakeToUp and checkCollisionBody([snakeHead[0] + 1, snakeHead[1]])),

            # 移动方向
            isSnakeToLeft, isSnakeToDown, isSnakeToRight, isSnakeToUp,

            # 食物相对的位置
            # 在左边
            self.food.getPosition()[0] < snakeHead[0],
            # 在右边
            self.food.getPosition()[0] > snakeHead[0],
            # 在上边
            self.food.getPosition()[1] < snakeHead[1],
            # 在下边
            self.food.getPosition()[1] > snakeHead[1],

            # 移动的步数
            self.notEatTime,

            # 移动上限
            self.notEatLimit * len(self.snake.getSnakeBody()) * self.notEatLimitUpRate,
        ]

        print("↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
        print(info)
        print("↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")

        state.extend(info)

        return np.array(state, dtype=int)
