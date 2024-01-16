import pygame
from config import *
from snake import *
from score import *
from food import *
from dead import *

pygame.init()

tileCountX = 50
tileCountY = 50
singleTileWidth = 10
singleTileHeight = 10

screen = pygame.display.set_mode((tileCountX * singleTileWidth, tileCountY * singleTileHeight))
clock  = pygame.time.Clock()
running = True
dt = 0

mapTile = []
snake = None
score = None
food = None
dead = None

def initGame():
    global mapTile, snake, score, food, dead  # 使用 global 声明全局变量

    mapTile = [[MapTileType.EMPTY for _ in range(tileCountY)] for _ in range(tileCountX)]
    snake = Snake(singleTileWidth, singleTileHeight, tileCountX, tileCountY, screen)
    score = Score(screen)
    food = Food(singleTileWidth, singleTileHeight)
    dead = Dead(screen)

initGame()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    if snake.checkIsDead():
        dead.draw()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            initGame()
    else:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            snake.changeDirection(SnakeDirection.UP)
        if keys[pygame.K_DOWN]:
            snake.changeDirection(SnakeDirection.DOWN)
        if keys[pygame.K_LEFT]:
            snake.changeDirection(SnakeDirection.LEFT)
        if keys[pygame.K_RIGHT]:
            snake.changeDirection(SnakeDirection.RIGHT)
            
        food.update(screen, snake.getSnakeBody(), tileCountX, tileCountY)
        snake.update(dt, food, score)
        score.draw()

    pygame.display.flip()
    dt = clock.tick(60) / 1000

pygame.quit()