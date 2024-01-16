import pygame
import random

class Food:
    def __init__(self, tileWidth, tileHeight):
        self.foodSprite = pygame.sprite.Sprite()
        self.foodSprite.image = pygame.Surface([tileWidth, tileHeight])
        self.foodSprite.image.fill("red")
        self.foodSprite.rect = self.foodSprite.image.get_rect()
        self.isInMap = False
        self.position = [-1, -1]
        self.foodSpriteGroup = pygame.sprite.Group()
        self.foodSpriteGroup.add(self.foodSprite)
    
    def createFood(self, snakeBodyInfo, tileCountX, tileCountY):
        if self.isInMap:
            return
        
        self.isInMap = True

        def checkInSnake(x, y):
            for bodyPart in snakeBodyInfo:
                if (x == -1 or y == -1) or (bodyPart[0] == x and bodyPart[1] == y):
                    return True
            return False

        while checkInSnake(self.position[0], self.position[1]):
            self.position[0] = random.randint(0, tileCountX - 1)
            self.position[1] = random.randint(0, tileCountY - 1)

        
        self.foodSprite.rect.x = self.position[0] * self.foodSprite.rect.width
        self.foodSprite.rect.y = self.position[1] * self.foodSprite.rect.height

    def update(self, screen, snakeBodyInfo, tileCountX, tileCountY):
        self.createFood(snakeBodyInfo, tileCountX, tileCountY)
        self.foodSpriteGroup.draw(screen)

    def getPosition(self):
        return self.position
    
    def eat(self):
        self.isInMap = False