import pygame

class SnakeBlock(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, color):
        pygame.sprite.Sprite.__init__(self)

        self.width = width
        self.height = height
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x * width
        self.rect.y = y * height

    def setPosition(self, x, y):
        self.rect.x = x * self.width
        self.rect.y = y * self.height