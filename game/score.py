import pygame

class Score:
    def __init__(self, screen):
        self.score = 0
        self.font = pygame.font.Font(None, 36)
        self.screen = screen

    def addScore(self, points = 1):
        self.score += points

    def getScore(self):
        return self.score

    def draw(self):
        text_surface =  self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        text_rect = text_surface.get_rect()
        text_rect.center = (text_rect.width // 2, text_rect.height // 2)
        self.screen.blit(text_surface, text_rect)

    