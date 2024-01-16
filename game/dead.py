import pygame

class Dead:
    def __init__(self, screen):
        self.font = pygame.font.Font(None, 36)
        self.screen = screen

    def draw(self):
        text_surface =  self.font.render("DEAD - PRESS ESC RESTART", True, (255, 255, 255))
        text_rect = text_surface.get_rect()
        text_rect.center = (self.screen.get_width() // 2, self.screen.get_height() // 2)
        self.screen.blit(text_surface, text_rect)

    