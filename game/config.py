from enum import Enum

class MapTileType(Enum):
    EMPTY = 0
    FOOD = 1
    SNAKE_BODY = 2

class SnakeDirection(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    RESET = 5