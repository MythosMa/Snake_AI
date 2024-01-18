import time
from game.game import *
from game.ai_game import *
from ai.controller import *

isAIPlay = True

tileCountX = 30
tileCountY = 30

aiContoller = SnakeAIController()

game = AIGame(tileCountX, tileCountY) if isAIPlay else Game(tileCountX, tileCountY)
game.initGame()
# game.startGame(ai.pushTrainData)

while True:
    isDead = False
    if isAIPlay:
        oldState = game.getGameState() 
        nextAction = aiContoller.getAction(oldState)
        reword, isDead = game.updateGame(nextAction)
        newState = game.getGameState()
        aiContoller.trainShortMemory(oldState, nextAction, reword, newState, isDead)    
        aiContoller.remember(oldState, nextAction, reword, newState, isDead)
    else:
        isDead = game.updateGame()

    if isDead:
        if isAIPlay:
            aiContoller.gameTimes += 1
            aiContoller.trainLongMemory()
            print("Game", aiContoller.gameTimes)

        time.sleep(1)
        game.initGame()
        