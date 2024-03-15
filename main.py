import time
from game.game import *
from game.ai_game import *
from ai.controller import *
from helper.helper import *

isAIPlay = True

tileCountX = 30
tileCountY = 30

aiContoller = SnakeAIController(tileCountX, tileCountY)

recordScore = 0
plot_scores = []
plot_mean_scores = []
total_score = 0

game = AIGame(tileCountX, tileCountY) if isAIPlay else Game(tileCountX, tileCountY)
game.initGame()
# game.startGame(ai.pushTrainData)

while True:
    isDead = False
    if isAIPlay:
        oldState = game.getGameState() 
        nextAction = aiContoller.getAction(oldState)
        reward, isDead, score = game.updateGame(nextAction)
        newState = game.getGameState()
        aiContoller.trainShortMemory(oldState, nextAction, reward, newState, isDead)    
        aiContoller.remember(oldState, nextAction, reward, newState, isDead)
    else:
        isDead = game.updateGame()

    if isDead:
        if isAIPlay:
            aiContoller.gameTimes += 1
            aiContoller.trainLongMemory()
            if score > recordScore:
                recordScore = score
                aiContoller.model.save()
                
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / aiContoller.gameTimes
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            print(f"Game {aiContoller.gameTimes} end. Score Record: {recordScore}")
            
        game.initGame()
    