import random
from collections import deque
from .modal import *


MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001 # 学习率

# 定义一个SnakeAIController类，用于控制AI游戏
class SnakeAIController:
    # 初始化类，设置游戏次数、epsilon、gamma、模型、记忆库、训练器
    def __init__(self, tileCountX, tileCountY):
        self.tileCountX = tileCountX,
        self.tileCountY = tileCountY,
        self.gameTimes = 0
        self.epsilon = 0 
        self.gamma = 0.9 
        self.model = SnakeAIModal(11 + tileCountX * tileCountY, 1280, 3)
        # self.model = SnakeAIModal(11 + tileCountX * tileCountY, 256, 3)
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.trainer = SnakeAITrainer(self.model, lr=LR, gamma=self.gamma)

    # 添加记忆功能，将状态、动作、奖励、下一个状态、是否结束添加到记忆库中
    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    # 训练长记忆，如果记忆库中的状态数量大于BATCH_SIZE，则从记忆库中随机抽取BATCH_SIZE个状态，否则使用全部状态；
    # 获取状态、动作、奖励、下一个状态、是否结束，并传入训练器中训练一步；
    def trainLongMemory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.trainStep(states, actions, rewards, next_states, dones)


    # 训练短记忆，将状态、动作、奖励、下一个状态、是否结束传入训练器中训练一步；
    def trainShortMemory(self, state, action, reward, nextState, done):
        self.trainer.trainStep(state, action, reward, nextState, done)

    # 根据状态获取动作，如果随机因子小于 epsilon，则随机获取动作，否则根据状态获取动作；
    def getAction(self, state):
        self.epsilon = 80 - self.gameTimes
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move