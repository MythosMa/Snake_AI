import random
from collections import deque
from .modal import *


MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001 # 学习率

class SnakeAIController:
    def __init__(self):
        self.gameTimes = 0
        self.epsilon = 0 
        self.gamma = 0.9 
        self.model = SnakeAIModal(11, 256, 3)
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.trainer = SnakeAITrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def trainLongMemory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def trainShortMemory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

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