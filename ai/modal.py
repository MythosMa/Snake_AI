import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# 定义SnakeAIModal类，继承nn.Module
class SnakeAIModal(nn.Module):
    # 初始化函数，输入参数为输入大小、隐藏层大小和输出大小
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 定义线性层，输入大小为input_size，输出大小为hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        # 定义线性层，输入大小为hidden_size，输出大小为output_size
        self.linear2 = nn.Linear(hidden_size, output_size)

    # 定义前向传播函数，输入参数为x
    def forward(self, x):
        # 计算线性层的输出，激活函数为relu
        x = F.relu(self.linear1(x))
        # 计算线性层的输出
        x = self.linear2(x)
        # 返回线性层的输出
        return x
    
    # 定义保存函数，输入参数为文件名，默认为model.pth
    def save(self, file_name='model.pth'):
        # 定义模型文件夹路径
        model_folder_path = './model'
        # 如果模型文件夹不存在，则创建模型文件夹
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        # 拼接模型文件路径
        file_name = os.path.join(model_folder_path, file_name)
        # 保存模型参数
        torch.save(self.state_dict(), file_name)
    

# 定义SnakeAITrainer类，输入参数为模型、学习率和折扣因子
class SnakeAITrainer:
    def __init__(self, model, lr, gamma):
        # 定义学习率
        self.lr = lr
        # 定义折扣因子
        self.gamma = gamma
        # 定义模型
        self.model = model
        # 定义优化器，优化器为Adam，学习率为lr
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # 定义损失函数，为均方误差
        self.criterion = nn.MSELoss()

    # 定义训练函数，输入参数为状态、动作、奖励、下一个状态和完成标志
    def trainStep(self, state, action, reward, next_state, done):
        # 将状态、下一个状态、动作、奖励转换为tensor，数据类型为float
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # 如果状态、下一个状态、动作、奖励的形状为（1，x），则将它们扩展为（x，1）
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 计算模型的输出
        pred = self.model(state)

        # 定义目标值，与模型输出相同
        target = pred.clone()
        # 遍历完成标志
        for idx in range(len(done)):
            # 计算新的Q值
            Q_new = reward[idx]
            # 如果完成标志为False，则新的Q值为奖励加上折扣因子乘以模型输出中最大Q值
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # 将新的Q值赋值给目标值
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    

        # 梯度归零
        self.optimizer.zero_grad()
        # 计算损失函数
        loss = self.criterion(target, pred)
        # 反向传播
        loss.backward()

        # 更新参数
        self.optimizer.step()