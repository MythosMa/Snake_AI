import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义SnakeAI类，继承nn.Module
class SnakeAI(nn.Module):
    # 初始化函数，输入参数为输入大小和输出大小
    def __init__(self, input_size, output_size):
        super(SnakeAI, self).__init__()
        # 定义第一个全连接层，输入大小为input_size，输出大小为128
        self.fc1 = nn.Linear(input_size, 128)
        # 定义第二个全连接层，输入大小为128，输出大小为output_size
        self.fc2 = nn.Linear(128, output_size)

        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)


    # 定义前向传播函数
    def forward(self, x):
        # 激活第一个全连接层
        x = torch.relu(self.fc1(x))
        # 激活第二个全连接层
        x = self.fc2(x)
        # 返回输出
        return x
    
    # 定义训练函数，输入参数为训练数据集、损失函数和优化器
    def train(self, train_loader, num_epochs=10):
        # 设置为训练模式
        self.train()
        # 开始训练
        for epoch in range(num_epochs):
            # 遍历训练数据集
            for input_data, target_label in train_loader:
                # 梯度归零
                self.optimizer.zero_grad()
                # 计算输出
                output = self(input_data)
                # 计算损失
                loss = self.criterion(output, target_label)
                # 反向传播
                loss.backward()
                # 更新参数
                self.optimizer.step()
            # 打印损失
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

        # 保存模型
        torch.save(self.state_dict(), 'snake_ai_model.pth')

    def pushTrainData(self, data):

        dataTensor =  torch.tensor(data['status'], dtype=torch.int64)
        controlMappingTensor = torch.tensor(data['controlMapping'], dtype=torch.int64)

        output = self(dataTensor)
        loss = self.criterion(output, controlMappingTensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(loss)