from game.game import *
from ai.ai import *


tileCountX = 50
tileCountY = 50
input_size = tileCountX * tileCountY + 2 + 1 + 1 + 1 + 5 # 输入特征数：地图格子数 + 食物坐标 + 蛇运动方向 + 得分 + 游戏是否结束 + 可输入操作
output_size = 1  # 输出特征数：操作情况 只有 上 下 左 右 重启 中的一个

ai = SnakeAI(input_size, output_size)

game = Game(tileCountX, tileCountY)
game.initGame()
game.startGame(ai.pushTrainData)

print("程序运行结束")