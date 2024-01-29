
"""
计算实际输出和目标之间的差距
为我们更新输出提供一定的依据（反向传播）
X: 1, 2, 3
Y: 1, 2, 5
L1Loss:相差取绝对值的平均,越小越好
L1Loss = (0+0+|-2|)/3=0.6
MSE = (0+0+2^2)/3=1.333

Person, Dog, Cat
  0,    1,    2
output [0.1 ,0.2, 0.3]
Target 1 class
Loss(x, class) = -0.2+ln(exp(0.1)+exp(0.2)+exp(0.3))
"""
import torch
from torch import nn
from torch.nn import L1Loss, MSELoss

input = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

input = torch.reshape(input, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss()
result = loss(input, targets)

loss_mse = nn.MSELoss()
result_mse = loss_mse(input, targets)


print(result)
print(result_mse)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])  # 计算损失函数的类别
x = torch.reshape(x, (1, 3))  # (batch-size,类)
loss_cross = nn.CrossEntropyLoss()
result_loss = loss_cross(x, y)
print(result_loss)
