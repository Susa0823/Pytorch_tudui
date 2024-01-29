import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.module1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.module1(x)
        return x

loss = nn.CrossEntropyLoss()
tudui = Tudui()

# 定义一个优化器
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)
# epoch 轮
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        # 对优化器每个参数梯度清零
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        # 调用损失参数的反向传播，求每个节点的梯度
        result_loss.backward()
        # 对模型参数每个调优
        optim.step()
        # 整体误差求和
        running_loss = running_loss + result_loss
    print(running_loss)
