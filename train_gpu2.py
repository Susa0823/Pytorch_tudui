import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time  # 计时
# from train_module import *

'''GPU
网络模型
数据（输入，标注）
损失函数
.cuda()
除了数据，图片可以不另外赋值
torch.device("cuda")
torch.device("cuda:0") 第一张显卡
torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
# 定义训练的设备
device = torch.device("cpu")

train_data = torchvision.datasets.CIFAR10("../data", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("../data", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

# get data_size
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集长度为：{}".format(train_data_size))
print("测试集长度为：{}".format(test_data_size))

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataLoader = DataLoader(test_data, batch_size=64)


# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 创建网络模型
tudui = Tudui()
tudui = tudui.to(device)

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 创建优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./logs_train")
for i in range(epoch):
    print("-----------第{}轮训练开始--------".format(i + 1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = tudui(imgs)
        loss = loss_fn(output, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数: {}，Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    # with里代码梯度没有，不用调优
    with torch.no_grad():
        for data in test_dataLoader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = imgs.to(device)

            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            # 整体测试集正确率横向
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的loss: {}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(tudui, "tudui_{}pth".format(i))
    print("模型已保存")

writer.close()
