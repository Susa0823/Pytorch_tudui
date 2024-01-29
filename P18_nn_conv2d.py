import torch
import torchvision

from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):  # 初始化
        super(Tudui, self).__init__()
        # 卷基层
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()  # 初始化网络

writer = SummaryWriter("./p18")
step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(imgs.shape)    # torch.Size([64张图片, 3 层, 32, 32])  32x32
    print(output.shape)  # torch.Size([64, 6, 30, 30]) 经过卷积后
writer.add_images("input", imgs, step)
# torch.size([64,6,30,30]) -> [xxx,3,30,30]  -1自动计算
output = torch.reshape(output, (-1, 3, 30, 30))
writer.add_images("output", output, step)
step = step + 1
