import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)  # convert float
# # (batch_size, channel, height, width)
# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

tudui = Tudui()
# output = tudui(input)
# print(output)

writer = SummaryWriter("./logs_maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)

    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step = step+1

writer.close()
