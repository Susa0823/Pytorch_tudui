import torch
import torchvision
from PIL import Image
from torch import nn

img_path = "img/flight.png"
img = Image.open(img_path)
img = img.convert('RGB')
print(img)
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

img = transform(img)
print(img.shape)


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
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("tudui_gpu26.pth", map_location=torch.device('cpu'))
print(model)
img = torch.reshape(img, (1, 3, 32, 32))

model.eval()  # 转化测试类别
with torch.no_grad():  # with自动处理对文件的关闭操作，节约内存性能
    output = model(img)
print(output)
print(output.argmax(1))
