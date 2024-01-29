import torch
import torchvision

# way1
model = torch.load("vgg16_method1.pth")
# print(model)


# way2
vgg16 = torchvision.models.vgg16(wights=None)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
print(vgg16)

