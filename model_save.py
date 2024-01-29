import torch
import torchvision

vgg16 = torchvision.models.vgg16(weights=None)
# 保存方式1 （保存网络结构+网络中参数）
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2 模型参数，状态转为字典形式（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
