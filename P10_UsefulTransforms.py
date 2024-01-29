from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
'''关注输入输出类型
多看官方文档
关注方法需要什么参数
使用print
'''

writer = SummaryWriter("logs")
img = Image.open("Dataset/dataset/train/ants_image/0013035.jpg")
# img.show()
print(img)

# Totensor
trans_totensor = transforms.ToTensor()
output = trans_totensor(img)
writer.add_image("ToTensor", output)


''' output[channel] = (input[channel] - mean[channel]) / std[channel]
    (input-0.5)/0.5 = 2*input-1
    input [0,1] -> result [-1,1]
'''
# Normalize 不同特征在数值上量纲可能不一样，归一化使量纲大致相同，输入PLR image
print((output[0][0][0]))        # 3通道平均值，标准差
trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1],True)
img_norm = trans_norm(output)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)


# Compose -resize -2
trans_resize_2 = transforms.Resize(512)
# Compose 参数需要列表，数据需要transform类型，Compose([transform参数1， transform参数2])
# PIL -> PIL -> Tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize2 = trans_compose(img)
writer.add_image("Resize", img_resize2, 1)

# RamdomCrop
trans_random = transforms.RandomCrop(500,1000)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
writer.add_image("RandomCrop", img_crop, i)


writer.close()
