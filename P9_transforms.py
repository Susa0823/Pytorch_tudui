from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import cv2
# tensorboard --logdir=logs 打开主机窗口
# tensorboard --logdir=logs --port=6007 指定端口
img_path = "Dataset/dataset/train/ants_image/0013035.jpg"
# cv_img = cv2.imread(img_path)  numpy image
img = Image.open(img_path)
print(img)
writer = SummaryWriter("logs")
#  1.transform how to use (python)
# create own tool ToTensor -> 输出传入图片
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()

'''
PIL Image.open()
tensor ToTensor()
narrays矩阵 cv.imread()
'''
