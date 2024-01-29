from torch.utils.data import Dataset
from PIL import Image
import os  # 关于系统的库，get location

# dataset类（1.获取label，data 2.知道有多少数据） dataloader类(打包，为网络提供不同的数据形式)
class MyData(Dataset):
    # picture -> input, "ant"-> label
    def __init__(self, root_dir, label_dir):
        root_dir = "Dataset/hymenoptera_data/train"
        label_dir = "ants"
        self.path = os.path.join(root_dir, label_dir)
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.img_path = os.listdir(self.path)  # 获取图片所有地址

    def __getitem__(self, idx):    # get location
        img_name = self.img_path[idx]  # get name
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "Dataset/train"
ants_label_dir = "ants"
ant_dataset = MyData(root_dir, ants_label_dir)


