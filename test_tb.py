from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
# writer.add_image()
# writer.add_scalar()  add number
#  y = x
for i in range(100):   # tag, x, y
    writer.add_scalar("y = 2x",3*i,i)
writer.close()

