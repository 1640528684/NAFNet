import torch
from torchvision import utils

# 创建一个示例张量
tensor_image = torch.randn(3, 256, 256)  # 3通道，256x256的示例张量
for i in range(64):
    tensor_image = x_v[0,i,:,:]
    # 将张量规范化到 [0, 1] 范围
    tensor_image = (tensor_image - tensor_image.min()) / (tensor_image.max() - tensor_image.min())

    # 将张量保存为图像
    utils.save_image(tensor_image, '/data/users/qingluhou/Neural_network/motion_deblur/Datesets/temp/x_v'+str(i)+'.png')

# 文件保存在当前工作目录下，名为 saved_image.png
