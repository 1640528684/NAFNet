from basicsr.models.archs.mtv4_arch import NAFNet
import torch
from thop import profile
from thop import clever_format
model_path = '/data/users/qingluhou/Neural_network/motion_deblur/NAFNet/experiments/mtv4-GoPro-width64/models/net_g_376000.pth'
model_state_dict = torch.load(model_path, map_location='cpu')

# 创建一个与原始模型具有相同结构的模型
# loaded_model = NAFNet(width=64, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
loaded_model = NAFNet(width=64, middle_blk_num=1, enc_blk_nums=[1, 1, 14], dec_blk_nums=[1, 1, 1])

loaded_model.load_state_dict(model_state_dict['params'])

# 创建一个示例输入
input_data = torch.randn(1, 3, 256, 256)

# 实例化模型
model = loaded_model

# 将模型转换为评估模式
model.eval()

# 使用 thop 计算 FLOPs
flops, params = profile(model, inputs=(input_data,))

# 使用 clever_format 函数将结果格式化为易读的形式
flops, params = clever_format([flops, params], "%.3f")

print(f"FLOPs: {flops}")
print(f"Params: {params}")
