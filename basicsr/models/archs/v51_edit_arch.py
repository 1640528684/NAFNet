import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from einops import rearrange
from torchvision import models
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from layers import MultiSpectralAttentionLayer, SimpleGate2

class SimpleGate(nn.Module):
    def __init__(self, dim):
        super(SimpleGate, self).__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=False):
        super(DFFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 
                               kernel_size=3, stride=1, padding=1, 
                               groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.sg = SimpleGate()  # 使用正确门控模块

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * self.sg(x2)
        x = self.project_out(x)
        return x

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight.unsqueeze(-1).unsqueeze(-1)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight.unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(-1).unsqueeze(-1)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        return self.body(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model):
        super().__init__()
        self.d_k = d_k
        self.fc = nn.Linear(d_v, d_model)

    def forward(self, q, k, v):
        attn = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return self.fc(output)

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, d_k, d_v):
        super().__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.W_q = nn.Linear(d_model, h * d_k)
        self.W_k = nn.Linear(d_model, h * d_k)
        self.W_v = nn.Linear(d_model, h * d_v)
        self.fc = nn.Linear(h * d_v, d_model)

    def forward(self, q, k, v):
        batch_size = q.size(0)
        Q = rearrange(self.W_q(q), 'b l (h d) -> b h l d', h=self.h)
        K = rearrange(self.W_k(k), 'b l (h d) -> b h l d', h=self.h)
        V = rearrange(self.W_v(v), 'b l (h d) -> b h l d', h=self.h)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = rearrange(out, 'b h l d -> b l (h d)', h=self.h)
        return self.fc(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1, bias=False, LayerNorm_type='WithBias', att=False):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.att = att
        if att:
            self.mha = MultiHeadAttention(
                h=8, 
                d_model=dim, 
                d_k=dim // 8, 
                d_v=dim // 8
            )
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        residual = x
        if self.att:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.mha(self.norm1(x), self.norm1(x), self.norm1(x))
            x = rearrange(x, 'b (h w) c -> b c h w', h=int(x.size(1)**0.5))
            x = x + residual
        x = x + self.ffn(self.norm2(x))
        return x

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.sg = SimpleGate(c)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, bias=True)
        self.dropout = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, bias=True)
        self.norm1 = LayerNorm(c, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(c, LayerNorm_type='WithBias')

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = shortcut + x
        y = self.norm1(x)
        y = self.conv4(y)
        y = self.sg(y)
        y = self.conv5(y)
        y = self.dropout(y)
        return x + y

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
            self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, x):
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        for i in range(len(x)-2, -1, -1):
            inner_lateral = self.inner_blocks[i](x[i])
            feat_shape = inner_lateral.shape[-2:]
            last_inner = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = last_inner + inner_lateral
            results.insert(0, self.layer_blocks[i](last_inner))
        return results

class VGGLoss(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        self.vgg = vgg_model
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        x_features = self.extract_features(x)
        y_features = self.extract_features(y.detach())
        loss = 0
        for xf, yf in zip(x_features, y_features):
            loss += self.criterion(xf, yf)
        return loss

    def extract_features(self, x):
        features = []
        for layer in self.vgg.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
        return features

class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan*2, 1),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
        self.padder_size = 2 ** len(self.encoders)
        self.vgg = models.vgg19(pretrained=True).features[:22].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.size()  # 确保 x 存在
        x = self.check_image_size(x)
        x = self.intro(x)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, reversed(encs)):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        mod_pad_h = (self.padder_size - x.size(2) % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - x.size(3) % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))

class v51editLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_size = train_size
        self.fast_imp = fast_imp
        base_size = (train_size[2], train_size[3])
        
        # 修正：显式创建虚拟输入并传递给 forward
        device = next(self.parameters()).device
        dummy_input = torch.randn(1, 3, *base_size).to(device)
        self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp, dummy_input=dummy_input)

    def convert(self, base_size, train_size, fast_imp, dummy_input):
        # 确保在调用 forward 时传递 dummy_input
        self.forward(dummy_input)  # 传递虚拟输入
        
        # 其他转换逻辑（如动态调整网络结构）
        # 示例：设置训练尺寸和快速模式
        self.train_size = train_size
        self.fast_imp = fast_imp

    def train(self, mode=True):
        super().train(mode)
        self.vgg.eval()  # 冻结VGG
        return self