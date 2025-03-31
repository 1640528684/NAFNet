# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from einops import rearrange
from mmcv.ops import ModulatedDeformConv2d, DeformConv2d
from .layers import *
from basicsr.models.DSConv import DSConv

class hallucination_module(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, norm_layer=nn.InstanceNorm2d):
        super(hallucination_module, self).__init__()

        self.dilation = dilation

        if self.dilation != 0:

            self.hallucination_conv = DeformConv(out_channels, out_channels, modulation=True, dilation=self.dilation)

        else:

            self.m_conv = nn.Conv2d(in_channels, 3 * 3, kernel_size=3, stride=1, bias=True)

            self.m_conv.weight.data.zero_()
            self.m_conv.bias.data.zero_()

            self.dconv = ModulatedDeformConv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):

        if self.dilation != 0:

            hallucination_output, hallucination_map = self.hallucination_conv(x)

        else:
            hallucination_map = 0

            mask = torch.sigmoid(self.m_conv(x))

            offset = torch.zeros_like(mask.repeat(1, 2, 1, 1))

            hallucination_output = self.dconv(x, offset, mask)

        return hallucination_output, hallucination_map

class hallucination_res_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=(0, 1, 2, 4), norm_layer=nn.InstanceNorm2d):
        super(hallucination_res_block, self).__init__()

        self.dilations = dilations

        self.res_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1),
                                      nn.LeakyReLU(0.1, inplace=True))

        self.hallucination_d0 = DSConv(in_channels, out_channels, 3, 1.0, 0, if_offset=True,device='cuda')
        self.hallucination_d1 = DSConv(in_channels, out_channels, 3, 1.0, 1, if_offset=True,device='cuda')
        self.hallucination_d2 = DSConv(in_channels, out_channels, 3, 1.0, 0, if_offset=True,device='cuda')
        self.hallucination_d3 = DSConv(in_channels, out_channels, 3, 1.0, 1, if_offset=True,device='cuda')

        self.mask_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       ResBlock_2(out_channels, out_channels, norm_layer=norm_layer),
                                       ResBlock_2(out_channels, out_channels, norm_layer=norm_layer),
                                       nn.Conv2d(out_channels, 4, 1, 1))

        self.fusion_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        res = self.res_conv(x)

        d0_out = self.hallucination_d0(res)
        d1_out = self.hallucination_d1(res)
        d2_out = self.hallucination_d2(res)
        d3_out = self.hallucination_d3(res)

        mask = self.mask_conv(x)
        mask = torch.softmax(mask, 1)

        sum_out = d0_out * mask[:, 0:1, :, :] + d1_out * mask[:, 1:2, :, :] + \
                  d2_out * mask[:, 2:3, :, :] + d3_out * mask[:, 3:4, :, :]

        res = self.fusion_conv(sum_out) + x

        # map = torch.cat([map1, map2, map3], 1)

        return res


class DeformConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=True, modulation=True, dilation=1):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(1)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, stride=stride, dilation=dilation,
                                padding=dilation, bias=bias)

        self.p_conv.weight.data.zero_()
        if bias:
            self.p_conv.bias.data.zero_()

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, stride=stride, dilation=dilation,
                                    padding=dilation, bias=bias)

            self.m_conv.weight.data.zero_()
            if bias:
                self.m_conv.bias.data.zero_()

            self.dconv = ModulatedDeformConv2d(inc, outc, kernel_size, padding=padding)
        else:
            self.dconv = DeformConv2d(inc, outc, kernel_size, padding=padding)

    def forward(self, x):
        offset = self.p_conv(x)

        if self.modulation:
            mask = torch.sigmoid(self.m_conv(x))
            x_offset_conv = self.dconv(x, offset, mask)
        else:
            x_offset_conv = self.dconv(x, offset)

        return x_offset_conv, offset

# 在通道维度上进行全局的pooling操作，再经过同一个MLP得到权重，相加作为最终的注意力向量（权重）。
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 1是输出特征图大小
        self.max_pool = nn.AdaptiveMaxPool2d(1)  #

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            # nn.Conv2d(输入channels，输出channels，卷积核kernels_size=3, padding=1, bias=False)
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()  #
        # self.softmax = nn.Softmax(dim = 1) #channel维度上处理 nn.Softmax(dim = 1)?????

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))  # torch.Size([1, 16, 1, 1])
        # print('【Channel】avgout.shape {}'.format(avgout.shape)) # torch.Size([1, 16, 1, 1])
        maxout = self.shared_MLP(self.max_pool(x))  # torch.Size([1, 16, 1, 1])
        return self.sigmoid(avgout + maxout)  # torch.Size([1, 16, 1, 1])
        # return self.softmax(avgout + maxout) # torch.Size([1, 16, 1, 1])


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        # self.conv2 = ResBlock(dw_channel, dw_channel, groups=dw_channel)
        self.conv3 = nn.Conv2d(in_channels=dw_channel//2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # self.sca = ChannelAttentionModule(dw_channel)

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        # self.conv5 = ResBlock(dw_channel, dw_channel, groups=dw_channel)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel//2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        # x = self.conv6(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)
        self.xx_ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                   groups=1,
                                   bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.decoders_2 = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.MFFs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

            self.decoders_2.append(
                nn.Sequential(
                    *[hallucination_res_block(chan, chan, dilations=(0, 1, 2, 4), norm_layer=nn.InstanceNorm2d) for _ in range(2*num)]
                )
            )
            # self.MFFs.append(
            #     nn.Sequential(
            #         *[MFF(chan, 1.5, False) for _ in range(num)]
            #     )
            # )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)

            encs.append(x)
            x = down(x)


        x_midle = self.middle_blks(x)
        x = x_midle
        x_t = x_midle
        x_x = x_midle
        # x_x = x_midle

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x_t = up(x_t)
            x_t = x_t + enc_skip
            x_t = x_t.transpose(2, 3).flip(2)
            x_t_1 = decoder(x_t)
            x_t = x_t_1.transpose(2, 3).flip(3)

        for decoder, up, enc_skip in zip(self.decoders_2, self.ups, encs[::-1]):
            x_x = up(x_x)
            x_x = x_x + enc_skip
            x_x = decoder(x_x)


        x = self.ending(x)
        x_t = self.ending(x_t_1).transpose(2, 3).flip(3)

        x_x = self.xx_ending(x_x)
        x_last = x + x_t + inp +x_x

        # return x, x_t, x_x, x_last[:, :, :H, :W]

        return x_last[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class mutv4Local(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        # base_size = (512, 512)

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                 enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)