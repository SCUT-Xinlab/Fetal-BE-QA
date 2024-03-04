"""
Model Blocks, Transformer and Conv
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .trans_layers import *


# ------Transformer Blocks------
# Thanks: https://github.com/yhygao/UTNet

class BasicTransBlock(nn.Module):

    def __init__(self, in_ch, heads, dim_head, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)

        self.attn = LinearAttention(in_ch, heads=heads, dim_head=in_ch//heads, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        # conv1x1 has not difference with mlp in performance

    def forward(self, x):

        out = self.bn1(x)
        out, q_k_attn = self.attn(out)
        
        out = out + x
        residue = out

        out = self.bn2(out)
        out = self.relu(out)
        out = self.mlp(out)

        out += residue

        return out


class down_block_trans(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, bottleneck=False, maxpool=True, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()

        block_list = []

        if bottleneck:
            block = BottleneckBlock
        else:
            block = ResConvBlock

        attn_block = BasicTransBlock

        if maxpool:
            block_list.append(nn.MaxPool2d(2))
            block_list.append(block(in_ch, out_ch, stride=1))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))
        
        assert num_block > 0
        for i in range(num_block):
            block_list.append(attn_block(out_ch, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos))
        self.blocks = nn.Sequential(*block_list)

    def forward(self, x):
        out = self.blocks(x)
        return out
    

class BasicTransDecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch, heads, dim_head, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()

        self.bn_l = nn.BatchNorm2d(in_ch)
        self.bn_h = nn.BatchNorm2d(out_ch)

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.attn = LinearAttentionDecoder(in_ch, out_ch, heads=heads, dim_head=out_ch//heads, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x1, x2):

        residue = F.interpolate(self.conv_ch(x1), size=x2.shape[-2:], mode='bilinear', align_corners=True)
        #x1: low-res, x2: high-res
        x1 = self.bn_l(x1)
        x2 = self.bn_h(x2)

        out, q_k_attn = self.attn(x2, x1)
        
        out = out + residue
        residue = out

        out = self.bn2(out)
        out = self.relu(out)
        out = self.mlp(out)

        out += residue

        return out


class up_block_trans(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, bottleneck=False, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()
 
        self.attn_decoder = BasicTransDecoderBlock(in_ch, out_ch, heads=heads, dim_head=dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)

        if bottleneck:
            block = BottleneckBlock
        else:
            block = ResConvBlock
        attn_block = BasicTransBlock
        
        block_list = []

        for i in range(num_block):
            block_list.append(attn_block(out_ch, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos))

        block_list.append(block(2*out_ch, out_ch, stride=1))

        self.blocks = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        # x1: low-res feature, x2: high-res feature
        out = self.attn_decoder(x1, x2)
        out = torch.cat([out, x2], dim=1)
        out = self.blocks(out)

        return out


class up_block_trans_upc(nn.Module):
    def __init__(self, in_ch, out_ch, upc_ch, num_block, bottleneck=False, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()
 
        self.attn_decoder = BasicTransDecoderBlock(in_ch, out_ch, heads=heads, dim_head=dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)

        if bottleneck:
            block = BottleneckBlock
        else:
            block = ResConvBlock
        attn_block = BasicTransBlock
        
        block_list = []

        for i in range(num_block):
            block_list.append(attn_block(out_ch, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos))

        block_list.append(block(2*out_ch + upc_ch, out_ch, stride=1))

        self.blocks = nn.Sequential(*block_list)

    def forward(self, x1, x2, xu):
        # x1: low-res feature, x2: high-res feature
        out = self.attn_decoder(x1, x2)
        out = torch.cat([out, x2, xu], dim=1)
        out = self.blocks(out)

        return out


# ------Conv Blocks------

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ResConvBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inplanes),
                    self.relu, 
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                    )

    def forward(self, x): 
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.shortcut(residue)

        return out


class BottleneckBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes//4, stride=1)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes//4, planes//4, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes//4)

        self.conv3 = conv1x1(planes//4, planes, stride=1)
        self.bn3 = nn.BatchNorm2d(planes//4)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inplanes),
                    self.relu,
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                    )

    def forward(self, x):
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += self.shortcut(residue)

        return out


class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, scale, num_block, bottleneck=False, pool=True):
        super().__init__()

        block_list = []

        if bottleneck:
            block = BottleneckBlock
        else:
            block = ResConvBlock

        if pool:
            block_list.append(nn.MaxPool2d(scale))
            block_list.append(block(in_ch, out_ch))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch, stride=1))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x):
        return self.conv(x)


class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, scale=(2,2),bottleneck=False):
        super().__init__()
        self.scale=scale

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        if bottleneck:
            block = BottleneckBlock
        else:
            block = ResConvBlock

        block_list = []
        block_list.append(block(2*out_ch, out_ch))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)

        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)

        return out


class up_block_upc(nn.Module):
    def __init__(self, in_ch, out_ch, upc_ch, num_block, scale=(2,2),bottleneck=False):
        super().__init__()
        self.scale=scale

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        if bottleneck:
            block = BottleneckBlock
        else:
            block = ResConvBlock


        block_list = []
        block_list.append(block(2*out_ch + upc_ch, out_ch))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2, xu):
        x1 = F.interpolate(x1, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)

        out = torch.cat([x2, x1, xu], dim=1)
        out = self.conv(out)

        return out
