"""
Dual-task module
"""
import torch.nn as nn
import torch.nn.functional as F

from .model_blocks import *


class JUTNetUpc(nn.Module):
    def __init__(
            self,
            in_chan=1,
            base_chan=32,
            seg_num_classes=2,
            qa_num_classes=2,
            reduce_size=16,
            block_list='1234',
            num_blocks=[1, 1, 1, 1],
            projection='interp',
            num_heads=[4, 4, 4, 4],
            attn_drop=0.1,
            proj_drop=0.1,
            bottleneck=False,
            maxpool=True,
            rel_pos=True,
            aux_loss=False,
            img_shape=[256, 256]
    ):
        super().__init__()
        
        self.enc = UTNetEnc(in_chan, base_chan, reduce_size, block_list, num_blocks, projection, num_heads, attn_drop, proj_drop, bottleneck, maxpool, rel_pos)
        self.seg_dec = UTNetDecUpc(base_chan, 8*base_chan, seg_num_classes, reduce_size, block_list, projection, num_heads, attn_drop, proj_drop, bottleneck, rel_pos, aux_loss, img_shape)
        self.qa_dec = QADec(16*base_chan, qa_num_classes, img_shape)
        self.feature_shape = [i//8 for i in img_shape]

    def forward(self, x):
        x_enc = self.enc(x)

        x_qa = self.qa_dec(x_enc[-1])
        qa_feat = F.interpolate(x_qa[0].clone().detach(), self.feature_shape)

        x_seg = self.seg_dec(*x_enc, qa_feat)

        return x_seg, x_qa[-1]


class QADec(nn.Module):
    def __init__(self, in_channels, num_class=2, img_shape=[256, 256]):
        super().__init__()

        self.conv1 = BasicConvBlock(in_channels, in_channels//2)
        self.conv2 = BasicConvBlock(256, 64)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(32 * (img_shape[0] // 16) * (img_shape[1] // 16), 256)
        self.fc2 = nn.Linear(256, 16)
        self.fc3 = nn.Linear(16, num_class)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        fc1 = self.relu(self.fc1(conv3))
        fc2 = self.relu(self.fc2(fc1))
        fc3 = self.relu(self.fc3(fc2))

        return conv1, conv2, fc3


# modified UTNet Encoder & Decoder
# Thanks: https://github.com/yhygao/UTNet

class UTNetEnc(nn.Module):
    
    def __init__(self, in_chan, base_chan, reduce_size=8, block_list='234', num_blocks=[1, 2, 4], projection='interp', num_heads=[2,4,8], attn_drop=0., proj_drop=0., bottleneck=False, maxpool=True, rel_pos=True):
        super().__init__()

        self.inc = [ResConvBlock(in_chan, base_chan)]
        if '0' in block_list:
            self.inc.append(BasicTransBlock(base_chan, heads=num_heads[-5], dim_head=base_chan//num_heads[-5], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos))
        else:
            self.inc.append(ResConvBlock(base_chan, base_chan))
        self.inc = nn.Sequential(*self.inc)

        if '1' in block_list:
            self.down1 = down_block_trans(base_chan, 2*base_chan, num_block=num_blocks[-4], bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-4], dim_head=2*base_chan//num_heads[-4], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            self.down1 = down_block(base_chan, 2*base_chan, (2,2), num_block=2)

        if '2' in block_list:
            self.down2 = down_block_trans(2*base_chan, 4*base_chan, num_block=num_blocks[-3], bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-3], dim_head=4*base_chan//num_heads[-3], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            self.down2 = down_block(2*base_chan, 4*base_chan, (2, 2), num_block=2)

        if '3' in block_list:
            self.down3 = down_block_trans(4*base_chan, 8*base_chan, num_block=num_blocks[-2], bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-2], dim_head=8*base_chan//num_heads[-2], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            self.down3 = down_block(4*base_chan, 8*base_chan, (2,2), num_block=2)

        if '4' in block_list:
            self.down4 = down_block_trans(8*base_chan, 16*base_chan, num_block=num_blocks[-1], bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-1], dim_head=16*base_chan//num_heads[-1], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            self.down4 = down_block(8*base_chan, 16*base_chan, (2,2), num_block=2)

    def forward(self, x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        return x1, x2, x3, x4, x5


class UTNetDecUpc(nn.Module):

    def __init__(self, base_chan, upc_ch=None, num_classes=1, reduce_size=8, block_list='234', projection='interp', num_heads=[2,4,8], attn_drop=0., proj_drop=0., bottleneck=False, rel_pos=True, aux_loss=False, img_shape=[256, 256]):
        super().__init__()

        if '0' in block_list:
            self.up4 = up_block_trans(2*base_chan, base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-4], dim_head=base_chan//num_heads[-4], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            self.up4 = up_block(2*base_chan, base_chan, scale=(2,2), num_block=2)

        if '1' in block_list:
            self.up3 = up_block_trans(4*base_chan, 2*base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-3], dim_head=2*base_chan//num_heads[-3], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            self.up3 = up_block(4*base_chan, 2*base_chan, scale=(2,2), num_block=2)

        if '2' in block_list:
            self.up2 = up_block_trans(8*base_chan, 4*base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-2], dim_head=4*base_chan//num_heads[-2], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            self.up2 = up_block(8*base_chan, 4*base_chan, scale=(2,2), num_block=2)

        if '3' in block_list:
            if upc_ch is None:
                self.up1 = up_block_trans(16*base_chan, 8*base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-1], dim_head=8*base_chan//num_heads[-1], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
            else:
                self.up1 = up_block_trans_upc(16*base_chan, 8*base_chan, upc_ch, num_block=0, bottleneck=bottleneck, heads=num_heads[-1], dim_head=8*base_chan//num_heads[-1], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            if upc_ch is None:
                self.up1 = up_block(16*base_chan, 8*base_chan, scale=(2,2), num_block=2)
            else:
                self.up1 = up_block_upc(16*base_chan, 8*base_chan, upc_ch, scale=(2,2), num_block=2)

        self.outc = nn.Conv2d(base_chan, num_classes, kernel_size=1, bias=True)

        self.aux_loss = aux_loss
        if aux_loss:
            self.out1 = nn.Conv2d(8*base_chan, num_classes, kernel_size=1, bias=True)
            self.out2 = nn.Conv2d(4*base_chan, num_classes, kernel_size=1, bias=True)
            self.out3 = nn.Conv2d(2*base_chan, num_classes, kernel_size=1, bias=True)
        
        self.img_shape = img_shape
        self.upc_ch = upc_ch
            
    def forward(self, x1, x2, x3, x4, x5, xu):
        
        if self.aux_loss:
            out = self.up1(x5, x4)
            out1 = F.interpolate(self.out1(out), size=self.img_shape, mode='bilinear', align_corners=True)

            out = self.up2(out, x3)
            out2 = F.interpolate(self.out2(out), size=self.img_shape, mode='bilinear', align_corners=True)

            out = self.up3(out, x2)
            out3 = F.interpolate(self.out3(out), size=self.img_shape, mode='bilinear', align_corners=True)

            out = self.up4(out, x1)
            out = self.outc(out)

            return out, out3, out2, out1

        else:
            if self.upc_ch is None:
                out = self.up1(x5, x4)
            else:
                out = self.up1(x5, x4, xu)
            out = self.up2(out, x3)
            out = self.up3(out, x2)

            out = self.up4(out, x1)
            out = self.outc(out)

            return out
