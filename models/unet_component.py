'''
Components of U-Net Model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.enabled = False
# 禁用 cudnn 加速 （否则报错）

class DoubleConv(nn.Module):
    '''
    Double Convolution Block
    (convolution => [BN] => ReLU) * 2
    '''
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels == None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False),
            # 后面跟着BatchNorm的时候，前面的bias可以不要，反正也会被消掉
            nn.BatchNorm2d(num_features=mid_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        # TODO: add dropout

    def forward(self, x):
        # print(x.shape)
        return self.double_conv(x)
    
class DownBlock(nn.Module):
    '''
    Down Block of U-Net
    Double conv then DownScaling with MaxPooling
    '''
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, mid_channels)
        self.max_pooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        skip = self.double_conv(x)
        x = self.max_pooling(skip)
        return x, skip
        
class UpBlock(nn.Module):
    '''
    Up Block of U-Net
    Upscaling with TransposeConv then double conv
    '''
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        # print(in_channels, out_channels)
        self.up_sampling = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels, mid_channels)
        # TODO: bilinear

    def forward(self, x, skip):
        x = self.up_sampling(x)
        assert x.shape[1] == skip.shape[1], "Channels of x and skip must be same"

        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        assert diff_h >= 0 and diff_w >= 0

        pad_h = (diff_h // 2, diff_h - diff_h // 2)
        pad_w = (diff_w // 2, diff_w - diff_w // 2)
        x = F.pad(x, [pad_w[0], pad_w[1], pad_h[0], pad_h[1]])  #按照左、右、上、下的顺序进行padding
        assert x.shape[2] == skip.shape[2] and x.shape[3] == skip.shape[3], "Physical dimensions of x and skip must be same"
        
        # print(x.shape, skip.shape)
        x = torch.cat([x, skip], dim=1)   # 使用torch进行concat的用法
        # print(x.shape)

        return self.double_conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))