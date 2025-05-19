'''
Pytorch implementation of U-Net Model
'''

from .unet_component import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_channels=32):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_channels = base_channels

        self.down_block1 = DownBlock(n_channels, base_channels)
        self.down_block2 = DownBlock(base_channels, base_channels*2)
        self.down_block3 = DownBlock(base_channels*2, base_channels*4)
        self.down_block4 = DownBlock(base_channels*4, base_channels*8)

        self.bottom_block = DoubleConv(base_channels*8, base_channels*16)

        self.up_block1 = UpBlock(base_channels*16, base_channels*8)
        self.up_block2 = UpBlock(base_channels*8, base_channels*4)
        self.up_block3 = UpBlock(base_channels*4, base_channels*2)
        self.up_block4 = UpBlock(base_channels*2, base_channels)

        self.output = OutConv(base_channels, n_classes)

    def forward(self, x):
        x, skip1 = self.down_block1(x)
        x, skip2 = self.down_block2(x)
        x, skip3 = self.down_block3(x)
        x, skip4 = self.down_block4(x)

        x = self.bottom_block(x)
        # print(x.shape)

        x = self.up_block1(x, skip4)
        x = self.up_block2(x, skip3)
        x = self.up_block3(x, skip2)
        x = self.up_block4(x, skip1)

        x = self.output(x)

        return x