import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil


class ResidualDenseBlock_2C(nn.Module):
    def __init__(self, filters=64, bias=True):
        super(ResidualDenseBlock_2C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(filters, filters, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(filters, filters, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        return x2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, filters=64):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_2C(filters)
        self.RDB2 = ResidualDenseBlock_2C(filters)
        self.RDB3 = ResidualDenseBlock_2C(filters)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out + x


class UpsampleConcatSqueeze(nn.Module):
    def __init__(self, filters_in, filters_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(filters_in, filters_in // 2, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(filters_in // 2 + filters_out, filters_out, 1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1) #CHW

        x = self.conv(x)
        return x


class USRGAN(nn.Module):
    def __init__(self, in_nc, out_nc, filters=64):
        super(USRGAN, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, filters, 3, 1, 1, bias=True)

        # Downsampling
        self.rrdb_1_1 = RRDB(filters)
        self.rrdb_1_2 = RRDB(filters)

        self.rrdb_2_1 = RRDB(filters)
        self.rrdb_2_2 = RRDB(filters)
        self.change_fn_1 = nn.Conv2d(filters, filters * 2, 1)

        self.rrdb_3_1 = RRDB(filters * 2)
        self.rrdb_3_2 = RRDB(filters * 2)

        self.rrdb_4_1 = RRDB(filters * 2)
        self.rrdb_4_2 = RRDB(filters * 2)
        self.change_fn_2 = nn.Conv2d(filters * 2, filters * 4, 1)

        # Bottleneck
        self.rrdb_5_1 = RRDB(filters * 4)
        self.rrdb_5_2 = RRDB(filters * 4)

        # Upsampling
        self.up_1 = UpsampleConcatSqueeze(filters * 4, filters * 2)
        self.rrdb_6_1 = RRDB(filters * 2)
        self.rrdb_6_2 = RRDB(filters * 2)

        self.up_2 = UpsampleConcatSqueeze(filters * 2, filters * 2)
        self.rrdb_7_1 = RRDB(filters * 2)
        self.rrdb_7_2 = RRDB(filters * 2)

        self.up_3 = UpsampleConcatSqueeze(filters * 2, filters)
        self.rrdb_8_1 = RRDB(filters)
        self.rrdb_8_2 = RRDB(filters)

        self.up_4 = UpsampleConcatSqueeze(filters, filters)
        self.rrdb_9_1 = RRDB(filters)
        self.rrdb_9_2 = RRDB(filters)

        self.upconv1 = nn.Conv2d(filters, filters, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(filters, filters, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(filters, filters, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(filters, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.avr_pool = nn.AvgPool2d(2)

    def forward(self, x):
        x_1 = self.conv_first(x)

        # Downsampling
        x_1_1 = self.rrdb_1_1(x_1)
        x_1_2 = self.rrdb_1_2(x_1_1)
        x_1_2_p = self.avr_pool(x_1_2)

        x_2_1 = self.rrdb_2_1(x_1_2_p)
        x_2_2 = self.rrdb_2_2(x_2_1)
        x_2_2_p = self.avr_pool(x_2_2)
        x_2_2_p = self.change_fn_1(x_2_2_p)

        x_3_1 = self.rrdb_3_1(x_2_2_p)
        x_3_2 = self.rrdb_3_2(x_3_1)
        x_3_2_p = self.avr_pool(x_3_2)

        x_4_1 = self.rrdb_4_1(x_3_2_p)
        x_4_2 = self.rrdb_4_2(x_4_1)
        x_4_2_p = self.avr_pool(x_4_2)
        x_4_2_p = self.change_fn_2(x_4_2_p)

        # Bottleneck
        x_5_1 = self.rrdb_5_1(x_4_2_p)
        x_5_2 = self.rrdb_5_2(x_5_1)

        # Upsampling
        x_6_1_u = self.up_1(x_5_2, x_4_2)
        x_6_1 = self.rrdb_6_1(x_6_1_u)
        x_6_2 = self.rrdb_6_2(x_6_1)

        x_7_1_u = self.up_2(x_6_2, x_3_2)
        x_7_1 = self.rrdb_7_1(x_7_1_u)
        x_7_2 = self.rrdb_7_2(x_7_1)

        x_8_1_u = self.up_3(x_7_2, x_2_2)
        x_8_1 = self.rrdb_8_1(x_8_1_u)
        x_8_2 = self.rrdb_8_2(x_8_1)

        x_9_1_u = self.up_4(x_8_2, x_1_2)
        x_9_1 = self.rrdb_9_1(x_9_1_u)
        x_9_2 = self.rrdb_9_2(x_9_1)

        x = x_9_2 + x_1

        x = self.lrelu(self.upconv1(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.lrelu(self.upconv2(F.interpolate(x, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(x)))

        return out
