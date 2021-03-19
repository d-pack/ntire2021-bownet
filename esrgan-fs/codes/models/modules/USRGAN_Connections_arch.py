import functools
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class UpsampleConcatSqueeze(nn.Module):
    def __init__(self, filters_in, filters_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(filters_in, filters_in // 2, kernel_size=2, stride=2)
        self.med_conn = nn.Sequential(
            nn.Conv2d(filters_out, filters_out, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv = nn.Conv2d(filters_in // 2 + filters_out, filters_out, 1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x2 = self.med_conn(x2)
        x = torch.cat([x2, x1], dim=1) #CHW

        x = self.conv(x)
        return x


class USRGANLarge(nn.Module):
    def __init__(self, in_nc, out_nc, nf, gc=32, nb=23):
        super(USRGANLarge, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        # Downsampling
        self.rrdb_1_1 = RRDB(nf=nf, gc=gc)
        self.rrdb_1_2 = RRDB(nf=nf, gc=gc)

        self.rrdb_2_1 = RRDB(nf=nf, gc=gc)
        self.rrdb_2_2 = RRDB(nf=nf, gc=gc)
        self.change_fn_1 = nn.Conv2d(nf, nf * 2, 1)

        self.rrdb_3_1 = RRDB(nf=nf * 2, gc=gc * 2)
        self.rrdb_3_2 = RRDB(nf=nf * 2, gc=gc * 2)

        self.rrdb_4_1 = RRDB(nf=nf * 2, gc=gc * 2)
        self.rrdb_4_2 = RRDB(nf=nf * 2, gc=gc * 2)
        self.change_fn_2 = nn.Conv2d(nf * 2, nf * 4, 1)

        # Bottleneck
        self.rrdb_5_1 = RRDB(nf=nf * 4, gc=gc * 4)
        self.rrdb_5_2 = RRDB(nf=nf * 4, gc=gc * 4)

        # Upsampling
        self.up_1 = UpsampleConcatSqueeze(nf * 4, nf * 2)
        self.rrdb_6_1 = RRDB(nf=nf * 2, gc=gc * 2)
        self.rrdb_6_2 = RRDB(nf=nf * 2, gc=gc * 2)

        self.up_2 = UpsampleConcatSqueeze(nf * 2, nf * 2)
        self.rrdb_7_1 = RRDB(nf=nf * 2, gc=gc * 2)
        self.rrdb_7_2 = RRDB(nf=nf * 2, gc=gc * 2)

        self.up_3 = UpsampleConcatSqueeze(nf * 2, nf)
        self.rrdb_8_1 = RRDB(nf=nf, gc=gc)
        self.rrdb_8_2 = RRDB(nf=nf, gc=gc)

        self.up_4 = UpsampleConcatSqueeze(nf, nf)
        self.rrdb_9_1 = RRDB(nf=nf, gc=gc)
        self.rrdb_9_2 = RRDB(nf=nf, gc=gc)

        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=True)

        self.conv_scale_x2 = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1),
            *[RRDB(nf=nf, gc=gc) for _ in range(nb)]
        )

        self.conv_scale_x4 = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1),
            *[RRDB(nf=nf, gc=gc) for _ in range(nb)]
        )

        self.HR_rrdb = RRDB(nf=nf * 3, gc=gc * 3)
        self.conv_last = nn.Conv2d(nf * 3, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.avr_pool = nn.AvgPool2d(2)

    def forward(self, x_in):
        x_in_x2 = F.interpolate(x_in, scale_factor=2, mode='nearest')
        x_in_x4 = F.interpolate(x_in_x2, scale_factor=2, mode='nearest')

        x_1 = self.conv_first(x_in)

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

        t0 = time.time()
        x = self.lrelu(self.upconv1(F.interpolate(x, scale_factor=2, mode='nearest')))
        x_x2 = self.conv_scale_x2(x_in_x2)
        x = torch.cat([x, x_x2], dim=1)

        x = self.lrelu(self.upconv2(F.interpolate(x, scale_factor=2, mode='nearest')))
        x_x4 = self.conv_scale_x4(x_in_x4)
        x = torch.cat([x, x_x4], dim=1)

        out = self.conv_last(self.HR_rrdb(x))
        print(f'UpTime: {time.time()-t0:.4f} sec')

        return out
