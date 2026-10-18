import sys
import math
import torch
import torch.nn as nn
from torch.nn import Conv2d, Conv3d


class ConvD2E7(nn.Module):
    def __init__(self, out_channels, nChannel, spanT, d_cnn2d, dropout):
        super().__init__()

        self.conv2d_m1r0 = Conv2d(1, out_channels[0], (nChannel, math.ceil(spanT[0])), stride=(1, math.ceil(spanT[0] / 2)))
        self.conv2d_m1r1 = Conv2d(1, out_channels[0], (nChannel, math.ceil(spanT[0])), stride=(1, math.ceil(spanT[0] / 2)))
        self.conv2d_m1r2 = Conv2d(1, out_channels[0], (nChannel, math.ceil(spanT[0])), stride=(1, math.ceil(spanT[0] / 2)))
        self.conv2d_m1r3 = Conv2d(1, out_channels[0], (nChannel, math.ceil(spanT[0])), stride=(1, math.ceil(spanT[0] / 2)))
        self.conv2d_m1r4 = Conv2d(1, out_channels[0], (nChannel, math.ceil(spanT[0])), stride=(1, math.ceil(spanT[0] / 2)))
        self.conv2d_m1r5 = Conv2d(1, out_channels[0], (nChannel, math.ceil(spanT[0])), stride=(1, math.ceil(spanT[0] / 2)))
        self.conv2d_m1r6 = Conv2d(1, out_channels[0], (nChannel, math.ceil(spanT[0])), stride=(1, math.ceil(spanT[0] / 2)))

        self.conv2d_m2r0 = Conv2d(out_channels[0], out_channels[1], (1, math.ceil(spanT[1])), stride=(1, math.ceil(spanT[1] / 2)))
        self.conv2d_m2r1 = Conv2d(out_channels[0], out_channels[1], (1, math.ceil(spanT[1])), stride=(1, math.ceil(spanT[1] / 2)))
        self.conv2d_m2r2 = Conv2d(out_channels[0], out_channels[1], (1, math.ceil(spanT[1])), stride=(1, math.ceil(spanT[1] / 2)))
        self.conv2d_m2r3 = Conv2d(out_channels[0], out_channels[1], (1, math.ceil(spanT[1])), stride=(1, math.ceil(spanT[1] / 2)))
        self.conv2d_m2r4 = Conv2d(out_channels[0], out_channels[1], (1, math.ceil(spanT[1])), stride=(1, math.ceil(spanT[1] / 2)))
        self.conv2d_m2r5 = Conv2d(out_channels[0], out_channels[1], (1, math.ceil(spanT[1])), stride=(1, math.ceil(spanT[1] / 2)))
        self.conv2d_m2r6 = Conv2d(out_channels[0], out_channels[1], (1, math.ceil(spanT[1])), stride=(1, math.ceil(spanT[1] / 2)))

        self.dropout = nn.Dropout(p=dropout)

        self.out_channels = out_channels[1]
        self.d_cnn2d = d_cnn2d

    def forward(self, x):

        y0 = self.conv2d_m1r0(x[:, 0, :, :].unsqueeze(1))
        y1 = self.conv2d_m1r1(x[:, 1, :, :].unsqueeze(1))
        y2 = self.conv2d_m1r2(x[:, 2, :, :].unsqueeze(1))
        y3 = self.conv2d_m1r3(x[:, 3, :, :].unsqueeze(1))
        y4 = self.conv2d_m1r4(x[:, 4, :, :].unsqueeze(1))
        y5 = self.conv2d_m1r5(x[:, 5, :, :].unsqueeze(1))
        y6 = self.conv2d_m1r6(x[:, 6, :, :].unsqueeze(1))

        y0 = nn.functional.relu(y0)
        y1 = nn.functional.relu(y1)
        y2 = nn.functional.relu(y2)
        y3 = nn.functional.relu(y3)
        y4 = nn.functional.relu(y4)
        y5 = nn.functional.relu(y5)
        y6 = nn.functional.relu(y6)

        y0 = self.conv2d_m2r0(y0)
        y1 = self.conv2d_m2r1(y1)
        y2 = self.conv2d_m2r2(y2)
        y3 = self.conv2d_m2r3(y3)
        y4 = self.conv2d_m2r4(y4)
        y5 = self.conv2d_m2r5(y5)
        y6 = self.conv2d_m2r6(y6)

        y0 = self.dropout(y0)
        y1 = self.dropout(y1)
        y2 = self.dropout(y2)
        y3 = self.dropout(y3)
        y4 = self.dropout(y4)
        y5 = self.dropout(y5)
        y6 = self.dropout(y6)

        d_bz = y0.size(0)
        d_out_channels = y0.size(1)
        d_nChannel = y0.size(2)
        d_nT = y0.size(3)

        if not d_out_channels == self.out_channels:
            print('in core.CNN.py:')
            print('tensor output, num_channel is ' + str(d_out_channels))
            print('tensor output, num_kernel is ' + str(self.out_channels))
            print('not equal!')
            sys.exit(0)

        d_out_cnn2d = d_out_channels * d_nChannel * d_nT
        if not d_out_cnn2d == self.d_cnn2d:
            print('in core.CNN.py:')
            print('d_cnn2d should be ' + str(d_out_cnn2d))
            print('d_cnn2d should not be ' + str(self.d_cnn2d))
            sys.exit(0)

        y0 = y0.view(d_bz, 1, d_out_cnn2d)
        y1 = y1.view(d_bz, 1, d_out_cnn2d)
        y2 = y2.view(d_bz, 1, d_out_cnn2d)
        y3 = y3.view(d_bz, 1, d_out_cnn2d)
        y4 = y4.view(d_bz, 1, d_out_cnn2d)
        y5 = y5.view(d_bz, 1, d_out_cnn2d)
        y6 = y6.view(d_bz, 1, d_out_cnn2d)

        y = torch.cat([y0, y1, y2, y3, y4, y5, y6], dim=1)
        return y


class ConvD3E7(nn.Module):
    def __init__(self, out_channels, nCloud, spanD, d_cnn3d, dropout):
        super().__init__()

        self.conv3d_m1r0 = Conv3d(1, out_channels[0], (spanD[0][0], spanD[0][1], nCloud), stride=(spanD[0][0] // 2, spanD[0][1] // 2, 1))
        self.conv3d_m1r1 = Conv3d(1, out_channels[0], (spanD[0][0], spanD[0][1], nCloud), stride=(spanD[0][0] // 2, spanD[0][1] // 2, 1))
        self.conv3d_m1r2 = Conv3d(1, out_channels[0], (spanD[0][0], spanD[0][1], nCloud), stride=(spanD[0][0] // 2, spanD[0][1] // 2, 1))
        self.conv3d_m1r3 = Conv3d(1, out_channels[0], (spanD[0][0], spanD[0][1], nCloud), stride=(spanD[0][0] // 2, spanD[0][1] // 2, 1))
        self.conv3d_m1r4 = Conv3d(1, out_channels[0], (spanD[0][0], spanD[0][1], nCloud), stride=(spanD[0][0] // 2, spanD[0][1] // 2, 1))
        self.conv3d_m1r5 = Conv3d(1, out_channels[0], (spanD[0][0], spanD[0][1], nCloud), stride=(spanD[0][0] // 2, spanD[0][1] // 2, 1))
        self.conv3d_m1r6 = Conv3d(1, out_channels[0], (spanD[0][0], spanD[0][1], nCloud), stride=(spanD[0][0] // 2, spanD[0][1] // 2, 1))

        self.conv3d_m2r0 = Conv3d(out_channels[0], out_channels[1], (spanD[1][0], spanD[1][1], 1), stride=(spanD[1][0] // 2, spanD[1][1] // 2, 1))
        self.conv3d_m2r1 = Conv3d(out_channels[0], out_channels[1], (spanD[1][0], spanD[1][1], 1), stride=(spanD[1][0] // 2, spanD[1][1] // 2, 1))
        self.conv3d_m2r2 = Conv3d(out_channels[0], out_channels[1], (spanD[1][0], spanD[1][1], 1), stride=(spanD[1][0] // 2, spanD[1][1] // 2, 1))
        self.conv3d_m2r3 = Conv3d(out_channels[0], out_channels[1], (spanD[1][0], spanD[1][1], 1), stride=(spanD[1][0] // 2, spanD[1][1] // 2, 1))
        self.conv3d_m2r4 = Conv3d(out_channels[0], out_channels[1], (spanD[1][0], spanD[1][1], 1), stride=(spanD[1][0] // 2, spanD[1][1] // 2, 1))
        self.conv3d_m2r5 = Conv3d(out_channels[0], out_channels[1], (spanD[1][0], spanD[1][1], 1), stride=(spanD[1][0] // 2, spanD[1][1] // 2, 1))
        self.conv3d_m2r6 = Conv3d(out_channels[0], out_channels[1], (spanD[1][0], spanD[1][1], 1), stride=(spanD[1][0] // 2, spanD[1][1] // 2, 1))

        self.dropout = nn.Dropout(p=dropout)

        self.out_channels = out_channels[1]
        self.d_cnn3d = d_cnn3d

    def forward(self, x):
        y0 = self.conv3d_m1r0(x[:, 0, :, :, :].unsqueeze(1))
        y1 = self.conv3d_m1r1(x[:, 1, :, :, :].unsqueeze(1))
        y2 = self.conv3d_m1r2(x[:, 2, :, :, :].unsqueeze(1))
        y3 = self.conv3d_m1r3(x[:, 3, :, :, :].unsqueeze(1))
        y4 = self.conv3d_m1r4(x[:, 4, :, :, :].unsqueeze(1))
        y5 = self.conv3d_m1r5(x[:, 5, :, :, :].unsqueeze(1))
        y6 = self.conv3d_m1r6(x[:, 6, :, :, :].unsqueeze(1))

        y0 = nn.functional.relu(y0)
        y1 = nn.functional.relu(y1)
        y2 = nn.functional.relu(y2)
        y3 = nn.functional.relu(y3)
        y4 = nn.functional.relu(y4)
        y5 = nn.functional.relu(y5)
        y6 = nn.functional.relu(y6)

        y0 = self.conv3d_m2r0(y0)
        y1 = self.conv3d_m2r1(y1)
        y2 = self.conv3d_m2r2(y2)
        y3 = self.conv3d_m2r3(y3)
        y4 = self.conv3d_m2r4(y4)
        y5 = self.conv3d_m2r5(y5)
        y6 = self.conv3d_m2r6(y6)

        y0 = self.dropout(y0)
        y1 = self.dropout(y1)
        y2 = self.dropout(y2)
        y3 = self.dropout(y3)
        y4 = self.dropout(y4)
        y5 = self.dropout(y5)
        y6 = self.dropout(y6)

        d_bz = y0.size(0)
        d_out_channels = y0.size(1)
        d_x = y0.size(2)
        d_y = y0.size(3)
        d_z = y0.size(4)

        if not d_out_channels == self.out_channels:
            print('in core.CNN.py:')
            print('tensor output, num_channel is ' + str(d_out_channels))
            print('tensor output, num_kernel is ' + str(self.out_channels))
            print('not equal!')
            sys.exit(0)

        d_out_cnn3d = d_out_channels * d_x * d_y * d_z
        if not d_out_cnn3d == self.d_cnn3d:
            print('in core.CNN.py:')
            print('d_cnn3d should be ' + str(d_out_cnn3d))
            print('d_cnn3d should not be ' + str(self.d_cnn3d))
            sys.exit(0)

        y0 = y0.view(d_bz, 1, d_out_cnn3d)
        y1 = y1.view(d_bz, 1, d_out_cnn3d)
        y2 = y2.view(d_bz, 1, d_out_cnn3d)
        y3 = y3.view(d_bz, 1, d_out_cnn3d)
        y4 = y4.view(d_bz, 1, d_out_cnn3d)
        y5 = y5.view(d_bz, 1, d_out_cnn3d)
        y6 = y6.view(d_bz, 1, d_out_cnn3d)

        y = torch.cat([y0, y1, y2, y3, y4, y5, y6], dim=1)
        return y
