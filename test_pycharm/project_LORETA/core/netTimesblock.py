import torch
import torch.nn as nn

import sys
sys.path.append(r'D:\test_pycharm\project_eegICAcloud\core\TimesNet')
from core.TimesNet.models.TimesNet import TimesBlock


class TimesBlockE7(nn.Module):
    def __init__(self, config, dropout):
        super().__init__()
        
        # self.timesBlock_r0 = TimesBlock(config)
        # self.timesBlock_r1 = TimesBlock(config)
        # self.timesBlock_r2 = TimesBlock(config)
        # self.timesBlock_r3 = TimesBlock(config)
        # self.timesBlock_r4 = TimesBlock(config)
        # self.timesBlock_r5 = TimesBlock(config)
        # self.timesBlock_r6 = TimesBlock(config)
        self.timesBlock_all = TimesBlock(config)

        self.dropout = nn.Dropout(p=dropout)

        self.seq_len = config.seq_len

    def forward(self, x):

        x = x.transpose(2, 3)

        # y0 = self.timesBlock_r0(x[:, 0, :, :])
        # y1 = self.timesBlock_r1(x[:, 1, :, :])
        # y2 = self.timesBlock_r2(x[:, 2, :, :])
        # y3 = self.timesBlock_r3(x[:, 3, :, :])
        # y4 = self.timesBlock_r4(x[:, 4, :, :])
        # y5 = self.timesBlock_r5(x[:, 5, :, :])
        # y6 = self.timesBlock_r6(x[:, 6, :, :])
        y0 = self.timesBlock_all(x[:, 0, :, :])
        y1 = self.timesBlock_all(x[:, 1, :, :])
        y2 = self.timesBlock_all(x[:, 2, :, :])
        y3 = self.timesBlock_all(x[:, 3, :, :])
        y4 = self.timesBlock_all(x[:, 4, :, :])
        y5 = self.timesBlock_all(x[:, 5, :, :])
        y6 = self.timesBlock_all(x[:, 6, :, :])

        y0 = nn.functional.relu(y0)
        y1 = nn.functional.relu(y1)
        y2 = nn.functional.relu(y2)
        y3 = nn.functional.relu(y3)
        y4 = nn.functional.relu(y4)
        y5 = nn.functional.relu(y5)
        y6 = nn.functional.relu(y6)

        y0 = self.dropout(y0.unsqueeze(1))
        y1 = self.dropout(y1.unsqueeze(1))
        y2 = self.dropout(y2.unsqueeze(1))
        y3 = self.dropout(y3.unsqueeze(1))
        y4 = self.dropout(y4.unsqueeze(1))
        y5 = self.dropout(y5.unsqueeze(1))
        y6 = self.dropout(y6.unsqueeze(1))

        y = torch.cat([y0, y1, y2, y3, y4, y5, y6], dim=1)
        y = y.transpose(2, 3)

        d_bz = y.size(0)
        d_embedding = y.size(1)
        d_nChannel = y.size(2)
        d_nT = y.size(3)

        if not d_nT == self.seq_len:
            print('in core.TBN.py:')
            print('seq_len should be ' + str(d_nT))
            print('seq_len should not be ' + str(self.d_cnnBlock))
            sys.exit(0)

        return y