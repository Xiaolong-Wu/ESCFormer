import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Linear, LSTM
from core.netConvolution import ConvD2E7, ConvD3E7
from core.netEncoder import Encoder
from core.netTimesblock import TimesBlockE7

class configSet:
    def __init__(self, config, fs, nChannel, dataNo):

        self.seq_len = math.ceil(fs*np.array(config['seq_len'][dataNo])/1000)
        self.pred_len = 0
        self.top_k = config['top_k'][dataNo]

        self.d_model = nChannel
        self.d_ff = config['d_ff'][dataNo]
        self.num_kernels = config['num_kernels'][dataNo]

class EEGNet(nn.Module):
    def __init__(self, config, fs, nChannel, nClass, dataNo):
        super().__init__()

        nEmbedding = config['nEmbedding']
        out_channels =  config['out_channels_2d']

        spanT = np.array(config['spanT'])*fs
        spanT[1] = 2*spanT[1]/spanT[0]

        d_cnn2d = config['d_cnn2d'][dataNo]
        d_dense= config['d_dense']
        dropout = config['dropout']

        if nEmbedding == 7:
            self.CNN = ConvD2E7(out_channels, nChannel, spanT, d_cnn2d, dropout)
        self.dense = Linear(nEmbedding * d_cnn2d, nEmbedding * d_dense)
        self.linear = Linear(nEmbedding * d_dense, nClass)

    def forward(self, x, x_cloud):

        y = self.CNN(x)

        d_bz = y.size(0)
        d_nEmbedding = y.size(1)
        d_cnn2d = y.size(2)

        y = y.view(d_bz, d_nEmbedding * d_cnn2d)
        y = self.dense(y)

        y = self.linear(y)
        y = nn.functional.softmax(y, dim=1)
        return y

class rEEGNet(nn.Module):
    def __init__(self, config, fs, nChannel, nClass, dataNo):
        super().__init__()

        nEmbedding = config['nEmbedding']
        out_channels =  config['out_channels_2d']

        spanT = np.array(config['spanT'])*fs
        spanT[1] = 2*spanT[1]/spanT[0]

        d_cnn2d = config['d_cnn2d'][dataNo]
        d_dense= config['d_dense']
        dropout = config['dropout']

        if nEmbedding == 7:
            self.CNN = ConvD2E7(out_channels, nChannel, spanT, d_cnn2d, dropout)
        self.lstm = LSTM(d_cnn2d, d_dense, 1)
        self.linear = Linear(nEmbedding * d_dense, nClass)

    def forward(self, x, x_cloud):

        y = self.CNN(x)

        y, (yh, yc) = self.lstm(y)

        d_bz = y.size(0)
        d_nEmbedding = y.size(1)
        d_dense = y.size(2)

        y = y.view(d_bz, d_nEmbedding * d_dense)
        y = self.linear(y)
        y = nn.functional.softmax(y, dim=1)
        return y

class tEEGNet(nn.Module):
    def __init__(self, config, fs, nChannel, nClass, dataNo):
        super().__init__()

        nEmbedding = config['nEmbedding']
        out_channels =  config['out_channels_2d']

        spanT = np.array(config['spanT'])*fs
        spanT[1] = 2*spanT[1]/spanT[0]

        d_cnn2d = config['d_cnn2d'][dataNo]
        n_layer = config['n_layer']
        n_head = config['n_head']

        dropout = config['dropout']

        self.nEmbedding = nEmbedding
        self.d_cnn2d = d_cnn2d
        d_k = math.ceil(d_cnn2d/2)
        d_v = d_cnn2d
        d_inner = d_k

        if nEmbedding == 7:
            self.CNN = ConvD2E7(out_channels, nChannel, spanT, d_cnn2d, dropout)
        self.encoder = Encoder(n_layer, d_cnn2d, n_head, d_k, d_v, d_inner, dropout)
        self.linear = Linear(nEmbedding * d_cnn2d, nClass)

    def forward(self, x, x_cloud):

        y = self.CNN(x)

        y, y_list = self.encoder(y, self.nEmbedding * self.d_cnn2d, None)

        y = self.linear(y)
        y = nn.functional.softmax(y, dim=1)
        return y

class EEGTimesBlock(nn.Module):
    def __init__(self, config, fs, nChannel, nClass, dataNo):
        super().__init__()

        configs = configSet(config, fs, nChannel, dataNo)

        nEmbedding = config['nEmbedding']
        out_channels =  config['out_channels_2d']

        spanT = np.array(config['spanT'])*fs
        spanT[1] = 2*spanT[1]/spanT[0]

        d_cnn2d = config['d_cnn2d'][dataNo]
        d_dense= config['d_dense']
        dropout = config['dropout']

        if nEmbedding == 7:
            self.TBN = TimesBlockE7(configs, dropout)
            self.CNN = ConvD2E7(out_channels, nChannel, spanT, d_cnn2d, dropout)
        self.dense = Linear(nEmbedding * d_cnn2d, nEmbedding * d_dense)
        self.linear = Linear(nEmbedding * d_dense, nClass)

    def forward(self, x, x_cloud):

        y = self.TBN(x)
        y = self.CNN(y)

        d_bz = y.size(0)
        d_nEmbedding = y.size(1)
        d_cnn2d = y.size(2)

        y = y.view(d_bz, d_nEmbedding * d_cnn2d)
        y = self.dense(y)

        y = self.linear(y)
        y = nn.functional.softmax(y, dim=1)
        return y

class rEEGTimesBlock(nn.Module):
    def __init__(self, config, fs, nChannel, nClass, dataNo):
        super().__init__()

        configs = configSet(config, fs, nChannel, dataNo)

        nEmbedding = config['nEmbedding']
        out_channels =  config['out_channels_2d']

        spanT = np.array(config['spanT'])*fs
        spanT[1] = 2*spanT[1]/spanT[0]

        d_cnn2d = config['d_cnn2d'][dataNo]
        d_dense= config['d_dense']
        dropout = config['dropout']

        if nEmbedding == 7:
            self.TBN = TimesBlockE7(configs, dropout)
            self.CNN = ConvD2E7(out_channels, nChannel, spanT, d_cnn2d, dropout)
        self.lstm = LSTM(d_cnn2d, d_dense, 1)
        self.linear = Linear(nEmbedding * d_dense, nClass)

    def forward(self, x, x_cloud):

        y = self.TBN(x)
        y = self.CNN(y)

        y, (yh, yc) = self.lstm(y)

        d_bz = y.size(0)
        d_nEmbedding = y.size(1)
        d_dense = y.size(2)

        y = y.view(d_bz, d_nEmbedding * d_dense)
        y = self.linear(y)
        y = nn.functional.softmax(y, dim=1)
        return y

class tEEGTimesBlock(nn.Module):
    def __init__(self, config, fs, nChannel, nClass, dataNo):
        super().__init__()

        configs = configSet(config, fs, nChannel, dataNo)

        nEmbedding = config['nEmbedding']
        out_channels =  config['out_channels_2d']

        spanT = np.array(config['spanT'])*fs
        spanT[1] = 2*spanT[1]/spanT[0]

        d_cnn2d = config['d_cnn2d'][dataNo]
        n_layer = config['n_layer']
        n_head = config['n_head']

        dropout = config['dropout']

        self.nEmbedding = nEmbedding
        self.d_cnn2d = d_cnn2d
        d_k = math.ceil(d_cnn2d/2)
        d_v = d_cnn2d
        d_inner = d_k

        if nEmbedding == 7:
            self.TBN = TimesBlockE7(configs, dropout)
            self.CNN = ConvD2E7(out_channels, nChannel, spanT, d_cnn2d, dropout)
        self.encoder = Encoder(n_layer, d_cnn2d, n_head, d_k, d_v, d_inner, dropout)
        self.linear = Linear(nEmbedding * d_cnn2d, nClass)

    def forward(self, x, x_cloud):

        y = self.TBN(x)
        y = self.CNN(y)

        y, y_list = self.encoder(y, self.nEmbedding * self.d_cnn2d, None)

        y = self.linear(y)
        y = nn.functional.softmax(y, dim=1)
        return y

class EEGNetC(nn.Module):
    def __init__(self, config, fs, nChannel, nCloud, nClass, dataNo):
        super().__init__()

        nEmbedding = config['nEmbedding']
        out_channels =  config['out_channels_2d']
        out_channels_cloud = config['out_channels_3d']

        spanT = np.array(config['spanT'])*fs
        spanT[1] = 2*spanT[1]/spanT[0]
        spanD = config['spanD']

        d_cnn2d = config['d_cnn2d'][dataNo]
        d_cnn3d = config['d_cnn3d'][dataNo]
        d_dense= config['d_dense']
        dropout = config['dropout']

        if nEmbedding == 7:
            self.CNN = ConvD2E7(out_channels, nChannel, spanT, d_cnn2d, dropout)
            self.CNNcloud = ConvD3E7(out_channels_cloud, nCloud, spanD, d_cnn3d, dropout)

        self.dense = Linear(nEmbedding * (d_cnn2d + d_cnn3d), nEmbedding * d_dense)
        self.linear = Linear(nEmbedding * d_dense, nClass)

    def forward(self, x, x_cloud):

        y = self.CNN(x)
        y_cloud = self.CNNcloud(x_cloud)

        y = torch.cat([y, y_cloud], dim=2)

        d_bz = y.size(0)
        d_nEmbedding = y.size(1)
        d_cnn2d_cloud = y.size(2)

        y = y.view(d_bz, d_nEmbedding * d_cnn2d_cloud)
        y = self.dense(y)

        y = self.linear(y)
        y = nn.functional.softmax(y, dim=1)
        return y

class rEEGNetC(nn.Module):
    def __init__(self, config, fs, nChannel, nCloud, nClass, dataNo):
        super().__init__()

        nEmbedding = config['nEmbedding']
        out_channels =  config['out_channels_2d']
        out_channels_cloud = config['out_channels_3d']

        spanT = np.array(config['spanT'])*fs
        spanT[1] = 2*spanT[1]/spanT[0]
        spanD = config['spanD']

        d_cnn2d = config['d_cnn2d'][dataNo]
        d_cnn3d = config['d_cnn3d'][dataNo]
        d_dense= config['d_dense']
        dropout = config['dropout']

        if nEmbedding == 7:
            self.CNN = ConvD2E7(out_channels, nChannel, spanT, d_cnn2d, dropout)
            self.CNNcloud = ConvD3E7(out_channels_cloud, nCloud, spanD, d_cnn3d, dropout)

        self.lstm = LSTM(d_cnn2d + d_cnn3d, d_dense, 1)
        self.linear = Linear(nEmbedding * d_dense, nClass)

    def forward(self, x, x_cloud):

        y = self.CNN(x)
        y_cloud = self.CNNcloud(x_cloud)

        y = torch.cat([y, y_cloud], dim=2)

        y, (yh, yc) = self.lstm(y)

        d_bz = y.size(0)
        d_nEmbedding = y.size(1)
        d_dense = y.size(2)

        y = y.view(d_bz, d_nEmbedding * d_dense)
        y = self.linear(y)
        y = nn.functional.softmax(y, dim=1)
        return y

class tEEGNetC(nn.Module):
    def __init__(self, config, fs, nChannel, nCloud, nClass, dataNo):
        super().__init__()

        nEmbedding = config['nEmbedding']
        out_channels =  config['out_channels_2d']
        out_channels_cloud = config['out_channels_3d']

        spanT = np.array(config['spanT'])*fs
        spanT[1] = 2*spanT[1]/spanT[0]
        spanD = config['spanD']

        d_cnn2d = config['d_cnn2d'][dataNo]
        d_cnn3d = config['d_cnn3d'][dataNo]

        n_layer = config['n_layer']
        n_head = config['n_head']

        dropout = config['dropout']

        self.nEmbedding = nEmbedding
        self.d_cnn = d_cnn2d + d_cnn3d
        d_k = math.ceil(self.d_cnn/2)
        d_v = self.d_cnn
        d_inner = d_k

        if nEmbedding == 7:
            self.CNN = ConvD2E7(out_channels, nChannel, spanT, d_cnn2d, dropout)
            self.CNNcloud = ConvD3E7(out_channels_cloud, nCloud, spanD, d_cnn3d, dropout)

        self.encoder = Encoder(n_layer, self.d_cnn, n_head, d_k, d_v, d_inner, dropout)
        self.linear = Linear(nEmbedding * self.d_cnn, nClass)

    def forward(self, x, x_cloud):

        y = self.CNN(x)
        y_cloud = self.CNNcloud(x_cloud)

        y = torch.cat([y, y_cloud], dim=2)

        y, y_list = self.encoder(y, self.nEmbedding * self.d_cnn, None)

        y = self.linear(y)
        y = nn.functional.softmax(y, dim=1)
        return y

class EEGTimesBlockC(nn.Module):
    def __init__(self, config, fs, nChannel, nCloud, nClass, dataNo):
        super().__init__()

        configs = configSet(config, fs, nChannel, dataNo)

        nEmbedding = config['nEmbedding']
        out_channels = config['out_channels_2d']
        out_channels_cloud = config['out_channels_3d']

        spanT = np.array(config['spanT'])*fs
        spanT[1] = 2*spanT[1]/spanT[0]
        spanD = config['spanD']

        d_cnn2d = config['d_cnn2d'][dataNo]
        d_cnn3d = config['d_cnn3d'][dataNo]

        d_dense= config['d_dense']
        dropout = config['dropout']

        if nEmbedding == 7:
            self.TBN = TimesBlockE7(configs, dropout)
            self.CNN = ConvD2E7(out_channels, nChannel, spanT, d_cnn2d, dropout)
            self.CNNcloud = ConvD3E7(out_channels_cloud, nCloud, spanD, d_cnn3d, dropout)

        self.dense = Linear(nEmbedding * (d_cnn2d + d_cnn3d), nEmbedding * d_dense)
        self.linear = Linear(nEmbedding * d_dense, nClass)

    def forward(self, x, x_cloud):

        y = self.TBN(x)
        y = self.CNN(y)

        y_cloud = self.CNNcloud(x_cloud)

        y = torch.cat([y, y_cloud], dim=2)

        d_bz = y.size(0)
        d_nEmbedding = y.size(1)
        d_cloud = y.size(2)

        y = y.view(d_bz, d_nEmbedding * d_cloud)
        y = self.dense(y)

        y = self.linear(y)
        y = nn.functional.softmax(y, dim=1)
        return y

class rEEGTimesBlockC(nn.Module):
    def __init__(self, config, fs, nChannel, nCloud, nClass, dataNo):
        super().__init__()

        configs = configSet(config, fs, nChannel, dataNo)

        nEmbedding = config['nEmbedding']
        out_channels = config['out_channels_2d']
        out_channels_cloud = config['out_channels_3d']

        spanT = np.array(config['spanT'])*fs
        spanT[1] = 2*spanT[1]/spanT[0]
        spanD = config['spanD']

        d_cnn2d = config['d_cnn2d'][dataNo]
        d_cnn3d = config['d_cnn3d'][dataNo]

        d_dense= config['d_dense']
        dropout = config['dropout']

        if nEmbedding == 7:
            self.TBN = TimesBlockE7(configs, dropout)
            self.CNN = ConvD2E7(out_channels, nChannel, spanT, d_cnn2d, dropout)
            self.CNNcloud = ConvD3E7(out_channels_cloud, nCloud, spanD, d_cnn3d, dropout)

        self.lstm = LSTM(d_cnn2d + d_cnn3d, d_dense, 1)
        self.linear = Linear(nEmbedding * d_dense, nClass)

    def forward(self, x, x_cloud):

        y = self.TBN(x)
        y = self.CNN(y)

        y_cloud = self.CNNcloud(x_cloud)

        y = torch.cat([y, y_cloud], dim=2)

        y, (yh, yc) = self.lstm(y)

        d_bz = y.size(0)
        d_nEmbedding = y.size(1)
        d_dense = y.size(2)

        y = y.view(d_bz, d_nEmbedding * d_dense)
        y = self.linear(y)
        y = nn.functional.softmax(y, dim=1)
        return y

class tEEGTimesBlockC(nn.Module):
    def __init__(self, config, fs, nChannel, nCloud, nClass, dataNo):
        super().__init__()

        configs = configSet(config, fs, nChannel, dataNo)

        nEmbedding = config['nEmbedding']
        out_channels = config['out_channels_2d']
        out_channels_cloud = config['out_channels_3d']

        spanT = np.array(config['spanT'])*fs
        spanT[1] = 2*spanT[1]/spanT[0]
        spanD = config['spanD']

        d_cnn2d = config['d_cnn2d'][dataNo]
        d_cnn3d = config['d_cnn3d'][dataNo]

        n_layer = config['n_layer']
        n_head = config['n_head']

        dropout = config['dropout']

        self.nEmbedding = nEmbedding
        self.d_cnn = d_cnn2d + d_cnn3d
        d_k = math.ceil(self.d_cnn/2)
        d_v = self.d_cnn
        d_inner = d_k

        if nEmbedding == 7:
            self.TBN = TimesBlockE7(configs, dropout)
            self.CNN = ConvD2E7(out_channels, nChannel, spanT, d_cnn2d, dropout)
            self.CNNcloud = ConvD3E7(out_channels_cloud, nCloud, spanD, d_cnn3d, dropout)

        self.encoder = Encoder(n_layer, self.d_cnn, n_head, d_k, d_v, d_inner, dropout)
        self.linear = Linear(nEmbedding * self.d_cnn, nClass)

    def forward(self, x, x_cloud):

        y = self.TBN(x)
        y = self.CNN(y)

        y_cloud = self.CNNcloud(x_cloud)

        y = torch.cat([y, y_cloud], dim=2)

        y, y_list = self.encoder(y, self.nEmbedding * self.d_cnn, None)

        y = self.linear(y)
        y = nn.functional.softmax(y, dim=1)
        return y
