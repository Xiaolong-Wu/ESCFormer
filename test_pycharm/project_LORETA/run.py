import os
import math
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from core.buildModel import EEGNet, rEEGNet, tEEGNet, EEGTimesBlock, rEEGTimesBlock, tEEGTimesBlock, EEGNetC, rEEGNetC, tEEGNetC, EEGTimesBlockC, rEEGTimesBlockC, tEEGTimesBlockC
from core.toolkit import strCode, loadData

nCode = 3
path_project = 'D:\\test_pycharm\\project_LORETA'

path_LORETAcloud = [
    'data_2024_Competition\\data_003_ICA(-100,700)\\0.8s(0)\\LORETAcloud',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\LORETAcloud',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\LORETAcloud',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\LORETAcloud'
]

path_modelCloud = [
    'data_2024_Competition\\data_003_ICA(-100,700)\\0.8s(0)\\LORETAcloud',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\LORETAcloud_1_sin1200',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\LORETAcloud_2_dogbark',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\LORETAcloud_3_xiaoming'
]

groupAll = [
    [[1],[2],[3],[4]],
    [[13],[22]],
    [[16],[25]],
    [[18],[27]]
]

nChannelAll = [
    22,
    59,
    59,
    59
]

fsAll = [
    250,
    1000,
    1000,
    1000
]

nCloud = 10

# 0, data_2024_Competition
# 1, data_2024_DoC sin1200
# 2, data_2024_DoC dogbark
# 3, data_2024_DoC xiaoming
dataNo = 2

# 0, EEGNet
# 1, CNN-LSTM
# 2, CNN-Transformer
# 3, TimesNet
# 4, TimesBlock-LSTM
# 5, TimesBlock-Transformer
modelNo = 2

modelEpoch = 4096
modelBatchSize = 32

is_GPU = torch.cuda.is_available()
# is_GPU = False

path1_model = os.path.join(path_project, 'dataset_model')
path2_model = os.path.join(path1_model, path_modelCloud[dataNo])

if not os.path.exists(path2_model):
    os.makedirs(path2_model)

path3_model = os.path.join(path2_model, 'modelNo(' + str(modelNo) + ')epoch(' + str(modelEpoch) + ')batch(' + str(modelBatchSize) + ').pth')
path3_model_loss = os.path.join(path2_model, 'modelNo(' + str(modelNo) + ')epoch(' + str(modelEpoch) + ')batch(' + str(modelBatchSize) + ')_loss.txt')

path3_model_target = os.path.join(path2_model, 'modelNo(' + str(modelNo) + ')epoch(' + str(modelEpoch) + ')batch(' + str(modelBatchSize) + ')_target.txt')
path3_model_test = os.path.join(path2_model, 'modelNo(' + str(modelNo) + ')epoch(' + str(modelEpoch) + ')batch(' + str(modelBatchSize) + ')_test.txt')

fs = fsAll[dataNo]
nChannel = nChannelAll[dataNo]
nClass = len(groupAll[dataNo])
isTrain = 0
if not os.path.isfile(path3_model):
    isTrain = 1

    config = json.load(open('config.json', 'r'))

    if modelNo == 0:
        modelWu2025 = EEGNetC(config, fs, nChannel, nCloud, nClass, dataNo)
    elif modelNo == 1:
        modelWu2025 = rEEGNetC(config, fs, nChannel, nCloud, nClass, dataNo)
    elif modelNo == 2:
        modelWu2025 = tEEGNetC(config, fs, nChannel, nCloud, nClass, dataNo)
    elif modelNo == 3:
        modelWu2025 = EEGTimesBlockC(config, fs, nChannel, nCloud, nClass, dataNo)
    elif modelNo == 4:
        modelWu2025 = rEEGTimesBlockC(config, fs, nChannel, nCloud, nClass, dataNo)
    else:
        modelWu2025 = tEEGTimesBlockC(config, fs, nChannel, nCloud, nClass, dataNo)

    modelWu2025.train()
    loss_fun = nn.CrossEntropyLoss()

    if (modelNo == 3) or (modelNo == 4) or (modelNo == 5):
        tempLr = 1e-6 * 1000
    else:
        tempLr = 1e-6 * 1
    optimizer = torch.optim.Adam(modelWu2025.parameters(), lr=tempLr)
else:
    modelEpoch = 1

    modelWu2025 = torch.load(path3_model, weights_only=False, map_location='cpu')
    modelWu2025.eval()

    is_GPU = False

if is_GPU:
    gpu_device = torch.device("cuda")
    modelWu2025.to(gpu_device)

    print('device: GPU')
else:
    print('device: CPU')

path1_train = os.path.join(path_project, 'dataset_train')
path1_test = os.path.join(path_project, 'dataset_test')

path2_train_LORETAcloud = os.path.join(path1_train, path_LORETAcloud[dataNo])
path2_test_LORETAcloud = os.path.join(path1_test, path_LORETAcloud[dataNo])

if not os.path.isfile(path3_model_test):
    for rEpoch in range(modelEpoch):
        for rClass in range(nClass):
            for rGroup in range(len(groupAll[dataNo][rClass])):
                path_group = 'group_' + strCode(groupAll[dataNo][rClass][rGroup],nCode)

                path3_train_LORETAcloud = os.path.join(path2_train_LORETAcloud, path_group)
                path3_test_LORETAcloud = os.path.join(path2_test_LORETAcloud, path_group)

                fileAll = os.listdir(path3_train_LORETAcloud)
                fileMat = []
                for rSubject in fileAll:
                    if rSubject.endswith('.mat'):
                        fileMat.append(rSubject)

                rSubject_No = 0
                for rSubject in fileMat:
                    path4_train_ICAcloud = os.path.join(path3_train_LORETAcloud, rSubject)
                    path4_test_ICAcloud = os.path.join(path3_test_LORETAcloud, rSubject)

                    if isTrain == 0:
                        EEG, cloud_LORETA = loadData(path4_test_ICAcloud)
                    else:
                        EEG, cloud_LORETA = loadData(path4_train_ICAcloud)

                    nSample = EEG.shape[0]
                    temp_label = np.zeros([nSample,nClass])
                    for rSample in range(nSample):
                        temp_label[rSample,rClass] = 1

                    label = torch.tensor(temp_label)

                    if isTrain == 0:

                        tempTrain_EEG = EEG
                        tempTrain_cloud_LORETA = cloud_LORETA

                        if is_GPU:
                            tempTrain_EEG = tempTrain_EEG.to(gpu_device)
                            tempTrain_cloud_LORETA = tempTrain_cloud_LORETA.to(gpu_device)

                        testOutput = modelWu2025(tempTrain_EEG, tempTrain_cloud_LORETA)

                        if is_GPU:
                            testOutput = testOutput.to('cpu')

                        if rEpoch == 0 and rClass == 0 and rGroup == 0 and rSubject_No == 0:
                            label_all = label.numpy()
                            testOutput_all = testOutput.detach().numpy()
                        else:
                            testOutput_all = np.vstack((testOutput_all, testOutput.detach().numpy()))
                            label_all = np.vstack((label_all, label.numpy()))
                    else:
                        nBatch = max([1,math.floor(nSample/modelBatchSize)])
                        for rBatch in range(nBatch):

                            first = rBatch*modelBatchSize
                            if rBatch == nBatch-1:
                                last = nSample
                            else:
                                last = first + modelBatchSize

                            tempTrain_EEG = EEG[first:last]
                            tempTrain_cloud_LORETA = cloud_LORETA[first:last]
                            tempTrain_label = label[first:last]

                            if is_GPU:
                                tempTrain_EEG = tempTrain_EEG.to(gpu_device)
                                tempTrain_cloud_LORETA = tempTrain_cloud_LORETA.to(gpu_device)
                                tempTrain_label = tempTrain_label.to(gpu_device)

                            trainOutput = modelWu2025(tempTrain_EEG, tempTrain_cloud_LORETA)

                            optimizer.zero_grad()
                            loss = loss_fun(tempTrain_label, trainOutput)
                            loss.backward()
                            optimizer.step()

                            if rEpoch == 0 and rClass == 0 and rGroup == 0 and rSubject_No == 0 and rBatch == 0:
                                loss_all = np.array(loss.item())
                            else:
                                loss_all = np.hstack((loss_all, np.array(loss.item())))

                            print('Loss: {}'.format(loss.item()))

                    rSubject_No = rSubject_No+1

    if isTrain == 0:
        np.savetxt(path3_model_target, label_all, delimiter=',')
        np.savetxt(path3_model_test, testOutput_all, delimiter=',')
    else:
        np.savetxt(path3_model_loss, loss_all, delimiter=',')
        modelWu2025.eval()

        if is_GPU:
            modelWu2025 = modelWu2025.to('cpu')
        torch.save(modelWu2025, path3_model)

loss_all = np.loadtxt(path3_model_loss)
# plt.plot(loss_all)