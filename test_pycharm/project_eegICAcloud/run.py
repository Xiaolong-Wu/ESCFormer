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
path_project = 'D:\\test_pycharm\\project_eegICAcloud'

path_ICAspace = [
    'data_2024_3D\\data_003_ICA(-100,700)\\0.8s(0)\\ICAspace_Norm1(1)',
    'data_2024_Competition\\data_003_ICA(-100,700)\\0.8s(0)\\ICAspace_Norm1(1)',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\ICAspace_Norm1(1)',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\ICAspace_Norm1(1)',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\ICAspace_Norm1(1)'
]
path_ICAcloud = [
    'data_2024_3D\\data_003_ICA(-100,700)\\0.8s(0)\\ICAcloud_Norm1(1)',
    'data_2024_Competition\\data_003_ICA(-100,700)\\0.8s(0)\\ICAcloud_Norm1(1)',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\ICAcloud_Norm1(1)',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\ICAcloud_Norm1(1)',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\ICAcloud_Norm1(1)'
]

path_modelSpace = [
    'data_2024_3D\\data_003_ICA(-100,700)\\0.8s(0)\\ICAspace_Norm1(1)',
    'data_2024_Competition\\data_003_ICA(-100,700)\\0.8s(0)\\ICAspace_Norm1(1)',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\ICAspace_Norm1(1)_1_sin1200',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\ICAspace_Norm1(1)_2_dogbark',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\ICAspace_Norm1(1)_3_xiaoming'
]

path_modelCloud = [
    'data_2024_3D\\data_003_ICA(-100,700)\\0.8s(0)\\ICAcloud_Norm1(1)',
    'data_2024_Competition\\data_003_ICA(-100,700)\\0.8s(0)\\ICAcloud_Norm1(1)',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\ICAcloud_Norm1(1)_1_sin1200',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\ICAcloud_Norm1(1)_2_dogbark',
    'data_2024_DoC\\data_003_ICA(-100,700)\\0.8s(0)\\ICAcloud_Norm1(1)_3_xiaoming'
]

groupAll = [
    [[0],[0]],
    [[1],[2],[3],[4]],
    [[13],[22]],
    [[16],[25]],
    [[18],[27]]
]

nChannelAll = [
    15,
    22,
    59,
    59,
    59
]

fsAll = [
    0,
    250,
    1000,
    1000,
    1000
]

nCloud = 10

# 0, data_2024_3D
# 1, data_2024_Competition
# 2, data_2024_DoC sin1200
# 3, data_2024_DoC dogbark
# 4, data_2024_DoC xiaoming
dataNo = 2

# modelNo[0]
# 0, modelSpace
# 1, modelCloud
# modelNo[1]
# 0, EEGNet
# 1, CNN-LSTM
# 2, CNN-Transformer
# 3, TimesNet
# 4, TimesBlock-LSTM
# 5, TimesBlock-Transformer
modelNo = [0,0]

modelEpoch = 4096
modelBatchSize = 32

is_GPU = torch.cuda.is_available()
# is_GPU = False

path1_model = os.path.join(path_project, 'dataset_model')

if modelNo[0] == 0:
    path2_model = os.path.join(path1_model, path_modelSpace[dataNo])
else:
    path2_model = os.path.join(path1_model, path_modelCloud[dataNo])

if not os.path.exists(path2_model):
    os.makedirs(path2_model)

path3_model = os.path.join(path2_model, 'modelNo(' + str(modelNo[1]) + ')epoch(' + str(modelEpoch) + ')batch(' + str(modelBatchSize) + ').pth')
path3_model_loss = os.path.join(path2_model, 'modelNo(' + str(modelNo[1]) + ')epoch(' + str(modelEpoch) + ')batch(' + str(modelBatchSize) + ')_loss.txt')

path3_model_target = os.path.join(path2_model, 'modelNo(' + str(modelNo[1]) + ')epoch(' + str(modelEpoch) + ')batch(' + str(modelBatchSize) + ')_target.txt')
path3_model_test = os.path.join(path2_model, 'modelNo(' + str(modelNo[1]) + ')epoch(' + str(modelEpoch) + ')batch(' + str(modelBatchSize) + ')_test.txt')

fs = fsAll[dataNo]
nChannel = nChannelAll[dataNo]
nClass = len(groupAll[dataNo])
isTrain = 0
if not os.path.isfile(path3_model):
    isTrain = 1

    config = json.load(open('config.json', 'r'))

    if modelNo[0] == 0:
        if modelNo[1] == 0:
            modelWu2025 = EEGNet(config, fs, nChannel, nClass, dataNo)
        elif modelNo[1] == 1:
            modelWu2025 = rEEGNet(config, fs, nChannel, nClass, dataNo)
        elif modelNo[1] == 2:
            modelWu2025 = tEEGNet(config, fs, nChannel, nClass, dataNo)
        elif modelNo[1] == 3:
            modelWu2025 = EEGTimesBlock(config, fs, nChannel, nClass, dataNo)
        elif modelNo[1] == 4:
            modelWu2025 = rEEGTimesBlock(config, fs, nChannel, nClass, dataNo)
        else:
            modelWu2025 = tEEGTimesBlock(config, fs, nChannel, nClass, dataNo)
    else:
        if modelNo[1] == 0:
            modelWu2025 = EEGNetC(config, fs, nChannel, nCloud, nClass, dataNo)
        elif modelNo[1] == 1:
            modelWu2025 = rEEGNetC(config, fs, nChannel, nCloud, nClass, dataNo)
        elif modelNo[1] == 2:
            modelWu2025 = tEEGNetC(config, fs, nChannel, nCloud, nClass, dataNo)
        elif modelNo[1] == 3:
            modelWu2025 = EEGTimesBlockC(config, fs, nChannel, nCloud, nClass, dataNo)
        elif modelNo[1] == 4:
            modelWu2025 = rEEGTimesBlockC(config, fs, nChannel, nCloud, nClass, dataNo)
        else:
            modelWu2025 = tEEGTimesBlockC(config, fs, nChannel, nCloud, nClass, dataNo)

    modelWu2025.train()
    loss_fun = nn.CrossEntropyLoss()

    if (modelNo[1] == 3) or (modelNo[1] == 4) or (modelNo[1] == 5):
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

path2_train_ICAspace = os.path.join(path1_train, path_ICAspace[dataNo])
path2_train_ICAcloud = os.path.join(path1_train, path_ICAcloud[dataNo])
path2_test_ICAspace = os.path.join(path1_test, path_ICAspace[dataNo])
path2_test_ICAcloud = os.path.join(path1_test, path_ICAcloud[dataNo])

if not os.path.isfile(path3_model_test):
    for rEpoch in range(modelEpoch):
        for rClass in range(nClass):
            for rGroup in range(len(groupAll[dataNo][rClass])):
                path_group = 'group_' + strCode(groupAll[dataNo][rClass][rGroup],nCode)

                path3_train_ICAspace = os.path.join(path2_train_ICAspace, path_group)
                path3_train_ICAcloud = os.path.join(path2_train_ICAcloud, path_group)
                path3_test_ICAspace = os.path.join(path2_test_ICAspace, path_group)
                path3_test_ICAcloud = os.path.join(path2_test_ICAcloud, path_group)

                fileAll = os.listdir(path3_train_ICAspace)
                fileMat = []
                for rSubject in fileAll:
                    if rSubject.endswith('.mat'):
                        fileMat.append(rSubject)

                rSubject_No = 0
                for rSubject in fileMat:
                    path4_train_ICAspace = os.path.join(path3_train_ICAspace, rSubject)
                    path4_train_ICAcloud = os.path.join(path3_train_ICAcloud, rSubject)
                    path4_test_ICAspace = os.path.join(path3_test_ICAspace, rSubject)
                    path4_test_ICAcloud = os.path.join(path3_test_ICAcloud, rSubject)

                    if isTrain == 0:
                        EEG, cloud_power, cloud_bregma, cloud_loss = loadData(path4_test_ICAspace, path4_test_ICAcloud)
                    else:
                        EEG, cloud_power, cloud_bregma, cloud_loss = loadData(path4_train_ICAspace, path4_train_ICAcloud)

                    nSample = EEG.shape[0]
                    temp_label = np.zeros([nSample,nClass])
                    for rSample in range(nSample):
                        temp_label[rSample,rClass] = 1

                    label = torch.tensor(temp_label)

                    if isTrain == 0:

                        tempTrain_EEG = EEG
                        tempTrain_cloud_power = cloud_power
                        tempTrain_cloud_bregma = cloud_bregma
                        tempTrain_cloud_loss = cloud_loss

                        if is_GPU:
                            tempTrain_EEG = tempTrain_EEG.to(gpu_device)
                            tempTrain_cloud_power = tempTrain_cloud_power.to(gpu_device)
                            tempTrain_cloud_bregma = tempTrain_cloud_bregma.to(gpu_device)
                            tempTrain_cloud_loss = tempTrain_cloud_loss.to(gpu_device)

                        testOutput = modelWu2025(tempTrain_EEG, tempTrain_cloud_power, tempTrain_cloud_bregma, tempTrain_cloud_loss)

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
                            tempTrain_cloud_power = cloud_power[first:last]
                            tempTrain_cloud_bregma = cloud_bregma[first:last]
                            tempTrain_cloud_loss = cloud_loss[first:last]
                            tempTrain_label = label[first:last]

                            if is_GPU:
                                tempTrain_EEG = tempTrain_EEG.to(gpu_device)
                                tempTrain_cloud_power = tempTrain_cloud_power.to(gpu_device)
                                tempTrain_cloud_bregma = tempTrain_cloud_bregma.to(gpu_device)
                                tempTrain_cloud_loss = tempTrain_cloud_loss.to(gpu_device)
                                tempTrain_label = tempTrain_label.to(gpu_device)

                            trainOutput = modelWu2025(tempTrain_EEG, tempTrain_cloud_power, tempTrain_cloud_bregma, tempTrain_cloud_loss)

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